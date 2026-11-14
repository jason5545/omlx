#!/usr/bin/env python3
"""Graft the native MTP head from a base checkpoint into a fine-tuned MLX model.

Fine-tune pipelines (heretic/abliteration/merge) frequently drop the
``mtp.*`` tensors of a Qwen3.5/3.6 release, leaving ``mtp_num_hidden_layers``
at 0. oMLX can then never enable Lightning MTP for that model even though the
architecture supports it. The official base checkpoint still carries the head,
and the head is architecturally independent of the tuned backbone weights, so
it can be transplanted.

What the script does:

1. Reads the target model's ``config.json`` / index and verifies the base
   repo's MTP tensor names *and shapes* line up with the target architecture.
2. Range-downloads only the ``mtp.*`` tensors from the base repo's shards
   (safetensors header + byte ranges, no full shard downloads).
3. Converts raw-HF norms to the MLX zero-centering convention (``+1``), and
   quantizes the MTP linears to the target checkpoint's scheme
   (default 8 bit / group 64 affine). ``mtp.fc`` stays bf16, matching the
   runtime ``quant_predicate``.
4. Writes ``mtp.safetensors`` next to the target shards, adds the keys to
   ``model.safetensors.index.json`` and sets
   ``text_config.mtp_num_hidden_layers`` in ``config.json``.

Originals are backed up as ``*.bak-graft-mtp``. Re-running is safe: the
sidecar is rewritten and the config value is reset to 1.

Usage:
    python scripts/graft_mtp_from_base.py \
        --model-dir ~/.lmstudio/models/SolsticeAI/Arete-27B-Unlocked-Uncensored-Heretic-mlx-8Bit \
        --base-repo Qwen/Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

HF = "https://huggingface.co"
# Must match mlx-lm's ``model*.safetensors`` glob so the loader picks it up.
SIDECAR = "model-mtp.safetensors"

# Norms whose HF checkpoints store a zero-centered gamma; MLX stores gamma + 1.
NORM_SUFFIXES = (
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)
# Kept in full precision by oMLX's MTP quant_predicate.
FULL_PRECISION_SUFFIXES = ("mtp.fc.weight",)


def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _safetensors_header(url: str) -> tuple[int, dict]:
    n = struct.unpack("<Q", _range(url, 0, 7))[0]
    return n, json.loads(_range(url, 8, 8 + n - 1))


def _bf16_array(buf: bytes, shape: tuple[int, ...]) -> mx.array:
    raw = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
    return mx.array(raw).view(mx.bfloat16)


def _is_norm(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in NORM_SUFFIXES)


def _is_full_precision(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in FULL_PRECISION_SUFFIXES)


def _local_header(model_dir: Path, shard: str) -> dict:
    with open(model_dir / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _local_tensor(model_dir: Path, shard: str, name: str) -> np.ndarray:
    with open(model_dir / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        start, end = hdr[name]["data_offsets"]
        fh.seek(8 + n + start)
        buf = fh.read(end - start)
    return np.frombuffer(buf, dtype=np.uint16)


def _mean_bf16(raw: np.ndarray) -> float:
    return float(((raw.astype(np.uint32) << 16).view(np.float32)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--base-repo", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--base-revision", default="main")
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir: Path = args.model_dir.expanduser()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        print(f"not an MLX sharded checkpoint: {model_dir}", file=sys.stderr)
        return 2

    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    text_config = config.get("text_config") or {}
    quant = config.get("quantization") or {}
    bits = args.bits or int(quant.get("bits") or 8)
    group_size = args.group_size or int(quant.get("group_size") or 64)
    if quant.get("mode", "affine") != "affine":
        print(f"unsupported quantization mode: {quant.get('mode')}", file=sys.stderr)
        return 2

    model_type = text_config.get("model_type") or config.get("model_type")
    if not str(model_type).startswith("qwen3_5"):
        print(f"unsupported target model_type: {model_type}", file=sys.stderr)
        return 2

    # --- base checkpoint: locate the MTP tensors -------------------------
    idx_url = f"{HF}/{args.base_repo}/resolve/{args.base_revision}/model.safetensors.index.json"
    with urllib.request.urlopen(idx_url, timeout=60) as resp:
        base_map = json.loads(resp.read())["weight_map"]
    mtp = {k: v for k, v in base_map.items() if k.startswith("mtp.")}
    if not mtp:
        print(f"{args.base_repo} carries no mtp.* tensors", file=sys.stderr)
        return 2

    by_shard: dict[str, list[str]] = {}
    for name, shard in mtp.items():
        by_shard.setdefault(shard, []).append(name)

    # --- shape / architecture cross-check against the target backbone ---
    hidden = int(text_config["hidden_size"])
    heads = int(text_config["num_attention_heads"])
    head_dim = int(text_config["head_dim"])
    kv_heads = int(text_config["num_key_value_heads"])
    inter = int(text_config["intermediate_size"])
    gate = bool(text_config.get("attn_output_gate", False))
    expected = {
        "mtp.fc.weight": (hidden, 2 * hidden),
        "mtp.layers.0.self_attn.q_proj.weight": (heads * head_dim * (2 if gate else 1), hidden),
        "mtp.layers.0.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, heads * head_dim),
        "mtp.layers.0.mlp.gate_proj.weight": (inter, hidden),
        "mtp.layers.0.mlp.up_proj.weight": (inter, hidden),
        "mtp.layers.0.mlp.down_proj.weight": (hidden, inter),
    }

    tensors: dict[str, mx.array] = {}
    norm_report: list[tuple[str, float, float]] = []
    fetched = 0
    total = len(mtp)
    for shard, names in by_shard.items():
        url = f"{HF}/{args.base_repo}/resolve/{args.base_revision}/{shard}"
        n, hdr = _safetensors_header(url)
        for name in sorted(names):
            entry = hdr.get(name)
            if entry is None:
                print(f"missing in {shard}: {name}", file=sys.stderr)
                return 2
            shape = tuple(entry["shape"])
            want = expected.get(name)
            if want is not None and shape != want:
                print(
                    f"shape mismatch for {name}: base {shape} vs target {want}",
                    file=sys.stderr,
                )
                return 2
            start, end = entry["data_offsets"]
            t0 = time.time()
            buf = _range(url, 8 + n + start, 8 + n + end - 1)
            arr = _bf16_array(buf, shape)
            dtype = entry["dtype"]
            if dtype != "BF16":
                print(f"unexpected dtype {dtype} for {name}", file=sys.stderr)
                return 2

            if _is_norm(name):
                raw_mean = float(arr.astype(mx.float32).mean().item())
                arr = (arr.astype(mx.float32) + 1.0).astype(mx.bfloat16)
                norm_report.append((name, raw_mean, raw_mean + 1.0))
            tensors["language_model." + name] = arr
            fetched += 1
            mb = len(buf) / 1e6
            print(f"  [{fetched}/{total}] {name:60s} {mb:8.1f} MB  {time.time()-t0:5.1f}s")

    # Sanity: grafted norms must sit near the backbone's own norm scale.
    anchor_name = "language_model.model.norm.weight"
    anchor_shard = index["weight_map"].get(anchor_name)
    if _is_norm("mtp.norm.weight") and anchor_shard:
        raw = _local_tensor(model_dir, anchor_shard, anchor_name)
        if raw.size:
            anchor = _mean_bf16(raw)
            print(f"\nbackbone {anchor_name} mean={anchor:.4f}")
            for name, raw_mean, shifted in norm_report:
                print(f"  {name:55s} raw={raw_mean:+.4f} -> {shifted:+.4f}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    # --- quantize MTP linears, keep norms / fc as bf16 -------------------
    out: dict[str, mx.array] = {}
    for name, arr in tensors.items():
        short = name[len("language_model."):]
        if arr.ndim != 2 or _is_full_precision(short):
            out[name] = arr.astype(mx.bfloat16)
            continue
        w, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        stem = name[: -len(".weight")]
        out[stem + ".weight"] = w
        out[stem + ".scales"] = scales
        out[stem + ".biases"] = biases

    mx.eval(*out.values())

    # --- write sidecar + index + config ----------------------------------
    sidecar = model_dir / SIDECAR
    if sidecar.exists():
        sidecar.unlink()
    mx.save_safetensors(str(sidecar), out)
    print(f"\nwrote {sidecar} ({sidecar.stat().st_size/1e9:.2f} GB, {len(out)} tensors)")

    for path in (index_path, config_path):
        bak = path.with_name(path.name + ".bak-graft-mtp")
        if not bak.exists():
            shutil.copy2(path, bak)

    weight_map = index["weight_map"]
    for name in out:
        weight_map[name] = SIDECAR
    sizes = {}
    for shard in set(weight_map.values()):
        p = model_dir / shard
        if p.exists():
            sizes[shard] = p.stat().st_size
        elif shard == SIDECAR:
            sizes[shard] = sidecar.stat().st_size
    if sizes:
        index.setdefault("metadata", {})["total_size"] = sum(sizes.values())
    index_path.write_text(json.dumps(index, indent=2) + "\n")

    text_config["mtp_num_hidden_layers"] = 1
    text_config.setdefault("mtp_use_dedicated_embeddings", False)
    config["text_config"] = text_config
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(
        f"config: text_config.mtp_num_hidden_layers=1\n"
        f"index: +{len(out)} keys -> {SIDECAR}\n"
        f"quantization: {bits} bit / group {group_size} (mtp.fc kept bf16)"
    )
    print("\nNext: set mtp_enabled=true for this model in ~/.omlx/model_settings.json, then reload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
