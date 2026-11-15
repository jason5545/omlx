#!/usr/bin/env python3
"""Graft the native vision tower from a base checkpoint into a text-only MLX model.

Text conversions of Qwen3.5/3.6/3.8 releases (``mlx_lm.convert``, mlx-my-repo)
drop ``model.visual.*`` and ``vision_config`` even though the architecture is
``Qwen3_5ForConditionalGeneration``, so oMLX loads the model as a plain LLM.
Text-only fine-tunes leave the vision tower untouched (Arete's heretic, Nous
Arete and Qwen3.8-27B carry byte-identical ``model.visual.*``), so the vendor
release's tower can be put back.

What the script does:

1. Resolves the vendor release from the model cards' ``base_model`` chain
   (or ``--base-repo``) and checks its text backbone shape against the target.
2. Range-downloads only the ``model.visual.*`` tensors.
3. Renames them to mlx-vlm's ``vision_tower.*`` and moves the Conv3d patch
   embed to MLX's NDHWC layout (what ``VisionModel.sanitize`` does). The tower
   stays bf16: mlx-vlm only quantizes modules whose ``.scales`` are present.
4. Writes ``model-vision.safetensors``, adds the keys to the index, adds the
   release's ``vision_config`` to ``config.json`` and copies its
   ``preprocessor_config.json`` / ``video_preprocessor_config.json`` when the
   target lacks them.

Originals are backed up as ``*.bak-graft-vision``. Re-running is safe: the
sidecar is rewritten and the config entries are reset.

Usage:
    python scripts/graft_vision_from_base.py \
        --model-dir ~/.lmstudio/models/SolsticeAI/Arete-27B-Unlocked-Uncensored-Heretic-mlx-8Bit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

from graft_mtp_from_base import HF, _range, _safetensors_header, resolve_base_repo

SIDECAR = "model-vision.safetensors"
PREFIX = "model.visual."
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def _array(buf: bytes, dtype: str, shape: tuple[int, ...]) -> mx.array:
    if dtype == "BF16":
        return mx.array(np.frombuffer(buf, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)
    if dtype == "F32":
        return mx.array(np.frombuffer(buf, dtype=np.float32).reshape(shape)).astype(mx.bfloat16)
    raise SystemExit(f"unsupported dtype {dtype}")


def _fetch_json(repo: str, revision: str, name: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{HF}/{repo}/resolve/{revision}/{name}", timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--base-repo", default="auto")
    ap.add_argument("--base-revision", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir: Path = args.model_dir.expanduser()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        print(f"not an MLX sharded checkpoint: {model_dir}", file=sys.stderr)
        return 2
    if args.base_repo == "auto":
        args.base_repo = resolve_base_repo(model_dir)

    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    text_config = config.get("text_config") or {}
    if not str(config.get("model_type")).startswith("qwen3_5"):
        print(f"unsupported target model_type: {config.get('model_type')}", file=sys.stderr)
        return 2

    base_config = _fetch_json(args.base_repo, args.base_revision, "config.json")
    vision_config = (base_config or {}).get("vision_config")
    if not vision_config:
        print(f"{args.base_repo} has no vision_config", file=sys.stderr)
        return 2
    base_text = base_config.get("text_config") or {}
    for key in ("hidden_size", "num_hidden_layers", "intermediate_size", "head_dim"):
        if base_text.get(key) != text_config.get(key):
            print(
                f"backbone mismatch on {key}: base {base_text.get(key)} vs target {text_config.get(key)}",
                file=sys.stderr,
            )
            return 2
    if vision_config.get("out_hidden_size") != text_config.get("hidden_size"):
        print("vision out_hidden_size does not match the target hidden_size", file=sys.stderr)
        return 2
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if base_config.get(key) != config.get(key):
            print(f"token id mismatch on {key}: base {base_config.get(key)} vs target {config.get(key)}", file=sys.stderr)
            return 2

    base_map = _fetch_json(args.base_repo, args.base_revision, "model.safetensors.index.json")["weight_map"]
    visual = {k: v for k, v in base_map.items() if k.startswith(PREFIX)}
    if not visual:
        print(f"{args.base_repo} carries no {PREFIX}* tensors", file=sys.stderr)
        return 2
    by_shard: dict[str, list[str]] = {}
    for name, shard in visual.items():
        by_shard.setdefault(shard, []).append(name)

    in_ch = int(vision_config.get("in_channels", 3))
    out: dict[str, mx.array] = {}
    total_bytes = 0
    t0 = time.time()
    for shard, names in by_shard.items():
        url = f"{HF}/{args.base_repo}/resolve/{args.base_revision}/{shard}"
        n, hdr = _safetensors_header(url)
        # One ranged read per contiguous run keeps 333 tensors to a few requests.
        entries = sorted((hdr[name]["data_offsets"][0], name) for name in names)
        lo = min(hdr[name]["data_offsets"][0] for name in names)
        hi = max(hdr[name]["data_offsets"][1] for name in names)
        span = hi - lo
        covered = sum(hdr[name]["data_offsets"][1] - hdr[name]["data_offsets"][0] for name in names)
        if span <= covered * 1.05:
            blob = _range(url, 8 + n + lo, 8 + n + hi - 1)
            reads = {name: blob[hdr[name]["data_offsets"][0] - lo : hdr[name]["data_offsets"][1] - lo] for _, name in entries}
        else:
            reads = {
                name: _range(url, 8 + n + hdr[name]["data_offsets"][0], 8 + n + hdr[name]["data_offsets"][1] - 1)
                for _, name in entries
            }
        for name, buf in reads.items():
            entry = hdr[name]
            arr = _array(buf, entry["dtype"], tuple(entry["shape"]))
            key = "vision_tower." + name[len(PREFIX) :]
            if key.endswith("patch_embed.proj.weight") and arr.ndim == 5:
                if arr.shape[-1] != in_ch and arr.shape[1] == in_ch:
                    arr = arr.transpose(0, 2, 3, 4, 1)
            out[key] = arr
            total_bytes += len(buf)
    print(f"fetched {len(out)} tensors, {total_bytes/1e9:.2f} GB in {time.time()-t0:.0f}s from {args.base_repo}")
    patch = out.get("vision_tower.patch_embed.proj.weight")
    if patch is not None:
        print(f"patch_embed.proj.weight {tuple(patch.shape)} (NDHWC)")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    mx.eval(*out.values())
    sidecar = model_dir / SIDECAR
    if sidecar.exists():
        sidecar.unlink()
    mx.save_safetensors(str(sidecar), out, metadata={"format": "mlx"})
    print(f"wrote {sidecar} ({sidecar.stat().st_size/1e9:.2f} GB)")

    for path in (index_path, config_path):
        bak = path.with_name(path.name + ".bak-graft-vision")
        if not bak.exists():
            shutil.copy2(path, bak)

    weight_map = index["weight_map"]
    for name in [k for k, v in weight_map.items() if v == SIDECAR]:
        del weight_map[name]
    for name in out:
        weight_map[name] = SIDECAR
    index.setdefault("metadata", {})["total_size"] = sum(
        (model_dir / shard).stat().st_size for shard in set(weight_map.values())
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")

    config["vision_config"] = vision_config
    config["language_model_only"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    for name in PROCESSOR_FILES:
        target = model_dir / name
        if target.exists():
            continue
        data = _fetch_json(args.base_repo, args.base_revision, name)
        if data is not None:
            target.write_text(json.dumps(data, indent=2) + "\n")
            print(f"copied {name} from {args.base_repo}")

    print(f"config: vision_config from {args.base_repo}\nindex: +{len(out)} keys -> {SIDECAR}")
    print("\nNext: reload the model in oMLX; it is now discovered as a VLM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
