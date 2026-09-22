# SPDX-License-Identifier: Apache-2.0
"""Load JANG mixed-precision bundles through the jang-tools runtime.

A JANG bundle stores per-tensor bit widths (attention 6-8 bit, routed experts
2-4 bit) and keeps them in a "jang_config.json" sidecar. mlx-lm and mlx-vlm
know nothing about that sidecar: they read the single top-level
"quantization" entry, so a JANG bundle either fails to load or loads at the
wrong precision. omlx sends these directories to
"jang_tools.loader.load_jang_model" / "load_jang_vlm_model" instead, then
hands the result to the regular "BatchedEngine" / "VLMBatchedEngine" so
prefix cache, vision, tool calling, and streaming keep working.

PrismML's ternary Bonsai packs -- including dealignai's "*-Ternary-JANG"
repacks -- also ship a "jang_config.json" but need mlx-vlm's prism Hadamard
runtime rather than this one. They are excluded here; see
"omlx.patches.prism_jangq_compat".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JANG_CONFIG_NAMES = (
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
)

# Mirrors jang_tools.loader.JANG_FORMAT_VALUES: the marker the jang runtime
# itself requires before it will load a directory.
JANG_FORMAT_VALUES = ("jang", "jjqf", "mxq")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_jang_config(model_dir: str | Path) -> dict:
    """Return the bundle's JANG sidecar, or an empty dict when it has none."""
    model_dir = Path(model_dir)
    for name in JANG_CONFIG_NAMES:
        path = model_dir / name
        if path.exists():
            return _read_json(path)
    return {}


def has_jang_config(model_dir: str | Path) -> bool:
    """True when the directory carries a JANG sidecar file."""
    model_dir = Path(model_dir)
    return any((model_dir / name).exists() for name in JANG_CONFIG_NAMES)


def is_prism_pack(model_dir: str | Path) -> bool:
    """True for PrismML ternary bundles, which use mlx-vlm's prism runtime.

    Covers both Prism's own schema-2 packs (model_type prism_hadamard_qwen35
    with a modules manifest) and JANGQ repacks that keep the Hadamard metadata
    but rename the model type and fold storage_bits into the quantization map.
    """
    from .prism_jangq_compat import PRISM_MODEL_TYPE, is_supported_config

    model_dir = Path(model_dir)
    if is_supported_config(model_dir):
        return True
    return _read_json(model_dir / "config.json").get("model_type") == PRISM_MODEL_TYPE


def is_jang_pack(model_dir: str | Path) -> bool:
    """True when the jang runtime should load this directory.

    Requires the sidecar's own format marker. Bundles that merely ship vMLX
    metadata next to MXFP8 or affine weights (dealignai's Ornith-1.5 MXFP8
    leaves format unset) keep loading through mlx-vlm's stock path.
    """
    config = read_jang_config(model_dir)
    if str(config.get("format", "")).lower() not in JANG_FORMAT_VALUES:
        return False
    return not is_prism_pack(model_dir)


def jang_has_vision(model_dir: str | Path) -> bool | None:
    """Return architecture.has_vision from the JANG sidecar when declared."""
    architecture = read_jang_config(model_dir).get("architecture")
    if not isinstance(architecture, dict):
        return None
    has_vision = architecture.get("has_vision")
    return has_vision if isinstance(has_vision, bool) else None


def is_jang_v2(model_dir: str | Path) -> bool:
    """True when the bundle uses the JANG v2 format (instant mmap load)."""
    version = read_jang_config(model_dir).get("format_version")
    try:
        return int(str(version).split(".")[0]) >= 2
    except (TypeError, ValueError):
        return False


def _model_config_dict(model: Any) -> dict:
    """Read the fields the JANG fixups need from a dict or object config."""
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return {
        attr: getattr(config, attr)
        for attr in (
            "architectures",
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "hidden_size",
            "model_type",
            "text_config",
        )
        if hasattr(config, attr)
    }


def _text_config(config: dict) -> dict:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _is_nemotron_h(model: Any) -> bool:
    """True for Nemotron-H architectures, whose JANG v1 gates need repair."""
    architectures = _model_config_dict(model).get("architectures") or []
    return any("Nemotron" in arch for arch in architectures)


def _needs_bfloat16(model: Any) -> bool:
    """True for 512+ expert models, where fp16 expert accumulation overflows."""
    config = _text_config(_model_config_dict(model))
    n_experts = config.get(
        "num_local_experts",
        config.get("num_experts", config.get("n_routed_experts", 0)),
    )
    hidden_size = config.get("hidden_size", 0)
    try:
        return int(n_experts) >= 512 and int(hidden_size) >= 4096
    except (TypeError, ValueError):
        return False


def _fix_nemotron_h_weights(model: Any, model_dir: Path) -> None:
    """Dequantize Nemotron-H mixer gate weights after a JANG v1 load.

    JANG stores those gates as quantized uint32 while mlx-lm's skeleton
    declares a plain nn.Linear. Loading with strict=False therefore keeps
    gate.weight but drops gate.scales / gate.biases. Read the pair back from
    the bundle's own shards and dequantize.
    """
    import mlx.core as mx

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        index_path = model_dir / "consolidated.safetensors.index.json"
    if not index_path.exists():
        logger.warning("Nemotron-H: no safetensors index found, skipping gate fixup")
        return

    weight_map = _read_json(index_path).get("weight_map", {})

    # Group the tensors by gate prefix (e.g. "backbone.layers.0.mixer.gate").
    gate_parts: dict[str, dict[str, str]] = {}
    for key, shard in weight_map.items():
        if ".gate." not in key:
            continue
        split_at = key.index(".gate.") + len(".gate")
        gate_parts.setdefault(key[:split_at], {})[key[split_at:].lstrip(".")] = shard

    if not gate_parts:
        logger.info("Nemotron-H: no gate weights in the index, skipping")
        return

    target_dtype = mx.bfloat16 if _needs_bfloat16(model) else mx.float16
    shard_cache: dict[str, dict[str, Any]] = {}
    dequantized_weights: list[tuple[str, Any]] = []

    for prefix, parts in gate_parts.items():
        if "weight" not in parts or "scales" not in parts or "biases" not in parts:
            # Not a quantized gate; mlx-lm already loaded it correctly.
            continue

        tensors: dict[str, Any] = {}
        for suffix in ("weight", "scales", "biases"):
            shard_file = parts[suffix]
            if shard_file not in shard_cache:
                shard_cache[shard_file] = mx.load(str(model_dir / shard_file))
            tensors[suffix] = shard_cache[shard_file][f"{prefix}.{suffix}"]

        gate_weight = tensors["weight"]
        scales = tensors["scales"]
        biases = tensors["biases"]

        # The sidecar records the gate's bit width but not the packed width, so
        # probe the widths JANG assigns to its critical tensors.
        dequantized = None
        for bits in (8, 6, 4, 3, 2):
            real_cols = gate_weight.shape[-1] * (32 // bits)
            group_size = real_cols // scales.shape[-1]
            if group_size > 0 and group_size * scales.shape[-1] == real_cols:
                dequantized = mx.dequantize(
                    gate_weight, scales, biases, group_size, bits
                ).astype(target_dtype)
                logger.info(
                    "Nemotron-H: dequantized %s.weight (%d-bit, group_size=%d) "
                    "%s -> %s",
                    prefix,
                    bits,
                    group_size,
                    gate_weight.shape,
                    dequantized.shape,
                )
                break

        if dequantized is None:
            logger.warning(
                "Nemotron-H: could not dequantize %s (weight=%s, scales=%s)",
                prefix,
                gate_weight.shape,
                scales.shape,
            )
            continue
        dequantized_weights.append((f"{prefix}.weight", dequantized))

    shard_cache.clear()

    if not dequantized_weights:
        logger.info("Nemotron-H: no gate weights needed dequantization")
        return
    model.load_weights(dequantized_weights, strict=False)
    logger.info(
        "Nemotron-H: dequantized %d gate weights to %s",
        len(dequantized_weights),
        target_dtype,
    )


def _manual_leaf_walk(module: Any):
    """Yield modules that have no children, descending through containers."""
    children = getattr(module, "children", None)
    if not callable(children):
        yield module
        return
    try:
        items = list(children().values())
    except Exception:
        items = []
    if not items:
        yield module
        return
    for child in items:
        if isinstance(child, (list, tuple)):
            for item in child:
                yield from _manual_leaf_walk(item)
        elif isinstance(child, dict):
            for item in child.values():
                yield from _manual_leaf_walk(item)
        else:
            yield from _manual_leaf_walk(child)


def _iter_modules(module: Any):
    """Yield every module in the tree, preferring mlx's own walker."""
    modules = getattr(module, "modules", None)
    if callable(modules):
        try:
            return iter(list(modules()))
        except Exception:
            pass
    return iter(list(_manual_leaf_walk(module)))


def _repair_hadamard_signs(model: Any) -> int:
    """Re-key the Hadamard signs jang records for non-32-divisor bit widths.

    jang derives the rotation width as packed_columns * (32 // bits), which
    only equals the true input width when bits divides 32. A 6-bit projection
    packed into 12 uint32 columns is 64 wide, not 60, so the signs jang caches
    fail to broadcast on the first forward pass. Regenerate them from the same
    seed jang itself uses (42) at the width the activations actually have.
    """
    import mlx.nn as nn
    from jang_tools.turboquant.rotation import generate_random_signs

    repaired = 0
    for module in _iter_modules(model):
        signs = getattr(module, "_hadamard_signs", None)
        if signs is None or not isinstance(module, nn.QuantizedLinear):
            continue
        bits = getattr(module, "bits", 0)
        packed_columns = module.weight.shape[-1]
        if not bits or (packed_columns * 32) % bits:
            continue
        in_dim = packed_columns * 32 // bits
        if signs.shape[0] == in_dim:
            continue
        object.__setattr__(
            module, "_hadamard_signs", generate_random_signs(in_dim, seed=42)
        )
        repaired += 1
    return repaired


def load_jang(
    model_dir: str | Path,
    *,
    is_vlm: bool,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a JANG bundle and return (model, processor_or_tokenizer).

    trust_remote_code is accepted for signature parity with the other loaders;
    jang-tools instantiates the architecture classes itself and exposes no
    equivalent switch.
    """
    from ..exceptions import JANGDependencyError, JANGLoadError

    path = Path(model_dir)
    try:
        from jang_tools.loader import load_jang_model, load_jang_vlm_model
    except ImportError as exc:
        raise JANGDependencyError(
            f"{path} is a JANG bundle but the jang runtime is missing; "
            f'install it with: pip install "jang[mlx]" ({exc})',
            model_name=str(path),
        ) from exc

    try:
        if is_vlm:
            model, processor = load_jang_vlm_model(str(path))
        else:
            model, tokenizer = load_jang_model(str(path))
            processor = tokenizer
    except Exception as exc:
        raise JANGLoadError(
            f"Failed to load JANG bundle {path}: {exc}",
            model_name=str(path),
        ) from exc

    # jang-tools repairs Nemotron-H gates itself on the v2 path.
    if not is_jang_v2(path) and _is_nemotron_h(model):
        logger.info("Nemotron-H JANG bundle: dequantizing mixer gate weights")
        _fix_nemotron_h_weights(model, path)

    quantization = read_jang_config(path).get("quantization")
    if isinstance(quantization, dict) and quantization.get("hadamard_rotation"):
        repaired = _repair_hadamard_signs(model)
        if repaired:
            logger.info(
                "Repaired Hadamard sign widths for %d quantized layers", repaired
            )

    if _needs_bfloat16(model):
        import mlx.core as mx

        logger.info(
            "JANG bundle has 512+ experts and hidden>=4096; switching to bfloat16"
        )
        model.set_dtype(mx.bfloat16)

    logger.info("Loaded JANG bundle: %s (vlm=%s)", path, is_vlm)
    return model, processor
