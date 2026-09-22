# SPDX-License-Identifier: Apache-2.0
"""Run JANGQ affine-ternary Bonsai packs on mlx-vlm's prism Hadamard runtime.

dealignai's Bonsai-2-27B-*-Ternary-JANG bundle repacks PrismML's ternary Bonsai
pack (prism-ml/Ternary-Bonsai-2-27B-mlx-2bit). The weights are a lossless
repack -- packed weight, scales, biases and signs tensors are byte-identical to
Prism's pack -- but the pack describes itself differently, so stock mlx-vlm
cannot load it:

* config.json declares model_type qwen3_5 although the language model is stored
  in Prism's blockwise Hadamard basis. mlx-vlm then builds the plain Qwen3.5
  model, never applies the input rotation, and silently emits garbage (the
  pack's own README warns about exactly this).
* Every per-module quantization entry carries a storage_bits key. MLX's
  to_quantized() accepts only bits/group_size/mode, so nn.quantize dies with
  "unexpected keyword argument 'storage_bits'" before a single weight is read.
* Language-model RMSNorm tensors are stored zero-centered (gamma - 1); the pack
  stamps this as layout.language_norms = "zero-centered-runtime-plus-one" plus
  shifted_norm_count. mlx-vlm folds the +1.0 only for keys outside
  language_model., so all 161 input_layernorm / post_attention_layernorm /
  q_norm / k_norm / model.norm tensors stay near zero and the model degenerates
  into repeated tokens.
* The vision tower is a JANGQ addition quantized at 6-bit/group-128 while the
  language model is 2-bit. Prism's own pack is text-only, so its schema-2
  ModelConfig demands one bare quantization dict and rejects the per-module
  entries the vision tower needs.

The fix is a load-time rewrite, scoped to a single load:

1. rebuild the config in the schema-2 prism_hadamard_qwen35 shape mlx-vlm
   already implements, regenerating the packed-module manifest from the pack's
   hadamard.forward_modules / inverse_modules metadata;
2. drop storage_bits and keep per-module quantization entries only for modules
   the Hadamard manifest does not own (the 6-bit vision tower);
3. relax Prism's exact-quantization guard to the base keys so those entries
   survive;
4. fold +1.0 into the zero-centered language-model RMSNorm tensors.

mlx_vlm.utils.load_config and the Prism Model/ModelConfig hooks are restored in
a finally block, so nothing leaks into other loads.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRISM_MODEL_TYPE = "prism_hadamard_qwen35"
PRISM_BASE_MODEL_TYPE = "qwen3_5"
PRISM_TENSOR_NAMESPACE = "mlx-vlm-qwen3_5"
PRISM_GDN_ACTIVATION_LAYOUT = "grouped"
PRISM_SCHEMA_VERSION = 2
PRISM_BASE_QUANTIZATION = {"bits": 2, "group_size": 128, "mode": "affine"}

# Mirrors jang_tools.zero_centered_norms.NORM_SUFFIXES, which mirrors mlx_lm's
# qwen3_5 sanitize. RMSNormGated (linear_attn.norm) is deliberately absent: it is
# never zero-centered and shifting it corrupts the GDN stack.
ZERO_CENTERED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

_JANG_CONFIG_NAMES = ("jang_config.json", "jjqf_config.json", "jang_cfg.json")
_QUANTIZATION_KEYS = ("bits", "group_size", "mode")
_LANGUAGE_PREFIX = "language_model."


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_jang_config(model_dir: Path) -> dict:
    for name in _JANG_CONFIG_NAMES:
        path = model_dir / name
        if path.exists():
            return _read_json(path)
    return {}


def _has_storage_bits(quantization: Any) -> bool:
    if not isinstance(quantization, dict):
        return False
    if "storage_bits" in quantization:
        return True
    return any(
        isinstance(value, dict) and "storage_bits" in value
        for value in quantization.values()
    )


def is_supported_config(model_dir: str | Path) -> bool:
    """Whether model_dir is a JANGQ Hadamard pack that needs the rewrite.

    Prism's own schema-2 packs declare model_type prism_hadamard_qwen35 with a
    modules manifest and are left alone; a JANGQ repack keeps Prism's hadamard
    metadata but renames the model type and folds storage_bits into the
    quantization map.
    """
    model_dir = Path(model_dir)
    config = _read_json(model_dir / "config.json")
    hadamard = config.get("hadamard")
    if not isinstance(hadamard, dict) or not hadamard.get("forward_modules"):
        return False
    if config.get("model_type") == PRISM_MODEL_TYPE:
        return _has_storage_bits(config.get("quantization"))
    return True


def _zero_centered_norms(config: dict, jang_config: dict) -> bool:
    """Whether the language-model RMSNorm tensors still need the +1.0 fold."""
    if config.get("jang_norms_pre_shifted"):
        return False
    layout = jang_config.get("layout")
    if not isinstance(layout, dict):
        return False
    if layout.get("language_norms") == "zero-centered-runtime-plus-one":
        return True
    try:
        return int(layout.get("shifted_norm_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def build_prism_config(config: dict) -> dict:
    """Rewrite a JANGQ pack config into mlx-vlm's prism schema-2 shape."""
    hadamard = config.get("hadamard")
    if not isinstance(hadamard, dict):
        raise ValueError("JANGQ prism pack is missing its hadamard metadata")
    forward = list(hadamard.get("forward_modules") or [])
    inverse = list(hadamard.get("inverse_modules") or [])
    if not forward:
        raise ValueError("JANGQ prism pack declares no Hadamard forward modules")
    block = int(hadamard.get("block_size") or 1024)
    owned = set(forward) | set(inverse)

    def packed_path(path: str) -> str:
        if path.startswith(_LANGUAGE_PREFIX):
            return path[len(_LANGUAGE_PREFIX) :]
        return path

    modules = [
        {
            "path": packed_path(path),
            "block": block,
            "embedding": False,
            "dtype": "float16",
        }
        for path in forward
    ]
    modules += [
        {
            "path": packed_path(path),
            "block": block,
            "embedding": True,
            "dtype": "float16",
        }
        for path in inverse
    ]

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("JANGQ prism pack is missing its quantization map")
    rewritten = {
        key: quantization[key] for key in _QUANTIZATION_KEYS if key in quantization
    }
    if rewritten != PRISM_BASE_QUANTIZATION:
        raise ValueError(
            "JANGQ prism packs require 2-bit affine weights, group size 128 "
            f"(found {rewritten})"
        )
    kept = 0
    for key, value in quantization.items():
        # Hadamard-owned modules are built by the prism Model itself; a
        # per-module entry for them makes mlx-vlm try to quantize a module that
        # has no to_quantized() and abort the load.
        if not isinstance(value, dict) or key in owned:
            continue
        per_module = {k: value[k] for k in _QUANTIZATION_KEYS if k in value}
        if per_module:
            rewritten[key] = per_module
            kept += 1
    if not kept:
        logger.warning(
            "JANGQ prism pack declares no non-Hadamard quantization entries; "
            "any extra (vision) tower would load unquantized"
        )

    rewritten_config = dict(config)
    rewritten_config.update(
        {
            "model_type": PRISM_MODEL_TYPE,
            "schema_version": PRISM_SCHEMA_VERSION,
            "base_model_type": PRISM_BASE_MODEL_TYPE,
            "tensor_namespace": PRISM_TENSOR_NAMESPACE,
            "gdn_activation_layout": PRISM_GDN_ACTIVATION_LAYOUT,
            "modules": modules,
            "quantization": rewritten,
        }
    )
    # The prism ModelConfig requires the bare base dict; per-module entries
    # travel through quantization and are read by mlx-vlm's class predicate.
    rewritten_config.pop("quantization_config", None)
    return rewritten_config


def _fold_norm_offsets(weights: dict, *, skip: bool = False) -> dict:
    if skip or not isinstance(weights, dict):
        return weights
    folded = 0
    out = {}
    for key, value in weights.items():
        if (
            key.startswith(_LANGUAGE_PREFIX)
            and getattr(value, "ndim", None) == 1
            and any(key.endswith(suffix) for suffix in ZERO_CENTERED_NORM_SUFFIXES)
        ):
            value = value + 1.0
            folded += 1
        out[key] = value
    if folded:
        logger.info(
            "JANGQ prism pack: folded +1.0 into %d zero-centered RMSNorm tensors",
            folded,
        )
    return out


def _mlx_vlm_would_shift_norms(weights: dict) -> bool:
    """Whether mlx-vlm's own sanitize already folds the +1.0 in.

    mlx_lm/mlx-vlm gate the fold on MTP keys or HF-layout conv1d weights. When
    the gate fires, mlx-vlm shifts language_model. norms itself and folding ours
    on top would double-shift.
    """
    try:
        from mlx_vlm.models.qwen3_5.qwen3_5 import should_shift_norm_weights
    except ImportError:
        return False
    try:
        return bool(should_shift_norm_weights(weights))
    except Exception:  # pragma: no cover - the gate only reads key metadata
        return False


def _relaxed_post_init(qwen35_model_config):
    """Prism's schema validation with the exact-quantization check relaxed.

    The prism guard demands quantization == {"bits": 2, "group_size": 128,
    "mode": "affine"}. A JANGQ pack also carries per-module entries for its
    6-bit vision tower, which mlx-vlm needs to keep. Validate the base keys and
    leave the full map in place.
    """

    def post_init(self):
        qwen35_model_config.__post_init__(self)
        if (
            self.schema_version != PRISM_SCHEMA_VERSION
            or self.base_model_type != PRISM_BASE_MODEL_TYPE
        ):
            raise ValueError("Only schema 2 Qwen3.5 Hadamard packs are supported")
        if self.tensor_namespace != PRISM_TENSOR_NAMESPACE:
            raise ValueError("Unsupported Hadamard tensor namespace")
        if self.gdn_activation_layout != PRISM_GDN_ACTIVATION_LAYOUT:
            raise ValueError("Hadamard packs require grouped GDN activations")
        quantization = self.quantization if isinstance(self.quantization, dict) else {}
        base = {key: quantization.get(key) for key in _QUANTIZATION_KEYS}
        if base != PRISM_BASE_QUANTIZATION:
            raise ValueError(
                "Hadamard packs require 2-bit affine weights, group size 128 "
                f"(found {base})"
            )
        if not self.modules:
            raise ValueError("Hadamard pack is missing its packed module manifest")

    return post_init


def load(model_name: str | Path, **kwargs) -> tuple[Any, Any]:
    """Load a JANGQ Hadamard pack through mlx-vlm's prism runtime."""
    import mlx_vlm.utils as vlm_utils
    from mlx_vlm.models.prism_hadamard_qwen35 import config as prism_config_module
    from mlx_vlm.models.prism_hadamard_qwen35 import (
        prism_hadamard_qwen35 as prism_model_module,
    )
    from mlx_vlm.models.qwen3_5.config import ModelConfig as Qwen35ModelConfig

    model_path = Path(model_name)
    raw_config = _read_json(model_path / "config.json")
    jang_config = _read_jang_config(model_path)
    prism_config = build_prism_config(raw_config)
    fold_norms = _zero_centered_norms(raw_config, jang_config)

    target = model_path.resolve()
    original_load_config = vlm_utils.load_config
    original_post_init = prism_config_module.ModelConfig.__post_init__
    original_sanitize = prism_model_module.Model.sanitize

    def load_config_normalized(path, *args, **kwargs):
        loaded = original_load_config(path, *args, **kwargs)
        try:
            is_target = Path(path).resolve() == target
        except (OSError, TypeError, ValueError):
            is_target = False
        if is_target:
            return copy.deepcopy(prism_config)
        return loaded

    def sanitize_with_norm_fold(self, weights, *args, **kwargs):
        sanitized = original_sanitize(self, weights, *args, **kwargs)
        if not fold_norms:
            return sanitized
        return _fold_norm_offsets(sanitized, skip=_mlx_vlm_would_shift_norms(weights))

    prism_config_module.ModelConfig.__post_init__ = _relaxed_post_init(
        Qwen35ModelConfig
    )
    prism_model_module.Model.sanitize = sanitize_with_norm_fold
    vlm_utils.load_config = load_config_normalized
    logger.info(
        "Loading JANGQ affine-ternary prism pack %s through the prism Hadamard "
        "runtime (%d packed modules, %d extra quantization entries, norm fold=%s)",
        model_path,
        len(prism_config["modules"]),
        len(prism_config["quantization"]) - len(_QUANTIZATION_KEYS),
        fold_norms,
    )
    try:
        return vlm_utils.load(model_path, **kwargs)
    finally:
        vlm_utils.load_config = original_load_config
        prism_model_module.Model.sanitize = original_sanitize
        prism_config_module.ModelConfig.__post_init__ = original_post_init

