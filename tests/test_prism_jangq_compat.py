# SPDX-License-Identifier: Apache-2.0
"""Tests for the JANGQ affine-ternary prism pack compatibility rewrite."""

import json

import numpy as np
import pytest

from omlx.patches.prism_jangq_compat import (
    PRISM_BASE_QUANTIZATION,
    _fold_norm_offsets,
    _zero_centered_norms,
    build_prism_config,
    is_supported_config,
)


def _jangq_config(**overrides):
    config = {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120},
        "vision_config": {"model_type": "qwen3_5"},
        "hadamard": {
            "block_size": 1024,
            "forward_modules": [
                "language_model.lm_head",
                "language_model.model.layers.0.mlp.gate_proj",
            ],
            "inverse_modules": ["language_model.model.embed_tokens"],
        },
        "quantization": {
            "group_size": 128,
            "bits": 2,
            "mode": "affine",
            "language_model.lm_head": {
                "group_size": 128,
                "bits": 2,
                "mode": "affine",
                "storage_bits": 2,
            },
            "language_model.model.embed_tokens": {
                "group_size": 128,
                "bits": 2,
                "mode": "affine",
                "storage_bits": 2,
            },
            "language_model.model.layers.0.mlp.gate_proj": {
                "group_size": 128,
                "bits": 2,
                "mode": "affine",
                "storage_bits": 2,
            },
            "vision_tower.blocks.0.attn.qkv": {
                "group_size": 128,
                "bits": 6,
                "mode": "affine",
                "storage_bits": 6,
            },
        },
    }
    config.update(overrides)
    return config


def _write_pack(directory, config, jang_config=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    if jang_config is not None:
        (directory / "jang_config.json").write_text(json.dumps(jang_config))
    return directory


def test_build_prism_config_rebuilds_the_packed_module_manifest():
    rewritten = build_prism_config(_jangq_config())
    assert rewritten["model_type"] == "prism_hadamard_qwen35"
    assert rewritten["schema_version"] == 2
    assert rewritten["base_model_type"] == "qwen3_5"
    assert rewritten["tensor_namespace"] == "mlx-vlm-qwen3_5"
    assert rewritten["gdn_activation_layout"] == "grouped"

    modules = {entry["path"]: entry for entry in rewritten["modules"]}
    assert set(modules) == {
        "lm_head",
        "model.layers.0.mlp.gate_proj",
        "model.embed_tokens",
    }
    assert modules["lm_head"]["embedding"] is False
    assert modules["model.embed_tokens"]["embedding"] is True
    assert all(entry["block"] == 1024 for entry in modules.values())
    assert all(entry["dtype"] == "float16" for entry in modules.values())


def test_build_prism_config_keeps_only_non_hadamard_quantization():
    rewritten = build_prism_config(_jangq_config())
    quantization = rewritten["quantization"]
    base = {key: quantization[key] for key in PRISM_BASE_QUANTIZATION}
    assert base == PRISM_BASE_QUANTIZATION
    assert quantization["vision_tower.blocks.0.attn.qkv"] == {
        "group_size": 128,
        "bits": 6,
        "mode": "affine",
    }
    # Hadamard-owned modules must not carry a per-module entry: mlx-vlm would
    # then try to quantize the prism module that has no to_quantized().
    assert "language_model.lm_head" not in quantization
    assert "language_model.model.embed_tokens" not in quantization
    assert "language_model.model.layers.0.mlp.gate_proj" not in quantization
    assert "storage_bits" not in json.dumps(quantization)
    assert "quantization_config" not in rewritten


def test_build_prism_config_rejects_foreign_bit_widths():
    config = _jangq_config()
    config["quantization"]["bits"] = 4
    with pytest.raises(ValueError, match="2-bit affine"):
        build_prism_config(config)


def test_build_prism_config_requires_hadamard_metadata():
    with pytest.raises(ValueError, match="hadamard"):
        build_prism_config({"model_type": "qwen3_5", "quantization": {}})


def test_is_supported_config_detects_jangq_and_skips_native_prism(tmp_path):
    jangq = _write_pack(tmp_path / "jangq", _jangq_config())
    assert is_supported_config(jangq) is True

    native = _jangq_config(model_type="prism_hadamard_qwen35", modules=[{"path": "lm_head"}])
    native["quantization"] = dict(PRISM_BASE_QUANTIZATION)
    assert is_supported_config(_write_pack(tmp_path / "prism", native)) is False

    # A prism pack that still carries storage_bits is rewritten too.
    stale = _jangq_config(model_type="prism_hadamard_qwen35")
    assert is_supported_config(_write_pack(tmp_path / "stale", stale)) is True

    assert is_supported_config(_write_pack(tmp_path / "plain", {"model_type": "qwen3_5"})) is False


def test_zero_centered_norms_follows_the_pack_layout():
    declared = {"layout": {"language_norms": "zero-centered-runtime-plus-one", "shifted_norm_count": 161}}
    assert _zero_centered_norms({}, declared) is True
    assert _zero_centered_norms({}, {"layout": {"shifted_norm_count": 161}}) is True
    assert _zero_centered_norms({}, {"layout": {"language_norms": "mlx-pre-shifted"}}) is False
    assert _zero_centered_norms({}, {}) is False
    assert _zero_centered_norms({"jang_norms_pre_shifted": True}, declared) is False


def test_fold_norm_offsets_touches_only_zero_centered_language_norms():
    weights = {
        "language_model.model.norm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.0.input_layernorm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.0.post_attention_layernorm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.3.q_norm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.3.k_norm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.0.linear_attn.norm.weight": np.zeros(4, dtype=np.float32),
        "language_model.model.layers.0.linear_attn.conv1d.weight": np.zeros((2, 4, 1), dtype=np.float32),
        "vision_tower.merger.norm.weight": np.zeros(4, dtype=np.float32),
    }
    folded = _fold_norm_offsets(weights)

    shifted = (
        "language_model.model.norm.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.layers.3.q_norm.weight",
        "language_model.model.layers.3.k_norm.weight",
    )
    for key in shifted:
        assert folded[key][0] == 1.0, key

    # RMSNormGated must never be shifted, and neither must vision norms
    # (mlx-vlm folds those itself) or non-norm tensors.
    for key in (
        "language_model.model.layers.0.linear_attn.norm.weight",
        "vision_tower.merger.norm.weight",
    ):
        assert folded[key][0] == 0.0, key
    conv = folded["language_model.model.layers.0.linear_attn.conv1d.weight"]
    assert conv.shape == (2, 4, 1) and float(conv[0][0][0]) == 0.0

    assert _fold_norm_offsets(weights, skip=True) is weights

