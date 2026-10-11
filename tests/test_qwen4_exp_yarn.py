# SPDX-License-Identifier: Apache-2.0
"""YaRN RoPE for qwen4_exp (Qwen3.8-Flash-Next long-context recipe).

The model card extends the 262,144-token window by adding ``rope_type: yarn``
with ``factor`` and ``original_max_position_embeddings`` to
``text_config.rope_parameters``. The frequencies must match Transformers'
``_compute_yarn_parameters``, the attention factor must reach every MRoPE
path (folded into the q/k and indexer norms), the default rope must stay
untouched, and discovery must report ``original * factor`` as the window.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.model_discovery import _read_model_context_length
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qwen4_exp = importlib.import_module("mlx_vlm.models.qwen4_exp")
rope_utils = importlib.import_module("mlx_vlm.models.rope_utils")

_DEFAULT_ROPE = {
    "type": "default",
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10_000_000,
}


def _yarn_rope(factor: float) -> dict:
    return {
        **_DEFAULT_ROPE,
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": 262144,
    }


_CONFIG = qwen4_exp.TextConfig(
    model_type="qwen4_exp_text",
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=256,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=3,
    num_experts=4,
    num_experts_per_tok=2,
    shared_expert_intermediate_size=16,
    moe_intermediate_size=16,
    rms_norm_eps=1e-6,
    vocab_size=64,
    max_position_embeddings=262144,
    hc_count=2,
    hc_lowrank=8,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    ple_layer_ids=[],
    ple_embed_dim=32,
    ple_conv_kernel_size=3,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=17,
    make_ngram_vocab_size_divisible_by=4,
    split_ngram_parts=4,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    eos_token_id=1,
    rope_parameters=dict(_DEFAULT_ROPE),
)


def _config(rope: dict):
    return replace(_CONFIG, rope_parameters=dict(rope))


def _norms(attn):
    return (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm)


@pytest.mark.parametrize("factor", [1.5, 2.0, 4.0])
def test_yarn_inv_freq_matches_transformers(factor):
    torch = pytest.importorskip("torch")
    hf_rope = pytest.importorskip("transformers.modeling_rope_utils")
    rope = _yarn_rope(factor)
    hf_params = {k: v for k, v in rope.items() if k != "type"}
    hf_params["rope_type"] = "yarn"
    hf_config = SimpleNamespace(
        rope_parameters=hf_params,
        standardize_rope_params=lambda: None,
        head_dim=256,
        hidden_size=2560,
        num_attention_heads=24,
        max_position_embeddings=262144,
    )
    expected, expected_factor = hf_rope._compute_yarn_parameters(hf_config, torch.device("cpu"))

    inv_freq, attention_factor = language._yarn_rope(_config(rope), 64)

    assert attention_factor == pytest.approx(expected_factor, rel=1e-12)
    assert attention_factor == pytest.approx(0.1 * math.log(factor) + 1.0)
    assert mx.allclose(inv_freq, mx.array(expected.numpy()), rtol=1e-6, atol=0).item()


def test_default_rope_is_untouched():
    attn = language.Qwen4ExpAttention(_config(_DEFAULT_ROPE))
    rotary = attn.rotary_emb
    assert mx.array_equal(rotary.inv_freq, rope_utils.compute_inv_freq(64, 10_000_000)).item()
    for norm in _norms(attn):
        assert norm._rope_scale is None
        assert mx.array_equal(norm._scale(), 1.0 + norm.weight.astype(mx.float32)).item()


def test_yarn_attention_uses_yarn_freqs_and_scaled_norms():
    attn = language.Qwen4ExpAttention(_config(_yarn_rope(2.0)))
    inv_freq, attention_factor = language._yarn_rope(_config(_yarn_rope(2.0)), 64)
    assert mx.array_equal(attn.rotary_emb.inv_freq, inv_freq).item()
    # The indexer rotates with the attention's own rotary.
    assert attn.indexer.rotary_emb is attn.rotary_emb
    for norm in _norms(attn):
        dim = norm.weight.shape[0]
        expected = mx.concatenate(
            [mx.full((64,), attention_factor), mx.ones((dim - 64,))]
        ) * (1.0 + norm.weight.astype(mx.float32))
        assert mx.allclose(norm._scale(), expected, rtol=1e-7, atol=0).item()


@pytest.mark.parametrize("fused", [False, True])
def test_folded_factor_equals_scaled_cos_sin(fused):
    """Norm-folded attention factor == Transformers' ``cos * factor`` rope."""
    rope = _yarn_rope(2.0)
    attn = language.Qwen4ExpAttention(_config(rope))
    mx.random.seed(0)
    for norm in _norms(attn):
        norm.weight = mx.random.normal(norm.weight.shape) * 0.1
    rotary = attn.rotary_emb
    if fused and not rotary.fused_apply:
        pytest.skip("fused MRoPE kernel unavailable")
    _, attention_factor = language._yarn_rope(_config(rope), 64)

    length = 16
    seq = mx.arange(300_000, 300_000 + length, dtype=mx.int32)[None]
    positions = mx.broadcast_to(seq[None], (3, 1, length))
    q = mx.random.normal((1, 4, length, 256))
    k = mx.random.normal((1, 2, length, 256))
    iq = mx.random.normal((1, 4, length, 128))

    saved = rotary.fused_apply
    rotary.fused_apply = fused
    try:
        out_q, out_k = rotary.apply_rotary(attn.q_norm(q), attn.k_norm(k), positions)
        out_iq, _ = rotary.apply_rotary(
            attn.indexer.q_layernorm(iq), attn.indexer.q_layernorm(iq), positions
        )
    finally:
        rotary.fused_apply = saved

    def plain(norm, x):
        return mx.fast.rms_norm(x, 1.0 + norm.weight.astype(mx.float32), norm.eps)

    def reference(x_q, x_k):
        cos, sin = rotary(x_k, positions)
        return rope_utils.apply_multimodal_rotary_pos_emb(
            x_q,
            x_k,
            cos * attention_factor,
            sin * attention_factor,
            mrope_section=rotary.mrope_section,
            style=rotary.style,
        )

    ref_q, ref_k = reference(plain(attn.q_norm, q), plain(attn.k_norm, k))
    ref_iq, _ = reference(
        plain(attn.indexer.q_layernorm, iq), plain(attn.indexer.q_layernorm, iq)
    )
    for got, want in ((out_q, ref_q), (out_k, ref_k), (out_iq, ref_iq)):
        assert mx.allclose(got, want, rtol=1e-5, atol=1e-5).item()


def _write_config(tmp_path, rope: dict, model_type: str = "qwen4_exp"):
    config = {
        "model_type": model_type,
        "text_config": {"max_position_embeddings": 262144, "rope_parameters": rope},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))


def test_discovery_reports_yarn_window(tmp_path):
    rope = {k: v for k, v in _yarn_rope(2.0).items() if k != "type"}
    rope["rope_type"] = "yarn"
    _write_config(tmp_path, rope)
    assert _read_model_context_length(tmp_path) == 524288


def test_discovery_yarn_factor_one_and_a_half(tmp_path):
    _write_config(tmp_path, _yarn_rope(1.5))
    assert _read_model_context_length(tmp_path) == 393216


def test_discovery_default_rope_keeps_native_window(tmp_path):
    _write_config(tmp_path, _DEFAULT_ROPE)
    assert _read_model_context_length(tmp_path) == 262144


def test_discovery_ignores_yarn_on_other_architectures(tmp_path):
    # Other loaders may drop the field silently; never advertise their window.
    _write_config(tmp_path, _yarn_rope(2.0), model_type="qwen3_5_moe")
    assert _read_model_context_length(tmp_path) == 262144
