# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5-MoE force-sanitize load shim (mixed shard metadata).

Regression: Ornith-1.5 MXFP8 ships shards with mixed ``format`` metadata
(``pt`` / ``mlx`` / missing). mlx-vlm reads the first unsorted ``glob()``
shard to decide ``is_mlx_format``; when an ``mlx`` shard sorts first,
``Model.sanitize`` is skipped, per-expert MTP MoE tensors never stack into
``switch_mlp``, strict ``load_weights`` fails and oMLX falls back to LLM,
silently dropping vision.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module


@pytest.mark.parametrize(
    ("model_type", "hidden"),
    [("qwen3_5_moe", True), ("qwen3_5", False), ("gemma4", False)],
    ids=["moe-hidden", "dense-untouched", "other-untouched"],
)
def test_qwen35_moe_mlx_metadata_hidden_only_for_moe(tmp_path, monkeypatch, model_type, hidden):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    weight_file = model_dir / "model-00001-of-00002.safetensors"
    weight_file.touch()
    outside_file = tmp_path / "outside.safetensors"
    outside_file.touch()

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {"format": "mlx", "source": "test"}

    import safetensors

    def fake_safe_open(*_args, **_kwargs):
        return FakeHandle()

    monkeypatch.setattr(safetensors, "safe_open", fake_safe_open)

    with vlm_module._force_qwen35_moe_sanitize_on_load(model_dir):
        target_metadata = safetensors.safe_open(weight_file).metadata()
        outside_metadata = safetensors.safe_open(outside_file).metadata()

    if hidden:
        assert target_metadata == {"source": "test"}
    else:
        assert target_metadata == {"format": "mlx", "source": "test"}
    assert outside_metadata == {"format": "mlx", "source": "test"}
    assert safetensors.safe_open is fake_safe_open
