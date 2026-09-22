# SPDX-License-Identifier: Apache-2.0
"""Tests for JANG bundle detection, routing, and the jang-tools load shim."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.exceptions import JANGDependencyError
from omlx.model_discovery import _is_model_dir, detect_model_type
from omlx.patches.jang_load import (
    is_jang_pack,
    is_jang_v2,
    is_prism_pack,
    jang_has_vision,
    load_jang,
    read_jang_config,
)
from omlx.utils.model_loading import maybe_load_jang, maybe_load_jangq_prism


def _write(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def _jang_dir(
    directory: Path,
    *,
    has_vision: bool | None = True,
    format_version: str = "2.0",
    config: dict | None = None,
    preprocessor: bool = False,
) -> Path:
    jang_config: dict = {"format": "jang", "format_version": format_version}
    if has_vision is not None:
        jang_config["architecture"] = {"type": "qwen3_5", "has_vision": has_vision}
    _write(directory, "jang_config.json", jang_config)
    _write(
        directory,
        "config.json",
        config if config is not None else {"model_type": "qwen3_5"},
    )
    if preprocessor:
        _write(directory, "preprocessor_config.json", {"patch_size": 16})
    return directory


def _jangq_prism_dir(directory: Path) -> Path:
    """A dealignai-style ternary repack: prism hadamard metadata + storage_bits."""
    _write(
        directory,
        "config.json",
        {
            "model_type": "qwen3_5",
            "hadamard": {"forward_modules": ["model.layers.0.self_attn.q_proj"]},
            "quantization": {
                "bits": 2,
                "group_size": 128,
                "mode": "affine",
                "model.layers.0.self_attn.q_proj": {
                    "bits": 2,
                    "group_size": 128,
                    "mode": "affine",
                    "storage_bits": 2,
                },
            },
        },
    )
    _write(
        directory,
        "jang_config.json",
        {
            "format": "jang",
            "format_version": "2.0",
            "architecture": {"type": "qwen3_5", "has_vision": True},
            "layout": {"language_norms": "zero-centered-runtime-plus-one"},
        },
    )
    return directory


def _native_prism_dir(directory: Path) -> Path:
    _write(
        directory,
        "config.json",
        {
            "model_type": "prism_hadamard_qwen35",
            "hadamard": {"forward_modules": ["model.layers.0.self_attn.q_proj"]},
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": [{"path": "lm_head"}],
        },
    )
    _write(directory, "jang_config.json", {"format": "jang", "format_version": "2.0"})
    return directory


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.dtype = None

    def astype(self, dtype):
        self.dtype = dtype
        return self


class _FakeModel:
    def __init__(self, config: dict):
        self.config = config
        self.dtype = None
        self.loaded: list[tuple[str, object]] = []

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def load_weights(self, weights, strict: bool = True) -> None:
        self.loaded.extend(weights)


def _fake_jang_tools(monkeypatch, *, model, processor_or_tokenizer, calls: list):
    def load_jang_vlm_model(path):
        calls.append(("vlm", str(path)))
        return model, processor_or_tokenizer

    def load_jang_model(path):
        calls.append(("text", str(path)))
        return model, processor_or_tokenizer

    package = types.ModuleType("jang_tools")
    loader = types.ModuleType("jang_tools.loader")
    loader.load_jang_vlm_model = load_jang_vlm_model
    loader.load_jang_model = load_jang_model
    package.loader = loader
    monkeypatch.setitem(sys.modules, "jang_tools", package)
    monkeypatch.setitem(sys.modules, "jang_tools.loader", loader)


class TestDetection:
    def test_jang_vision_bundle_is_a_vlm(self, tmp_path):
        directory = _jang_dir(tmp_path / "bundle", has_vision=True)
        assert detect_model_type(directory) == "vlm"
        assert is_jang_pack(directory) is True

    def test_jang_text_bundle_stays_llm_despite_vlm_architecture(self, tmp_path):
        directory = _jang_dir(
            tmp_path / "bundle",
            has_vision=False,
            config={"model_type": "qwen3_5", "architectures": ["Qwen3VLForConditionalGeneration"]},
        )
        assert detect_model_type(directory) == "llm"

    def test_jang_vision_without_declared_modality_uses_preprocessor(self, tmp_path):
        directory = _jang_dir(
            tmp_path / "bundle", has_vision=None, preprocessor=True
        )
        assert detect_model_type(directory) == "vlm"

    def test_preprocessor_config_alone_does_not_force_vlm(self, tmp_path):
        directory = tmp_path / "plain"
        _write(directory, "config.json", {"model_type": "qwen3_5"})
        _write(directory, "preprocessor_config.json", {"patch_size": 16})
        assert detect_model_type(directory) == "llm"

    def test_jang_sidecar_without_config_is_not_a_model_dir(self, tmp_path):
        directory = tmp_path / "bundle"
        _write(directory, "jang_config.json", {"format": "jang", "format_version": "2.0"})
        assert _is_model_dir(directory) is False

    def test_adapter_dir_with_jang_sidecar_is_not_a_model_dir(self, tmp_path):
        directory = tmp_path / "adapter"
        _write(directory, "jang_config.json", {"format": "jang", "format_version": "2.0"})
        _write(directory, "adapter_config.json", {})
        assert _is_model_dir(directory) is False

    def test_prism_repack_is_not_a_jang_pack(self, tmp_path):
        directory = _jangq_prism_dir(tmp_path / "jangq")
        assert is_prism_pack(directory) is True
        assert is_jang_pack(directory) is False

    def test_native_prism_pack_is_not_a_jang_pack(self, tmp_path):
        directory = _native_prism_dir(tmp_path / "prism")
        assert is_jang_pack(directory) is False

    def test_plain_model_is_not_a_jang_pack(self, tmp_path):
        directory = tmp_path / "plain"
        _write(directory, "config.json", {"model_type": "llama"})
        assert is_jang_pack(directory) is False
        assert read_jang_config(directory) == {}
        assert jang_has_vision(directory) is None

    def test_mxfp8_bundle_with_vmlx_sidecar_is_not_a_jang_pack(self, tmp_path):
        """Ornith-1.5 MXFP8 ships jang_config.json but no JANG format marker."""
        directory = tmp_path / "ornith"
        _write(
            directory,
            "config.json",
            {
                "model_type": "qwen3_5_moe",
                "vision_config": {"depth": 27},
                "quantization": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
            },
        )
        _write(
            directory,
            "jang_config.json",
            {
                "format": None,
                "format_version": None,
                "weight_format": "mxfp8",
                "capabilities": {"has_vision": True, "family": "qwen3_5_moe"},
                "quantization": {"method": "mxfp8", "mode": "mxfp8"},
            },
        )
        assert is_jang_pack(directory) is False
        assert maybe_load_jang(str(directory), is_vlm=True) is None
        assert detect_model_type(directory) == "vlm"

    def test_format_version_detection(self, tmp_path):
        assert is_jang_v2(_jang_dir(tmp_path / "v2", format_version="2.0")) is True
        assert is_jang_v2(_jang_dir(tmp_path / "v1", format_version="1.0")) is False
        assert is_jang_v2(tmp_path / "missing") is False


class TestMaybeLoadJang:
    def test_returns_none_for_plain_model(self, tmp_path):
        directory = tmp_path / "plain"
        _write(directory, "config.json", {"model_type": "llama"})
        assert maybe_load_jang(str(directory), is_vlm=False) is None

    def test_returns_none_for_prism_repack(self, tmp_path):
        directory = _jangq_prism_dir(tmp_path / "jangq")
        assert maybe_load_jang(str(directory), is_vlm=True) is None

    def test_jangq_prism_refuses_text_only_load(self, tmp_path):
        directory = _jangq_prism_dir(tmp_path / "jangq")
        with pytest.raises(ValueError, match="cannot be served text-only"):
            maybe_load_jangq_prism(str(directory), is_vlm=False)

    def test_jangq_prism_loader_returns_none_for_plain_model(self, tmp_path):
        directory = tmp_path / "plain"
        _write(directory, "config.json", {"model_type": "llama"})
        assert maybe_load_jangq_prism(str(directory), is_vlm=False) is None

    @pytest.mark.asyncio
    async def test_batched_engine_refuses_jangq_before_other_loaders(
        self, tmp_path, monkeypatch
    ):
        from omlx.engine import batched as batched_module
        from omlx.utils import model_loading

        directory = _jangq_prism_dir(tmp_path / "jangq")
        calls: list[str] = []

        def record_call(name):
            def recorder(*args, **kwargs):
                calls.append(name)
                return None

            return recorder

        monkeypatch.setattr(
            batched_module, "get_tokenizer_config", lambda *args, **kwargs: {}
        )
        monkeypatch.setattr(
            model_loading, "maybe_apply_pre_load_patches", lambda *args, **kwargs: None
        )
        for name in (
            "maybe_load_jang",
            "maybe_load_custom_quantization",
            "lm_load_compat",
        ):
            monkeypatch.setattr(model_loading, name, record_call(name))

        with pytest.raises(ValueError, match="cannot be served text-only"):
            await BatchedEngine(model_name=str(directory)).start()

        assert calls == []

    def test_refuses_text_only_load_of_vision_bundle(self, tmp_path):
        directory = _jang_dir(tmp_path / "bundle", has_vision=True)
        with pytest.raises(ValueError, match="vision bundle"):
            maybe_load_jang(str(directory), is_vlm=False)

    def test_dispatches_text_bundle_to_text_loader(self, tmp_path, monkeypatch):
        directory = _jang_dir(tmp_path / "bundle", has_vision=False)
        model = _FakeModel({"model_type": "qwen3_5", "hidden_size": 64})
        calls: list = []
        _fake_jang_tools(
            monkeypatch, model=model, processor_or_tokenizer="tok", calls=calls
        )
        loaded = maybe_load_jang(str(directory), is_vlm=False)
        assert loaded == (model, "tok")
        assert calls == [("text", str(directory))]
        assert model.dtype is None

    def test_dispatches_vision_bundle_to_vlm_loader(self, tmp_path, monkeypatch):
        directory = _jang_dir(tmp_path / "bundle", has_vision=True)
        model = _FakeModel({"model_type": "qwen3_5", "hidden_size": 64})
        calls: list = []
        _fake_jang_tools(
            monkeypatch, model=model, processor_or_tokenizer="proc", calls=calls
        )
        loaded = maybe_load_jang(str(directory), is_vlm=True)
        assert loaded == (model, "proc")
        assert calls == [("vlm", str(directory))]

    def test_switches_large_expert_bundle_to_bfloat16(self, tmp_path, monkeypatch):
        directory = _jang_dir(tmp_path / "bundle", has_vision=False)
        model = _FakeModel(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"num_experts": 512, "hidden_size": 4096},
            }
        )
        _fake_jang_tools(
            monkeypatch, model=model, processor_or_tokenizer="tok", calls=[]
        )
        with mock.patch.dict(
            sys.modules,
            {
                "mlx": types.ModuleType("mlx"),
                "mlx.core": types.SimpleNamespace(bfloat16="bf16"),
            },
        ):
            maybe_load_jang(str(directory), is_vlm=False)
        assert model.dtype == "bf16"

    def test_missing_runtime_raises_dependency_error(self, tmp_path, monkeypatch):
        directory = _jang_dir(tmp_path / "bundle", has_vision=False)
        monkeypatch.setitem(sys.modules, "jang_tools", None)
        with pytest.raises(JANGDependencyError, match="jang"):
            maybe_load_jang(str(directory), is_vlm=False)


class TestHadamardSignRepair:
    """jang derives rotation widths as packed_cols * (32 // bits)."""

    @staticmethod
    def _fake_mlx(quantized_linear, generate_random_signs):
        mlx = types.ModuleType("mlx")
        mlx_nn = types.ModuleType("mlx.nn")
        mlx_nn.QuantizedLinear = quantized_linear
        mlx.nn = mlx_nn
        rotation = types.ModuleType("jang_tools.turboquant.rotation")
        rotation.generate_random_signs = generate_random_signs
        turboquant = types.ModuleType("jang_tools.turboquant")
        turboquant.rotation = rotation
        jang_tools = types.ModuleType("jang_tools")
        jang_tools.turboquant = turboquant
        return {
            "mlx": mlx,
            "mlx.nn": mlx_nn,
            "jang_tools": jang_tools,
            "jang_tools.turboquant": turboquant,
            "jang_tools.turboquant.rotation": rotation,
        }

    def test_rekeys_only_mismatched_quantized_layers(self, monkeypatch):
        class QuantizedLinear:
            def __init__(self, packed_columns, bits, signs_len):
                self.weight = types.SimpleNamespace(shape=(8, packed_columns))
                self.bits = bits
                self._hadamard_signs = (
                    None if signs_len is None else types.SimpleNamespace(shape=(signs_len,))
                )

        class FakeModel:
            def __init__(self, modules):
                self._modules = modules

            def modules(self):
                return self._modules

        wrong = QuantizedLinear(12, 6, 60)      # 12 * (32 // 6) = 60, should be 64
        right = QuantizedLinear(16, 8, 64)      # 16 * (32 // 8) = 64, already correct
        unrotated = QuantizedLinear(4, 2, None)  # 2-bit layers are never rotated
        generated: list[int] = []

        def fake_generate(dim, seed=0):
            generated.append((dim, seed))
            return types.SimpleNamespace(shape=(dim,))

        monkeypatch.setitem(
            sys.modules, "jang_tools", types.ModuleType("jang_tools")
        )
        with mock.patch.dict(
            sys.modules, self._fake_mlx(QuantizedLinear, fake_generate)
        ):
            from omlx.patches.jang_load import _repair_hadamard_signs

            repaired = _repair_hadamard_signs(FakeModel([wrong, right, unrotated]))

        assert repaired == 1
        assert generated == [(64, 42)]
        assert wrong._hadamard_signs.shape == (64,)
        assert right._hadamard_signs.shape == (64,)
        assert unrotated._hadamard_signs is None

    def test_load_jang_repairs_hadamard_bundles_only(self, tmp_path, monkeypatch):
        directory = _jang_dir(tmp_path / "bundle", has_vision=False)
        model = _FakeModel({"model_type": "qwen3", "hidden_size": 64})
        _fake_jang_tools(
            monkeypatch, model=model, processor_or_tokenizer="tok", calls=[]
        )
        with mock.patch(
            "omlx.patches.jang_load._repair_hadamard_signs", return_value=0
        ) as repair:
            load_jang(str(directory), is_vlm=False)
        repair.assert_not_called()

        config = json.loads((directory / "jang_config.json").read_text())
        config["quantization"] = {"hadamard_rotation": True}
        (directory / "jang_config.json").write_text(json.dumps(config))
        with mock.patch(
            "omlx.patches.jang_load._repair_hadamard_signs", return_value=3
        ) as repair:
            load_jang(str(directory), is_vlm=False)
        repair.assert_called_once()


class TestNemotronFixup:
    def _write_shards(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "backbone.layers.0.mixer.gate.weight": "model-00001.safetensors",
                        "backbone.layers.0.mixer.gate.scales": "model-00001.safetensors",
                        "backbone.layers.0.mixer.gate.biases": "model-00001.safetensors",
                        "backbone.layers.1.mixer.gate.weight": "model-00001.safetensors",
                    }
                }
            )
        )

    def test_dequantizes_only_gates_with_scales_and_biases(self, tmp_path, monkeypatch):
        directory = tmp_path / "nemotron"
        self._write_shards(directory)
        model = _FakeModel(
            {"model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"]}
        )
        dequantized_calls: list = []
        shard = {
            "backbone.layers.0.mixer.gate.weight": _FakeTensor((8, 4)),
            "backbone.layers.0.mixer.gate.scales": _FakeTensor((8, 1)),
            "backbone.layers.0.mixer.gate.biases": _FakeTensor((8, 1)),
        }

        def fake_dequantize(weight, scales, biases, group_size, bits):
            dequantized_calls.append((group_size, bits))
            return _FakeTensor((weight.shape[0], weight.shape[-1] * (32 // bits)))

        fake_core = types.SimpleNamespace(
            bfloat16="bf16",
            float16="fp16",
            load=lambda path: shard,
            dequantize=fake_dequantize,
        )
        with mock.patch.dict(
            sys.modules,
            {"mlx": types.ModuleType("mlx"), "mlx.core": fake_core},
        ):
            from omlx.patches.jang_load import _fix_nemotron_h_weights

            _fix_nemotron_h_weights(model, directory)

        assert dequantized_calls == [(16, 8)]
        assert [key for key, _ in model.loaded] == [
            "backbone.layers.0.mixer.gate.weight"
        ]
        assert model.loaded[0][1].dtype == "fp16"

    def test_skips_fixup_for_jang_v2_bundles(self, tmp_path, monkeypatch):
        directory = _jang_dir(
            tmp_path / "bundle",
            has_vision=False,
            config={
                "model_type": "nemotron_h",
                "architectures": ["NemotronHForCausalLM"],
            },
        )
        model = _FakeModel(
            {"model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"]}
        )
        _fake_jang_tools(
            monkeypatch, model=model, processor_or_tokenizer="tok", calls=[]
        )
        with mock.patch(
            "omlx.patches.jang_load._fix_nemotron_h_weights"
        ) as fixup:
            load_jang(str(directory), is_vlm=False)
        fixup.assert_not_called()
        assert model.loaded == []
