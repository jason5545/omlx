# SPDX-License-Identifier: Apache-2.0
"""Image generation: discovery, prompt expansion, engine and /v1/images/*."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omlx import image_generation as ig
from omlx.engine.image_generation import (
    GeneratedImage,
    ImageGenerationEngine,
    ImageJob,
)
from omlx.exceptions import ModelBusyError
from omlx.image_generation import ImageCapabilities

H3 = ImageCapabilities(generate=True, edit=True, family="minimax_h3_image")


# ---------------------------------------------------------------------------
# Fixtures: a MiniMax-H3-Image-shaped checkpoint on disk
# ---------------------------------------------------------------------------


def _write_component(root: Path, name: str, config: dict, shards: int = 1) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config))
    names = (
        ["model.safetensors"]
        if shards == 1
        else [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    )
    for shard in names:
        (folder / shard).write_bytes(b"\0" * 64)
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}})
    )


def _h3_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ImagePipeline",
                "model_type": "minimax_h3_image",
                "mlx_vlm_format": "minimax_h3_image",
            }
        )
    )
    _write_component(
        root,
        "transformer",
        {"_class_name": "MiniMaxH3ImageTransformer2DModel"},
        shards=2,
    )
    _write_component(root, "text_encoder", {"model_type": "qwen3_vl"}, shards=2)
    _write_component(root, "vae", {"_class_name": "AutoencoderKLMiniMaxH3Image"})
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    return root


def _llm_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (root / "model.safetensors").write_bytes(b"\0" * 64)
    return root


@pytest.fixture
def mlx_vlm_h3():
    pytest.importorskip("mlx_vlm.models.minimax_h3_image.model")
    ig._capabilities_cached.cache_clear()
    yield
    ig._capabilities_cached.cache_clear()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_h3_checkpoint_is_an_image_generation_model(tmp_path, mlx_vlm_h3):
    from omlx.model_discovery import detect_model_type, discover_models

    _h3_checkpoint(tmp_path / "models" / "nativ-community" / "MiniMax-H3-Image-8bit")
    models = discover_models(tmp_path / "models")
    assert list(models) == ["MiniMax-H3-Image-8bit"]
    info = models["MiniMax-H3-Image-8bit"]
    assert (info.model_type, info.engine_type) == ("image_generation", "image_generation")
    assert info.config_model_type == "minimax_h3_image"
    assert info.estimated_size > 0
    assert detect_model_type(Path(info.model_path)) == "image_generation"
    caps = ig.image_capabilities(info.model_path)
    assert (caps.generate, caps.edit) == (True, True)


def test_vendor_folders_in_front_of_mlx_vlm_do_not_hide_image_models(
    tmp_path, mlx_vlm_h3, monkeypatch
):
    import mlx_vlm.models

    # oMLX's compat patches put vendor/mlx_vlm/models first on the path.
    vendor = tmp_path / "vendor" / "mlx_vlm" / "models"
    (vendor / "minimax_m3_vl").mkdir(parents=True)
    monkeypatch.setattr(
        mlx_vlm.models, "__path__", [str(vendor), *mlx_vlm.models.__path__]
    )
    root = _h3_checkpoint(tmp_path / "H3")
    assert ig.image_capabilities(root).generate


def test_image_components_are_not_registered_as_models(tmp_path, mlx_vlm_h3):
    from omlx.model_discovery import discover_models

    # Level-1 placement: before, transformer/ and text_encoder/ (which carry a
    # config.json) were registered as LLMs of their own.
    _h3_checkpoint(tmp_path / "models" / "MiniMax-H3-Image-8bit")
    assert list(discover_models(tmp_path / "models")) == ["MiniMax-H3-Image-8bit"]
    # Pointing the model directory at the pipeline itself.
    direct = discover_models(tmp_path / "models" / "MiniMax-H3-Image-8bit")
    assert list(direct) == ["MiniMax-H3-Image-8bit"]


def test_incomplete_image_download_is_skipped(tmp_path, mlx_vlm_h3):
    from omlx.model_discovery import discover_models

    root = _h3_checkpoint(tmp_path / "models" / "org" / "MiniMax-H3-Image-8bit")
    (root / "transformer" / "model-00002-of-00002.safetensors").unlink()
    assert discover_models(tmp_path / "models") == {}

    root2 = _h3_checkpoint(tmp_path / "models2" / "org" / "H3")
    partial = root2 / ".cache" / "huggingface" / "download" / "vae"
    partial.mkdir(parents=True)
    (partial / "abc.incomplete").write_bytes(b"")
    assert discover_models(tmp_path / "models2") == {}


def test_ordinary_checkpoints_never_reach_the_mlx_vlm_check(tmp_path, monkeypatch):
    from omlx.model_discovery import discover_models

    _llm_checkpoint(tmp_path / "models" / "org" / "Llama-3")

    def boom(_path):
        raise AssertionError("mlx-vlm image check ran for an LLM folder")

    monkeypatch.setattr(ig, "_candidate_types", boom)
    ig._capabilities_cached.cache_clear()
    models = discover_models(tmp_path / "models")
    assert models["Llama-3"].model_type == "llm"


def test_pipeline_metadata_without_an_mlx_vlm_image_package(tmp_path, monkeypatch):
    root = tmp_path / "SomePipeline"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": "Unknown"}))
    monkeypatch.setattr(ig, "_candidate_types", lambda _p: ("unknown_family",))
    ig._capabilities_cached.cache_clear()
    assert not ig.is_image_generation_dir(root)


# ---------------------------------------------------------------------------
# Prompt expansion
# ---------------------------------------------------------------------------


def _pe_folder(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    return root


@pytest.fixture
def h3_default_expander(monkeypatch):
    monkeypatch.setattr(
        ig,
        "default_expander",
        lambda family: "MiniMaxAI/MiniMax-H3-Image-PE"
        if family == "minimax_h3_image"
        else None,
    )


def _no_hf_cache(monkeypatch):
    import huggingface_hub

    def missing(*_a, **_k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)


def test_default_expansion_is_skipped_when_the_expander_is_missing(
    tmp_path, monkeypatch, h3_default_expander
):
    _no_hf_cache(monkeypatch)
    resolved = ig.resolve_prompt_expansion("minimax_h3_image", None, None, [tmp_path])
    assert resolved.extra() == {"upsample_prompt": False}


def test_requested_expansion_without_the_expander_is_an_error(
    tmp_path, monkeypatch, h3_default_expander
):
    _no_hf_cache(monkeypatch)
    with pytest.raises(ig.PromptExpanderMissing) as error:
        ig.resolve_prompt_expansion("minimax_h3_image", True, None, [tmp_path])
    assert "MiniMaxAI/MiniMax-H3-Image-PE" in str(error.value)
    with pytest.raises(ig.PromptExpanderMissing):
        ig.resolve_prompt_expansion(
            "minimax_h3_image", None, "someone/other-PE", [tmp_path]
        )


def test_local_expander_is_passed_as_a_path(tmp_path, monkeypatch, h3_default_expander):
    _no_hf_cache(monkeypatch)
    local = _pe_folder(tmp_path / "MiniMaxAI" / "MiniMax-H3-Image-PE")
    resolved = ig.resolve_prompt_expansion("minimax_h3_image", None, None, [tmp_path])
    assert resolved.extra() == {"prompt_expansion_model": str(local)}
    explicit = ig.resolve_prompt_expansion(
        "minimax_h3_image", True, str(local), [tmp_path / "elsewhere"]
    )
    assert explicit.extra() == {
        "upsample_prompt": True,
        "prompt_expansion_model": str(local),
    }


def test_disabled_expansion_needs_no_expander(tmp_path, h3_default_expander):
    resolved = ig.resolve_prompt_expansion(
        "minimax_h3_image", False, "missing/PE", [tmp_path]
    )
    assert resolved.extra() == {"upsample_prompt": False}


def test_families_without_a_default_expander_pass_through(tmp_path):
    assert ig.resolve_prompt_expansion("z_image", None, None, [tmp_path]).extra() == {}


def test_default_expanders_come_from_mlx_vlm():
    upsampler = pytest.importorskip("mlx_vlm.models.minimax_h3_image.upsampler")
    assert ig.default_expander("minimax_h3_image") == upsampler.DEFAULT_UPSAMPLER
    assert ig.default_expander("qwen_image") is None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _FakeEngine(ImageGenerationEngine):
    def __init__(self, delay: float = 0.0, fail: Exception | None = None):
        super().__init__("/models/fake-h3")
        self._capabilities = H3
        self.delay = delay
        self.fail = fail
        self.calls: list[tuple[str, ImageJob]] = []

    def _run_one_sync(self, task, job):
        self.calls.append((task, job))
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        width = job.width or 8
        height = job.height or 6
        return GeneratedImage(
            pixels=np.full((height, width, 3), job.seed % 255, dtype=np.uint8),
            width=width,
            height=height,
            seed=job.seed,
            revised_prompt=None,
            metadata={},
            peak_memory_gb=1.5,
            elapsed_s=self.delay,
        )


def _job(seed: int = 1) -> ImageJob:
    return ImageJob("a cat", seed, None, 8, 6, None, {})


def test_generation_does_not_block_the_event_loop():
    engine = _FakeEngine(delay=0.3)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        images = await engine.run("generate", [_job(1), _job(2)])
        task.cancel()
        return images, ticks

    images, ticks = asyncio.run(main())
    assert [image.seed for image in images] == [1, 2]
    assert ticks >= 10
    assert not engine.has_active_requests()


def test_cancellation_skips_the_remaining_images():
    from omlx.engine.image_generation import ImageGenerationCancelled

    engine = _FakeEngine()
    cancelled = threading.Event()

    async def main():
        def stop_after_first(index, image):
            cancelled.set()

        return await engine.run(
            "generate",
            [_job(1), _job(2), _job(3)],
            cancelled=cancelled,
            on_image=stop_after_first,
        )

    with pytest.raises(ImageGenerationCancelled):
        asyncio.run(main())
    assert len(engine.calls) == 1


def test_unsupported_task_is_rejected():
    engine = _FakeEngine()
    engine._capabilities = ImageCapabilities(True, False, "z_image")
    with pytest.raises(ValueError):
        asyncio.run(engine.run("edit", [_job()]))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class _Pool:
    def __init__(self, engine: ImageGenerationEngine, model_dirs=()):
        self.engine = engine
        self._model_dirs = list(model_dirs)
        self.entries = {
            "MiniMax-H3-Image-8bit": SimpleNamespace(
                model_id="MiniMax-H3-Image-8bit",
                model_path="/models/MiniMax-H3-Image-8bit",
                model_type="image_generation",
            ),
            "chat": SimpleNamespace(
                model_id="chat", model_path="/models/chat", model_type="llm"
            ),
        }
        self.leases = 0
        self.released = 0
        self.busy = False

    def resolve_model_id(self, model_id, settings_manager, **_kwargs):
        return model_id

    def get_entry(self, model_id):
        return self.entries.get(model_id)

    def get_model_ids(self):
        return list(self.entries)

    async def get_engine(self, model_id, _lease=False, **_kwargs):
        if self.busy:
            raise ModelBusyError("chat", "unload for MiniMax-H3-Image-8bit")
        self.leases += 1
        return self.engine

    async def release_engine(self, model_id):
        self.released += 1

    def get_status(self):
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": 2,
            "loaded_count": 0,
            "models": [
                {"id": mid, "model_path": e.model_path, "model_type": e.model_type}
                for mid, e in self.entries.items()
            ],
        }


@pytest.fixture
def server(tmp_path, monkeypatch, h3_default_expander):
    from omlx.model_settings import ModelSettingsManager
    from omlx.server import ServerState, app

    _no_hf_cache(monkeypatch)
    engine = _FakeEngine()
    pool = _Pool(engine, model_dirs=[tmp_path / "models"])
    state = ServerState()
    state.engine_pool = pool
    state.settings_manager = ModelSettingsManager(base_path=tmp_path)
    state.api_key = "secret"
    monkeypatch.setattr("omlx.api.image_routes.image_capabilities", lambda _p: H3)
    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        client.headers["Authorization"] = "Bearer secret"
        yield SimpleNamespace(client=client, pool=pool, engine=engine, tmp=tmp_path)


def _wait_released(pool, count=1):
    deadline = time.monotonic() + 5
    while pool.released < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return pool.released


def _png(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


def test_generation_returns_png_and_releases_the_lease(server):
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "n": 2,
              "size": "16x12", "seed": 42},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["size"] == "16x12" and body["output_format"] == "png"
    assert [item["seed"] for item in body["data"]] == [42, 43]
    image = _png(body["data"][0]["b64_json"])
    assert image.size == (16, 12) and image.format == "PNG"
    assert body["data"][0]["mime_type"] == "image/png"
    assert _wait_released(server.pool) == 1 and server.pool.leases == 1
    task, job = server.engine.calls[0]
    assert task == "generate" and (job.width, job.height) == (16, 12)
    # No expander on disk: the H3 default expansion is switched off.
    assert job.extra == {"upsample_prompt": False}


def test_generation_requires_the_api_key(server):
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
    defaults = server.client.get(
        "/v1/images/defaults",
        params={"model": "MiniMax-H3-Image-8bit"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert defaults.status_code == 401
    assert server.pool.leases == 0


def test_requested_expansion_without_the_expander_is_400(server):
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat",
              "upsample_prompt": True},
    )
    assert response.status_code == 400
    assert "MiniMaxAI/MiniMax-H3-Image-PE" in response.text
    assert server.pool.leases == 0


def test_local_expander_is_used(server):
    local = _pe_folder(server.tmp / "models" / "MiniMaxAI" / "MiniMax-H3-Image-PE")
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat"},
    )
    assert response.status_code == 200, response.text
    assert server.engine.calls[0][1].extra == {"prompt_expansion_model": str(local)}


@pytest.mark.parametrize(
    "extra",
    [{"response_format": "path"}, {"output_path": "/tmp/x.png"}, {"output_dir": "/tmp"}],
)
def test_server_side_file_outputs_are_rejected(server, extra):
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", **extra},
    )
    assert response.status_code == 400
    assert server.pool.leases == 0


def test_wrong_model_kinds(server):
    not_image = server.client.post(
        "/v1/images/generations", json={"model": "chat", "prompt": "a cat"}
    )
    assert not_image.status_code == 400
    missing = server.client.post(
        "/v1/images/generations", json={"model": "nope", "prompt": "a cat"}
    )
    assert missing.status_code == 404
    assert "MiniMax-H3-Image-8bit" in missing.text


def test_busy_pool_answers_409(server):
    server.pool.busy = True
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat"},
    )
    assert response.status_code == 409


def test_model_validation_errors_are_400(server):
    server.engine.fail = ValueError("H3-Image is distilled for exactly six steps")
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "steps": 4},
    )
    assert response.status_code == 400
    assert "six steps" in response.text
    assert _wait_released(server.pool) == 1


def _events(response) -> list:
    events = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_streaming_sends_started_result_done(server):
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "size": "8x6",
              "seed": 7, "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert [e if e == "[DONE]" else e["type"] for e in events] == [
        "started", "result", "[DONE]"]
    assert events[0] == {"type": "started", "model": "MiniMax-H3-Image-8bit", "n": 1}
    result = events[1]
    assert result["size"] == "8x6" and result["data"][0]["seed"] == 7
    assert _png(result["data"][0]["b64_json"]).size == (8, 6)
    assert _wait_released(server.pool) == 1


def test_streaming_keepalive_and_error(server, monkeypatch):
    monkeypatch.setattr("omlx.api.image_routes.KEEPALIVE_SECONDS", 0.05)
    server.engine.delay = 0.2
    server.engine.fail = RuntimeError("Metal ran out")
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "stream": True},
    )
    assert ": keepalive" in response.text
    events = _events(response)
    assert events[0]["type"] == "started"
    assert events[1]["type"] == "error" and "Metal ran out" in events[1]["message"]
    assert events[-1] == "[DONE]"


def test_edit_accepts_paths_and_data_urls(server, tmp_path):
    reference = tmp_path / "ref.png"
    Image.new("RGB", (4, 4), "red").save(reference)
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    seen = {}

    original = server.engine._run_one_sync

    def capture(task, job):
        seen["paths"] = job.image_paths
        seen["exist"] = [Path(p).is_file() for p in job.image_paths]
        return original(task, job)

    server.engine._run_one_sync = capture
    response = server.client.post(
        "/v1/images/edits",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "<Picture 1> blue",
              "image": [str(reference), data_url], "negative_prompt": "blurry"},
    )
    assert response.status_code == 200, response.text
    assert seen["paths"][0] == str(reference) and seen["exist"] == [True, True]
    assert _wait_released(server.pool) == 1
    # The decoded data URL lives only for the request.
    assert not Path(seen["paths"][1]).exists()
    task, job = server.engine.calls[0]
    assert task == "edit" and job.width is None and job.extra["negative_prompt"] == "blurry"
    assert response.json()["size"] == "8x6"


def test_edit_rejects_bad_images(server):
    for image in ["relative.png", "/does/not/exist.png", "data:image/png;base64,AAAA"]:
        response = server.client.post(
            "/v1/images/edits",
            json={"model": "MiniMax-H3-Image-8bit", "prompt": "x", "image": image},
        )
        assert response.status_code == 400, image
    assert server.pool.leases == 0


def test_defaults_endpoint(server, monkeypatch):
    import importlib

    # ``import mlx_vlm.generate.x as y`` trips over mlx_vlm.generate, which
    # mlx_vlm/__init__ rebinds to the generate() function.
    defaults_module = importlib.import_module("mlx_vlm.generate.image_defaults")

    calls = []

    def fake_defaults(model, task="generate"):
        calls.append((model, task))
        if task == "edit":
            raise NotImplementedError("no edit defaults")
        return SimpleNamespace(steps=6, guidance=1.0)

    monkeypatch.setattr(defaults_module, "resolve_image_defaults", fake_defaults, raising=False)
    response = server.client.get(
        "/v1/images/defaults", params={"model": "MiniMax-H3-Image-8bit"}
    )
    assert response.status_code == 200
    assert response.json() == {"steps": 6, "guidance": 1.0}
    assert calls == [("/models/MiniMax-H3-Image-8bit", "generate")]
    edit = server.client.get(
        "/v1/images/defaults",
        params={"model": "MiniMax-H3-Image-8bit", "task": "edit"},
    )
    assert edit.status_code == 501
    assert server.pool.leases == 0


def test_models_list_marks_image_generation(server):
    with patch("omlx.server.get_max_context_window", return_value=4096):
        response = server.client.get("/v1/models")
    assert response.status_code == 200
    data = {m["id"]: m for m in response.json()["data"]}
    assert data["MiniMax-H3-Image-8bit"]["model_type"] == "image_generation"
    assert data["MiniMax-H3-Image-8bit"]["max_model_len"] is None
    assert data["chat"]["max_model_len"] == 4096


def test_request_schemas_match_mlx_vlm():
    try:
        schemas = __import__("mlx_vlm.server.schemas", fromlist=["x"])
    except Exception as exc:  # pragma: no cover - mlx-vlm without image API
        pytest.skip(f"mlx_vlm.server.schemas unavailable: {exc}")
    if not hasattr(schemas, "ImageEditRequest"):
        pytest.skip("mlx-vlm without the image API")
    from omlx.api import image_routes

    for name in (
        "ImageGenerationRequest",
        "ImageEditRequest",
        "ImageGenerationResponseData",
        "ImageGenerationResponse",
    ):
        ours = set(getattr(image_routes, name).model_fields) - {"stream"}
        theirs = set(getattr(schemas, name).model_fields)
        assert ours == theirs, name
