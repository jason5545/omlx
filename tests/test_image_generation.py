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
    # Production applies the progress patch before any image module loads.
    ig.ensure_image_patches()
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

    def _run_one_sync(self, task, job, progress=None):
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

    def capture(task, job, progress=None):
        seen["paths"] = job.image_paths
        seen["exist"] = [Path(p).is_file() for p in job.image_paths]
        return original(task, job, progress)

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


def test_engine_pool_loads_an_image_pipeline_without_root_config(
    tmp_path, mlx_vlm_h3, monkeypatch
):
    """The pool's stale-path guard must not drop a pipeline for lacking config.json."""
    from omlx import engine_pool as engine_pool_module
    from omlx.engine_pool import EnginePool

    started = []

    class _Stub(ImageGenerationEngine):
        async def start(self):
            started.append(self.model_name)

        async def stop(self):
            pass

    monkeypatch.setattr(engine_pool_module, "ImageGenerationEngine", _Stub)
    _h3_checkpoint(tmp_path / "models" / "org" / "MiniMax-H3-Image-8bit")
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path / "models"))
    entry = pool.get_entry("MiniMax-H3-Image-8bit")
    assert (entry.model_type, entry.engine_type) == ("image_generation", "image_generation")

    async def main():
        engine = await pool.get_engine("MiniMax-H3-Image-8bit", _lease=True)
        await pool.release_engine("MiniMax-H3-Image-8bit")
        await asyncio.sleep(0)
        return engine

    engine = asyncio.run(main())
    assert isinstance(engine, _Stub) and started == [entry.model_path]
    assert pool.get_entry("MiniMax-H3-Image-8bit") is entry


# ---------------------------------------------------------------------------
# Memory: component estimate, own peak, pool charge, unload settle
# ---------------------------------------------------------------------------

GB = 10**9


def _write_safetensors(path: Path, tensors: dict[str, int]) -> None:
    """Write a real safetensors header with zero-filled data of the given sizes."""
    header, offset = {}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * offset)


def _sized_h3_checkpoint(root: Path) -> Path:
    """H3 layout with headers: transformer 4748 B (2608 B AdaLN), te 2749, vae 234."""
    _h3_checkpoint(root)
    for folder in ("transformer", "text_encoder", "vae"):
        for shard in (root / folder).glob("*.safetensors"):
            shard.unlink()
        (root / folder / "model.safetensors.index.json").unlink()
    _write_safetensors(
        root / "transformer" / "model.safetensors",
        {
            "transformer_blocks.0.adaln_proj.linear.weight": 2600,
            "norm_out.linear.weight": 8,
            "transformer_blocks.0.ff.weight": 1228,
            "transformer_blocks.0.attn.to_q.weight": 912,
        },
    )
    _write_safetensors(
        root / "text_encoder" / "model.safetensors",
        {"language_model.model.layers.0.weight": 2749},
    )
    _write_safetensors(root / "vae" / "model.safetensors", {"decoder.weight": 234})
    return root


def _file_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.glob("*.safetensors"))


def test_h3_estimate_is_the_steady_state_request_peak(tmp_path, mlx_vlm_h3):
    from omlx.model_discovery import discover_models, estimate_model_size

    root = _sized_h3_checkpoint(tmp_path / "models" / "org" / "MiniMax-H3-Image-8bit")
    expected = int(
        (
            _file_bytes(root / "transformer")
            - 2608
            + _file_bytes(root / "vae")
            + _file_bytes(root / "text_encoder")
        )
        * 1.05
    )
    assert ig.estimate_image_memory(root, "minimax_h3_image") == expected
    info = discover_models(tmp_path / "models")["MiniMax-H3-Image-8bit"]
    assert info.estimated_size == expected
    assert info.estimated_size < estimate_model_size(root)


def test_h3_8bit_component_sizes_give_an_estimate_between_51_4_and_60_gb(
    tmp_path, monkeypatch
):
    """The MiniMax-H3-Image-8bit sizes read on 11/16 (bytes from the headers)."""
    files = {"transformer": 47.48 * GB, "text_encoder": 27.49 * GB, "vae": 2.34 * GB}
    monkeypatch.setattr(ig, "_files_bytes", lambda folder: int(files[folder.name]))
    monkeypatch.setattr(ig, "_tensor_bytes", lambda folder, predicate: int(26.08 * GB))
    estimate = ig.estimate_image_memory(tmp_path, "minimax_h3_image")
    # 51.4 GB is the largest peak measured on 11/15 (a 4096x4096 edit).
    assert 51.4 * GB <= estimate <= 60 * GB
    assert estimate == int((47.48 - 26.08 + 2.34 + 27.49) * GB * 1.05)


def test_unknown_families_and_unreadable_h3_shards_keep_the_file_size(
    tmp_path, mlx_vlm_h3
):
    from omlx.model_discovery import discover_models, estimate_model_size

    assert ig.estimate_image_memory(tmp_path, "flux3") is None
    assert ig.estimate_image_memory(tmp_path, None) is None
    # The fixture's shards are 64 zero bytes: no valid header, so no rule.
    root = _h3_checkpoint(tmp_path / "models" / "org" / "MiniMax-H3-Image-8bit")
    assert ig.estimate_image_memory(root, "minimax_h3_image") is None
    info = discover_models(tmp_path / "models")["MiniMax-H3-Image-8bit"]
    assert info.estimated_size == estimate_model_size(root)


def test_h3_without_adaln_weights_keeps_the_file_size(tmp_path, mlx_vlm_h3):
    root = _sized_h3_checkpoint(tmp_path / "MiniMax-H3-Image-8bit")
    _write_safetensors(root / "transformer" / "model.safetensors", {"proj_in.weight": 64})
    assert ig.estimate_image_memory(root, "minimax_h3_image") is None


class _FakeMX:
    """Process-wide MLX gauges as one counter, like mx.get_active_memory()."""

    def __init__(self, active: int):
        self.active = active
        self.peak = 0

    def get_active_memory(self):
        return self.active

    def get_peak_memory(self):
        return self.peak

    def reset_peak_memory(self):
        self.peak = 0

    def clear_cache(self):
        pass

    def alloc(self, size: int):
        self.active += size
        self.peak = max(self.peak, self.active)

    def free(self, size: int):
        self.active -= size


class _LazyH3Engine(ImageGenerationEngine):
    """Replays the H3 pipeline's allocations on a fake MLX gauge."""

    def __init__(self, fake: _FakeMX, observer=None, fail: bool = False):
        super().__init__("/models/fake-h3", memory_observer=observer)
        self._capabilities = H3
        self.fake = fake
        self.fail = fail
        self.loaded = False

    def _generate_sync(self, task, job, progress=None):
        fake = self.fake
        fake.alloc(27 * GB)  # text encoder, loaded for every request
        fake.free(27 * GB)
        if not self.loaded:
            fake.alloc(26 * GB)  # AdaLN weights, dropped after the cache
            fake.free(26 * GB)
            fake.alloc(23 * GB)  # transformer remainder + VAE, kept
            self.loaded = True
        fake.alloc(2 * GB)  # activations
        if self.fail:
            raise RuntimeError("metal out of memory")
        fake.free(2 * GB)
        return GeneratedImage(
            pixels=np.zeros((6, 8, 3), np.uint8),
            width=8,
            height=6,
            seed=job.seed,
            revised_prompt=None,
            metadata={},
            peak_memory_gb=fake.peak / GB,
            elapsed_s=0.0,
        )


def test_engine_measures_its_own_peak_without_other_models(monkeypatch):
    from omlx.engine import image_generation as engine_module

    fake = _FakeMX(active=30 * GB)  # an LLM already loaded in the process
    monkeypatch.setattr(engine_module, "mx", fake)
    seen = []
    engine = _LazyH3Engine(fake, observer=seen.append)

    first = asyncio.run(engine.run("generate", [_job(1)]))[0]
    assert first.footprint_peak_bytes == 27 * GB  # nothing resident yet
    assert first.peak_memory_gb == 57  # the process-wide number, unchanged
    assert engine.resident_bytes == 23 * GB

    second = asyncio.run(engine.run("generate", [_job(2)]))[0]
    assert second.footprint_peak_bytes == (23 + 27) * GB  # resident + encoder
    assert engine.resident_bytes == 23 * GB
    assert seen == [27 * GB, 50 * GB]
    assert engine.observed_peak_bytes == 50 * GB


def test_a_failed_request_still_reports_its_peak(monkeypatch):
    from omlx.engine import image_generation as engine_module

    fake = _FakeMX(active=0)
    monkeypatch.setattr(engine_module, "mx", fake)
    seen = []
    engine = _LazyH3Engine(fake, observer=seen.append, fail=True)
    with pytest.raises(RuntimeError):
        asyncio.run(engine.run("generate", [_job(1)]))
    assert seen == [27 * GB]
    assert engine.resident_bytes == 25 * GB  # what it loaded before failing


def _image_pool(tmp_path, monkeypatch, *, sized: bool):
    from omlx import engine_pool as engine_pool_module
    from omlx.engine_pool import EnginePool

    class _Stub(ImageGenerationEngine):
        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(engine_pool_module, "ImageGenerationEngine", _Stub)
    root = tmp_path / "models" / "org" / "MiniMax-H3-Image-8bit"
    (_sized_h3_checkpoint if sized else _h3_checkpoint)(root)
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path / "models"))
    return pool, "MiniMax-H3-Image-8bit"


def _load(pool, model_id):
    async def main():
        engine = await pool.get_engine(model_id, _lease=True)
        await pool.release_engine(model_id)
        return engine

    return asyncio.run(main())


@pytest.mark.parametrize("sized", [True, False])
def test_pool_raises_the_charge_only_above_the_estimate(
    tmp_path, mlx_vlm_h3, monkeypatch, sized
):
    pool, model_id = _image_pool(tmp_path, monkeypatch, sized=sized)
    entry = pool.get_entry(model_id)
    estimate = entry.estimated_size
    engine = _load(pool, model_id)
    assert pool._entry_resident_size(entry) == estimate
    assert pool._current_model_memory == estimate
    assert entry.actual_size == estimate  # lazy start: no longer the file size

    # A peak that fits (with 5% headroom) changes nothing.
    engine._observe_memory(int(estimate / 1.06))
    assert pool._entry_resident_size(entry) == estimate
    assert entry.actual_size == int(estimate / 1.06)  # measured, not the charge
    assert pool._current_model_memory == estimate

    # A larger peak raises the charge and the pool total together.
    bigger = estimate * 2
    engine._observe_memory(bigger)
    assert pool._entry_resident_size(entry) == int(bigger * 1.05)
    assert pool._current_model_memory == int(bigger * 1.05)
    assert entry.actual_size == bigger

    # A later smaller peak never lowers it.
    engine._observe_memory(estimate // 2)
    assert pool._entry_resident_size(entry) == int(bigger * 1.05)


def test_pool_reload_keeps_the_learned_charge(tmp_path, mlx_vlm_h3, monkeypatch):
    pool, model_id = _image_pool(tmp_path, monkeypatch, sized=True)
    entry = pool.get_entry(model_id)
    engine = _load(pool, model_id)
    learned = int(entry.estimated_size * 3 * 1.05)
    engine._observe_memory(entry.estimated_size * 3)

    async def unload():
        async def no_clear(_model_id):
            pass

        pool._metal_sync_clear_cache = no_clear
        await pool._stop_and_unload_engine(model_id)

    asyncio.run(unload())
    assert entry.engine is None and pool._current_model_memory == 0
    # Admission for the next load already uses the learned charge.
    assert pool._entry_resident_size(entry) == learned
    _load(pool, model_id)
    assert pool._current_model_memory == learned


def test_unload_settles_against_the_measured_resident_bytes(
    tmp_path, mlx_vlm_h3, monkeypatch, caplog
):
    """11/15: 21.55 GiB freed against need>=71.82 GiB stalled the barrier."""
    from omlx import engine_pool as engine_pool_module

    pool, model_id = _image_pool(tmp_path, monkeypatch, sized=True)
    engine = _load(pool, model_id)
    entry = pool.get_entry(model_id)
    # Charge the real H3-8bit estimate, as a load of that checkpoint would.
    entry.runtime_estimated_size = entry.runtime_settle_size = 53_788_367_769
    gauge = {"active": 40 * GB}
    engine._resident_bytes = 23 * GB

    async def stop():
        gauge["active"] -= int(21.55 * 1024**3)

    async def no_clear(_model_id):
        pass

    engine.stop = stop
    pool._metal_sync_clear_cache = no_clear
    monkeypatch.setattr(engine_pool_module.mx, "get_active_memory", lambda: gauge["active"])
    monkeypatch.setattr(engine_pool_module, "get_phys_footprint", lambda: 0)
    with caplog.at_level("INFO", logger="omlx.engine_pool"):
        asyncio.run(pool._stop_and_unload_engine(model_id))
    assert "(settled)" in caplog.text
    assert "stalled" not in caplog.text


# ---------------------------------------------------------------------------
# Progress: pipeline callback -> engine relay -> SSE
# ---------------------------------------------------------------------------


DENOISE = [("denoise", step, 6) for step in range(7)]
H3_STAGES = (
    [("cache_adaln", unit, 30) for unit in range(31)]
    + DENOISE
    + [("decode", 0, 1), ("decode", 1, 1)]
)
H3_CACHED_STAGES = [("cache_adaln", 30, 30)] + DENOISE + [
    ("decode", 0, 1), ("decode", 1, 1)]


class _ReportingModel:
    """An mlx-vlm image model whose pipeline reports progress.

    It calls ``request.progress_callback`` the way H3-Image's
    ``generate_array`` does: on the thread that runs the model, once per
    report, before returning.
    """

    is_image_generation_model = True
    is_image_edit_model = True

    def __init__(self, reports_per_image):
        self.reports_per_image = list(reports_per_image)
        self.requests = []
        self.threads = []

    def _run(self, request):
        self.requests.append(request)
        self.threads.append(threading.get_ident())
        reports = self.reports_per_image[
            min(len(self.requests), len(self.reports_per_image)) - 1
        ]
        for stage, completed, total in reports:
            if request.progress_callback is not None:
                request.progress_callback(stage, completed, total)
        width = request.width or 8
        height = request.height or 6
        return SimpleNamespace(
            array=np.zeros((height, width, 3), np.uint8),
            seed=request.seed,
            peak_memory=0.0,
            metadata={},
        )

    def generate(self, request):
        return self._run(request)

    def edit(self, request):
        return self._run(request)


class _ReportingEngine(ImageGenerationEngine):
    """The real engine around a reporting model (no weights)."""

    def __init__(self, reports_per_image):
        super().__init__("/models/fake-h3")
        self._capabilities = H3
        self._model = _ReportingModel(reports_per_image)


def test_engine_relays_progress_from_the_mlx_thread_to_the_loop(caplog):
    engine = _ReportingEngine([DENOISE])
    seen = []

    async def main():
        loop_thread = threading.get_ident()

        def on_progress(index, stage, completed, total):
            seen.append((index, stage, completed, total, threading.get_ident()))

        await engine.run("generate", [_job(1), _job(2)], on_progress=on_progress)
        return loop_thread

    with caplog.at_level("INFO", logger="omlx.engine.image_generation"):
        loop_thread = asyncio.run(main())
    assert [entry[:4] for entry in seen] == [
        (index, *report) for index in (0, 1) for report in DENOISE
    ]
    assert {entry[4] for entry in seen} == {loop_thread}
    assert loop_thread not in engine._model.threads  # pipeline ran elsewhere
    steps = [r.getMessage() for r in caplog.records if "progress:" in r.getMessage()]
    assert len(steps) == 12 and "image=1 step 6/6" in steps[-1]
    # The callback is carried by mlx-vlm's own request classes (a subclass
    # until the pin has the field), so the pipeline's checks still pass.
    from mlx_vlm.generate.image import ImageGenerationRequest

    assert all(isinstance(r, ImageGenerationRequest) for r in engine._model.requests)


def test_progress_handler_failures_never_fail_the_image():
    engine = _ReportingEngine([DENOISE])

    def broken(*_args):
        raise RuntimeError("client handler bug")

    images = asyncio.run(engine.run("generate", [_job(3)], on_progress=broken))
    assert [image.seed for image in images] == [3]


def test_requests_without_a_handler_still_get_a_logging_callback():
    engine = _ReportingEngine([DENOISE])
    asyncio.run(engine.run("generate", [_job(1)]))
    assert callable(engine._model.requests[0].progress_callback)


def _progress_server(server, reports_per_image):
    engine = _ReportingEngine(reports_per_image)
    server.pool.engine = engine
    return engine


def _types(events):
    return [e if e == "[DONE]" else e["type"] for e in events]


def test_streaming_sends_one_progress_event_per_denoising_step(server):
    _progress_server(server, [DENOISE])
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "size": "8x6",
              "seed": 7, "stream": True},
    )
    assert response.status_code == 200
    events = _events(response)
    assert _types(events) == ["started"] + ["progress"] * 6 + ["result", "[DONE]"]
    progress = events[1:7]
    assert [(e["image_index"], e["stage"], e["step"], e["total_steps"])
            for e in progress] == [(0, "denoise", step, 6) for step in range(1, 7)]
    elapsed = [e["elapsed_s"] for e in progress]
    assert elapsed == sorted(elapsed) and all(isinstance(v, float) for v in elapsed)
    assert set(progress[0]) == {
        "type", "image_index", "stage", "step", "total_steps", "elapsed_s"}
    assert events[7]["data"][0]["seed"] == 7
    assert _wait_released(server.pool) == 1


def test_streaming_progress_numbers_each_image(server):
    _progress_server(server, [DENOISE])
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "size": "8x6",
              "seed": 7, "n": 2, "stream": True},
    )
    events = _events(response)
    assert _types(events) == ["started"] + ["progress"] * 12 + ["result", "[DONE]"]
    assert [(e["image_index"], e["step"]) for e in events[1:13]] == [
        (index, step) for index in (0, 1) for step in range(1, 7)]
    assert [item["seed"] for item in events[13]["data"]] == [7, 8]


def test_streaming_other_stages_carry_stage_not_step(server):
    _progress_server(server, [H3_STAGES, H3_CACHED_STAGES])
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "size": "8x6",
              "n": 2, "stream": True},
    )
    events = _events(response)
    progress = [e for e in events if e != "[DONE]" and e["type"] == "progress"]
    summary = [
        (e["image_index"], e["stage"], e.get("step", e.get("completed")))
        for e in progress
    ]
    # First image: the AdaLN cache is built (start and end; the 29 reports in
    # between arrive within the throttle interval), then six steps, then the
    # VAE decode. Second image: the cache is reused, so it is not reported.
    assert summary == (
        [(0, "cache_adaln", 0), (0, "cache_adaln", 30)]
        + [(0, "denoise", step) for step in range(1, 7)]
        + [(0, "decode", 0), (0, "decode", 1)]
        + [(1, "denoise", step) for step in range(1, 7)]
        + [(1, "decode", 0), (1, "decode", 1)]
    )
    for event in progress:
        if event["stage"] == "denoise":
            assert "completed" not in event and event["total_steps"] == 6
        else:
            assert "step" not in event and "total_steps" not in event
            assert event["total"] == (30 if event["stage"] == "cache_adaln" else 1)
    assert _types(events)[-2:] == ["result", "[DONE]"]


def test_streaming_edit_reports_progress(server, tmp_path):
    _progress_server(server, [DENOISE])
    reference = tmp_path / "ref.png"
    Image.new("RGB", (4, 4), "red").save(reference)
    response = server.client.post(
        "/v1/images/edits",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "blue",
              "image": str(reference), "stream": True},
    )
    events = _events(response)
    assert _types(events) == ["started"] + ["progress"] * 6 + ["result", "[DONE]"]


def test_non_streaming_requests_ignore_progress(server):
    _progress_server(server, [H3_STAGES])
    response = server.client.post(
        "/v1/images/generations",
        json={"model": "MiniMax-H3-Image-8bit", "prompt": "a cat", "size": "8x6"},
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_progress_request_classes_keep_mlx_vlm_semantics():
    import dataclasses

    from mlx_vlm.generate.edit_image import ImageEditRequest
    from mlx_vlm.generate.image import ImageGenerationRequest

    from omlx.patches.mlx_vlm_image_progress import (
        _with_progress_field,
        request_classes,
    )

    generation, edit = request_classes()
    request = generation("a cat", progress_callback=print)
    copy = dataclasses.replace(request, seed=4)
    assert isinstance(copy, ImageGenerationRequest)
    assert copy.progress_callback is print and copy.seed == 4
    assert edit("blue", "one.png").image_paths == ("one.png",)  # __post_init__
    assert isinstance(edit("blue", ("a.png",)), ImageEditRequest)

    @dataclasses.dataclass
    class Upstream:  # a pin that already has the field
        prompt: str
        progress_callback: object = None

    assert _with_progress_field(Upstream) is Upstream


_VENDOR_CHECK = r"""
import importlib, inspect, json, logging
from types import SimpleNamespace
import mlx.core as mx
from omlx.image_generation import ensure_image_patches
ensure_image_patches()
from omlx.patches.mlx_vlm_image_progress import request_classes
import mlx_vlm.models.minimax_h3_image as package
pipeline_module = importlib.import_module("mlx_vlm.models.minimax_h3_image.pipeline")
model_module = importlib.import_module("mlx_vlm.models.minimax_h3_image.model")
from mlx_vlm.generate.image import generate_image
from mlx_vlm.models.minimax_h3_image import (
    MiniMaxH3ImageTransformer, MiniMaxH3ImageTransformerConfig,
    MiniMaxH3ImageVAE, MiniMaxH3ImageVAEConfig,
)

class Tokenizer:
    def encode(self, text, **kwargs):
        return [12]

pipeline = pipeline_module.MiniMaxH3ImagePipeline(
    "/synthetic", evict_text_encoder=False, upsample_prompt=False)
pipeline._conditioner = SimpleNamespace(
    tokenizer=Tokenizer(),
    _encode=lambda ids, tags, **kw: SimpleNamespace(
        hidden_states=mx.ones((1, len(ids), 32)) * 0.01))
pipeline._transformer = MiniMaxH3ImageTransformer(MiniMaxH3ImageTransformerConfig(
    hidden_size=32, num_layers=2, num_refiner_layers=1, num_attention_heads=4,
    attention_head_dim=8, ffn_dim=64, in_channels=2, text_dim=32, freq_dim=8,
    time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=1))
pipeline._vae = MiniMaxH3ImageVAE(MiniMaxH3ImageVAEConfig(
    block_out_channels=(4, 4, 4, 4, 4, 4), layers_per_block=1, norm_num_groups=2,
    latent_channels=2, latents_mean=(0.2, -0.3), latents_std=(0.8, 1.2),
    decoder_num_layers=1, decoder_num_attention_heads=2,
    decoder_attention_head_dim=8))
model = model_module.MiniMaxH3ImageGenerationModel(pipeline, "synthetic")
events = []
generation, _ = request_classes()
request = generation("a teapot", seed=1, width=64, height=64,
                     extra={"min_generation_area": 1024},
                     progress_callback=lambda *report: events.append(report))
first = generate_image(model, request)
second = generate_image(model, request)
print(json.dumps({
    "package": package.__file__,
    "pipeline": pipeline_module.__file__,
    "model": model_module.__file__,
    "transformer": importlib.import_module(
        "mlx_vlm.models.minimax_h3_image.transformer").__file__,
    "events": events,
    "shape": list(first.array.shape),
    "same": bool(mx.array_equal(first.array, second.array).item()),
}))
"""


def test_vendored_h3_pipeline_reports_progress_on_the_pinned_mlx_vlm():
    """Fresh interpreter: the patch has to run before H3 modules are imported."""
    import subprocess
    import sys

    pytest.importorskip("mlx_vlm.models.minimax_h3_image")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", _VENDOR_CHECK],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "PYTHONPATH": str(root)},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    vendor = root / "omlx/patches/mlx_vlm_image_progress/vendor"
    assert Path(result["pipeline"]).is_relative_to(vendor)
    assert Path(result["model"]).is_relative_to(vendor)
    assert not Path(result["package"]).is_relative_to(vendor)
    assert not Path(result["transformer"]).is_relative_to(vendor)
    events = [tuple(event) for event in result["events"]]
    cache_work = 6 * (2 + 1)  # six tables x (two transformer blocks + output)
    first = (
        [("cache_adaln", unit, cache_work) for unit in range(cache_work + 1)]
        + [("denoise", step, 6) for step in range(7)]
        + [("decode", 0, 1), ("decode", 1, 1)]
    )
    second = (
        [("cache_adaln", cache_work, cache_work)]
        + [("denoise", step, 6) for step in range(7)]
        + [("decode", 0, 1), ("decode", 1, 1)]
    )
    assert events == first + second
    assert result["shape"] == [64, 64, 3] and result["same"]
