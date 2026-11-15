# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style image endpoints: generations, edits and sampling defaults.

Request and response bodies follow mlx-vlm's own server
(``mlx_vlm/server/schemas.py``: ``ImageGenerationRequest``,
``ImageEditRequest``, ``ImageGenerationResponse``; ``ImageSamplingDefaults``).
oMLX does not import ``mlx_vlm.server`` (that builds mlx-vlm's own app), so the
fields are mirrored here and ``tests/test_image_generation.py`` compares them
with mlx-vlm's schemas. oMLX extensions:

- ``stream: true`` answers with Server-Sent Events (``started``, ``progress``,
  ``result``, ``error``, then ``[DONE]``; SSE comment lines keep the
  connection alive). ``progress`` needs a pipeline that reports it (see
  :class:`_ProgressEvents`).
- ``/v1/images/edits`` also accepts ``data:image/...;base64,`` URLs. They are
  written to a temporary folder for the request and removed afterwards.
- Images are returned as ``b64_json`` only. ``response_format: "path"``,
  ``output_path`` and ``output_dir`` (mlx-vlm writes server-side files there)
  are rejected, so an API key cannot write files on the server.
- Prompt expansion never downloads: see :mod:`omlx.image_generation`.

The model runs inside the EnginePool like any other model: a request leases
the engine, so it cannot be evicted mid-generation, and a request for another
model meanwhile gets the pool's 409 ``ModelBusyError`` instead of waiting.
Requests for the same image model run one after another.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import random
import re
import shutil
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..image_generation import (
    MODEL_TYPE,
    PromptExpanderMissing,
    image_capabilities,
    resolve_prompt_expansion,
)

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_IMAGE_SIZE = "512x512"
KEEPALIVE_SECONDS = 15.0
# Stages other than denoising can report many small units (H3-Image's AdaLN
# cache counts every block); between their start and end send at most this
# often.
PROGRESS_STAGE_INTERVAL_SECONDS = 1.0
_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+)?(;[\w=.+-]+)*;base64,", re.I)


# ---------------------------------------------------------------------------
# Schemas (mirrors mlx_vlm.server.schemas)
# ---------------------------------------------------------------------------


class _FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ImageGenerationRequest(_FlexibleModel):
    prompt: str = Field(..., description="Text prompt for image generation.")
    model: str = Field("", description="Image generation model id (/v1/models).")
    n: int = Field(1, ge=1, le=10, description="Number of images to generate.")
    size: Optional[str] = Field(
        DEFAULT_IMAGE_SIZE,
        description="Image size as WIDTHxHEIGHT. Width/height fields override this.",
    )
    width: Optional[int] = Field(None, description="Generated image width.")
    height: Optional[int] = Field(None, description="Generated image height.")
    steps: Optional[int] = Field(None, ge=1, description="Inference steps.")
    seed: Optional[int] = Field(None, description="Base seed; output i uses seed+i.")
    guidance: Optional[float] = Field(None, description="Guidance scale.")
    auto_json_caption: Optional[bool] = Field(None)
    prompt_expansion_model: Optional[str] = Field(None)
    upsample_prompt: Optional[bool] = Field(None)
    boxes: Optional[list[dict[str, Any]]] = Field(None)
    response_format: Literal["b64_json", "path"] = Field("b64_json")
    output_format: Literal["png"] = Field("png")
    output_path: Optional[str] = Field(None)
    output_dir: Optional[str] = Field(None)
    user: Optional[str] = Field(None)
    # oMLX extension
    stream: bool = Field(False, description="Stream started/result events (SSE).")


class ImageEditRequest(_FlexibleModel):
    prompt: str = Field(..., description="Text prompt for image editing.")
    image: Union[str, List[str]] = Field(
        ..., description="Local paths or data URLs of the reference images."
    )
    model: str = Field(..., description="Image edit model id (/v1/models).")
    n: int = Field(1, ge=1, le=10)
    size: Optional[str] = Field(None)
    width: Optional[int] = Field(None)
    height: Optional[int] = Field(None)
    steps: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = Field(None)
    guidance: Optional[float] = Field(None)
    negative_prompt: Optional[str] = Field(None)
    output_resolution: Optional[int] = Field(None, ge=256)
    use_kv_cache: Optional[bool] = Field(None)
    prompt_expansion_model: Optional[str] = Field(None)
    upsample_prompt: Optional[bool] = Field(None)
    boxes: Optional[list[dict[str, Any]]] = Field(None)
    response_format: Literal["b64_json", "path"] = Field("b64_json")
    output_format: Literal["png"] = Field("png")
    output_path: Optional[str] = Field(None)
    output_dir: Optional[str] = Field(None)
    user: Optional[str] = Field(None)
    # oMLX extension
    stream: bool = Field(False)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value):
        images = [value] if isinstance(value, str) else list(value)
        if not images:
            raise ValueError("At least one image is required.")
        if not all(isinstance(item, str) and item for item in images):
            raise ValueError("Image paths must be non-empty strings.")
        return value


class ImageGenerationResponseData(BaseModel):
    b64_json: Optional[str] = None
    path: Optional[str] = None
    revised_prompt: Optional[str] = None
    mime_type: str = "image/png"
    width: int
    height: int
    seed: int


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageGenerationResponseData]
    output_format: Literal["png"] = "png"
    size: str


# ---------------------------------------------------------------------------
# Server access (lazy: omlx.server imports this module)
# ---------------------------------------------------------------------------


def _get_engine_pool():
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _wake_enforcer() -> None:
    try:
        from omlx.server import _wake_process_memory_enforcer
    except Exception:  # pragma: no cover - partial server in tests
        return
    _wake_process_memory_enforcer(active=True)


def _log_request_policy(http_request: Request, model_id: str) -> None:
    """Record who asked; no sub-key policy applies to image generation.

    The policies (max_context_window, enable_thinking, thinking_budget_tokens)
    bound a language model's context and reasoning. An image request has no
    chat context or reasoning to bound, so they are logged and not applied.
    """
    try:
        from omlx.server import _build_request_policy, _request_policy_has_overrides
    except Exception:  # pragma: no cover - partial server in tests
        return
    policy = _build_request_policy(http_request)
    if _request_policy_has_overrides(policy):
        logger.info(
            "Request policy not applied to image generation: client=%s "
            "source=%s model=%s (policies bound LLM context and thinking only)",
            policy.get("client") or "(unknown)",
            policy.get("source") or "(unknown)",
            model_id,
        )


def _image_entry(model_id: str):
    pool = _get_engine_pool()
    resolved = _resolve_model(model_id)
    entry = pool.get_entry(resolved)
    if entry is None:
        available = [
            mid
            for mid in pool.get_model_ids()
            if getattr(pool.get_entry(mid), "model_type", None) == MODEL_TYPE
        ]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_id}' not found. Available image models: "
                f"{', '.join(available) if available else '(none)'}"
            ),
        )
    if entry.model_type != MODEL_TYPE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{resolved}' is not an image generation model "
                f"(model_type: {entry.model_type})."
            ),
        )
    return pool, resolved, entry


async def _lease_engine(pool, model_id: str):
    from ..engine.image_generation import ImageGenerationEngine
    from ..exceptions import (
        EnginePoolError,
        InsufficientMemoryError,
        ModelBusyError,
        ModelLoadingError,
        ModelNotFoundError,
        ModelTooLargeError,
        ModelUnavailableError,
    )

    _wake_enforcer()
    try:
        engine = await pool.get_engine(model_id, _lease=True)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ModelTooLargeError, InsufficientMemoryError) as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except (ModelUnavailableError, ModelLoadingError, ModelBusyError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except EnginePoolError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Loading image model %s failed", model_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to load image model: {e}"
        ) from e
    if not isinstance(engine, ImageGenerationEngine):
        await pool.release_engine(model_id)
        raise HTTPException(
            status_code=400, detail=f"Model '{model_id}' is not an image model."
        )
    return engine


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------


def _parse_size(value: str) -> tuple[int, int]:
    from mlx_vlm.generate.image import parse_size

    try:
        return parse_size(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _resolve_size(body, *, required: bool) -> tuple[int | None, int | None]:
    if body.width is not None or body.height is not None:
        if body.width is None or body.height is None:
            raise HTTPException(
                status_code=400,
                detail="Both width and height are required when either is set.",
            )
        return int(body.width), int(body.height)
    if body.size is None:
        if required:
            return _parse_size(DEFAULT_IMAGE_SIZE)
        return None, None
    return _parse_size(body.size)


def _reject_server_paths(body) -> None:
    if body.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail='oMLX returns images as b64_json only; send "response_format": "b64_json".',
        )
    for field in ("output_path", "output_dir"):
        if getattr(body, field) is not None:
            raise HTTPException(
                status_code=400,
                detail=f"{field} is not supported by oMLX; images are returned as b64_json.",
            )


def _model_dirs(pool) -> list[Path]:
    dirs = list(getattr(pool, "_model_dirs", None) or [])
    return [Path(d) for d in dirs]


def _expansion_extra(pool, entry, body) -> dict[str, Any]:
    capabilities = image_capabilities(entry.model_path)
    try:
        expansion = resolve_prompt_expansion(
            capabilities.family,
            body.upsample_prompt,
            body.prompt_expansion_model,
            _model_dirs(pool),
        )
    except PromptExpanderMissing as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return expansion.extra()


def _write_data_url(data_url: str, folder: Path, index: int) -> str:
    match = _DATA_URL_RE.match(data_url)
    if match is None:
        raise HTTPException(
            status_code=400,
            detail=f"image[{index}] is not a base64 data URL.",
        )
    try:
        payload = base64.b64decode(data_url[match.end():], validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"image[{index}] has invalid base64 data."
        ) from e
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            image.verify()
            fmt = (image.format or "png").lower()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"image[{index}] is not a readable image: {e}"
        ) from e
    path = folder / f"image-{index + 1:02d}.{fmt}"
    path.write_bytes(payload)
    return str(path)


def _edit_image_paths(body: ImageEditRequest, folder_factory) -> tuple[str, ...]:
    images = [body.image] if isinstance(body.image, str) else list(body.image)
    paths: list[str] = []
    folder: Path | None = None
    for index, image in enumerate(images):
        if image.startswith("data:"):
            if folder is None:
                folder = folder_factory()
            paths.append(_write_data_url(image, folder, index))
            continue
        path = Path(image).expanduser()
        if not path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail=f"image[{index}] must be an absolute local path or a data URL.",
            )
        if not path.is_file():
            raise HTTPException(
                status_code=400, detail=f"image[{index}] not found: {image}"
            )
        paths.append(str(path))
    return tuple(paths)


def _png_b64(pixels) -> str:
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _response(images, created: int, size: str) -> dict[str, Any]:
    data = []
    for image in images:
        b64 = await asyncio.to_thread(_png_b64, image.pixels)
        data.append(
            ImageGenerationResponseData(
                b64_json=b64,
                revised_prompt=image.revised_prompt,
                width=image.width,
                height=image.height,
                seed=image.seed,
            )
        )
    return ImageGenerationResponse(
        created=created, data=data, output_format="png", size=size
    ).model_dump()


def _http_error(exc: BaseException) -> HTTPException:
    from ..engine.image_generation import ImageGenerationCancelled

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ImageGenerationCancelled):
        return HTTPException(status_code=499, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError, FileNotFoundError)):
        # mlx-vlm validates options inside the pipeline (e.g. H3-Image only
        # accepts six steps and guidance 1).
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Image generation failed: {exc}")


class _ProgressEvents:
    """Turn pipeline progress reports into SSE ``progress`` payloads.

    Denoising steps become ``{"type": "progress", "image_index", "stage":
    "denoise", "step", "total_steps", "elapsed_s"}``, one per finished step
    (``step`` 1..``total_steps``; the "about to start" report is dropped).
    Other stages (H3-Image: ``cache_adaln``, ``decode``) carry ``stage``,
    ``completed`` and ``total`` instead of ``step``: their start, their end,
    and in between at most one per ``PROGRESS_STAGE_INTERVAL_SECONDS``. A
    stage whose first report is already complete (H3-Image reusing its AdaLN
    cache) did no work and is not sent. ``elapsed_s`` counts from the start of
    the generation, after the model was loaded.
    """

    def __init__(self, started: float):
        self._started = started
        self._stages: dict[tuple[int, str], float] = {}

    def __call__(self, index: int, stage: str, completed: int, total: int):
        now = time.monotonic()
        elapsed = round(now - self._started, 1)
        if stage == "denoise":
            if completed <= 0:
                return None
            return {
                "type": "progress",
                "image_index": index,
                "stage": stage,
                "step": completed,
                "total_steps": total,
                "elapsed_s": elapsed,
            }
        key = (index, stage)
        last = self._stages.get(key)
        if last is None:
            if total > 0 and completed >= total:
                return None
        elif completed < total and now - last < PROGRESS_STAGE_INTERVAL_SECONDS:
            return None
        self._stages[key] = now
        return {
            "type": "progress",
            "image_index": index,
            "stage": stage,
            "completed": completed,
            "total": total,
            "elapsed_s": elapsed,
        }


def _release_when_done(pool, model_id: str, task: asyncio.Task, cleanup) -> None:
    """Release the lease once the MLX work is over, even after a disconnect.

    The executor keeps running an image that has started; unloading the model
    under it would free arrays it is still using.
    """

    async def _wait_and_release():
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        finally:
            try:
                cleanup()
            finally:
                await pool.release_engine(model_id)

    asyncio.get_running_loop().create_task(
        _wait_and_release(), name=f"image-lease-release:{model_id}"
    )


async def _serve(
    http_request: Request,
    body,
    *,
    task: str,
    jobs_factory,
    size_for,
    folder_holder: dict,
):
    pool, model_id, entry = _image_entry(body.model)
    capabilities = image_capabilities(entry.model_path)
    if not (capabilities.generate if task == "generate" else capabilities.edit):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' does not support image {'editing' if task == 'edit' else 'generation'}.",
        )
    _reject_server_paths(body)
    _log_request_policy(http_request, model_id)
    extra = _expansion_extra(pool, entry, body)
    jobs = jobs_factory(extra)

    def cleanup():
        folder = folder_holder.get("folder")
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)

    engine = await _lease_engine(pool, model_id)

    cancelled = threading.Event()
    created = int(time.time())
    started = time.monotonic()
    # Streaming only: progress payloads, then None once the work is over.
    # The engine schedules on_progress on this loop before the work finishes,
    # so the None always comes after the last report.
    updates: asyncio.Queue | None = asyncio.Queue() if body.stream else None
    on_progress = None
    if updates is not None:
        progress_events = _ProgressEvents(started)

        def on_progress(index: int, stage: str, completed: int, total: int) -> None:
            payload = progress_events(index, stage, completed, total)
            if payload is not None:
                updates.put_nowait(payload)

    work = asyncio.get_running_loop().create_task(
        engine.run(task, jobs, cancelled=cancelled, on_progress=on_progress),
        name=f"image-{task}:{model_id}",
    )
    if updates is not None:
        work.add_done_callback(lambda _work: updates.put_nowait(None))
    _release_when_done(pool, model_id, work, cleanup)
    folder_holder["handed_off"] = True

    def finished(images) -> None:
        peak = max((image.peak_memory_gb for image in images), default=0.0)
        logger.info(
            "Image %s request: model=%s n=%d %.1fs peak=%.1fGB",
            task,
            model_id,
            len(images),
            time.monotonic() - started,
            peak,
        )

    if not body.stream:
        try:
            images = await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except BaseException as exc:
            raise _http_error(exc) from exc
        finished(images)
        return await _response(images, created, size_for(images))

    async def events() -> AsyncIterator[str]:
        def event(payload: Any) -> str:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            return f"data: {text}\n\n"

        try:
            yield event({"type": "started", "model": model_id, "n": len(jobs)})
            while True:
                try:
                    update = await asyncio.wait_for(
                        updates.get(), timeout=KEEPALIVE_SECONDS
                    )
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if update is None:
                    break
                yield event(update)
            try:
                images = work.result()
            except BaseException as exc:
                error = _http_error(exc)
                yield event({"type": "error", "message": str(error.detail)})
            else:
                finished(images)
                result = await _response(images, created, size_for(images))
                yield event({"type": "result", **result})
            yield event("[DONE]")
        finally:
            if not work.done():
                cancelled.set()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _base_seed(seed: int | None) -> int:
    return int(seed) if seed is not None else random.randrange(2**32)


@router.post("/v1/images/generations")
async def images_generations(http_request: Request, body: ImageGenerationRequest):
    from ..engine.image_generation import ImageJob

    if not body.prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")
    width, height = _resolve_size(body, required=True)
    base_seed = _base_seed(body.seed)

    def jobs_factory(expansion: dict[str, Any]):
        extra: dict[str, Any] = {}
        if body.auto_json_caption is not None:
            extra["auto_json_caption"] = body.auto_json_caption
        if body.boxes is not None:
            extra["boxes"] = body.boxes
        extra.update(expansion)
        return [
            ImageJob(
                prompt=body.prompt,
                seed=base_seed + index,
                steps=body.steps,
                width=width,
                height=height,
                guidance=body.guidance,
                extra=extra,
            )
            for index in range(body.n)
        ]

    return await _serve(
        http_request,
        body,
        task="generate",
        jobs_factory=jobs_factory,
        size_for=lambda images: f"{width}x{height}",
        folder_holder={},
    )


@router.post("/v1/images/edits")
async def images_edits(http_request: Request, body: ImageEditRequest):
    from ..engine.image_generation import ImageJob

    if not body.prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")
    width, height = _resolve_size(body, required=False)
    base_seed = _base_seed(body.seed)
    holder: dict[str, Any] = {}

    def folder_factory() -> Path:
        holder["folder"] = Path(tempfile.mkdtemp(prefix="omlx-image-edit-"))
        return holder["folder"]

    try:
        image_paths = _edit_image_paths(body, folder_factory)
    except BaseException:
        if "folder" in holder:
            shutil.rmtree(holder["folder"], ignore_errors=True)
        raise

    def jobs_factory(expansion: dict[str, Any]):
        extra = {
            key: value
            for key in ("negative_prompt", "output_resolution", "use_kv_cache", "boxes")
            if (value := getattr(body, key)) is not None
        }
        extra.update(expansion)
        return [
            ImageJob(
                prompt=body.prompt,
                seed=base_seed + index,
                steps=body.steps,
                width=width,
                height=height,
                guidance=body.guidance,
                extra=extra,
                image_paths=image_paths,
            )
            for index in range(body.n)
        ]

    def size_for(images) -> str:
        if images:
            return f"{images[0].width}x{images[0].height}"
        return f"{width or 0}x{height or 0}"

    try:
        return await _serve(
            http_request,
            body,
            task="edit",
            jobs_factory=jobs_factory,
            size_for=size_for,
            folder_holder=holder,
        )
    except BaseException:
        # Once the work starts, the lease release removes the folder; before
        # that point nothing else will.
        if "folder" in holder and not holder.get("handed_off"):
            shutil.rmtree(holder["folder"], ignore_errors=True)
        raise


@router.get("/v1/images/defaults")
async def images_defaults(
    model: str = Query(..., min_length=1),
    task: Literal["generate", "edit"] = "generate",
):
    """Model sampling defaults from metadata only; weights are not loaded."""
    _, model_id, entry = _image_entry(model)

    def _resolve():
        from ..image_generation import ensure_image_patches

        ensure_image_patches()
        from mlx_vlm.generate.image_defaults import resolve_image_defaults

        return resolve_image_defaults(entry.model_path, task=task)

    try:
        defaults = await asyncio.to_thread(_resolve)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"steps": defaults.steps, "guidance": defaults.guidance}
