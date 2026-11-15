# SPDX-License-Identifier: Apache-2.0
"""Image generation engine for oMLX, backed by mlx-vlm's image pipelines.

The engine holds one mlx-vlm image model (``load_image_generation_model`` or
``load_image_edit_model``). All MLX work runs on the global MLX executor, like
every other engine, so a generation never blocks the event loop and never
races another model's Metal work. Generations on one engine therefore run one
after another; the EnginePool lease keeps the model resident meanwhile.

Progress. Pipelines that take a ``progress_callback(stage, completed, total)``
(MiniMax-H3-Image, through ``omlx.patches.mlx_vlm_image_progress`` until the
mlx-vlm pin has Neywa-Labs/mlx-vlm#2742) call it on the MLX executor thread
after each stage's work is evaluated. The engine logs denoising steps there
and hands every report to ``run(on_progress=...)`` on the event loop with
``call_soon_threadsafe``, which never blocks the generation. Other pipelines
ignore the callback, so only the start and end of each image are known.

Memory. ``mx.get_peak_memory()`` and ``mx.get_active_memory()`` count every
model in the process. Because all MLX work is serialized on the one MLX
executor thread, nothing else allocates while ``_run_one_sync`` runs, so the
engine can separate its own share: it tracks the bytes it keeps resident
(``resident_bytes``) and, per image, the peak it reached on top of the other
models (``GeneratedImage.footprint_peak_bytes``). Pipelines such as
MiniMax-H3-Image load components lazily and drop some after each request, so
neither number is the checkpoint's file size. The pool charges the larger of
its estimate and the peaks reported through ``memory_observer``, and settles
an unload against ``resident_bytes``.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Sequence

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from ..image_generation import ImageCapabilities, image_capabilities
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

ImageTask = Literal["generate", "edit"]
# (image_index, stage, completed, total); runs on the event loop.
ImageProgressHandler = Callable[[int, str, int, int], Any]


class ImageGenerationCancelled(Exception):
    """The client went away before the remaining images were generated."""


@dataclass
class GeneratedImage:
    """One finished image, converted off the MLX array."""

    pixels: np.ndarray  # HWC uint8
    width: int
    height: int
    seed: int
    revised_prompt: str | None
    metadata: Dict[str, Any]
    peak_memory_gb: float
    elapsed_s: float
    # This engine's own peak: what it held before the request plus the most
    # the request allocated on top. Excludes other models in the process.
    footprint_peak_bytes: int = 0


@dataclass(frozen=True)
class ImageJob:
    """Everything one image needs, already validated by the API layer."""

    prompt: str
    seed: int
    steps: int | None
    width: int | None
    height: int | None
    guidance: float | None
    extra: Dict[str, Any]
    image_paths: tuple[str, ...] = ()


def _supports(model: Any, task: ImageTask) -> bool:
    if model is None:
        return False
    cls = type(model)
    if task == "generate":
        return bool(getattr(cls, "is_image_generation_model", False)) and callable(
            getattr(model, "generate", None)
        )
    return bool(getattr(cls, "is_image_edit_model", False)) and callable(
        getattr(model, "edit", None)
    )


class ImageGenerationEngine(BaseNonStreamingEngine):
    """Serve ``/v1/images/generations`` and ``/v1/images/edits`` for one model."""

    def __init__(
        self,
        model_name: str,
        *,
        memory_observer: Callable[[int], Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._model_name = model_name
        self._model: Any = None
        self._capabilities: ImageCapabilities = image_capabilities(model_name)
        self._kwargs = kwargs
        self._generated = 0
        self._memory_observer = memory_observer
        self._resident_bytes = 0
        self._observed_peak_bytes = 0
        self._failed_peak_bytes = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ImageCapabilities:
        return self._capabilities

    @property
    def family(self) -> str | None:
        return self._capabilities.family

    def supports(self, task: ImageTask) -> bool:
        return self._capabilities.generate if task == "generate" else self._capabilities.edit

    @property
    def resident_bytes(self) -> int:
        """Bytes this engine keeps allocated between requests (measured)."""
        return self._resident_bytes

    @property
    def observed_peak_bytes(self) -> int:
        """Largest own peak any request on this engine has reached."""
        return self._observed_peak_bytes

    def _track_resident(self, active_before: int) -> None:
        """Fold the allocation change of one executor call into the tally."""
        delta = mx.get_active_memory() - active_before
        self._resident_bytes = max(0, self._resident_bytes + delta)

    # -- lifecycle -----------------------------------------------------------

    def _load_sync(self, task: ImageTask) -> Any:
        from ..image_generation import ensure_image_patches

        ensure_image_patches()
        if task == "generate":
            from mlx_vlm.generate.image import load_image_generation_model

            return load_image_generation_model(self._model_name)
        from mlx_vlm.generate.edit_image import load_image_edit_model

        return load_image_edit_model(self._model_name)

    def _ensure_model_sync(self, task: ImageTask) -> Any:
        """Return a model for ``task``; runs on the MLX executor."""
        if _supports(self._model, task):
            return self._model
        if self._model is not None:
            # Families whose edit pipeline is a different class: hold one at a time.
            logger.info("Reloading %s for image %s", self._model_name, task)
            self._model = None
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        self._model = self._load_sync(task)
        return self._model

    async def start(self) -> None:
        if self._model is not None:
            return
        if not self._capabilities.any:
            raise ValueError(
                f"mlx-vlm does not recognize {self._model_name} as an image "
                f"generation or editing model"
            )
        task: ImageTask = "generate" if self._capabilities.generate else "edit"
        logger.info("Starting image engine: %s (%s)", self._model_name, task)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._start_sync, task)
        logger.info("Image engine started: %s", self._model_name)

    def _start_sync(self, task: ImageTask) -> None:
        active_before = mx.get_active_memory()
        self._ensure_model_sync(task)
        self._track_resident(active_before)

    async def stop(self) -> None:
        if self._model is None:
            return
        logger.info("Stopping image engine: %s", self._model_name)
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        self._resident_bytes = 0
        logger.info("Image engine stopped: %s", self._model_name)

    # -- generation ----------------------------------------------------------

    def _run_one_sync(
        self,
        task: ImageTask,
        job: ImageJob,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> GeneratedImage:
        # Measure from before a possible generate/edit reload so its load
        # counts toward this request.
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()
        try:
            image = self._generate_sync(task, job, progress)
        except BaseException:
            # A failed request still left what it loaded and still peaked;
            # run() reports the peak so a near-OOM raises the pool's charge.
            mx.clear_cache()
            self._failed_peak_bytes = self._footprint_peak(active_before)
            self._track_resident(active_before)
            raise
        image.footprint_peak_bytes = self._footprint_peak(active_before)
        self._track_resident(active_before)
        return image

    def _footprint_peak(self, active_before: int) -> int:
        return int(
            self._resident_bytes + max(0, mx.get_peak_memory() - active_before)
        )

    def _generate_sync(
        self,
        task: ImageTask,
        job: ImageJob,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> GeneratedImage:
        from ..patches.mlx_vlm_image_progress import request_classes

        model = self._ensure_model_sync(task)
        started = time.monotonic()
        generation_request, edit_request = request_classes()
        if task == "generate":
            from mlx_vlm.generate.image import generate_image

            result = generate_image(
                model,
                generation_request(
                    prompt=job.prompt,
                    seed=job.seed,
                    steps=job.steps,
                    width=int(job.width),
                    height=int(job.height),
                    guidance=job.guidance,
                    extra=dict(job.extra),
                    progress_callback=progress,
                ),
            )
        else:
            from mlx_vlm.generate.edit_image import edit_image

            result = edit_image(
                model,
                edit_request(
                    prompt=job.prompt,
                    image_paths=job.image_paths,
                    seed=job.seed,
                    steps=job.steps,
                    width=job.width,
                    height=job.height,
                    guidance=job.guidance,
                    extra=dict(job.extra),
                    progress_callback=progress,
                ),
            )
        pixels = np.array(result.array)
        if pixels.ndim == 4:
            pixels = pixels[0]
        elapsed = time.monotonic() - started
        peak = max(float(result.peak_memory or 0.0), mx.get_peak_memory() / 1e9)
        metadata = dict(result.metadata or {})
        seed = int(result.seed)
        del result
        mx.clear_cache()
        return GeneratedImage(
            pixels=pixels,
            width=int(pixels.shape[1]),
            height=int(pixels.shape[0]),
            seed=seed,
            revised_prompt=metadata.get("revised_prompt"),
            metadata=metadata,
            peak_memory_gb=peak,
            elapsed_s=elapsed,
        )

    async def run(
        self,
        task: ImageTask,
        jobs: Sequence[ImageJob],
        *,
        cancelled: threading.Event | None = None,
        on_image: Callable[[int, GeneratedImage], Any] | None = None,
        on_progress: ImageProgressHandler | None = None,
    ) -> list[GeneratedImage]:
        """Generate ``jobs`` in order on the MLX executor.

        ``cancelled`` is checked between images: an image already running
        finishes, the rest are skipped. ``on_progress(index, stage, completed,
        total)`` receives the pipeline's progress reports on the event loop.
        """
        if not self.supports(task):
            raise ValueError(f"{self._model_name} does not support image {task}")
        activity_id = self._begin_activity(
            f"image {task}",
            detail=f"Image {task}",
            total_items=len(jobs),
            metadata={"completed_items": 0},
        )
        loop = asyncio.get_running_loop()
        results: list[GeneratedImage] = []
        try:
            for index, job in enumerate(jobs):
                if cancelled is not None and cancelled.is_set():
                    raise ImageGenerationCancelled(
                        f"cancelled after {index} of {len(jobs)} images"
                    )
                self._failed_peak_bytes = 0
                progress = self._progress_relay(loop, task, index, on_progress)
                try:
                    image = await loop.run_in_executor(
                        get_mlx_executor(), self._run_one_sync, task, job, progress
                    )
                except BaseException:
                    self._observe_memory(self._failed_peak_bytes)
                    raise
                results.append(image)
                self._generated += 1
                self._observe_memory(image.footprint_peak_bytes)
                self._update_activity(activity_id, completed_items=index + 1)
                logger.info(
                    "Image %s done: model=%s %dx%d seed=%d %.1fs peak=%.1fGB "
                    "(own %.1fGB, resident %.1fGB)%s",
                    task,
                    self._model_name,
                    image.width,
                    image.height,
                    image.seed,
                    image.elapsed_s,
                    image.peak_memory_gb,
                    image.footprint_peak_bytes / 1e9,
                    self._resident_bytes / 1e9,
                    " (prompt expanded)" if image.revised_prompt else "",
                )
                if on_image is not None:
                    outcome = on_image(index, image)
                    if asyncio.iscoroutine(outcome):
                        await outcome
            return results
        finally:
            await self._finish_activity(activity_id)

    def _progress_relay(
        self,
        loop: asyncio.AbstractEventLoop,
        task: ImageTask,
        index: int,
        handler: ImageProgressHandler | None,
    ) -> Callable[[str, int, int], None]:
        """The callback the pipeline calls on the MLX executor thread.

        It only logs and schedules ``handler`` on ``loop``; it never waits for
        the loop, so a slow or vanished client cannot stall the generation,
        and its own failures never fail the image.
        """
        started = time.monotonic()
        model_name = self._model_name

        def deliver(stage: str, completed: int, total: int) -> None:
            try:
                outcome = handler(index, stage, completed, total)
                if asyncio.iscoroutine(outcome):
                    loop.create_task(outcome)
            except Exception:
                logger.exception("Image progress handler failed for %s", model_name)

        def relay(stage: str, completed: int, total: int) -> None:
            try:
                stage, completed, total = str(stage), int(completed), int(total)
                elapsed = time.monotonic() - started
                if stage == "denoise" and completed > 0:
                    logger.info(
                        "Image %s progress: model=%s image=%d step %d/%d %.1fs",
                        task, model_name, index, completed, total, elapsed,
                    )
                elif completed in (0, total):
                    logger.debug(
                        "Image %s progress: model=%s image=%d %s %d/%d %.1fs",
                        task, model_name, index, stage, completed, total, elapsed,
                    )
                if handler is not None:
                    loop.call_soon_threadsafe(deliver, stage, completed, total)
            except RuntimeError:
                pass  # the event loop is closed: nobody is listening anymore
            except Exception:
                logger.exception("Image progress relay failed for %s", model_name)

        return relay

    def _observe_memory(self, footprint_peak_bytes: int) -> None:
        if footprint_peak_bytes <= 0:
            return
        self._observed_peak_bytes = max(
            self._observed_peak_bytes, footprint_peak_bytes
        )
        if self._memory_observer is None:
            return
        try:
            self._memory_observer(footprint_peak_bytes)
        except Exception:  # accounting must never fail a finished image
            logger.exception("Image memory observer failed for %s", self._model_name)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "family": self._capabilities.family,
            "generate": self._capabilities.generate,
            "edit": self._capabilities.edit,
            "images_generated": self._generated,
            "resident_bytes": self._resident_bytes,
            "observed_peak_bytes": self._observed_peak_bytes,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<ImageGenerationEngine model={self._model_name} status={status}>"
