# SPDX-License-Identifier: Apache-2.0
"""Image generation engine for oMLX, backed by mlx-vlm's image pipelines.

The engine holds one mlx-vlm image model (``load_image_generation_model`` or
``load_image_edit_model``). All MLX work runs on the global MLX executor, like
every other engine, so a generation never blocks the event loop and never
races another model's Metal work. Generations on one engine therefore run one
after another; the EnginePool lease keeps the model resident meanwhile.

mlx-vlm's image pipelines run their denoising loop internally and expose no
per-step callback, so the engine reports when an image starts and finishes,
not each step.
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

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__()
        self._model_name = model_name
        self._model: Any = None
        self._capabilities: ImageCapabilities = image_capabilities(model_name)
        self._kwargs = kwargs
        self._generated = 0

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

    # -- lifecycle -----------------------------------------------------------

    def _load_sync(self, task: ImageTask) -> Any:
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
        await loop.run_in_executor(get_mlx_executor(), self._ensure_model_sync, task)
        logger.info("Image engine started: %s", self._model_name)

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
        logger.info("Image engine stopped: %s", self._model_name)

    # -- generation ----------------------------------------------------------

    def _run_one_sync(self, task: ImageTask, job: ImageJob) -> GeneratedImage:
        model = self._ensure_model_sync(task)
        mx.reset_peak_memory()
        started = time.monotonic()
        if task == "generate":
            from mlx_vlm.generate.image import ImageGenerationRequest, generate_image

            result = generate_image(
                model,
                ImageGenerationRequest(
                    prompt=job.prompt,
                    seed=job.seed,
                    steps=job.steps,
                    width=int(job.width),
                    height=int(job.height),
                    guidance=job.guidance,
                    extra=dict(job.extra),
                ),
            )
        else:
            from mlx_vlm.generate.edit_image import ImageEditRequest, edit_image

            result = edit_image(
                model,
                ImageEditRequest(
                    prompt=job.prompt,
                    image_paths=job.image_paths,
                    seed=job.seed,
                    steps=job.steps,
                    width=job.width,
                    height=job.height,
                    guidance=job.guidance,
                    extra=dict(job.extra),
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
    ) -> list[GeneratedImage]:
        """Generate ``jobs`` in order on the MLX executor.

        ``cancelled`` is checked between images: an image already running
        finishes, the rest are skipped.
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
                image = await loop.run_in_executor(
                    get_mlx_executor(), self._run_one_sync, task, job
                )
                results.append(image)
                self._generated += 1
                self._update_activity(activity_id, completed_items=index + 1)
                logger.info(
                    "Image %s done: model=%s %dx%d seed=%d %.1fs peak=%.1fGB%s",
                    task,
                    self._model_name,
                    image.width,
                    image.height,
                    image.seed,
                    image.elapsed_s,
                    image.peak_memory_gb,
                    " (prompt expanded)" if image.revised_prompt else "",
                )
                if on_image is not None:
                    outcome = on_image(index, image)
                    if asyncio.iscoroutine(outcome):
                        await outcome
            return results
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "family": self._capabilities.family,
            "generate": self._capabilities.generate,
            "edit": self._capabilities.edit,
            "images_generated": self._generated,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<ImageGenerationEngine model={self._model_name} status={status}>"
