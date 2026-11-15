# SPDX-License-Identifier: Apache-2.0
"""Per-step progress for mlx-vlm's MiniMax-H3-Image pipeline.

The pinned mlx-vlm runs H3-Image's denoising loop with no callback:
``MiniMaxH3ImagePipeline.generate_array`` takes none, and
``MiniMaxH3ImageGenerationModel._run`` rejects unknown ``extra`` options.
Neywa-Labs/mlx-vlm#2742 adds ``progress_callback(stage, completed, total)`` to
``ImageGenerationRequest``/``ImageEditRequest`` and reports ``cache_adaln``,
``denoise`` (six steps) and ``decode``, each after its ``mx.eval``.

Until the pin includes that PR this patch

- puts ``vendor/mlx_vlm/models/minimax_h3_image`` in front of the *stock*
  package's ``__path__``, so its ``model`` and ``pipeline`` submodules load
  from the PR versions. The package ``__init__`` and every other submodule
  (transformer, VAE, upsampler, packing, weights) stay the installed ones;
- provides request classes that carry ``progress_callback``
  (:func:`request_classes`).

It must run before anything imports ``minimax_h3_image.model`` or
``.pipeline``; oMLX calls it from :mod:`omlx.image_generation` before every
mlx-vlm image lookup. With a pin whose request classes already have the field,
the patch does nothing and the stock classes are used. Remove the patch when
the pin includes the PR (``PATCHES.md``).
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import sys
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PACKAGE = "mlx_vlm.models.minimax_h3_image"
_VENDOR_PACKAGE = (
    Path(__file__).resolve().parent
    / "vendor"
    / "mlx_vlm"
    / "models"
    / "minimax_h3_image"
)
_VENDORED = ("pipeline", "model")

_LOCK = threading.Lock()
# None: not tried yet; "vendor", "upstream", "unavailable" afterwards.
_STATE: str | None = None

ImageProgressCallback = Callable[[str, int, int], None]


def _has_progress_field(cls: type) -> bool:
    return dataclasses.is_dataclass(cls) and any(
        field.name == "progress_callback" for field in dataclasses.fields(cls)
    )


def apply_mlx_vlm_image_progress_patch() -> bool:
    """Load the vendored H3-Image modules once; True if they are in use."""
    global _STATE
    if _STATE is not None:
        return _STATE == "vendor"
    with _LOCK:
        if _STATE is not None:
            return _STATE == "vendor"
        _STATE = _apply()
        return _STATE == "vendor"


def _apply() -> str:
    try:
        from mlx_vlm.generate.image import ImageGenerationRequest
    except Exception as exc:  # noqa: BLE001 - mlx-vlm missing or too old
        logger.debug("H3-Image progress patch skipped: %s", exc)
        return "unavailable"
    if _has_progress_field(ImageGenerationRequest):
        logger.info(
            "mlx-vlm image requests already carry progress_callback; "
            "the H3-Image progress vendor patch is unused"
        )
        return "upstream"
    try:
        package = importlib.import_module(_PACKAGE)
    except Exception as exc:  # noqa: BLE001
        logger.debug("H3-Image progress patch skipped: %s", exc)
        return "unavailable"
    stale = [name for name in _VENDORED if f"{_PACKAGE}.{name}" in sys.modules]
    if stale:
        logger.warning(
            "H3-Image progress patch not applied: %s already imported from %s",
            ", ".join(f"{_PACKAGE}.{name}" for name in stale),
            Path(package.__file__).parent,
        )
        return "unavailable"
    path = getattr(package, "__path__", None)
    if path is None:
        return "unavailable"
    vendor = str(_VENDOR_PACKAGE)
    if vendor not in path:
        path.insert(0, vendor)
    try:
        for name in _VENDORED:
            module = importlib.import_module(f"{_PACKAGE}.{name}")
            if Path(module.__file__).resolve().parent != _VENDOR_PACKAGE:
                raise ImportError(f"{module.__name__} loaded from {module.__file__}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("H3-Image progress patch failed: %s", exc)
        for name in _VENDORED:
            sys.modules.pop(f"{_PACKAGE}.{name}", None)
        if vendor in path:
            path.remove(vendor)
        return "unavailable"
    logger.info("H3-Image progress patch applied (Neywa-Labs/mlx-vlm#2742)")
    return "vendor"


def _with_progress_field(base: type) -> type:
    if _has_progress_field(base):
        return base
    # A subclass passes every isinstance check mlx-vlm makes, and
    # dataclasses.replace() keeps the subclass and the callback.
    cls = dataclasses.make_dataclass(
        base.__name__,
        [("progress_callback", Any, dataclasses.field(default=None))],
        bases=(base,),
        slots=True,
        namespace={"__module__": __name__},
    )
    cls.__qualname__ = base.__qualname__
    return cls


@lru_cache(maxsize=1)
def request_classes() -> tuple[type, type]:
    """``(ImageGenerationRequest, ImageEditRequest)`` with ``progress_callback``."""
    from mlx_vlm.generate.edit_image import ImageEditRequest
    from mlx_vlm.generate.image import ImageGenerationRequest

    return _with_progress_field(ImageGenerationRequest), _with_progress_field(
        ImageEditRequest
    )
