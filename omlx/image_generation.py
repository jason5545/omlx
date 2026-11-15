# SPDX-License-Identifier: Apache-2.0
"""Image generation support shared by discovery, the engine and the API.

mlx-vlm decides which checkpoints are image generation models
(``mlx_vlm.generate.image.is_image_generation_model`` and
``mlx_vlm.generate.edit_image.is_image_edit_model``). oMLX keeps no list of
image model types. It only skips the mlx-vlm check for directories that carry
none of the metadata files an image pipeline ships, because the check imports
``mlx_vlm.models.<type>.model`` for every candidate type, and importing an
ordinary model package during discovery would pull model code (and possibly a
stock copy of a package oMLX vendors) in before oMLX's own load-time patches.

The prompt expanders some pipelines run before generation (MiniMax-H3-Image,
Flux3) are separate Hugging Face models that mlx-vlm downloads on first use.
A request must never download anything, so :func:`resolve_prompt_expansion`
looks for the expander locally first and the API answers 400 when expansion is
requested but the expander is not on disk.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MODEL_TYPE = "image_generation"

# Metadata an mlx-vlm image pipeline reads to identify its family (see
# ``mlx_vlm.generate.image._local_image_model_types``): a Diffusers
# ``model_index.json``, a Bonsai ``manifest.json``, a native config with a
# pipeline class or a Flux3 ``dit`` section, or a transformer component index.
_PIPELINE_INDEX_FILES = ("model_index.json", "manifest.json")
_TRANSFORMER_INDEXES = (
    "transformer/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
)

# Families whose pipeline expands the prompt with a separate model by default,
# mapped to the module that names that model. The repo id itself is read from
# mlx-vlm at runtime, so an upstream change of the default is picked up.
_DEFAULT_EXPANDER_MODULES = {
    "minimax_h3_image": "mlx_vlm.models.minimax_h3_image.upsampler",
    "flux3": "mlx_vlm.models.flux3.upsampler",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def has_pipeline_metadata(path: Path) -> bool:
    """Cheap pre-check: does ``path`` carry any image pipeline metadata?"""
    if any((path / name).is_file() for name in _PIPELINE_INDEX_FILES):
        return True
    if any((path / name).is_file() for name in _TRANSFORMER_INDEXES):
        return True
    config = _load_json(path / "config.json")
    return bool(config) and ("_class_name" in config or "dit" in config)


def _mlx_vlm_models_dirs() -> list[Path]:
    """Every folder ``mlx_vlm.models.<type>`` can be imported from.

    oMLX's compat patches put their vendor folders in front of the package
    paths (``omlx/patches/mlx_vlm_*_compat/vendor/mlx_vlm``), so the stock
    install is not necessarily the first entry. Read the live paths when the
    package is imported; ``find_spec`` on the top-level package does not
    import it otherwise.
    """
    import sys

    models = sys.modules.get("mlx_vlm.models")
    if models is not None and getattr(models, "__path__", None):
        return [Path(p) for p in models.__path__]
    package = sys.modules.get("mlx_vlm")
    locations = getattr(package, "__path__", None) if package is not None else None
    if not locations:
        try:
            spec = importlib.util.find_spec("mlx_vlm")
        except (ImportError, ValueError):
            return []
        locations = spec.submodule_search_locations if spec is not None else None
    return [Path(p) / "models" for p in (locations or [])]


def _candidate_types(path: Path) -> tuple[str, ...]:
    from mlx_vlm.generate.image import _local_image_model_types, _model_type_from_id

    return (*_local_image_model_types(str(path)), _model_type_from_id(str(path)))


def _has_image_package(candidates: Iterable[str]) -> bool:
    # mlx-vlm image pipelines live in ``models/<type>/model.py``; language and
    # vision-language packages do not have that module.
    names = [c for c in candidates if c and c.isidentifier()]
    return any(
        (models_dir / name / "model.py").is_file()
        for models_dir in _mlx_vlm_models_dirs()
        for name in names
    )


@dataclass(frozen=True)
class ImageCapabilities:
    generate: bool
    edit: bool
    family: str | None

    @property
    def any(self) -> bool:
        return self.generate or self.edit


_NONE = ImageCapabilities(False, False, None)


def _signature(path: Path) -> tuple:
    stamps = []
    for name in ("model_index.json", "manifest.json", "config.json"):
        try:
            stamps.append((path / name).stat().st_mtime_ns)
        except OSError:
            stamps.append(None)
    return tuple(stamps)


@lru_cache(maxsize=256)
def _capabilities_cached(path_str: str, _signature: tuple) -> ImageCapabilities:
    path = Path(path_str)
    try:
        candidates = _candidate_types(path)
    except Exception as exc:  # mlx-vlm missing or too old
        logger.debug("Image model check unavailable for %s: %s", path, exc)
        return _NONE
    if not _has_image_package(candidates):
        return _NONE
    try:
        from mlx_vlm.generate.edit_image import (
            image_edit_model_class,
            is_image_edit_model,
        )
        from mlx_vlm.generate.image import (
            image_generation_model_class,
            is_image_generation_model,
        )

        generate = bool(is_image_generation_model(path_str))
        edit = bool(is_image_edit_model(path_str))
        family = None
        if generate or edit:
            model_class = (
                image_generation_model_class(path_str)
                if generate
                else image_edit_model_class(path_str)
            )
            family = getattr(model_class, "model_type", None)
    except Exception as exc:
        logger.warning("mlx-vlm could not classify %s: %s", path, exc)
        return _NONE
    return ImageCapabilities(generate, edit, family)


def image_capabilities(path: Path | str) -> ImageCapabilities:
    """Return what mlx-vlm can do with the checkpoint at ``path``."""
    path = Path(path)
    if not path.is_dir() or not has_pipeline_metadata(path):
        return _NONE
    return _capabilities_cached(str(path), _signature(path))


def is_image_generation_dir(path: Path | str) -> bool:
    return image_capabilities(path).any


def incomplete_download(path: Path) -> str | None:
    """Describe why an image checkpoint looks half downloaded, or None.

    A component folder with ``config.json`` but no weights, a shard set with
    gaps, or a Hugging Face ``*.incomplete`` file means the weights are not
    all there yet.
    """
    from .model_discovery import _missing_weight_shards

    partial = path / ".cache" / "huggingface" / "download"
    if partial.is_dir() and next(partial.rglob("*.incomplete"), None) is not None:
        return "a Hugging Face download is still in progress"
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        if child.name.startswith(".") or not (child / "config.json").is_file():
            continue
        if not any(child.glob("*.safetensors")):
            if (child / "model.safetensors.index.json").is_file() or any(
                child.glob("*.safetensors.index.json")
            ):
                return f"{child.name}/ has no weight files yet"
            continue
        missing = _missing_weight_shards(child)
        if missing:
            count, examples, _ = missing
            return (
                f"{child.name}/ is missing {count} weight shard(s) "
                f"(e.g. {', '.join(examples)})"
            )
    return None


# ---------------------------------------------------------------------------
# Prompt expansion
# ---------------------------------------------------------------------------


def default_expander(family: str | None) -> str | None:
    """The Hub repo a family expands prompts with by default, or None."""
    module_name = _DEFAULT_EXPANDER_MODULES.get(family or "")
    if module_name is None:
        return None
    try:
        return str(importlib.import_module(module_name).DEFAULT_UPSAMPLER)
    except Exception as exc:  # pragma: no cover - mlx-vlm layout changed
        logger.warning("Cannot read the default prompt expander of %s: %s", family, exc)
        return None


def _is_model_folder(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def find_local_model(reference: str, model_dirs: Iterable[Path]) -> Path | None:
    """Find a model folder for a local path or Hub id without the network.

    Looks at the path itself, then ``<model_dir>/<org>/<name>`` and
    ``<model_dir>/<name>`` in oMLX's model folders, then the Hugging Face
    cache (``local_files_only``).
    """
    path = Path(reference).expanduser()
    if path.is_absolute() or reference.startswith((".", "~")):
        return path if _is_model_folder(path) else None
    name = reference.rstrip("/").rsplit("/", 1)[-1]
    for root in model_dirs:
        for candidate in (Path(root) / reference, Path(root) / name):
            if _is_model_folder(candidate):
                return candidate
    try:
        from huggingface_hub import snapshot_download

        cached = Path(snapshot_download(reference, local_files_only=True))
    except Exception:
        return None
    return cached if _is_model_folder(cached) else None


class PromptExpanderMissing(Exception):
    def __init__(self, reference: str, requested: bool):
        self.reference = reference
        self.requested = requested
        how = (
            "Prompt expansion was requested"
            if requested
            else "A prompt_expansion_model was given"
        )
        super().__init__(
            f"{how}, but the prompt expansion model '{reference}' is not on "
            f"this machine and oMLX does not download models during a request. "
            f"Download it first (Hugging Face repo '{reference}', e.g. "
            f"`hf download {reference}` or the admin downloader), or send "
            f'"upsample_prompt": false.'
        )


@dataclass(frozen=True)
class PromptExpansion:
    """How a request expands its prompt, resolved before the model is used."""

    upsample_prompt: bool | None
    prompt_expansion_model: str | None
    expander: str | None  # Hub id or path of the expander that will run

    def extra(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if self.upsample_prompt is not None:
            values["upsample_prompt"] = self.upsample_prompt
        if self.prompt_expansion_model is not None:
            values["prompt_expansion_model"] = self.prompt_expansion_model
        return values


def resolve_prompt_expansion(
    family: str | None,
    upsample_prompt: bool | None,
    prompt_expansion_model: str | None,
    model_dirs: Iterable[Path],
) -> PromptExpansion:
    """Resolve the expander to a local folder, or raise PromptExpanderMissing.

    ``upsample_prompt`` follows mlx-vlm: unset means the family default
    (MiniMax-H3-Image and Flux3 expand by default). oMLX changes that one
    default: when the default expander is not on disk the prompt is used as
    is, so a machine without the expander still generates. An explicit
    ``upsample_prompt: true`` or an explicit ``prompt_expansion_model`` that is
    not on disk is an error instead.
    """
    model_dirs = list(model_dirs)
    if upsample_prompt is False:
        return PromptExpansion(False, None, None)
    default = default_expander(family)
    if prompt_expansion_model:
        local = find_local_model(prompt_expansion_model, model_dirs)
        if local is None:
            raise PromptExpanderMissing(
                prompt_expansion_model, requested=bool(upsample_prompt)
            )
        return PromptExpansion(upsample_prompt, str(local), str(local))
    if default is None:
        # No implicit expander for this family; nothing will be fetched.
        return PromptExpansion(upsample_prompt, None, None)
    local = find_local_model(default, model_dirs)
    if local is not None:
        return PromptExpansion(upsample_prompt, str(local), str(local))
    if upsample_prompt:
        raise PromptExpanderMissing(default, requested=True)
    logger.info(
        "Prompt expansion model %s is not on this machine; generating from "
        "the prompt as given (send upsample_prompt=true to require expansion)",
        default,
    )
    return PromptExpansion(False, None, None)


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------

# Same allowance for runtime buffers as model_discovery.estimate_model_size.
_RUNTIME_OVERHEAD = 1.05
# A safetensors header is a few KB to a few MB; anything larger is not one.
_MAX_SAFETENSORS_HEADER = 256 * 1024 * 1024


def _safetensors_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.safetensors")) if folder.is_dir() else []


def _files_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in _safetensors_files(folder))


def _tensor_bytes(folder: Path, predicate) -> int:
    """Sum the stored bytes of the tensors in ``folder`` whose name matches.

    Reads only the JSON header at the front of each shard, never tensor data.
    """
    total = 0
    for path in _safetensors_files(folder):
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path} is not a safetensors file")
            length = int.from_bytes(raw, "little")
            if not 0 < length <= _MAX_SAFETENSORS_HEADER:
                raise ValueError(f"{path} has an implausible header length")
            header = json.loads(handle.read(length))
        for name, info in header.items():
            if name == "__metadata__" or not predicate(name):
                continue
            start, end = info["data_offsets"]
            total += int(end) - int(start)
    return total


def _h3_component(root: Path, name: str) -> Path:
    # Mirrors mlx_vlm.models.minimax_h3_image.weights.component_path.
    diffusers = root / f"{name}_diffusers"
    if name != "text_encoder" and (diffusers / "config.json").is_file():
        return diffusers
    return root / name


def _is_h3_adaln_weight(name: str) -> bool:
    # The projections MiniMaxH3Transformer.drop_adaln_weights() deletes.
    return ".adaln_proj." in name or name.startswith("norm_out.linear.")


def _h3_image_memory(root: Path) -> int | None:
    """Peak memory of one MiniMax-H3-Image request, from component sizes.

    mlx-vlm's H3-Image pipeline loads nothing up front. Per request it loads
    the text encoder, encodes the prompt and drops the encoder again
    (``evict_text_encoder`` defaults to True). The transformer and the VAE
    load on first use and stay. The first request materializes the AdaLN
    tables and then drops the AdaLN projection weights (about 26 GB of the
    8-bit transformer). From the second request on, the text encoder is
    loaded next to the resident transformer and VAE, which is the peak:

        transformer - AdaLN weights + VAE + text encoder

    The first request peaks lower, at the larger of the text encoder and the
    AdaLN weights alone.
    """
    transformer = _h3_component(root, "transformer")
    text_encoder = _h3_component(root, "text_encoder")
    vae = _h3_component(root, "vae")
    transformer_bytes = _files_bytes(transformer)
    text_encoder_bytes = _files_bytes(text_encoder)
    vae_bytes = _files_bytes(vae)
    if not (transformer_bytes and text_encoder_bytes and vae_bytes):
        return None  # e.g. a text encoder kept in another repo
    adaln_bytes = _tensor_bytes(transformer, _is_h3_adaln_weight)
    if not 0 < adaln_bytes < transformer_bytes:
        return None
    steady = transformer_bytes - adaln_bytes + vae_bytes + text_encoder_bytes
    first = max(text_encoder_bytes, adaln_bytes)
    return int(max(steady, first) * _RUNTIME_OVERHEAD)


# Families whose pipelines never hold every component at once. Any other
# family is charged its full file size.
_MEMORY_ESTIMATORS = {
    "minimax_h3_image": _h3_image_memory,
}


def estimate_image_memory(path: Path | str, family: str | None) -> int | None:
    """Return the peak bytes one request of this pipeline needs, if known.

    ``None`` means "no family rule": the caller charges the file size.
    """
    estimator = _MEMORY_ESTIMATORS.get(family or "")
    if estimator is None:
        return None
    try:
        return estimator(Path(path))
    except Exception as exc:  # unreadable header, odd layout: fall back
        logger.warning("Could not size %s from its components: %s", path, exc)
        return None
