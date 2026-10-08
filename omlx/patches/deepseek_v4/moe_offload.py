# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the DeepSeek V4 / GLM-5.x-flash MoE block.

The ``deepseek_v4`` and ``glm5_next`` checkpoints run their routed experts
through this package's own :class:`~omlx.patches.deepseek_v4.switch_layers
.SwitchGLU`, which the common adapter cannot wrap: it is not the stock
mlx-lm module, and its forward carries native Metal fast paths (the
``deepseek_*_gather_qmm`` block kernels, the gate/up pair kernels and the
native ``glm_moe_weighted_sum``). Re-implementing that forward would fork
it, so this adapter keeps the module and swaps what it computes on.

Each projection's parameters are replaced, before lazy weights materialize,
by slot tensors of ``capacity`` experts; expert ids are translated to slot
ids and the module's own ``__call__`` runs unchanged on them. Every kernel
decision inside it is a function of the routes, the projections' quantization
metadata and the tensors' shapes — ``num_experts`` is a property of the
weight shape, so it follows the slots — and every use of an index is a
gather, so a route computes the same numbers against its slot as it would
against its expert. A miss reads the expert's gate, up and down slabs from
the checkpoint's own safetensors with positional reads on a bounded pool, as
the GLM DSA and DeepSeek V4.1 adapters do. The slots are the common
adapter's :class:`~omlx.patches.moe_expert_offload.ExpertCache`, so a miss
evicts the expert with the lowest decayed routing count, and a decode step
whose reads are slow keeps the GPU clocked while it waits.

Over-capacity prefill, where one call routes to more distinct experts than
the cache holds, is chunked on expert boundaries exactly as the other
adapters do: resident experts first, each expert installed at most once per
call, the next chunk's reads started before the current chunk is evaluated,
chunks run under the module's own forward with one route per row, and the
weighted sum, when the module's own forward would have applied it natively,
is applied to the reassembled routes the way the caller does when the kernel
is unavailable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...custom_kernels.glm_moe_dsa import fast as glm_fast
from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    _SORT_MIN_ROUTES,
    CheckpointExpertStore,
    ExpertCache,
    _drain,
    _GLUStoreView,
    _io_batch,
    _io_pool,
    _is_mtp_path,
    _minimum_experts,
    _resolve_model_dir,
)
from .switch_layers import _sort_threshold

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")


def is_deepseek_v4_switch_glu(obj) -> bool:
    """This package's SwitchGLU (never an already-wrapped one)."""
    return (
        type(obj).__name__ == "SwitchGLU"
        and type(obj).__module__ == "omlx.patches.deepseek_v4.switch_layers"
    )


def _fields(lin) -> tuple[str, ...]:
    return ("weight", "scales") + (("biases",) if lin.get("biases") is not None else ())


def _is_quantized(lin) -> bool:
    return type(lin).__name__ == "QuantizedSwitchLinear" and all(
        hasattr(lin, a) for a in ("group_size", "bits", "mode")
    )


def resolve_view(glu, store: CheckpointExpertStore, path: str):
    """Validate the checkpoint against the module; ``(view, None)`` or ``(None, reason)``.

    The checkpoint must hold the stacked ``gate_proj``/``up_proj``/``down_proj``
    tensors under the module's tree path, in shapes the module's projections
    match and a storage dtype the store can read.
    """
    for proj in _PROJS:
        lin = glu.get(proj)
        if lin is None or not _is_quantized(lin):
            return None, f"{proj} is not a QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{proj} has per-expert bias (unsupported)"
    view = _GLUStoreView(store, path)
    for proj in _PROJS:
        lin = glu[proj]
        for field in _fields(lin):
            name = view._name(proj, field, 0)
            if not store.has(name):
                return None, f"checkpoint has no tensor {name!r}"
            shape, dtype = store.spec(name)
            if shape != tuple(lin[field].shape):
                return (
                    None,
                    f"{name!r} shape {shape} != expected {tuple(lin[field].shape)}",
                )
            if dtype not in _DTYPES:
                return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


class _SlotCache(ExpertCache):
    """:class:`ExpertCache` slots living inside the module's own projection tensors.

    The common cache's slot tensors are installed as the projections'
    parameters, so the module's own forward computes on them, and its
    eviction (lowest decayed routing count), read-ahead and decode keepalive
    apply unchanged.
    """

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        self.glu = glu  # read by _allocate, which the base constructor calls
        super().__init__(glu, capacity, view)

    def _allocate(self, capacity: int) -> None:
        super()._allocate(capacity)
        for proj in self.projs:
            lin = self.glu[proj]
            for field, slots in zip(("weight", "scales", "biases"), self.resident[proj]):
                if slots is not None:
                    setattr(lin, field, slots)  # drops the lazy full-size array

    def release_slots(self) -> int:
        # Prefill memory borrowing is not wired for this module: the cache
        # keeps its full size, as it always has.
        return 0


class OffloadedSwitchGLU(nn.Module):
    """DeepSeek V4 SwitchGLU whose experts live in a :class:`_SlotCache`."""

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        super().__init__()
        # A plain attribute, like the other adapters: the module with the slot
        # tensors stays out of the tree so parameter walks see the wrapper.
        self.cache = _SlotCache(glu, capacity, view)
        # Keep the GPU clocked while a decode step waits on slow reads (see
        # moe_expert_offload._GpuKeepalive). OMLX_MOE_OFFLOAD_OVERLAP=0 keeps
        # the plain wait.
        self._overlap = os.environ.get("OMLX_MOE_OFFLOAD_OVERLAP", "1") != "0"

    def overlaps_decode(self, n_routes: int) -> bool:
        """Whether a call of ``n_routes`` routes takes the decode branch.

        That branch reads the routes back to the host before it computes, and
        starts the routed experts on the GPU as soon as they are built, so
        the host builds the rest of the layer while they run. glm5_next's MoE
        block asks this to put its shared expert ahead of the read-back.
        """
        return (
            self._overlap
            and not self.cache.warm
            and n_routes < _SORT_MIN_ROUTES
            and _io_pool() is not None
        )

    def _forward_expert_major(self, flat_x: mx.array, ids: list[int], k: int):
        """Routes grouped by expert, cut into chunks of ``capacity`` experts.

        Resident experts come first, so a miss evicts only experts this call
        has already used: without that, the first chunk's installs evicted
        resident experts a later chunk still needed, and that chunk read them
        back. The next chunk's first reads start before this chunk's eval;
        their slot writes wait for it.
        """
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        resident = np.zeros(c.n_experts, dtype=np.bool_)
        resident[c.slot_expert[c.slot_expert >= 0]] = True
        rank = ids_np + (~resident[ids_np]) * c.n_experts
        order = np.argsort(rank, kind="stable")  # routes grouped by expert
        sorted_ids = ids_np[order]
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        chunks = list(zip(cuts[:-1], cuts[1:]))
        outs = []
        ahead: dict = {}
        try:
            for n, (start, end) in enumerate(chunks):
                chunk_ids = sorted_ids[start:end]
                c._ensure_ids(np.unique(chunk_ids).tolist(), pending=ahead)
                ahead = {}
                slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32))
                t_idx = mx.array(order[start:end] // k, dtype=mx.int32)
                xe = mx.take(flat_x, t_idx, axis=0)
                o = c.glu(xe, slots.reshape(-1, 1))[:, 0, :]
                if n + 1 < len(chunks):
                    s1, e1 = chunks[n + 1]
                    ahead = c._read_ahead(
                        np.unique(sorted_ids[s1:e1]).tolist(), _io_batch()
                    )
                mx.eval(o)
                outs.append(o)
        finally:
            _drain(ahead)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        return mx.take(out, inverse, axis=0).reshape(-1, k, d_model)

    def __call__(self, x: mx.array, indices: mx.array, scores=None, weighted_sum=False):
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        fits = n_tok * k <= c.capacity or n_tok == 1
        ids = None
        if not fits:
            ids = flat_i.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            decode = self.overlaps_decode(indices.size)
            if decode:
                # Decode: when the first miss is still being read after the
                # grace period, the rest of the wait keeps the GPU clocked.
                c._ensure_ids(indices.reshape(-1).tolist(), lambda: None)
            else:
                c.ensure(indices)
            slots = mx.take(c.map, indices)
            y = c.glu(x, slots, scores=scores, weighted_sum=weighted_sum)
            if decode:
                # Every layer reads its routes back, so nothing runs on the
                # GPU between that read-back and the next layer's until it is
                # started: start the routed experts now, and the host builds
                # the rest of this layer and the next one's attention while
                # they run. Scheduling only, same kernels on the same inputs
                # (measured 5% less time per MTP cycle on GLM-5.3, M5 Max).
                mx.async_eval(y)
            return y
        y = self._forward_expert_major(x.reshape(-1, x.shape[-1]), ids, k)
        y = y.reshape(indices.shape + (x.shape[-1],))
        # Sum exactly when the module's own forward would have: it returns the
        # routes unsummed unless the call is sorted, carries float32 scores of
        # top-k 6 or 8 on half-precision activations, and the native kernel is
        # present — and the caller applies the scores itself otherwise.
        if (
            weighted_sum
            and scores is not None
            and indices.size
            >= _sort_threshold(c.glu.gate_proj, c.glu.up_proj, c.glu.down_proj)
            and scores.shape[-1] in (6, 8)
            and scores.dtype == mx.float32
            and y.dtype in (mx.float16, mx.bfloat16)
            and glm_fast.has_symbol("glm_moe_weighted_sum")
        ):
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y


def _iter_deepseek_v4_switch_glus(model):
    seen = set()

    def walk(parent, key, obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if is_deepseek_v4_switch_glu(obj):
            yield (parent, key, obj, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(obj, k, v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(obj, i, v, f"{path}.{i}")

    yield from walk(None, None, model, "")


def apply_deepseek_v4_moe_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Wrap every covered DeepSeek V4 SwitchGLU; returns the number wrapped.

    Same contract as ``apply_moe_expert_offload``: runs before lazy weights
    materialize, honors the kill switch, and skips (with a logged reason)
    any module the checkpoint does not cover.

    ``mtp_resident`` keeps the embedded MTP draft head's experts fully
    resident (glm5_next Lightning MTP + offload): the head is one decoder
    layer whose drafts the streamed backbone verifies, so streaming its
    experts would add SSD latency to every draft step. With the flag off,
    the head wraps like any other layer, exactly as before.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return 0
    targets = list(_iter_deepseek_v4_switch_glus(model))
    if not targets:
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    minimum = _minimum_experts(model_dir)
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning(
            "deepseek_v4 moe expert offload: no safetensors under %s", model_dir
        )
        return 0
    wrapped = 0
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in targets:
        if mtp_resident and _is_mtp_path(path):
            continue
        view, reason = resolve_view(glu, store, path)
        if view is None:
            logger.info(
                "deepseek_v4 moe expert offload: skipping %s (%s)", path, reason
            )
            continue
        n_experts = glu[_PROJS[0]]["weight"].shape[0]
        capacity = min(n_experts, max(minimum, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(glu[proj][field].shape)) * glu[proj][field].dtype.size
            for proj in _PROJS
            for field in _fields(glu[proj])
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        new = OffloadedSwitchGLU(glu, capacity, view)
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)
        else:
            parent[key] = new
        wrapped += 1
        _sync_and_clear_cache()

    # glm5_next decoder layers compile their FFN block at decode shapes
    # (mlx_vlm glm5_next language.py ``compile_ffn``). The offloaded block
    # manages slots host-side — LRU map, pread fetches — and cannot be
    # traced into a compiled graph: ``tolist()`` inside a trace dies with
    # "eval during function transformations". Keep those layers eager; the
    # native gather kernels still run, only the graph fusion is lost, and
    # offload trades speed for memory anyway.
    for module in model.modules():
        if not getattr(module, "compile_ffn", False):
            continue
        if any(isinstance(c, OffloadedSwitchGLU) for c in module.modules()):
            module.compile_ffn = False
            module._ffn_c = None

    if wrapped:
        logger.info(
            "deepseek_v4 moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)%s",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
            ", draft head resident" if mtp_resident else "",
        )
    return wrapped


__all__ = [
    "OffloadedSwitchGLU",
    "apply_deepseek_v4_moe_expert_offload",
    "is_deepseek_v4_switch_glu",
    "resolve_view",
]
