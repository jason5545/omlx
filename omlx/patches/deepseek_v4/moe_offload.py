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
from concurrent.futures import Future
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
from . import route_trace
from .switch_layers import _sort_threshold

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

# glm5_next decode reads the next layer's predicted misses ahead (see
# OffloadedSwitchGLU.stage_next_routes). OMLX_MOE_OFFLOAD_PREFETCH=0 turns
# it off; at most _PREFETCH_MAX experts are read ahead per layer.
_PREFETCH = os.environ.get("OMLX_MOE_OFFLOAD_PREFETCH", "1") != "0"
_PREFETCH_MAX = 8


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


class _Prefetch:
    """Read-ahead state of one wrapper, kept out of the module tree.

    ``gate``/``next`` are the next offloaded MoE layer's router and wrapper;
    ``staged`` is that router's prediction for the call in flight;
    ``incoming`` holds the reads the previous layer started for this one.
    """

    __slots__ = ("gate", "next", "staged", "incoming")

    def __init__(self):
        self.gate = self.next = self.staged = self.incoming = None


def _fill_reads(store_view, items) -> None:
    """One IO task for an expert's slabs, in plan order. ``store_view`` is
    only held: it keeps the shard descriptors open while the task runs
    (a dropped read may outlive the call that started it). A cancelled
    slab is skipped."""
    for plan, future in items:
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(CheckpointExpertStore.read(plan))
        except BaseException as exc:
            future.set_exception(exc)


def _read_expert(pool, cache: ExpertCache, e: int) -> list:
    """Start reading expert ``e`` as one task; the group has the layout of
    ``ExpertCache._submit`` (one future per slab)."""
    group = [(name, field, plan, Future()) for name, field, plan in cache._plans(e)]
    pool.submit(_fill_reads, cache.disk, [(plan, f) for _, _, plan, f in group])
    return group


def _flat_ids(a: mx.array) -> list:
    """``a``'s ids as a flat list, read from ``a`` itself.

    ``a.reshape(-1).tolist()`` evaluates a new (empty) reshape on the stream,
    so the host waits for everything submitted before it; after the router
    and the shared expert went out in separate command buffers, that meant
    waiting for the shared expert too. Reading the evaluated ``a`` waits for
    its own command buffer only.
    """
    out = a.tolist()
    while out and isinstance(out[0], list):
        out = [v for row in out for v in row]
    return out


def _cancel_reads(reads) -> None:
    """Drop read-ahead groups without waiting (a running slab finishes in
    the background and its bytes are discarded)."""
    for group in (reads or {}).values():
        for _, _, _, future in group:
            future.cancel()


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
        self._prefetch = _Prefetch()
        # The decoder layer index, for the routing trace (route_trace); 255
        # when the module path names none.
        self._layer = 255

    def stage_next_routes(self, x: mx.array) -> list:
        """The next offloaded layer's routes predicted from this layer's
        input, for the caller to start together with this layer's router.

        The residual stream changes little from one layer to the next, so
        the next layer's router applied to this layer's FFN input names
        about half of the next layer's misses (54-57% on GLM-5.3 oQ2e at
        85% residency, one wasted read per hit). After this layer's own
        misses are installed, the decode branch starts reading the predicted
        non-resident ones, which the next layer then finds in flight or
        done. Reads only: nothing is installed or evicted on a prediction,
        so the cache and every output stay as without it. Returns ``[]``
        when nothing is predicted.
        """
        pf = self._prefetch
        pf.staged = None
        if not _PREFETCH or pf.gate is None or _io_pool() is None:
            return []
        pf.staged = pf.gate(x)[0]
        return [pf.staged]

    def _claim_reads(self, incoming, ids) -> dict | None:
        """The read-ahead groups this call misses on; the rest are dropped."""
        if not incoming:
            return None
        slot_of = self.cache.slot_of
        need = {e for e in ids if e not in slot_of}
        pending = {e: g for e, g in incoming.items() if e in need}
        _cancel_reads({e: g for e, g in incoming.items() if e not in pending})
        return pending or None

    def _read_next(self, staged: mx.array) -> None:
        """Start reading the next layer's predicted non-resident experts."""
        nxt = self._prefetch.next
        cache = nxt.cache
        pool = _io_pool()
        if cache.warm or pool is None:
            return
        cand = []
        for e in dict.fromkeys(_flat_ids(staged)):
            if e not in cache.slot_of:
                cand.append(e)
                if len(cand) >= _PREFETCH_MAX:
                    break
        if cand:
            _cancel_reads(nxt._prefetch.incoming)
            nxt._prefetch.incoming = {e: _read_expert(pool, cache, e) for e in cand}

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
        pf = self._prefetch
        incoming, pf.incoming = pf.incoming, None
        staged, pf.staged = pf.staged, None
        trace = route_trace.active()
        if fits:
            decode = self.overlaps_decode(indices.size)
            if decode:
                # Decode: when the first miss is still being read after the
                # grace period, the rest of the wait keeps the GPU clocked.
                ids = _flat_ids(indices)
                if trace is not None:
                    route_trace.record(
                        trace, self._layer, c, ids, n_tok, k, route_trace.ROUTES,
                        decode=True, x=x,
                    )
                pending = self._claim_reads(incoming, ids)
                c._ensure_ids(ids, lambda: None, pending=pending)
            else:
                if trace is not None:
                    route_trace.record(
                        trace, self._layer, c, _flat_ids(indices), n_tok, k,
                        route_trace.ROUTES, x=x,
                    )
                _cancel_reads(incoming)
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
                if staged is not None and pf.next is not None:
                    self._read_next(staged)
            return y
        _cancel_reads(incoming)
        if trace is not None:
            route_trace.record(
                trace, self._layer, c, ids, n_tok, k, route_trace.EXPERT_MAJOR
            )
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


def _layer_index(path: str):
    """The decoder layer index in a module path (``...layers.<i>....``)."""
    parts = path.split(".")
    for name, value in zip(parts, parts[1:]):
        if name == "layers" and value.isdigit():
            return int(value)
    return None


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
    chain = []  # (layer index, wrapper, its router) of glm5_next MoE blocks
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
        layer = _layer_index(path)
        if layer is not None and layer < 255:
            new._layer = layer
        gate = parent.get("gate") if isinstance(parent, dict) else None
        if layer is not None and type(gate).__name__ == "Glm5NextMoEGate":
            chain.append((layer, new, gate))
        wrapped += 1
        _sync_and_clear_cache()

    # Each glm5_next layer predicts the next one's routes (stage_next_routes).
    chain.sort(key=lambda t: t[0])
    for (_, cur, _), (_, nxt, gate) in zip(chain, chain[1:]):
        cur._prefetch.gate = gate
        cur._prefetch.next = nxt

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
