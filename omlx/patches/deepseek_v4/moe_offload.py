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

import copy
import logging
import os
from concurrent.futures import Future, wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...custom_kernels.glm_moe_dsa import fast as glm_fast
from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    _SCORE_DECAY,
    _SCORE_DECAY_EVERY,
    _SORT_MIN_ROUTES,
    CheckpointExpertStore,
    ExpertCache,
    _drain,
    _env_int,
    _gpu_keepalive,
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

# glm5_next decode can read the next layer's predicted misses ahead (see
# OffloadedSwitchGLU.stage_next_routes); at most _PREFETCH_MAX experts per
# layer. Off unless OMLX_MOE_OFFLOAD_PREFETCH=1: on GLM-5.3 oQ3.5e at 54%
# residency (M5 Max) it made decode 12-20% slower per MTP cycle. Each read
# ahead holds an IO worker for its expert's slabs in turn, and the next
# layer's own misses queue behind them on the same pool, while about half
# the reads go unused.
_PREFETCH = os.environ.get("OMLX_MOE_OFFLOAD_PREFETCH", "0") == "1"
_PREFETCH_MAX = 8
# Decode misses are read straight into their slots (_SlotCache._ensure_decode).
# OMLX_MOE_OFFLOAD_READ_INTO_SLOT=0 keeps the read-then-install path.
_INTO_SLOT = os.environ.get("OMLX_MOE_OFFLOAD_READ_INTO_SLOT", "1") != "0"
# How long those reads may run before the wait keeps the GPU clocked (the
# common adapter waits 0.5 ms; this path was measured at 0).
_KEEPALIVE_GRACE_S = _env_int("OMLX_MOE_OFFLOAD_KEEPALIVE_GRACE_US", 0, 0) * 1e-6

# Over-capacity prefill reads the experts the cache does not hold into
# temporary weights and computes them group by group while the next groups
# are read, instead of installing them over resident experts (see
# OffloadedSwitchGLU._forward_streamed). _STREAM_GROUP experts per group,
# _STREAM_RING groups' weights allocated at once (the read-ahead depth and
# the extra memory: 4 x 16 experts of ~13 MB on GLM-5.3 oQ3.5e).
# OMLX_MOE_OFFLOAD_STREAM_PREFILL=0 keeps the installing expert-major path.
_STREAM = os.environ.get("OMLX_MOE_OFFLOAD_STREAM_PREFILL", "1") != "0"
_STREAM_GROUP = max(1, _env_int("OMLX_MOE_OFFLOAD_STREAM_GROUP", 16, 16))
_STREAM_RING = max(2, _env_int("OMLX_MOE_OFFLOAD_STREAM_RING", 4, 4))
_FIELDS = ("weight", "scales", "biases")


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
        self.rows = self._row_layout()

    def _row_layout(self):
        """Per projection field ``(name, fi)``: ``(shape, dtype, row bytes)``
        of one slot row, or ``None`` when a field's checkpoint bytes are not
        that row's layout (then nothing may be read straight into a slot)."""
        rows = {}
        for name, fi, plan in self._plans(0):
            shape, dtype = self._slot_specs[name][fi]
            row = int(np.prod(shape)) * dtype.size
            if plan.nbytes != row or _plan_dtype(plan) != dtype:
                return None
            rows[(name, fi)] = (shape, dtype, row)
        return rows

    def _ensure_decode(self, ids, pending=None) -> None:
        """:meth:`_ensure_ids` for a decode step, reading misses into their slots.

        The slots are claimed before the reads start, in miss order and by
        the installing path's rule (a free slot, else the lowest decayed
        count outside the call's experts), so the cache ends in the same
        state. Each slab is then read with ``os.preadv`` straight into its
        slot row, one IO task per slab, instead of into bytes the main thread
        turns into an array and the GPU copies into the slot (on GLM-5.3
        oQ3.5e expert shapes, M5 Max: 0.3-0.38 ms less per miss, from the SSD
        and from page cache alike). Once a read is still running
        ``_KEEPALIVE_GRACE_S`` after they start, the rest of the wait keeps
        the GPU clocked. A read started ahead (``pending``) holds bytes,
        copied into the slot once done. A failed read waits out the others
        and gives back the slots whose expert did not arrive. Falls back to
        :meth:`_ensure_ids` (with the keepalive) when reads are serial or a
        slab's checkpoint bytes are not its slot row's layout.
        """
        pool = _io_pool()
        if not _INTO_SLOT or pool is None or self.rows is None:
            self._ensure_ids(ids, lambda: None, pending=pending)
            return
        pending = dict(pending or {})
        needed = list(dict.fromkeys(int(e) for e in ids))
        if len(needed) > self.capacity:
            _drain(pending)
            raise ValueError("Expert cache capacity is smaller than the call's routes")
        np.add.at(self.score, np.asarray(ids, dtype=np.int64), 1.0)
        self._calls += 1
        if self._calls % _SCORE_DECAY_EVERY == 0:
            self.score *= _SCORE_DECAY
        misses = [e for e in needed if e not in self.slot_of]
        self.hits += len(needed) - len(misses)
        if not misses:
            _drain(pending)
            return
        protected = frozenset(needed)
        claimed = []  # (expert, slot); resident once its reads are done
        groups = []  # per claimed expert: [(slot row view, future, copy bytes)]
        done = 0
        try:
            for e in misses:
                slot = self._reserve(protected)
                # Owned from now on, so the next claim cannot take it back.
                self.slot_of[e] = slot
                self.slot_expert[slot] = e
                claimed.append((e, slot))
            # A write to a slot tensor still pending lands before the reads.
            mx.eval(*(self.resident[name][fi] for name, fi in self.rows))
            views = {
                key: memoryview(self.resident[key[0]][key[1]]).cast("B")
                for key in self.rows
            }
            for e, slot in claimed:
                ahead = pending.pop(e, None)
                group = []
                for n, (name, fi, plan) in enumerate(self._plans(e)):
                    row = self.rows[(name, fi)][2]
                    dst = views[(name, fi)][slot * row : (slot + 1) * row]
                    if ahead is not None:
                        group.append((dst, ahead[n][3], True))
                    else:
                        future = pool.submit(_pread_into, self.disk, plan, dst)
                        group.append((dst, future, False))
                groups.append(group)
            keepalive = None
            reads = [f for group in groups for _, f, _ in group]
            if wait(reads, timeout=_KEEPALIVE_GRACE_S).not_done:
                keepalive = _gpu_keepalive()
            for (e, slot), group in zip(claimed, groups):
                futures = [f for _, f, _ in group]
                if keepalive is not None:
                    keepalive.wait(futures)
                else:
                    wait(futures)
                for dst, future, copy_bytes in group:
                    raw = future.result()
                    if copy_bytes:
                        dst[:] = raw
                self.map[e] = slot
                self.misses += 1
                self.fetched_bytes += self.expert_bytes
                done += 1
        finally:
            if done < len(claimed):
                # Nothing may write a slot once it is given back.
                futures = [f for group in groups for _, f, _ in group]
                for future in futures:
                    future.cancel()
                wait(futures)
                for e, slot in claimed[done:]:
                    del self.slot_of[e]
                    self.slot_expert[slot] = -1
                    self.free.append(slot)
            _drain(pending)
        self.warm = len(self.slot_of) == self.n_experts

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

    def note_streamed(self, ids, hits: int, streamed: int) -> None:
        """Count a streamed prefill call (``OffloadedSwitchGLU._forward_streamed``):
        the decayed routing counts move as for one ``_ensure_ids`` over its
        distinct experts ``ids``; ``streamed`` experts were read, none was
        installed or evicted."""
        np.add.at(self.score, np.asarray(ids, dtype=np.int64), 1.0)
        self._calls += 1
        if self._calls % _SCORE_DECAY_EVERY == 0:
            self.score *= _SCORE_DECAY
        self.hits += hits
        self.misses += streamed
        self.fetched_bytes += streamed * self.expert_bytes


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


def _plan_dtype(plan):
    """The mlx dtype a plan's bytes hold."""
    return plan.mx_view if plan.mx_view is not None else mx.array(
        np.zeros(0, dtype=plan.np_dtype)
    ).dtype


def _pread_into(store_view, plan, view) -> None:
    """Read a plan's bytes straight into ``view`` (positional; any thread).
    ``store_view`` is only held, so the shard descriptors stay open."""
    got = 0
    while got < plan.nbytes:
        n = os.preadv(plan.fd, [view[got : plan.nbytes]], plan.offset + got)
        if n <= 0:
            raise OSError(f"short read of {plan.nbytes} bytes at {plan.offset}")
        got += n


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

    def _ring(self, n_groups: int):
        """Temporary weights for streamed expert groups: per group, every
        projection field as a ``[_STREAM_GROUP, ...]`` array of the slot
        dtype, evaluated, with a writable byte view of its buffer. ``None``
        when a field's checkpoint bytes are not the slot's layout (the
        caller then installs, as before)."""
        rows = self.cache.rows
        if rows is None:
            return None
        ring = []
        for _ in range(min(_STREAM_RING, n_groups)):
            arrays = {
                key: mx.zeros((_STREAM_GROUP,) + tuple(shape), dtype=dtype)
                for key, (shape, dtype, _) in rows.items()
            }
            mx.eval(*arrays.values())
            views = {key: memoryview(a).cast("B") for key, a in arrays.items()}
            ring.append((arrays, views))
        return ring, {key: row for key, (_, _, row) in rows.items()}

    def _forward_streamed(self, flat_x: mx.array, ids: list[int], k: int):
        """Over-capacity prefill that leaves the resident experts in place.

        The routes are grouped by expert, resident experts first, as in
        :meth:`_forward_expert_major`. Free slots (a cold cache) take the
        call's most-routed experts. The resident routes run on the slots;
        every other expert is read straight into temporary weights, in groups
        of ``_STREAM_GROUP``, and its routes run on them through a copy of the
        module that shares everything but those weights, while the next
        groups are read. Nothing is evicted, so the experts the decode keeps
        hot survive the prompt, and no slot is written while the GPU reads
        the slots. Each route still runs once, one route per row, under the
        module's own forward (rounding may differ from the installing path,
        as between residencies).
        """
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        used, counts = np.unique(ids_np, return_counts=True)
        fill = []
        if c.free:
            new = used[[int(e) not in c.slot_of for e in used]]
            if len(new):
                top = np.argsort(-counts[np.searchsorted(used, new)], kind="stable")
                fill = [int(e) for e in new[top[: len(c.free)]]]
                c._ensure_ids(fill)
        resident = np.zeros(c.n_experts, dtype=np.bool_)
        resident[c.slot_expert[c.slot_expert >= 0]] = True
        stream = used[~resident[used]]
        groups = [
            stream[i : i + _STREAM_GROUP] for i in range(0, len(stream), _STREAM_GROUP)
        ]
        ring = self._ring(len(groups)) if groups else ([], {})
        if ring is None:
            return self._forward_expert_major(flat_x, ids, k)
        ring, row_bytes = ring
        filled = set(fill)
        rest = [int(e) for e in used if int(e) not in filled]
        c.note_streamed(rest, int(resident[used].sum()) - len(fill), len(stream))

        rank = ids_np + (~resident[ids_np]) * c.n_experts
        order = np.argsort(rank, kind="stable")  # resident first, then by id
        sorted_ids = ids_np[order]
        n_res = int(resident[ids_np].sum())
        tail = sorted_ids[n_res:]
        pool = _io_pool()
        reads: dict[int, list] = {}

        def start(g: int) -> None:
            _, views = ring[g % len(ring)]
            futures = []
            for j, e in enumerate(groups[g]):
                for name, fi, plan in c._plans(int(e)):
                    row = row_bytes[(name, fi)]
                    view = views[(name, fi)][j * row : (j + 1) * row]
                    if pool is None:
                        _pread_into(c.disk, plan, view)
                    else:
                        futures.append(pool.submit(_pread_into, c.disk, plan, view))
            reads[g] = futures

        def run(glu, a: int, b: int, slots: mx.array) -> mx.array:
            t_idx = mx.array(order[a:b] // k, dtype=mx.int32)
            xe = mx.take(flat_x, t_idx, axis=0)
            return glu(xe, slots.reshape(-1, 1))[:, 0, :]

        outs = []
        try:
            for g in range(len(ring)):
                start(g)
            if n_res:
                chunk = mx.array(sorted_ids[:n_res], dtype=mx.int32)
                o = run(c.glu, 0, n_res, mx.take(c.map, chunk))
                mx.async_eval(o)
                outs.append(o)
            if groups:
                glu = copy.copy(c.glu)
                for name in _PROJS:
                    glu[name] = copy.copy(c.glu[name])
            for g, group in enumerate(groups):
                futures = reads.pop(g)
                wait(futures)
                for f in futures:
                    f.result()
                arrays, _ = ring[g % len(ring)]
                for (name, fi), a in arrays.items():
                    glu[name][_FIELDS[fi]] = a
                a = n_res + int(np.searchsorted(tail, group[0], side="left"))
                b = n_res + int(np.searchsorted(tail, group[-1], side="right"))
                local = np.searchsorted(group, sorted_ids[a:b])
                o = run(glu, a, b, mx.array(local, dtype=mx.int32))
                mx.async_eval(o)
                outs.append(o)
                if g + len(ring) < len(groups):
                    mx.eval(o)  # its weights take the group read next
                    start(g + len(ring))
            mx.eval(outs)
        finally:
            pending = [f for futures in reads.values() for f in futures]
            for f in pending:
                f.cancel()
            if pending:
                wait(pending)
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
                # Decode: misses are read straight into their slots, and a
                # wait on them keeps the GPU clocked.
                ids = _flat_ids(indices)
                if trace is not None:
                    route_trace.record(
                        trace, self._layer, c, ids, n_tok, k, route_trace.ROUTES,
                        decode=True, x=x,
                    )
                pending = self._claim_reads(incoming, ids)
                c._ensure_decode(ids, pending=pending)
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
        forward = self._forward_streamed if _STREAM else self._forward_expert_major
        y = forward(x.reshape(-1, x.shape[-1]), ids, k)
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
