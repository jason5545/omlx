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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...custom_kernels.glm_moe_dsa import fast as glm_fast
from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    _SCORE_DECAY_EVERY,
    _SORT_MIN_ROUTES,
    CheckpointExpertStore,
    ExpertCache,
    _drain,
    _gpu_keepalive,
    _GLUStoreView,
    _io_batch,
    _io_pool,
    _is_mtp_path,
    _minimum_experts,
    _resolve_model_dir,
)
from . import capacity_profile, route_trace
from .switch_layers import _sort_threshold

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, invalid: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return invalid


_PROJS = ("gate_proj", "up_proj", "down_proj")

# glm5_next decode can read the next layer's predicted misses ahead (see
# OffloadedSwitchGLU.stage_next_routes) while the GPU computes and the SSD
# would otherwise wait: the _AHEAD_MAX experts with the most predicted routing
# weight, on a pool of their own, the unused ones dropped when the next
# layer's routes arrive. That gap is 0.85-1.9 ms per layer on GLM-5.3 oQ3.5e
# (M5 Max), about one expert read (1.1 ms, then 0.93 ms per more), hence 2.
# An earlier version (up to 8 per layer, each a task reading its slabs in
# turn, on the demand pool) made decode 12-20% slower: the next layer's own
# misses queued behind reads that ran past its routes. On by default
# (OMLX_MOE_OFFLOAD_PREFETCH=0 turns it off; the file named by
# OMLX_MOE_OFFLOAD_PREFETCH_FILE, ~/.omlx/moe_offload_prefetch by default,
# holding 0 or 1 overrides that while the server runs, re-read at most once
# a second); OMLX_MOE_OFFLOAD_PREFETCH_MAX sets the count.
_PREFETCH = os.environ.get("OMLX_MOE_OFFLOAD_PREFETCH", "1") != "0"
_AHEAD_MAX = max(1, _env_int("OMLX_MOE_OFFLOAD_PREFETCH_MAX", 2, 2))
# Each Python reader takes the GIL from the main thread, which builds the
# next layer's graph while they read, every time a read returns: an expert's
# three big slabs get a task each and its small ones share one, on four
# workers, so one expert is read at a time and the next waits in the queue,
# where dropping it costs nothing (on GLM-5.3 oQ3.5e shapes, M5 Max: 88 us
# of main-thread time per layer reading two experts ahead, against 160 us
# for a task per slab on nine workers). POSIX AIO would take no GIL, but
# macOS cannot cancel a queued AIO read, and the dropped ones then delay the
# next layer's own reads by about a millisecond.
_AHEAD_WORKERS = 4
_AHEAD_BIG = 1 << 20  # slabs read by a task of their own
_AHEAD_BUFFERS = 6  # host buffers read into, reused once their reads are done
_SWITCH = {"checked": float("-inf"), "value": None}
# Decode misses are read straight into their slots (_SlotCache._ensure_decode).
# OMLX_MOE_OFFLOAD_READ_INTO_SLOT=0 keeps the read-then-install path.
_INTO_SLOT = os.environ.get("OMLX_MOE_OFFLOAD_READ_INTO_SLOT", "1") != "0"
# How long those reads may run before the wait keeps the GPU clocked (the
# common adapter waits 0.5 ms; this path was measured at 0).
_KEEPALIVE_GRACE_S = _env_int("OMLX_MOE_OFFLOAD_KEEPALIVE_GRACE_US", 0, 0) * 1e-6


def _env_decay(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if 0.0 < value <= 1.0 else default


# Eviction decay of these slot caches (the common cache keeps _SCORE_DECAY,
# 0.7): routing counts x_SLOT_SCORE_DECAY every _SCORE_DECAY_EVERY calls.
# Replaying a 15-minute agent-loop route trace of GLM-5.3 oQ3.5e at 54%
# residency (M5 Max) through an exact copy of this cache: 0.97 reads 9.3%
# fewer experts in decode than 0.7 (0.95-0.98 within 0.5% of it; LRU 4% and
# LFU 60% more than 0.7). OMLX_MOE_OFFLOAD_SLOT_SCORE_DECAY overrides it.
_SLOT_SCORE_DECAY = _env_decay("OMLX_MOE_OFFLOAD_SLOT_SCORE_DECAY", 0.97)

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
    apply unchanged, with the counts decaying by ``_SLOT_SCORE_DECAY``.
    """

    score_decay = _SLOT_SCORE_DECAY

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        self.glu = glu  # read by _allocate, which the base constructor calls
        super().__init__(glu, capacity, view)
        self.rows = self._row_layout()
        self._ahead_layout = None  # see ahead_slabs

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
        the GPU clocked. A read started ahead (``pending``, :class:`_Ahead`)
        lands in a host buffer and is copied into its slot first, while the
        other misses are still being read (read again if it failed). A failed
        read waits out the others and gives back the slots whose expert did
        not arrive. Falls back to
        :meth:`_ensure_ids` (with the keepalive) when reads are serial or a
        slab's checkpoint bytes are not its slot row's layout.
        """
        pool = _io_pool()
        if not _INTO_SLOT or pool is None or self.rows is None:
            _cancel_reads(pending)
            self._ensure_ids(ids, lambda: None)
            return
        pending = dict(pending or {})
        needed = list(dict.fromkeys(int(e) for e in ids))
        if len(needed) > self.capacity:
            _cancel_reads(pending)
            raise ValueError("Expert cache capacity is smaller than the call's routes")
        np.add.at(self.score, np.asarray(ids, dtype=np.int64), 1.0)
        self._calls += 1
        if self._calls % _SCORE_DECAY_EVERY == 0:
            self.score *= self.score_decay
        misses = [e for e in needed if e not in self.slot_of]
        self.hits += len(needed) - len(misses)
        if not misses:
            _cancel_reads(pending)
            return
        protected = frozenset(needed)
        claimed = []  # (expert, slot); resident once its bytes are in the slot
        groups = []  # per claimed expert: (its read ahead or None, [(slot row, plan, read)])
        landed = set()  # indices into claimed whose expert is in its slot
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
                rows = []
                for name, fi, plan in self._plans(e):
                    row = self.rows[(name, fi)][2]
                    dst = views[(name, fi)][slot * row : (slot + 1) * row]
                    read = None
                    if ahead is None:
                        read = pool.submit(_pread_into, self.disk, plan, dst)
                    rows.append((dst, plan, read))
                groups.append((ahead, rows))
            # Copy what was read ahead while the other misses are being read.
            for i, (ahead, rows) in enumerate(groups):
                if ahead is None:
                    continue
                if ahead.arrived():
                    for n, (dst, _, _) in enumerate(rows):
                        _copy_into(dst, ahead.slab(n))
                    self._landed(*claimed[i])
                    landed.add(i)
                else:  # the read ahead failed: read the expert now
                    groups[i] = (None, [
                        (dst, plan, pool.submit(_pread_into, self.disk, plan, dst))
                        for dst, plan, _ in rows
                    ])
            keepalive = None
            reads = [f for i, (_, rows) in enumerate(groups) if i not in landed
                     for _, _, f in rows]
            if reads and wait(reads, timeout=_KEEPALIVE_GRACE_S).not_done:
                keepalive = _gpu_keepalive()
            for i, (_, rows) in enumerate(groups):
                if i in landed:
                    continue
                futures = [f for _, _, f in rows]
                if keepalive is not None:
                    keepalive.wait(futures)
                else:
                    wait(futures)
                for future in futures:
                    future.result()
                self._landed(*claimed[i])
                landed.add(i)
        finally:
            if len(landed) < len(claimed):
                # Nothing may write a slot once it is given back.
                futures = [f for _, rows in groups for _, _, f in rows if f is not None]
                for future in futures:
                    future.cancel()
                wait(futures)
                for i, (e, slot) in enumerate(claimed):
                    if i not in landed:
                        del self.slot_of[e]
                        self.slot_expert[slot] = -1
                        self.free.append(slot)
            _cancel_reads(pending)
        self.warm = len(self.slot_of) == self.n_experts

    def _landed(self, e: int, slot: int) -> None:
        """Expert ``e``'s bytes are in ``slot``: route to it."""
        self.map[e] = slot
        self.misses += 1
        self.fetched_bytes += self.expert_bytes

    def ahead_slabs(self, e: int) -> list:
        """Expert ``e``'s slabs as ``(fd, offset, bytes)`` in plan order: by
        offset arithmetic when every slab is a row of a stacked checkpoint
        tensor (worked out once), else from the read plans."""
        if self._ahead_layout is None:
            self._ahead_layout = self._stacked_layout() or False
        if self._ahead_layout:
            return [(fd, base + e * nbytes, nbytes) for fd, base, nbytes in self._ahead_layout]
        return [(plan.fd, plan.offset, plan.nbytes) for *_, plan in self._plans(e)]

    def _stacked_layout(self):
        if self.n_experts < 2:
            return None
        last = self.n_experts - 1
        out = []
        for (*_, a), (*_, b), (*_, z) in zip(self._plans(0), self._plans(1), self._plans(last)):
            if not (
                a.fd == b.fd == z.fd
                and a.nbytes == b.nbytes == z.nbytes
                and b.offset == a.offset + a.nbytes
                and z.offset == a.offset + last * a.nbytes
            ):
                return None
            out.append((a.fd, a.offset, a.nbytes))
        return out

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
            self.score *= self.score_decay
        self.hits += hits
        self.misses += streamed
        self.fetched_bytes += streamed * self.expert_bytes


class _Prefetch:
    """Read-ahead state of one wrapper, kept out of the module tree.

    ``gate``/``next`` are the next offloaded MoE layer's router and wrapper;
    ``staged`` is that router's ``(indices, scores)`` for the call in flight;
    ``incoming`` maps each expert the previous layer started reading for this
    one to its :class:`_Ahead`.
    """

    __slots__ = ("gate", "next", "staged", "incoming")

    def __init__(self):
        self.gate = self.next = self.staged = self.incoming = None


_AHEAD_LOCK = threading.Lock()
_AHEAD_POOL: ThreadPoolExecutor | None = None


def _ahead_pool() -> ThreadPoolExecutor | None:
    """The read-ahead pool: apart from the demand pool, so a layer's own
    misses never queue behind reads ahead. ``None`` when reads are serial."""
    global _AHEAD_POOL
    if _io_pool() is None:
        return None
    with _AHEAD_LOCK:
        if _AHEAD_POOL is None:
            _AHEAD_POOL = ThreadPoolExecutor(
                max_workers=_AHEAD_WORKERS, thread_name_prefix="omlx-moe-ahead"
            )
        return _AHEAD_POOL


def _shutdown_ahead_pool() -> None:
    """Drop the read-ahead pool once its reads finish (tests)."""
    global _AHEAD_POOL
    with _AHEAD_LOCK:
        pool, _AHEAD_POOL = _AHEAD_POOL, None
    if pool is not None:
        pool.shutdown(wait=True)


class _Ahead:
    """One expert read ahead into a host buffer: ``slabs`` holds each slab's
    ``(offset, bytes)`` in the buffer, in plan order; ``futures`` its reads."""

    __slots__ = ("buf", "slabs", "futures", "released")

    def __init__(self, nbytes: int):
        # Written once, so the reads into it take no page faults.
        self.buf = np.zeros(nbytes, dtype=np.uint8)
        self.slabs = []
        self.futures = []
        self.released = False

    def idle(self) -> bool:
        """Released, and nothing is still reading into it."""
        return self.released and all(f.done() for f in self.futures)

    def arrived(self) -> bool:
        """Wait for the reads; whether every slab arrived."""
        wait(self.futures)
        return all(not f.cancelled() and f.exception() is None for f in self.futures)

    def slab(self, n: int) -> memoryview:
        offset, nbytes = self.slabs[n]
        return memoryview(self.buf)[offset : offset + nbytes]

    def drop(self) -> None:
        """Cancel the queued reads; a running one finishes in the background,
        and the buffer is not handed out again before it does."""
        for future in self.futures:
            future.cancel()
        self.released = True


class _AheadBuffers:
    """At most ``limit`` read-ahead buffers, shared by the layers. A buffer
    is handed out again only once it was released and every read into it
    (a dropped read may still be running) has finished; a larger one serves
    a smaller expert."""

    def __init__(self, limit: int):
        self.limit = limit
        self._lock = threading.Lock()
        self._all: list[_Ahead] = []

    def take(self, nbytes: int) -> _Ahead | None:
        """A buffer of at least ``nbytes``, or ``None`` when all are busy."""
        with self._lock:
            idle = [i for i, a in enumerate(self._all) if a.idle()]
            fits = [i for i in idle if self._all[i].buf.nbytes >= nbytes]
            if fits:
                a = self._all[min(fits, key=lambda i: self._all[i].buf.nbytes)]
                a.slabs, a.futures, a.released = [], [], False
                return a
            if len(self._all) >= self.limit and not idle:
                return None
            a = _Ahead(nbytes)
            if len(self._all) < self.limit:
                self._all.append(a)
            else:  # an idle buffer too small for this expert makes room
                self._all[idle[0]] = a
            return a


_AHEAD = _AheadBuffers(_AHEAD_BUFFERS)


def _prefetch_on() -> bool:
    """``_PREFETCH``, unless the switch file says otherwise ("0" off, anything
    else on); the file is re-read at most once a second."""
    now = time.monotonic()
    if now - _SWITCH["checked"] >= 1.0:
        _SWITCH["checked"] = now
        path = os.environ.get("OMLX_MOE_OFFLOAD_PREFETCH_FILE") or (
            Path.home() / ".omlx" / "moe_offload_prefetch"
        )
        try:
            _SWITCH["value"] = Path(path).expanduser().read_text().strip() != "0"
        except OSError:
            _SWITCH["value"] = None
    value = _SWITCH["value"]
    return _PREFETCH if value is None else value


def _read_slabs(store_view, items) -> None:
    """Read ``(fd, offset, view)`` items in turn (positional; any thread).
    ``store_view`` is only held, so the shard descriptors stay open."""
    for fd, offset, view in items:
        got, nbytes = 0, len(view)
        while got < nbytes:
            n = os.preadv(fd, [view[got:]], offset + got)
            if n <= 0:
                raise OSError(f"short read of {nbytes} bytes at {offset}")
            got += n


def _start_ahead(pool, cache: ExpertCache, e: int, ahead: _Ahead) -> None:
    """Start reading expert ``e`` into ``ahead``'s buffer: a task per big
    slab, one for the small ones."""
    buf = memoryview(ahead.buf)
    small, pos = [], 0
    for fd, offset, nbytes in cache.ahead_slabs(e):
        item = (fd, offset, buf[pos : pos + nbytes])
        ahead.slabs.append((pos, nbytes))
        pos += nbytes
        if nbytes >= _AHEAD_BIG:
            # Kept as each starts, so the buffer waits for every read into it.
            ahead.futures.append(pool.submit(_read_slabs, cache.disk, [item]))
        else:
            small.append(item)
    if small:
        ahead.futures.append(pool.submit(_read_slabs, cache.disk, small))


def _copy_into(dst, src) -> None:
    """Copy a slab read ahead into its slot row (numpy releases the GIL)."""
    np.copyto(np.frombuffer(dst, dtype=np.uint8), np.frombuffer(src, dtype=np.uint8))


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
    """Drop reads ahead without waiting (see :meth:`_Ahead.drop`)."""
    for ahead in (reads or {}).values():
        ahead.drop()


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
        about half of the next layer's misses (53-56% on GLM-5.3 oQ3.5e at
        54% residency, with about one unused read per used one). After this
        layer's own misses are read, the decode branch starts reading the
        predicted non-resident ones with the most routing weight, which the
        next layer then finds done or in flight. Reads only: nothing is
        installed or evicted on a prediction, so the cache and every output
        stay as without it. Returns ``[]`` when nothing is predicted.
        """
        pf = self._prefetch
        pf.staged = None
        if pf.gate is None or not _prefetch_on() or _ahead_pool() is None:
            return []
        pf.staged = tuple(pf.gate(x))
        return list(pf.staged)

    def _claim_reads(self, incoming, ids) -> dict | None:
        """The reads ahead this call misses on; the rest are dropped."""
        if not incoming:
            return None
        slot_of = self.cache.slot_of
        need = {e for e in ids if e not in slot_of}
        claimed = {e: a for e, a in incoming.items() if e in need}
        _cancel_reads({e: a for e, a in incoming.items() if e not in claimed})
        return claimed or None

    def _read_next(self, staged) -> None:
        """Start reading the next layer's predicted non-resident experts, the
        ``_AHEAD_MAX`` with the most predicted routing weight over the rows."""
        nxt = self._prefetch.next
        cache = nxt.cache
        _cancel_reads(nxt._prefetch.incoming)
        nxt._prefetch.incoming = None
        pool = _ahead_pool()
        # Only the read-into-slot path takes a buffer's bytes.
        if cache.warm or pool is None or cache.rows is None or not _INTO_SLOT:
            return
        indices, weights = staged
        weight = {}
        for e, w in zip(_flat_ids(indices), _flat_ids(weights)):
            if e not in cache.slot_of:
                weight[e] = weight.get(e, 0.0) + w
        incoming = {}
        for e in sorted(weight, key=weight.get, reverse=True)[:_AHEAD_MAX]:
            ahead = _AHEAD.take(cache.expert_bytes)
            if ahead is None:
                break
            _start_ahead(pool, cache, e, ahead)
            incoming[e] = ahead
        nxt._prefetch.incoming = incoming or None

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
                claimed = self._claim_reads(incoming, ids) or {}
                try:
                    c._ensure_decode(ids, pending=claimed or None)
                finally:
                    # Copied into their slots, or drained on the way out.
                    for ahead in claimed.values():
                        ahead.released = True
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


def _profiled_capacities(covered, uniform, model_dir, minimum):
    """Per-layer capacities from the model's capacity profile (see
    capacity_profile.py), keyed like ``uniform``; ``None`` keeps the uniform
    split (no profile, a profile for other layers or expert counts, or
    layers without a decoder index)."""
    path = capacity_profile.profile_path(model_dir)
    profile = capacity_profile.load_profile(path)
    if profile is None or not covered:
        return None
    layers = [_layer_index(path_) for *_, path_, _view, _n, _b in covered]
    counts = {n for *_, n, _ in covered}
    if None in layers or len(set(layers)) != len(layers) or len(counts) != 1:
        return None
    plan = capacity_profile.plan_capacities(
        profile,
        {L: b // n for L, (*_, n, b) in zip(layers, covered)},
        {L: uniform[i] for i, L in enumerate(layers)},
        counts.pop(),
        minimum,
    )
    if plan is None:
        logger.info(
            "deepseek_v4 moe expert offload: capacity profile %s does not fit "
            "this model; uniform capacity",
            path,
        )
        return None
    logger.info(
        "deepseek_v4 moe expert offload: per-layer capacity from %s (%d..%d slots, "
        "uniform %d): %s",
        path,
        min(plan.values()),
        max(plan.values()),
        uniform[0],
        " ".join(f"{L}:{plan[L]}" for L in sorted(plan)),
    )
    return {i: plan[L] for i, L in enumerate(layers)}


def apply_deepseek_v4_moe_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Wrap every covered DeepSeek V4 SwitchGLU; returns the number wrapped.

    Same contract as ``apply_moe_expert_offload``: runs before lazy weights
    materialize and skips (with a logged reason) any module the checkpoint
    does not cover.

    ``mtp_resident`` keeps the embedded MTP draft head's experts fully
    resident (glm5_next Lightning MTP + offload): the head is one decoder
    layer whose drafts the streamed backbone verifies, so streaming its
    experts would add SSD latency to every draft step. With the flag off,
    the head wraps like any other layer, exactly as before.
    """
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
    covered = []
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
        layer_bytes = sum(
            int(np.prod(glu[proj][field].shape)) * glu[proj][field].dtype.size
            for proj in _PROJS
            for field in _fields(glu[proj])
        )
        covered.append((parent, key, glu, path, view, n_experts, layer_bytes))
    uniform = {
        i: min(n, max(minimum, round(n * resident_fraction)))
        for i, (*_, n, _) in enumerate(covered)
    }
    capacities = _profiled_capacities(covered, uniform, model_dir, minimum) or uniform

    wrapped = 0
    total_bytes = resident_bytes = 0
    chain = []  # (layer index, wrapper, its router) of glm5_next MoE blocks
    for i, (parent, key, glu, path, view, n_experts, layer_bytes) in enumerate(covered):
        capacity = capacities[i]
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
