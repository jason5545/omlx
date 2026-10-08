# SPDX-License-Identifier: Apache-2.0
"""Routing trace of the offloaded glm5_next / DeepSeek V4 MoE layers.

Off unless asked for, and never part of what the model computes: while on,
every call of an offloaded layer appends the expert ids it routed to (and,
optionally, the FFN input its router saw) to a binary file, so expert cache
policies, per-layer capacities and route predictors can be replayed offline
against real routing instead of being estimated from hit counters.

Turning it on needs no restart. ``OMLX_MOE_ROUTE_TRACE`` names the directory
(default ``~/.omlx/route_trace``); recording runs while the file ``ENABLE``
exists in it, checked at most once a second. ``ENABLE`` may hold ``x=1`` to
also record decode-sized FFN inputs (bf16 bits; it adds one host sync per
layer, so the decode is slower while it records) and ``max_mb=<n>`` to cap
the file (default 2048). Each switch-on starts a new file, named by process
id and start time.

Recording reads only what the wrapper already has or what is already
evaluated; it installs, evicts and reorders nothing. The first record of
each layer in a file is that layer's cache state (resident experts, decayed
counts, call count), so a replay can start from where the live cache was.

File layout (little-endian): ``b"OMLXRT01"``, a u32-length JSON header
(``pid``, ``t0_ns`` on ``time.perf_counter_ns``, ``t0_epoch``, ``x``), then
records of ``<BBBBIIQ`` — kind, layer, top-k, flags, rows, n, t_ns — and a
payload:

- ``ROUTES`` / ``EXPERT_MAJOR``: ``uint16[rows * k]`` expert ids in route
  order; with ``FLAG_X``, a u32 hidden size and ``uint16[rows * hidden]``.
  ``ROUTES`` is one ``_ensure_ids`` over all routes (decode, verify and
  in-capacity prefill); ``EXPERT_MAJOR`` is the over-capacity prefill path.
- ``STATE``: rows = capacity, n = experts; ``int32[capacity]`` slot experts
  (-1 empty), ``float32[n]`` decayed counts, u32 call count.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"OMLXRT01"
ROUTES, EXPERT_MAJOR, STATE = 0, 1, 2
FLAG_WARM, FLAG_X, FLAG_DECODE = 1, 2, 4
_HEAD = struct.Struct("<BBBBIIQ")
_CHECK_EVERY_S = 1.0
_X_MAX_ROWS = 8  # decode/verify only: a prefill's inputs would be gigabytes
_FLUSH_EVERY = 512


def _trace_dir() -> Path:
    return Path(
        os.environ.get("OMLX_MOE_ROUTE_TRACE") or Path.home() / ".omlx" / "route_trace"
    ).expanduser()


class _Writer:
    """One trace file. ``record`` runs on the generating thread under a lock."""

    def __init__(self, directory: Path, with_x: bool, max_bytes: int):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"routes-{os.getpid()}-{stamp}.bin"
        self.with_x = with_x
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.seen: set[int] = set()
        self.records = 0
        self.full = False
        self.f = open(self.path, "wb", buffering=4 << 20)
        header = json.dumps(
            {
                "pid": os.getpid(),
                "t0_ns": time.perf_counter_ns(),
                "t0_epoch": time.time(),
                "x": with_x,
            }
        ).encode()
        self.f.write(MAGIC + struct.pack("<I", len(header)) + header)
        self.written = self.f.tell()

    def _put(self, kind, layer, k, flags, rows, n, payload: list) -> None:
        size = _HEAD.size + sum(len(p) for p in payload)
        if self.written + size > self.max_bytes:
            if not self.full:
                self.full = True
                logger.warning("moe route trace: %s reached its size cap", self.path)
            return
        self.f.write(
            _HEAD.pack(kind, layer, k, flags, rows, n, time.perf_counter_ns())
        )
        for p in payload:
            self.f.write(p)
        self.written += size
        self.records += 1
        if self.records % _FLUSH_EVERY == 0:
            self.f.flush()

    def state(self, layer: int, cache) -> None:
        slots = np.asarray(cache.slot_expert, dtype=np.int32)
        score = np.asarray(cache.score, dtype=np.float32)
        self._put(
            STATE,
            layer,
            0,
            FLAG_WARM if cache.warm else 0,
            len(slots),
            len(score),
            [slots.tobytes(), score.tobytes(), struct.pack("<I", cache._calls)],
        )

    def close(self) -> None:
        with self.lock:
            if not self.f.closed:
                self.f.close()


class _Tracer:
    """Polls the switch; holds the open writer while it is on."""

    def __init__(self):
        self.lock = threading.Lock()
        self.writer: _Writer | None = None
        self.next_check = 0.0

    def _poll(self) -> _Writer | None:
        now = time.monotonic()
        if now < self.next_check:
            return self.writer
        with self.lock:
            if now < self.next_check:
                return self.writer
            self.next_check = now + _CHECK_EVERY_S
            directory = _trace_dir()
            enable = directory / "ENABLE"
            try:
                on = enable.is_file()
                opts = dict(
                    kv.split("=", 1)
                    for kv in enable.read_text().split()
                    if "=" in kv
                ) if on else {}
            except OSError:
                on, opts = False, {}
            if on and self.writer is None:
                try:
                    self.writer = _Writer(
                        directory,
                        opts.get("x", "0") not in ("0", ""),
                        int(float(opts.get("max_mb", 2048)) * (1 << 20)),
                    )
                    logger.info("moe route trace: recording to %s", self.writer.path)
                except Exception:
                    logger.warning("moe route trace: cannot start", exc_info=True)
                    self.writer = None
            elif not on and self.writer is not None:
                writer, self.writer = self.writer, None
                writer.close()
                logger.info(
                    "moe route trace: stopped, %d records in %s",
                    writer.records,
                    writer.path,
                )
            return self.writer

    def active(self) -> _Writer | None:
        writer = self._poll()
        return None if writer is None or writer.full else writer

    def close(self) -> None:
        with self.lock:
            if self.writer is not None:
                self.writer.close()
                self.writer = None


_TRACER = _Tracer()
atexit.register(_TRACER.close)


def active() -> _Writer | None:
    """The open trace, or ``None`` (recording off). Cheap while off."""
    return _TRACER.active()


def record(writer: _Writer, layer: int, cache, ids, rows: int, k: int, kind: int,
           decode: bool = False, x=None) -> None:
    """Append one call of ``layer``. Call before the cache handles ``ids``.

    ``x`` (the call's FFN input) is written only when the trace asks for it
    and the call is decode-sized; reading it waits for what is already
    queued on the stream.
    """
    payload = [np.asarray(ids, dtype=np.uint16).tobytes()]
    flags = (FLAG_WARM if cache.warm else 0) | (FLAG_DECODE if decode else 0)
    if writer.with_x and x is not None and rows <= _X_MAX_ROWS:
        import mlx.core as mx

        bits = x.reshape(rows, -1)
        bits = bits.view(mx.uint16) if bits.dtype.size == 2 else bits.astype(
            mx.float16
        ).view(mx.uint16)
        host = np.array(bits)
        payload += [struct.pack("<I", host.shape[-1]), host.tobytes()]
        flags |= FLAG_X
    with writer.lock:
        if writer.f.closed:
            return
        if layer not in writer.seen:
            writer.seen.add(layer)
            writer.state(layer, cache)
        writer._put(kind, layer, k, flags, rows, len(ids), payload)


def read(path) -> tuple[dict, list[dict]]:
    """``(header, records)`` of a trace file; each record is a dict with
    ``kind``, ``layer``, ``k``, ``flags``, ``rows``, ``t_ns`` and ``ids``
    (``[rows, k]``), ``x`` (``uint16[rows, hidden]`` or ``None``), or for
    ``STATE`` ``slots``, ``score`` and ``calls``. A truncated tail (the
    process stopped mid-write) is dropped."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path} is not a route trace")
    (hlen,) = struct.unpack_from("<I", data, 8)
    header = json.loads(data[12 : 12 + hlen])
    pos = 12 + hlen
    out = []
    while pos + _HEAD.size <= len(data):
        kind, layer, k, flags, rows, n, t_ns = _HEAD.unpack_from(data, pos)
        p = pos + _HEAD.size
        rec = {"kind": kind, "layer": layer, "k": k, "flags": flags,
               "rows": rows, "t_ns": t_ns}
        try:
            if kind == STATE:
                end = p + 4 * rows + 4 * n + 4
                if end > len(data):
                    break
                rec["slots"] = np.frombuffer(data, np.int32, rows, p)
                rec["score"] = np.frombuffer(data, np.float32, n, p + 4 * rows)
                (rec["calls"],) = struct.unpack_from("<I", data, p + 4 * rows + 4 * n)
            else:
                end = p + 2 * n
                if end > len(data):
                    break
                rec["ids"] = np.frombuffer(data, np.uint16, n, p).reshape(rows, k)
                rec["x"] = None
                if flags & FLAG_X:
                    (hidden,) = struct.unpack_from("<I", data, end)
                    xs = end + 4
                    end = xs + 2 * rows * hidden
                    if end > len(data):
                        break
                    bits = np.frombuffer(data, np.uint16, rows * hidden, xs)
                    rec["x"] = bits.reshape(rows, hidden)
        except struct.error:
            break
        out.append(rec)
        pos = end
    return header, out


__all__ = ["EXPERT_MAJOR", "ROUTES", "STATE", "active", "read", "record"]
