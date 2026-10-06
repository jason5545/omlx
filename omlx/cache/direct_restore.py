# SPDX-License-Identifier: Apache-2.0
"""Read paged-SSD block tensors straight into one restore buffer.

A prefix restore used to let ``mx.load`` read every block's tensors into
their own Metal buffers (a 167k-token Qwen4 prefix is ~80 blocks x 12 QSA
layers x 4 tensors, thousands of allocations, each a residency-set commit)
and then concatenate them on the GPU into a second, full-size copy. Here the
final array is allocated once and each block's bytes are ``preadv``-ed into
their place in it, so a restore allocates one buffer per tensor and copies
the prefix once.

Safetensors stores a tensor C-contiguously; a block of ``L`` tokens along
the sequence axis is ``outer`` runs of ``L * inner`` bytes (``outer`` = the
product of the dims before the axis), which land ``capacity * inner`` bytes
apart in the destination. One ``preadv`` with ``outer`` iovecs reads a
block's tensor.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - MLX-less environments
    mx = None

_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool_",
}
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_HEADER_CACHE_ENTRIES = 8192
_READ_THREADS = 8


@dataclass(frozen=True)
class TensorSpan:
    dtype: str  # mlx dtype name
    shape: tuple[int, ...]
    offset: int  # absolute file offset of the first byte
    nbytes: int


@dataclass(frozen=True)
class SafetensorsFile:
    """A parsed safetensors header: where each tensor's bytes sit."""

    path: str
    tensors: dict[str, TensorSpan]

    def span(self, name: str) -> TensorSpan | None:
        return self.tensors.get(name)


_headers: OrderedDict[str, SafetensorsFile] = OrderedDict()
_headers_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def safetensors_file(path: str | Path) -> SafetensorsFile | None:
    """Parse (and cache) a block file's header, or None if unreadable.

    Block files are content-addressed and never rewritten in place, so a
    header is cached by path.
    """
    key = str(path)
    with _headers_lock:
        hit = _headers.get(key)
        if hit is not None:
            _headers.move_to_end(key)
            return hit
    try:
        with open(key, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (size,) = struct.unpack("<Q", raw)
            if not 0 < size <= _MAX_HEADER_BYTES:
                return None
            header = json.loads(f.read(size))
    except (OSError, ValueError):
        return None
    data_start = 8 + size
    tensors: dict[str, TensorSpan] = {}
    try:
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(info["dtype"])
            if dtype is None:
                continue
            begin, end = info["data_offsets"]
            tensors[name] = TensorSpan(
                dtype, tuple(int(d) for d in info["shape"]), data_start + int(begin), int(end) - int(begin)
            )
    except (KeyError, TypeError, ValueError):
        return None
    parsed = SafetensorsFile(key, tensors)
    with _headers_lock:
        _headers[key] = parsed
        _headers.move_to_end(key)
        while len(_headers) > _HEADER_CACHE_ENTRIES:
            _headers.popitem(last=False)
    return parsed


def forget(path: str | Path) -> None:
    """Drop a cached header (the file was removed or replaced)."""
    with _headers_lock:
        _headers.pop(str(path), None)


def _read_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=_READ_THREADS, thread_name_prefix="omlx-restore-read"
            )
        return _pool


def _preadv_all(path: str, offset: int, views: list[memoryview]) -> None:
    expected = sum(v.nbytes for v in views)
    fd = os.open(path, os.O_RDONLY)
    try:
        done = 0
        while done < expected:
            # Skip what earlier (short) reads already filled.
            skip = done
            pending = []
            for v in views:
                if skip >= v.nbytes:
                    skip -= v.nbytes
                    continue
                pending.append(v[skip:] if skip else v)
                skip = 0
            n = os.preadv(fd, pending, offset + done)
            if n <= 0:
                raise OSError(f"short read from {path}: {done}/{expected} bytes")
            done += n
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Piece:
    """One block's tensor: read from ``file`` at ``span``, or copied from
    ``array`` (a block served from memory)."""

    length: int  # tokens along the gather axis
    file: SafetensorsFile | None = None
    span: TensorSpan | None = None
    array: Any = None


def gather_along_axis(
    pieces: Sequence[Piece],
    *,
    axis: int,
    capacity: int,
    shape: Sequence[int],
    dtype: Any,
) -> Any:
    """``concatenate(pieces, axis)`` zero-padded to ``capacity`` on ``axis``.

    ``shape`` is any block's shape (the gather axis is ignored). File pieces
    are read straight into the result; array pieces are copied on the host.
    Raises on any mismatch or I/O error; the caller falls back to
    ``mx.concatenate``.
    """
    total = sum(p.length for p in pieces)
    if total > capacity:
        raise ValueError("gather capacity is smaller than the pieces")
    out_shape = list(shape)
    out_shape[axis] = capacity
    dest = mx.zeros(out_shape, dtype=dtype)
    mx.eval(dest)
    flat = memoryview(dest).cast("B")
    itemsize = dtype.size
    outer = math.prod(out_shape[:axis])
    inner = math.prod(out_shape[axis + 1 :]) * itemsize
    stride = capacity * inner
    jobs = []
    start = 0
    for piece in pieces:
        run = piece.length * inner
        views = [flat[o * stride + start * inner : o * stride + start * inner + run] for o in range(outer)]
        if piece.file is not None:
            span = piece.span
            if span is None or span.nbytes != run * outer:
                raise ValueError("block tensor size does not match its shape")
            jobs.append((piece.file.path, span.offset, views))
        else:
            src = mx.contiguous(piece.array)
            mx.eval(src)
            raw = memoryview(src).cast("B")
            if raw.nbytes != run * outer:
                raise ValueError("block array size does not match its shape")
            for o, view in enumerate(views):
                view[:] = raw[o * run : (o + 1) * run]
        start += piece.length
    if jobs:
        futures = [_read_pool().submit(_preadv_all, *job) for job in jobs]
        for future in futures:
            future.result()
    return dest
