# SPDX-License-Identifier: Apache-2.0
"""Paged-SSD block tensors read straight into one restore buffer."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cache import direct_restore
from omlx.cache.direct_restore import Piece, gather_along_axis, safetensors_file
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler


def _block(start: int, n: int, seed: int, channels: int = 1):
    mx.random.seed(seed)
    return (
        mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
        mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
        mx.random.normal((1, n, 4)).astype(mx.bfloat16),
        mx.broadcast_to(mx.arange(start, start + n, dtype=mx.int32)[None, None], (1, channels, n)),
    )


def _write(tmp_path, name, layer, elements):
    path = tmp_path / f"{name}.safetensors"
    tensors = {f"layer_{layer}_state_{k}": mx.contiguous(e) for k, e in enumerate(elements)}
    tensors["layer_0_other"] = mx.ones((3,))  # unrelated tensor in the same file
    mx.save_safetensors(str(path), tensors, metadata={"format": "test"})
    return path


def _lazy(path, layer):
    arrays = mx.load(str(path))
    return tuple(arrays[f"layer_{layer}_state_{k}"] for k in range(4))


def test_header_spans_point_at_each_tensor(tmp_path):
    elements = _block(0, 5, seed=1)
    path = _write(tmp_path, "a", 3, elements)
    parsed = safetensors_file(path)
    assert parsed is safetensors_file(path)  # cached
    span = parsed.span("layer_3_state_0")
    assert span.dtype == "bfloat16" and span.shape == (1, 2, 5, 8) and span.nbytes == 1 * 2 * 5 * 8 * 2
    with open(path, "rb") as f:
        f.seek(span.offset)
        assert f.read(span.nbytes) == bytes(memoryview(mx.contiguous(elements[0])).cast("B"))
    assert safetensors_file(tmp_path / "missing.safetensors") is None


@pytest.mark.parametrize("k,axis", [(0, 2), (1, 2), (2, 1), (3, 2)])
@pytest.mark.parametrize("channels", [1, 3])
def test_gather_matches_concatenate_with_zero_padding(tmp_path, k, axis, channels):
    lengths = (2048, 2048, 700)
    blocks, pieces, start = [], [], 0
    for i, n in enumerate(lengths):
        elements = _block(start, n, seed=10 + i, channels=channels)
        blocks.append(elements[k])
        if i == 1:  # served from memory
            pieces.append(Piece(n, array=elements[k]))
        else:
            parsed = safetensors_file(_write(tmp_path, f"b{i}", 7, elements))
            pieces.append(Piece(n, file=parsed, span=parsed.span(f"layer_7_state_{k}")))
        start += n
    total = sum(lengths)
    capacity = total + 1500
    got = gather_along_axis(pieces, axis=axis, capacity=capacity, shape=blocks[0].shape, dtype=blocks[0].dtype)
    pad_shape = list(blocks[0].shape)
    pad_shape[axis] = capacity - total
    want = mx.concatenate([*blocks, mx.zeros(pad_shape, dtype=blocks[0].dtype)], axis=axis)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert mx.array_equal(got, want).item()


def test_gather_rejects_a_span_that_does_not_match(tmp_path):
    elements = _block(0, 16, seed=3)
    parsed = safetensors_file(_write(tmp_path, "c", 0, elements))
    wrong = Piece(8, file=parsed, span=parsed.span("layer_0_state_0"))
    with pytest.raises(ValueError):
        gather_along_axis([wrong], axis=2, capacity=16, shape=(1, 2, 8, 8), dtype=mx.bfloat16)


def _states(tmp_path, lengths, *, memory_index=None, channels=1):
    states, expected, start = [], [], 0
    for i, n in enumerate(lengths):
        elements = _block(start, n, seed=20 + i, channels=channels)
        expected.append(elements)
        path = _write(tmp_path, f"s{i}", 5, elements)
        lazy = _lazy(path, 5)
        source = None if i == memory_index else safetensors_file(path)
        states.append({"states": lazy, "direct_source": (source, "layer_5_state")})
        start += n
    return states, expected


@pytest.mark.parametrize("memory_index", [None, 2])
def test_handler_reads_blocks_in_place_and_matches_concatenate(tmp_path, memory_index, monkeypatch):
    lengths = (2048, 2048, 2048, 913)
    states, expected = _states(tmp_path, lengths, memory_index=memory_index)
    calls = []
    real = direct_restore.gather_along_axis

    def counting(*args, **kwargs):
        calls.append(kwargs["axis"])
        return real(*args, **kwargs)

    monkeypatch.setattr(direct_restore, "gather_along_axis", counting)
    handler = Qwen4QSAKVCacheHandler()
    direct = handler.concatenate_states(states)
    assert calls == [2, 2, 1, 2]

    plain = handler.concatenate_states([{"states": s["states"]} for s in states])
    assert direct["qsa_length"] == plain["qsa_length"] == sum(lengths)
    for got, want in zip(direct["states"], plain["states"]):
        assert got.shape == want.shape and mx.array_equal(got, want).item()

    cache = handler.reconstruct_cache(direct)
    keys, values, index_keys, positions = cache.state
    assert mx.array_equal(keys, mx.concatenate([e[0] for e in expected], axis=2)).item()
    assert mx.array_equal(index_keys, mx.concatenate([e[2] for e in expected], axis=1)).item()
    assert mx.array_equal(positions, mx.arange(sum(lengths), dtype=mx.int32)[None]).item()


def test_handler_falls_back_without_file_sources_or_on_mixed_positions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(direct_restore, "gather_along_axis", lambda *a, **k: calls.append(1))
    handler = Qwen4QSAKVCacheHandler()
    states, _ = _states(tmp_path, (64, 64))
    for state in states:
        state["direct_source"] = (SimpleNamespace(span=lambda name: None), "layer_5_state")
    handler.concatenate_states(states)
    mixed, _ = _states(tmp_path, (64, 64))
    mixed[1]["states"] = _block(64, 64, seed=9, channels=3)
    handler.concatenate_states(mixed)
    assert calls == []


def test_direct_read_source_skips_blocks_served_from_memory(tmp_path):
    path = _write(tmp_path, "d", 0, _block(0, 4, seed=4))
    fake = SimpleNamespace(
        _hot_cache_lock=threading.Lock(),
        _hot_cache={b"hot": {}},
        _pending_write_hashes_lock=threading.Lock(),
        _pending_write_buffers={b"pending": {}},
        _index=SimpleNamespace(get=lambda h: SimpleNamespace(file_path=path) if h != b"none" else None),
    )
    read = PagedSSDCacheManager.direct_read_source
    assert read(fake, b"hot") is None
    assert read(fake, b"pending") is None
    assert read(fake, b"none") is None
    assert read(fake, b"disk").span("layer_0_state_0").shape == (1, 2, 4, 8)
