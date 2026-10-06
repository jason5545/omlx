"""Exactness and lifecycle tests for incremental Qwen4 QSA block caching."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


def _identity_rope(x, position_ids):
    del position_ids
    return x


def _append(cache, raw_keys, start, stop):
    length = stop - start
    keys = raw_keys[:, start:stop, :4].reshape(1, 1, length, 4)
    values = (keys + 1).astype(keys.dtype)
    cache.update_and_fetch(keys, values)
    cache.update_indexer(
        raw_keys[:, start:stop],
        mx.arange(start, stop, dtype=mx.int32)[None],
    )


def test_qsa_kv_and_raw_index_buffers_grow_geometrically_with_logical_views():
    cache = language.QSAKVCache()
    raw = mx.arange(8194 * 8, dtype=mx.float32).reshape(1, 8194, 8)

    _append(cache, raw, 0, 2050)
    kv_backing = cache.keys
    index_backing = cache._index_keys
    assert cache.keys.shape[2] == 8192
    assert cache._index_keys.shape[1] == 8192
    assert cache.index_keys.shape == (1, 2050, 8)

    _append(cache, raw, 2050, 4098)
    assert cache.keys is kv_backing
    assert cache._index_keys is index_backing
    assert cache.state[0].shape[2] == 4098
    assert cache.state[2].shape[1] == 4098

    _append(cache, raw, 4098, 8194)
    assert cache.keys.shape[2] == 16384
    assert cache._index_keys.shape[1] == 16384
    assert cache.state[0].shape[2] == 8194
    assert cache.state[2].shape[1] == 8194
    assert language.QSAQuantizedKVCache.step == 8192
    assert language.QSAQuantizedKVCache.geometric_growth is True


@pytest.mark.parametrize("chunks", [(2048, 2048, 2048), (2050, 2048, 2047)])
def test_completed_qsa_blocks_match_one_shot_and_only_compute_new_suffix(chunks):
    total = sum(chunks)
    raw = mx.sin(mx.arange(total * 8, dtype=mx.float32)).reshape(1, total, 8)
    incremental = language.QSAKVCache()
    block_calls = []

    def tracked_norm(x):
        block_calls.append(int(x.shape[1]))
        return x * mx.array(1.25, dtype=x.dtype)

    start = 0
    for length in chunks:
        stop = start + length
        incremental.update_indexer(
            raw[:, start:stop],
            mx.arange(start, stop, dtype=mx.int32)[None],
        )
        actual = incremental.pooled_indexer_keys(
            4,
            tracked_norm,
            _identity_rope,
            cache_tag=tracked_norm,
        )
        start = stop

    calls_before_noop = list(block_calls)
    cached_again = incremental.pooled_indexer_keys(
        4,
        tracked_norm,
        _identity_rope,
        cache_tag=tracked_norm,
    )
    assert block_calls == calls_before_noop
    assert block_calls == [512, 512, 512]

    one_shot = qsa_fast.pool_completed_index_keys(
        raw,
        mx.arange(total, dtype=mx.int32)[None],
        compress_ratio=4,
        index_key_norm=lambda x: x * mx.array(1.25, dtype=x.dtype),
        apply_index_rope=_identity_rope,
    )
    mx.eval(actual, cached_again, one_shot)
    assert mx.array_equal(actual, one_shot).item()
    assert mx.array_equal(cached_again, one_shot).item()


@pytest.mark.parametrize("chunks", [(8, 8, 8), (6, 7, 12)])
def test_gathered_qsa_one_shot_and_incremental_appends_are_exact(chunks, monkeypatch):
    total = sum(chunks)
    mx.random.seed(431)
    queries = mx.random.normal((1, 4, total, 8)).astype(mx.float16)
    keys = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    values = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    index_queries = mx.random.normal((1, total, 3, 8)).astype(mx.float16)
    index_keys = mx.random.normal((1, total, 8)).astype(mx.float16)
    positions = mx.arange(total, dtype=mx.int32)[None]
    kwargs = dict(
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=8,
        compress_ratio=4,
        token_budget=8,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
        query_chunk=3,
    )
    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", lambda *a, **k: None)

    expected = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        **kwargs,
    )

    cache = language.QSAKVCache()
    outputs = []
    start = 0
    for length in chunks:
        stop = start + length
        cache.update_indexer(index_keys[:, start:stop], positions[:, start:stop])
        pooled = cache.pooled_indexer_keys(
            4,
            kwargs["index_key_norm"],
            kwargs["apply_index_rope"],
            cache_tag=kwargs["index_key_norm"],
        )
        outputs.append(
            qsa_fast.contiguous_causal_gathered_qsa(
                queries[:, :, start:stop],
                keys[:, :, :stop],
                values[:, :, :stop],
                index_queries[:, start:stop],
                cache.index_keys,
                cache.index_position_ids,
                pooled_index_keys=pooled,
                **kwargs,
            )
        )
        start = stop
    actual = mx.concatenate(outputs, axis=1)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_qsa_ephemeral_pool_rebuilds_after_restore_extract_and_trim():
    cache = language.QSAKVCache()
    raw = mx.sin(mx.arange(13 * 8, dtype=mx.float32)).reshape(1, 13, 8)
    _append(cache, raw, 0, 13)
    pooled = cache.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=cache
    )
    mx.eval(pooled)
    assert cache._pooled_index_offset == 3
    assert len(cache.state) == 4

    restored = language.QSAKVCache()
    restored.prefix_cache_restore(cache.prefix_cache_snapshot())
    assert restored._pooled_index_keys is None
    restored_pool = restored.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=restored
    )

    extracted = cache.extract(0)
    assert extracted._pooled_index_keys is None
    extracted_pool = extracted.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=extracted
    )
    mx.eval(restored_pool, extracted_pool, pooled)
    assert mx.array_equal(restored_pool, pooled).item()
    assert mx.array_equal(extracted_pool, pooled).item()

    assert cache.trim(3) == 3
    # trim keeps the pooled blocks below the new complete count (10 // 4 = 2)
    # and only clamps the pooled frontier; the tail is re-pooled lazily.
    assert cache._pooled_index_keys is not None
    assert cache._pooled_index_offset == 2
    replacement = mx.cos(mx.arange(3 * 8, dtype=mx.float32)).reshape(1, 3, 8)
    _append(cache, mx.concatenate([raw[:, :10], replacement], axis=1), 10, 13)
    rebuilt = cache.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=cache
    )
    expected = qsa_fast.pool_completed_index_keys(
        cache.index_keys,
        cache.index_position_ids,
        compress_ratio=4,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
    )
    mx.eval(rebuilt, expected)
    assert mx.array_equal(rebuilt, expected).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_portable_qsa_flattened_gemm_is_exactly_the_broadcast_reference(dtype):
    mx.random.seed(909)
    queries = mx.random.normal((1, 7, 4, 128)).astype(dtype)
    pooled = mx.random.normal((1, 19, 128)).astype(dtype)
    broadcast = queries.astype(mx.float32) @ pooled[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    expected = mx.sum(mx.maximum(broadcast, 0), axis=-2) / math.sqrt(128)
    actual = qsa_fast._portable_indexer_scores(queries, pooled, 128)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


cache_module = importlib.import_module("mlx_vlm.models.qwen4_exp.cache")


def test_kv_capacity_ladder_depends_only_on_length_and_stays_within_a_quarter():
    bucket = cache_module.kv_capacity_bucket
    assert [bucket(n, 8192) for n in (1, 8192, 8193, 32768, 32769)] == [8192, 8192, 16384, 32768, 40960]
    # Consecutive agent turns at 90k and 220k ask for one size.
    assert bucket(89_891, 8192) == bucket(90_925, 8192) == 98_304
    assert bucket(220_087, 8192) == bucket(222_778, 8192) == 229_376
    previous = 0
    for n in range(1, 600_000, 997):
        capacity = bucket(n, 8192)
        assert capacity >= n and capacity % 8192 == 0
        assert capacity >= previous  # monotonic in the length
        assert capacity - n < max(8192, 0.25 * n) + 1  # one step, or a quarter past 32k
        previous = capacity


def test_qsa_kv_growth_follows_the_ladder_instead_of_doubling():
    cache = language.QSAKVCache()
    keys = mx.zeros((1, 1, 70_000, 4), dtype=mx.float16)
    cache.update_and_fetch(keys, keys)
    assert cache.keys.shape[2] == 81_920  # doubling from 65,536 would be 131,072
    cache.update_and_fetch(keys[:, :, :12_000], keys[:, :, :12_000])
    assert cache.keys.shape[2] == cache_module.kv_capacity_bucket(82_000, 8192) == 98_304
    assert cache.state[0].shape[2] == 82_000


def _qsa_blocks(lengths, seed=7):
    mx.random.seed(seed)
    blocks = []
    start = 0
    for n in lengths:
        blocks.append(
            {
                "states": (
                    mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
                    mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
                    mx.random.normal((1, n, 4)).astype(mx.bfloat16),
                    mx.arange(start, start + n, dtype=mx.int32)[None, None],
                )
            }
        )
        start += n
    return blocks


@pytest.mark.parametrize("lengths", [(2048, 2048, 1500), (2048,) * 44 + (1906,)])
def test_prefix_restore_lands_in_ladder_capacity_with_exact_state(lengths):
    from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler

    handler = Qwen4QSAKVCacheHandler()
    blocks = _qsa_blocks(lengths)
    total = sum(lengths)
    cache = handler.reconstruct_cache(handler.concatenate_states(blocks))

    capacity = language.QSAKVCache.restore_capacity(total)
    assert cache.keys.shape[2] == cache.values.shape[2] == capacity > total
    assert cache._index_keys.shape[1] == capacity
    assert cache.offset == total and cache._index_offset == total
    expected = [mx.concatenate([b["states"][i] for b in blocks], axis=a) for i, a in ((0, 2), (1, 2), (2, 1))]
    keys, values, index_keys, positions = cache.state
    for got, want in zip((keys, values, index_keys), expected):
        assert got.shape == want.shape and mx.array_equal(got, want).item()
    assert positions.shape == (1, total)
    assert mx.array_equal(positions, mx.arange(total, dtype=mx.int32)[None]).item()

    # The suffix prefill appends into the restored buffers in place.
    backing, index_backing = cache.keys, cache._index_keys
    suffix = mx.ones((1, 2, 300, 8), dtype=mx.bfloat16)
    cache.update_and_fetch(suffix, suffix)
    cache.update_indexer(mx.ones((1, 300, 4), dtype=mx.bfloat16), mx.arange(total, total + 300, dtype=mx.int32)[None])
    assert cache.keys is backing and cache._index_keys is index_backing
    assert cache.state[0].shape[2] == total + 300
    assert mx.array_equal(cache.state[0][:, :, :total], expected[0]).item()


def test_consecutive_turn_restores_ask_for_one_buffer_size():
    from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler

    handler = Qwen4QSAKVCacheHandler()
    first = handler.reconstruct_cache(handler.concatenate_states(_qsa_blocks((2048,) * 43 + (1907,))))
    second = handler.reconstruct_cache(handler.concatenate_states(_qsa_blocks((2048,) * 44 + (337,))))
    assert first.offset != second.offset
    assert first.keys.shape == second.keys.shape
    assert first._index_keys.shape == second._index_keys.shape
