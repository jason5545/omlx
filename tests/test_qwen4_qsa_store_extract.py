# SPDX-License-Identifier: Apache-2.0
"""Request-end QSA extraction: views of the live cache, same SSD blocks.

``QSAKVCache.extract`` / ``BatchQSAKVCache.extract`` used to copy the whole
prefix (``mx.contiguous``, or a fresh ladder bank for the batch cache) before
the store sliced out the one or two new blocks. They now return views. These
tests run the old extraction (kept here as the reference) and the new one
through the real store path (``Scheduler._extract_cache_states``, the dispatch
``mx.eval``, ``BlockAwarePrefixCache.store_cache``, ``PagedSSDCacheManager``)
into separate cache directories and compare every block file byte for byte,
then restore the new blocks and compare them with the live cache.
"""

from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.scheduler import Scheduler, _BoundarySnapshotProvider

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")

BLOCK = 64
HEADS = 2
HEAD_DIM = 32
INDEX_DIM = 16
MODEL = "qsa-store-extract"


# --------------------------------------------------------------------------
# Live caches
# --------------------------------------------------------------------------


_MAX_TOKENS = 1024
_STREAMS: dict[tuple[int, int], tuple[mx.array, ...]] = {}


def _stream(seed: int, rows: int):
    """Per-position random K/V/index keys, so chunking does not change data."""
    if (seed, rows) not in _STREAMS:
        k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
        shape = (rows, HEADS, _MAX_TOKENS, HEAD_DIM)
        _STREAMS[(seed, rows)] = (
            mx.random.normal(shape, key=k1).astype(mx.bfloat16),
            mx.random.normal(shape, key=k2).astype(mx.bfloat16),
            mx.random.normal((rows, _MAX_TOKENS, INDEX_DIM), key=k3).astype(
                mx.bfloat16
            ),
        )
    return _STREAMS[(seed, rows)]


def _chunk(seed: int, rows: int, start: int, n: int):
    all_keys, all_values, all_index = _stream(seed, rows)
    keys = all_keys[..., start : start + n, :]
    values = all_values[..., start : start + n, :]
    index_keys = all_index[:, start : start + n]
    text = mx.arange(start, start + n, dtype=mx.int32)
    # Three MRoPE channels that differ, so a transposed or mixed-up channel
    # cannot pass as the right one.
    positions = mx.stack([text, text * 2 + 1, text * 3 + 2])[:, None, :]
    positions = mx.broadcast_to(positions, (3, rows, n))
    return keys, values, index_keys, positions


def _grow(cache, seed: int, start: int, stop: int, *, rows: int = 1):
    """Append ``[start, stop)`` like prefill chunks and decode steps do."""
    pos = start
    sizes = (37, 1, 1, 90, 3, 64, 1)
    i = 0
    while pos < stop:
        n = min(sizes[i % len(sizes)], stop - pos)
        keys, values, index_keys, positions = _chunk(seed, rows, pos, n)
        cache.update_and_fetch(keys, values)
        cache.update_indexer(index_keys, positions)
        pos += n
        i += 1
    if isinstance(cache, language.BatchQSAKVCache):
        mx.eval(cache.kv_cache.keys, cache.kv_cache.values)
        mx.eval(cache._index_keys, cache._index_position_ids)
    else:
        mx.eval(cache.keys, cache.values, cache._index_keys, cache._index_position_ids)


def _live_single(seed: int, tokens: int):
    cache = language.QSAKVCache()
    _grow(cache, seed, 0, tokens)
    # The live bank is wider than its tokens, so a row slice is strided.
    assert cache.keys.shape[2] > cache.offset
    return cache


# --------------------------------------------------------------------------
# The extraction before this change (reference)
# --------------------------------------------------------------------------


def _old_single_extract(cache, idx=0):
    out = language.QSAKVCache()
    out.keys = mx.contiguous(cache.keys[idx : idx + 1, :, : cache.offset, :])
    out.values = mx.contiguous(cache.values[idx : idx + 1, :, : cache.offset, :])
    out.offset = cache.offset
    out._geometric_capacity_managed = False
    index_keys = mx.contiguous(cache.index_keys[idx : idx + 1])
    positions = cache.index_position_ids
    if positions.ndim == 3:
        positions = mx.contiguous(positions[:, idx : idx + 1])
    else:
        positions = mx.contiguous(positions[idx : idx + 1])
    out._restore_indexer_state(index_keys, positions)
    return out


def _old_batch_extract(batch, idx):
    out = language.QSAKVCache()
    base = batch.kv_cache.extract(idx)  # ladder-bank copy of the row
    out.keys, out.values, out.offset = base.keys, base.values, base.offset
    out._geometric_capacity_managed = False
    padding = int(batch.left_padding[idx].item())
    out.index_keys = mx.contiguous(
        batch.index_keys[idx : idx + 1, padding : batch.index_offset]
    )
    out.index_position_ids = mx.contiguous(
        batch.index_position_ids[:, idx : idx + 1, padding : batch.index_offset]
    )
    return out


def _extract(path: str, cache, idx=0):
    if isinstance(cache, language.BatchQSAKVCache):
        return _old_batch_extract(cache, idx) if path == "old" else cache.extract(idx)
    return _old_single_extract(cache, idx) if path == "old" else cache.extract(idx)


# --------------------------------------------------------------------------
# Store pipeline (one per extraction path)
# --------------------------------------------------------------------------


_EXTRACTOR = SimpleNamespace(model_name=MODEL)


def _scheduler_extract(raw_cache):
    """``_extract_cache_states`` plus the dispatch-phase full eval."""
    extracted, config = Scheduler._extract_cache_states(_EXTRACTOR, raw_cache)
    assert extracted and len(extracted) == len(raw_cache)
    return extracted, config


def _dispatch_eval(cache_to_store, snapshots=None):
    arrays = Scheduler._collect_arrays_from_extracted_cache(cache_to_store)
    if snapshots is not None:
        for snapshot in snapshots.iter_in_memory_extracted():
            arrays.extend(Scheduler._collect_arrays_from_extracted_cache(snapshot))
    mx.eval(*arrays)


class _Pipeline:
    def __init__(self, root: Path, layer_types: list[str], *, split: bool = False):
        self.root = root
        self.layer_types = layer_types
        self.paged = PagedCacheManager(
            block_size=BLOCK,
            max_blocks=256,
            model_name=MODEL,
            initial_blocks=256,
        )
        self.ssd = PagedSSDCacheManager(
            cache_dir=root,
            max_size_bytes=512 * 1024**2,
            expected_model_name=MODEL,
            expected_num_layers=len(layer_types),
            expected_block_size=BLOCK,
            expected_layer_cache_types=layer_types,
            gdn_ssd_split_enabled=split,
        )
        self.boundary = (
            BoundarySnapshotSSDStore(root, pending_max_bytes=64 * 1024**2)
            if split
            else None
        )
        model = SimpleNamespace(layers=[MagicMock() for _ in layer_types])
        self.prefix = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=self.paged,
            paged_ssd_cache_manager=self.ssd,
            gdn_ssd_split_enabled=split,
        )
        if split:
            self.prefix.set_gdn_checkpoint_loader(self.boundary.load_file)

    def fetch(self, request_id: str, tokens: list[int]) -> int:
        table, _ = self.prefix.fetch_cache(request_id, tokens)
        return 0 if table is None else table.num_tokens

    def store(self, request_id, tokens, cache_to_store, config, snapshots=None, **kw):
        table = self.prefix.store_cache(
            request_id,
            tokens,
            cache_to_store,
            model_cache_config=config,
            boundary_snapshots=snapshots,
            **kw,
        )
        assert table is not None
        # The worker hands the request's blocks back for eviction afterwards.
        self.paged.release_for_eviction(table.block_ids)
        self.prefix.clear_request_entry(request_id)
        assert self.ssd.wait_for_pending_writes(30)
        return table

    def restore(self, request_id: str, tokens: list[int]):
        table, _ = self.prefix.fetch_cache(request_id, tokens)
        assert table is not None
        restored = self.prefix.reconstruct_cache(table)
        assert restored is not None
        return table, restored

    def close(self):
        if self.boundary is not None:
            self.boundary.shutdown()
        self.ssd.close()


def _read_safetensors(path: Path):
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    metadata = header.pop("__metadata__", {})
    metadata.pop("created_at", None)
    return header, metadata, raw[8 + header_len :]


def _block_files(root: Path) -> dict[str, Path]:
    """Committed block payloads and GDN sidecars, keyed by relative path.

    ``_boundary_snapshots/<pid>-<uuid>/`` is the request-local staging area of
    the boundary snapshot store, not part of the paged cache.
    """
    return {
        str(p.relative_to(root)): p
        for p in root.rglob("*.safetensors")
        if "_tmp_" not in p.name
        and p.relative_to(root).parts[0] != "_boundary_snapshots"
    }


def _assert_same_files(old_root: Path, new_root: Path) -> dict[str, Path]:
    old_files, new_files = _block_files(old_root), _block_files(new_root)
    assert old_files and sorted(old_files) == sorted(new_files)
    for name, old_path in old_files.items():
        old_header, old_meta, old_data = _read_safetensors(old_path)
        new_header, new_meta, new_data = _read_safetensors(new_files[name])
        assert new_header == old_header, name
        assert new_meta == old_meta, name
        assert new_data == old_data, name
    return new_files


def _qsa_tensor_names(header: dict, layer: int) -> list[str]:
    return sorted(name for name in header if name.startswith(f"layer_{layer}_"))


def _assert_restored_prefix(back, live, tokens: int):
    assert back.offset == tokens
    keys, values, index_keys, positions = back.state
    assert mx.array_equal(keys, live.keys[..., :tokens, :]).item()
    assert mx.array_equal(values, live.values[..., :tokens, :]).item()
    assert mx.array_equal(index_keys, live.index_keys[:, :tokens]).item()
    assert mx.array_equal(positions, live.index_position_ids[..., :tokens]).item()


# --------------------------------------------------------------------------
# Extraction itself
# --------------------------------------------------------------------------


def test_single_extract_returns_views_without_copying_the_prefix():
    live = _live_single(1, 700)
    mx.synchronize()
    before = mx.get_active_memory()
    row = live.extract(0)
    state = row.state
    mx.eval(*state)
    mx.synchronize()
    logical = sum(int(x.nbytes) for x in state)
    assert mx.get_active_memory() - before < logical // 100

    old = _old_single_extract(live)
    for new_x, old_x in zip(row.state, old.state):
        assert new_x.shape == old_x.shape and new_x.dtype == old_x.dtype
        assert mx.array_equal(new_x, old_x).item()
    assert row.offset == old.offset == 700
    assert row.index_keys.shape[1] == 700


def test_batch_extract_returns_padded_row_views_without_copying():
    first, second = _live_single(2, 300), _live_single(3, 517)
    batch = language.BatchQSAKVCache.merge([first, second])
    _grow(batch, 4, 517, 530, rows=2)  # decode with both rows on the ladder
    paddings = [int(p) for p in batch.left_padding.tolist()]
    assert paddings == [217, 0]

    mx.synchronize()
    before = mx.get_active_memory()
    rows = [batch.extract(0), batch.extract(1)]
    for row in rows:
        mx.eval(*row.state)
    mx.synchronize()
    logical = sum(int(x.nbytes) for row in rows for x in row.state)
    assert mx.get_active_memory() - before < logical // 100

    for idx, row in enumerate(rows):
        old = _old_batch_extract(batch, idx)
        assert row.offset == old.offset == 530 - paddings[idx]
        for new_x, old_x in zip(row.state, old.state):
            assert new_x.shape == old_x.shape and new_x.dtype == old_x.dtype
            assert mx.array_equal(new_x, old_x).item()


def test_views_survive_rewrites_of_the_live_bank_and_their_own_appends():
    """MLX copies instead of writing through a buffer a view still holds."""
    live = _live_single(5, 400)
    row = live.extract(0)
    expected = [mx.contiguous(x) for x in row.state]
    mx.eval(*row.state, *expected)

    # MTP rollback: trim the live cache and overwrite the same positions.
    live.trim(40)
    keys, values, index_keys, positions = _chunk(99, 1, 360, 40)
    live.update_and_fetch(keys, values)
    live.update_indexer(index_keys, positions)
    mx.eval(live.keys, live.values, live._index_keys, live._index_position_ids)
    assert not mx.array_equal(live.state[0][..., 360:, :], expected[0][..., 360:, :])

    for got, want in zip(row.state, expected):
        assert mx.array_equal(got, want).item()

    # The extracted cache can keep decoding; it grows into its own buffer.
    keys, values, index_keys, positions = _chunk(77, 1, 400, 3)
    row.update_and_fetch(keys, values)
    row.update_indexer(index_keys, positions)
    mx.eval(*row.state)
    assert row.offset == 403 and row.index_keys.shape[1] == 403
    assert mx.array_equal(row.state[0][..., :400, :], expected[0]).item()
    assert mx.array_equal(live.state[0][..., :360, :], expected[0][..., :360, :]).item()


# --------------------------------------------------------------------------
# Store: QSA-only model, two turns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "new_blocks"),
    [
        pytest.param(3 * BLOCK, 5 * BLOCK, 2, id="on-block-boundaries"),
        pytest.param(3 * BLOCK + 17, 4 * BLOCK + 40, 1, id="off-boundaries"),
        pytest.param(BLOCK + 5, 7 * BLOCK + 3, 6, id="many-blocks-in-one-store"),
    ],
)
def test_second_turn_writes_the_same_new_blocks(tmp_path, first, second, new_blocks):
    tokens = list(range(1000, 1000 + second))
    layer_types = ["QSAKVCache", "QSAKVCache"]
    pipes = {
        path: _Pipeline(tmp_path / path, layer_types) for path in ("old", "new")
    }
    try:
        for path, pipe in pipes.items():
            layers = [language.QSAKVCache(), language.QSAKVCache()]
            for seed, layer in enumerate(layers):
                _grow(layer, 10 + seed, 0, first)

            # Turn 1: nothing cached yet.
            assert pipe.fetch("turn-1", tokens[:first]) == 0
            data, config = _scheduler_extract([_extract(path, c) for c in layers])
            _dispatch_eval(data)
            pipe.store("turn-1", tokens[:first], data, config)
            files_after_first = set(_block_files(pipe.root))
            assert len(files_after_first) == first // BLOCK

            # Turn 2 hits turn 1's blocks; the cache grows past them.
            hit = pipe.fetch("turn-2", tokens)
            assert hit == (first // BLOCK) * BLOCK
            for seed, layer in enumerate(layers):
                _grow(layer, 10 + seed, first, second)
            data, config = _scheduler_extract([_extract(path, c) for c in layers])
            _dispatch_eval(data)
            pipe.store("turn-2", tokens, data, config)

            written = set(_block_files(pipe.root)) - files_after_first
            assert len(written) == new_blocks

        files = _assert_same_files(pipes["old"].root, pipes["new"].root)
        header, _, _ = _read_safetensors(next(iter(files.values())))
        # Every block carries K, V, index keys and positions for each layer.
        assert len(_qsa_tensor_names(header, 0)) == 4
        assert len(_qsa_tensor_names(header, 1)) == 4

        # Round trip: the new blocks restore to the live cache's prefix.
        layers = [language.QSAKVCache(), language.QSAKVCache()]
        for seed, layer in enumerate(layers):
            _grow(layer, 10 + seed, 0, second)
        table, restored = pipes["new"].restore("restore", tokens)
        stored = (second // BLOCK) * BLOCK
        assert table.num_tokens == stored
        for live, back in zip(layers, restored):
            assert isinstance(back, language.QSAKVCache)
            _assert_restored_prefix(back, live, stored)
    finally:
        for pipe in pipes.values():
            pipe.close()


def test_batch_rows_write_the_same_blocks(tmp_path):
    layer_types = ["QSAKVCache"]
    pipes = {
        path: _Pipeline(tmp_path / path, layer_types) for path in ("old", "new")
    }
    try:
        for path, pipe in pipes.items():
            first, second = _live_single(20, 2 * BLOCK + 9), _live_single(21, 4 * BLOCK)
            batch = language.BatchQSAKVCache.merge([first, second])
            _grow(batch, 22, 4 * BLOCK, 5 * BLOCK + 2, rows=2)
            lengths = [5 * BLOCK + 2 - int(p) for p in batch.left_padding.tolist()]
            for idx, length in enumerate(lengths):
                tokens = list(range(idx * 10_000, idx * 10_000 + length))
                data, config = _scheduler_extract([_extract(path, batch, idx)])
                _dispatch_eval(data)
                pipe.store(f"row-{idx}", tokens, data, config)
        _assert_same_files(pipes["old"].root, pipes["new"].root)
    finally:
        for pipe in pipes.values():
            pipe.close()


# --------------------------------------------------------------------------
# Store: hybrid model with boundary snapshots (the scheduler's override path)
# --------------------------------------------------------------------------


def _recurrent_layer(value: float):
    layer = ArraysCache(size=2)
    layer.cache[0] = mx.full((1, 3, 8), value)
    layer.cache[1] = mx.full((1, 2, 4, 8), value + 0.5)
    return layer


def _snapshot(token_count: int):
    """A boundary snapshot as captured: QSA layer skipped, recurrent kept."""
    extracted, _ = _scheduler_extract([_recurrent_layer(float(token_count))])
    placeholder = {
        "state": (),
        "meta_state": (),
        "class_name": "QSAKVCache",
        "cache_type": "QSAKVCache",
    }
    mx.eval(*Scheduler._collect_arrays_from_extracted_cache(extracted))
    return [extracted[0], placeholder]


def _store_turn(pipe, path, qsa, request_id, tokens, latest, tail, split):
    """The scheduler's boundary-override store for one finished turn."""
    boundaries = [tc for tc in range(BLOCK, latest + 1, BLOCK)]
    if tail:
        boundaries.append(latest)
    snapshots = {tc: _snapshot(tc) for tc in boundaries}
    full, config = _scheduler_extract(
        [_recurrent_layer(float(qsa.offset)), _extract(path, qsa)]
    )
    cache_to_store = Scheduler._merge_boundary_with_full_cache(snapshots[latest], full)
    assert cache_to_store[1] is full[1]  # QSA layer from the extracted cache
    intermediate = [tc for tc in boundaries if tc != latest]
    tail_count = latest if tail else None
    if split:
        for tc, snapshot in snapshots.items():
            assert pipe.boundary.save(
                request_id, tc, [MagicMock()], lambda _s, e=snapshot: (e, None)
            )
        provider = _BoundarySnapshotProvider(
            pipe.boundary,
            request_id,
            intermediate,
            {},
            paged_ssd_manager=pipe.ssd,
            tail_terminal_token_count=tail_count,
        )
    else:
        provider = _BoundarySnapshotProvider(
            None,
            request_id,
            intermediate,
            {tc: snapshots[tc] for tc in intermediate},
            tail_terminal_token_count=tail_count,
        )
    _dispatch_eval(cache_to_store, provider)
    before = set(_block_files(pipe.root))
    pipe.store(
        request_id,
        tokens[:latest],
        cache_to_store,
        config,
        provider,
        _store_tail_terminal=tail,
    )
    return set(_block_files(pipe.root)) - before


@pytest.mark.parametrize("split", [False, True], ids=["embedded-gdn", "split-gdn"])
@pytest.mark.parametrize(
    ("total", "latest", "tail"),
    [
        pytest.param(6 * BLOCK + 30, 6 * BLOCK, False, id="block-aligned"),
        pytest.param(6 * BLOCK + 30, 6 * BLOCK + 21, True, id="tail-terminal"),
    ],
)
def test_boundary_snapshot_store_writes_the_same_blocks(
    tmp_path, split, total, latest, tail
):
    layer_types = ["ArraysCache", "QSAKVCache"]
    first = 2 * BLOCK
    tokens = list(range(5000, 5000 + total))
    pipes = {
        path: _Pipeline(tmp_path / path, layer_types, split=split)
        for path in ("old", "new")
    }
    try:
        for path, pipe in pipes.items():
            # Turn 1 leaves two blocks; turn 2 hits them and goes on.
            qsa = language.QSAKVCache()
            _grow(qsa, 30, 0, first + 11)
            written = _store_turn(
                pipe, path, qsa, "turn-1", tokens, first, False, split
            )
            assert written
            assert pipe.fetch("turn-2", tokens) == first

            _grow(qsa, 30, first + 11, total)
            written = _store_turn(
                pipe, path, qsa, "turn-2", tokens, latest, tail, split
            )
            # Only the blocks past the hit: (latest - first) // BLOCK full
            # blocks, plus the tail block, plus a GDN sidecar per block when
            # recurrent state is split out.
            new_blocks = (latest - first) // BLOCK + (1 if tail else 0)
            assert len(written) == new_blocks * (2 if split else 1)

        _assert_same_files(pipes["old"].root, pipes["new"].root)

        # Round trip through the new blocks.
        qsa = language.QSAKVCache()
        _grow(qsa, 30, 0, total)
        table, restored = pipes["new"].restore("restore", tokens[:latest])
        assert table.num_tokens == latest
        _assert_restored_prefix(restored[1], qsa, latest)
        recurrent = restored[0].cache[0]
        assert float(recurrent[0, 0, 0]) == pytest.approx(float(latest))
    finally:
        for pipe in pipes.values():
            pipe.close()
