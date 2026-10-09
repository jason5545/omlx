# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the DeepSeek V4 / glm5_next MoE block.

(omlx/patches/deepseek_v4/moe_offload.py)

The adapter swaps the module's projection tensors for resident slots and
runs the module's own forward on slot indices, so every path the resident
model takes (unsorted decode, sorted prefill through the native block/pair
kernels, the native weighted sum) is compared against the untouched module
on the same routes. Bit-exact where the kernel path and the per-row inputs
are identical; rounding-scale only where an over-capacity prefill
reassembles routes the model's fallback way.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.deepseek_v4 import moe_offload as dsv4
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    estimate_offload_admission_bytes,
    materialize_offload_state,
    moe_offload_stats,
)

# top-6 like DeepSeek V4 Flash: the native weighted-sum kernel accepts
# top-k 6 or 8 on half-precision activations, which is what the real model
# feeds it.
E, D, INTER, K, GROUP = 32, 64, 32, 6, 32
PREFIX = "model.layers.0.ffn.switch_mlp"


def _make_glu(seed=0, e=E, d=D, inter=INTER, group=GROUP):
    mx.random.seed(seed)
    glu = SwitchGLU(d, inter, e)
    # bf16 weights, as shipped: quantizing them yields bf16 scales and
    # biases, so bf16 activations stay bf16 through gather_qmm.
    for lin in glu.values():
        if isinstance(lin, nn.Module) and "weight" in lin:
            lin.weight = lin.weight.astype(mx.bfloat16)
    nn.quantize(glu, group_size=group, bits=4)
    mx.eval(glu.parameters())
    return glu


def _tensors(glu, prefix=PREFIX):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            out[f"{prefix}.{proj}.{field}"] = getattr(glu, proj)[field]
    return out


def _write(tmp_path, tensors, top_k=K, model_type="deepseek_v4"):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "num_experts_per_tok": top_k})
    )
    return tmp_path


class _FFN(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.ffn = _FFN(glu)


class _Inner(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.layers = [_Layer(g) for g in glus]


class _Model(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.model = _Inner(glus)


def _copy(glu):
    """A second module instance sharing no arrays with ``glu``."""
    twin = _make_glu(seed=99)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            setattr(twin[proj], field, mx.array(glu[proj][field]))
    mx.eval(twin.parameters())
    return twin


def _wrapped(tmp_path, reference, fraction):
    model = _Model([_copy(reference)])
    n = dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, fraction)
    assert n == 1
    return model.model.layers[0].ffn.switch_mlp


def _routes(shape, e=E, seed=1):
    mx.random.seed(seed)
    return mx.random.randint(0, e, shape)


def _x(*shape):
    return mx.random.normal(shape).astype(mx.bfloat16)


def _scores(indices):
    s = mx.random.uniform(shape=indices.shape)
    return s / s.sum(axis=-1, keepdims=True)


def _slow_reads(patch, seconds=0.02):
    """Slow every checkpoint read, into bytes or straight into a slot."""
    import time

    import omlx.patches.moe_expert_offload as meo

    read = meo.CheckpointExpertStore.read
    read_into = dsv4._pread_into

    def slow_read(plan):
        time.sleep(seconds)
        return read(plan)

    def slow_read_into(*args):
        time.sleep(seconds)
        return read_into(*args)

    patch.setattr(meo.CheckpointExpertStore, "read", staticmethod(slow_read))
    patch.setattr(dsv4, "_pread_into", slow_read_into)


@pytest.fixture(params=["native", "fallback"])
def kernels(request, monkeypatch):
    """With the native GLM/DSv4 kernels, or without them as on a CI runner:
    the module then returns sorted routes unsummed and the caller applies the
    scores, and the adapter must follow the same rule."""
    from omlx.custom_kernels.glm_moe_dsa import fast as k

    if request.param == "fallback":
        monkeypatch.setattr(k, "_ext", None)
        monkeypatch.setattr(k, "has_symbol", lambda name: False)
    elif not k.has_symbol("glm_moe_weighted_sum"):
        pytest.skip("native GLM kernels are not built here")
    return request.param


@pytest.fixture()
def reference(tmp_path):
    glu = _make_glu()
    _write(tmp_path, _tensors(glu))
    return glu


def test_wrap_replaces_module_and_keeps_only_slots(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)
    assert isinstance(wrapped, dsv4.OffloadedSwitchGLU)
    cache = wrapped.cache
    assert cache.capacity == 8 and cache.n_experts == E
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = cache.glu[proj]
        for field in ("weight", "scales", "biases"):
            assert lin[field].shape[0] == 8
            assert lin[field].shape[1:] == reference[proj][field].shape[1:]
    # num_experts is a property of the weight shape: it follows the slots
    assert cache.glu.gate_proj.num_experts == 8
    # the wrapper registers no parameters of its own: the slots live off-tree
    assert not wrapped.parameters()
    assert materialize_offload_state(_Model([wrapped])) == 1


def test_decode_bit_exact_at_quarter_residency(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)
    x = _x(4, 1, D)
    i = _routes((4, 1, K))
    ref, got = reference(x, i), wrapped(x, i)
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))
    assert moe_offload_stats(_Model([wrapped]))["misses"] == len(
        set(i.reshape(-1).tolist())
    )


def test_sorted_prefill_bit_exact_at_full_residency(tmp_path, reference, kernels):
    wrapped = _wrapped(tmp_path, reference, 1.0)
    x = _x(2, 40, D)
    i = _routes((2, 40, K))  # 160 routes: the sorted path
    for weighted in (False, True):
        s = _scores(i)
        ref = reference(x, i, scores=s, weighted_sum=weighted)
        got = wrapped(x, i, scores=s, weighted_sum=weighted)
        mx.eval(ref, got)
        assert ref.shape == got.shape
        assert ref.ndim == (3 if weighted and kernels == "native" else 4)
        assert bool(mx.array_equal(ref, got)), f"weighted_sum={weighted}"


def test_sorted_prefill_within_capacity_bit_exact(tmp_path, reference):
    """Routes that fit the cache take the module's own sorted path, keyed by
    slot instead of expert: every row still meets its own expert."""
    wrapped = _wrapped(tmp_path, reference, 0.5)  # 16 slots
    x = _x(1, 48, D)
    i = _routes((1, 48, K), e=12)  # 96 routes over 12 distinct experts
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=True)
    got = wrapped(x, i, scores=s, weighted_sum=True)
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))


@pytest.mark.parametrize("weighted", [False, True])
def test_over_capacity_prefill_rounding_bounded(tmp_path, reference, kernels, weighted):
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    x = _x(2, 64, D)
    i = _routes((2, 64, K))  # far more distinct experts than slots
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=weighted)
    got = wrapped(x, i, scores=s, weighted_sum=weighted)
    mx.eval(ref, got)
    assert ref.shape == got.shape
    assert ref.ndim == (3 if weighted and kernels == "native" else 4)
    assert float(mx.abs(ref - got).max()) < 2e-2
    # every distinct expert was installed exactly once for this call, and
    # each install read exactly one expert's worth of bytes
    assert wrapped.cache.misses == len(set(i.reshape(-1).tolist()))
    assert (
        wrapped.cache.fetched_bytes == wrapped.cache.misses * wrapped.cache.expert_bytes
    )


def test_least_used_eviction_and_counters(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    x = _x(1, 1, D)
    wrapped(x, mx.arange(8).reshape(1, 1, 8))
    assert (wrapped.cache.hits, wrapped.cache.misses) == (0, 8)
    for _ in range(2):
        wrapped(x, mx.array([[[0, 1]]]))
    # 0 and 1 are now the least recently used, but the most used
    wrapped(x, mx.arange(2, 8).reshape(1, 1, 6))
    assert (wrapped.cache.hits, wrapped.cache.misses) == (10, 8)
    wrapped(x, mx.array([[[8, 9]]]))  # evicts two of the least used, 2..7
    assert wrapped.cache.misses == 10 and len(wrapped.cache.slot_of) == 8
    assert 0 in wrapped.cache.slot_of and 1 in wrapped.cache.slot_of
    assert len(set(range(2, 8)) - set(wrapped.cache.slot_of)) == 2
    got = wrapped(x, mx.array([[[0, 9]]]))
    ref = reference(x, mx.array([[[0, 9]]]))
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref, got))


def test_slot_cache_decays_by_its_own_factor(tmp_path, reference, monkeypatch):
    """The slot caches decay their routing counts by _SLOT_SCORE_DECAY on
    every path that counts (decode read into the slots, decode through the
    installing path, streamed prefill); the common cache keeps 0.7."""
    import numpy as np

    import omlx.patches.moe_expert_offload as meo

    assert meo.ExpertCache.score_decay == meo._SCORE_DECAY == 0.7
    assert dsv4._SlotCache.score_decay == dsv4._SLOT_SCORE_DECAY == 0.97
    decay = np.float32(dsv4._SLOT_SCORE_DECAY)
    steps = _decode_steps(n=11)
    prefill = (_x(1, 64, D), _routes((1, 64, K), seed=5))
    assert len(set(prefill[1].reshape(-1).tolist())) > 24  # over capacity: streamed
    for into in (True, False):
        wrapped = _wrapped(tmp_path, reference, 0.75)  # 24 slots
        with monkeypatch.context() as patch:
            patch.setattr(dsv4, "_INTO_SLOT", into)
            for x, i in steps:
                mx.eval(wrapped(x, i))
            assert not wrapped.cache.free  # so the prefill fills no slot
            mx.eval(wrapped(*prefill))
        expected = np.zeros(E, dtype=np.float32)
        calls = [i.reshape(-1).tolist() for _, i in steps]
        calls.append(sorted(set(prefill[1].reshape(-1).tolist())))  # once per expert
        for n, ids in enumerate(calls, start=1):
            np.add.at(expected, np.asarray(ids, dtype=np.int64), np.float32(1.0))
            if n % meo._SCORE_DECAY_EVERY == 0:
                expected *= decay
        assert wrapped.cache.score.tolist() == expected.tolist()
        assert wrapped.cache.misses > 24


def test_slot_score_decay_override(monkeypatch):
    for raw, want in (("0.9", 0.9), ("1", 1.0), ("0", 0.97), ("1.5", 0.97), ("x", 0.97)):
        monkeypatch.setenv("OMLX_TEST_DECAY", raw)
        assert dsv4._env_decay("OMLX_TEST_DECAY", 0.97) == want
    monkeypatch.delenv("OMLX_TEST_DECAY")
    assert dsv4._env_decay("OMLX_TEST_DECAY", 0.97) == 0.97


def test_over_capacity_prefill_never_rereads_a_resident_expert(tmp_path, reference):
    """Resident experts run first, so the call's installs evict only experts
    it has already used. Sorting by expert id alone let the first chunk evict
    the high-id residents, and the last chunk read them back."""
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    warm = list(range(E - 8, E))  # the highest ids, as the id order hurts most
    wrapped(_x(1, 1, D), mx.array(warm).reshape(1, 1, 8))
    assert wrapped.cache.misses == 8
    x = _x(2, 64, D)
    i = _routes((2, 64, K))
    distinct = set(i.reshape(-1).tolist())
    assert set(warm) <= distinct  # the call needs every resident expert
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=True)
    got = wrapped(x, i, scores=s, weighted_sum=True)
    mx.eval(ref, got)
    assert float(mx.abs(ref - got).max()) < 2e-2
    assert wrapped.cache.misses == 8 + len(distinct - set(warm))


@pytest.mark.parametrize("group,ring", [(16, 4), (3, 2)])
def test_streamed_prefill_leaves_resident_experts_in_place(
    tmp_path, reference, kernels, monkeypatch, group, ring
):
    """An over-capacity prefill reads the experts the cache does not hold
    into temporary weights: the resident experts keep their slots, every
    non-resident expert is read once, and the output matches the module
    within the rounding the installing path already allows. Groups of 3 in
    a ring of 2 reuse each group's weights several times."""
    monkeypatch.setattr(dsv4, "_STREAM_GROUP", group)
    monkeypatch.setattr(dsv4, "_STREAM_RING", ring)
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    warm = list(range(E - 8, E))
    wrapped(_x(1, 1, D), mx.array(warm).reshape(1, 1, 8))
    c = wrapped.cache
    before = dict(c.slot_of)
    x = _x(2, 64, D)
    i = _routes((2, 64, K))
    distinct = set(i.reshape(-1).tolist())
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=True)
    got = wrapped(x, i, scores=s, weighted_sum=True)
    mx.eval(ref, got)
    assert ref.shape == got.shape
    assert float(mx.abs(ref - got).max()) < 2e-2
    assert c.slot_of == before  # nothing evicted, nothing moved
    assert c.misses == 8 + len(distinct - set(warm))
    assert c.fetched_bytes == c.misses * c.expert_bytes
    # the cache still serves its experts bit-exactly afterwards
    idx = mx.array(warm[:K]).reshape(1, 1, K)
    x1 = _x(1, 1, D)
    after_ref, after = reference(x1, idx), wrapped(x1, idx)
    mx.eval(after_ref, after)
    assert bool(mx.array_equal(after_ref, after))


def test_streamed_prefill_fills_a_cold_cache_with_its_most_routed_experts(
    tmp_path, reference
):
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 empty slots
    x = _x(1, 120, D)
    i = _routes((1, 120, K))
    flat = i.reshape(-1).tolist()
    counts = {e: flat.count(e) for e in set(flat)}
    top = sorted(counts, key=lambda e: (-counts[e], e))[:8]
    ref = reference(x, i)
    got = wrapped(x, i)
    mx.eval(ref, got)
    assert float(mx.abs(ref - got).max()) < 2e-2
    assert sorted(wrapped.cache.slot_of) == sorted(top)
    assert wrapped.cache.misses == len(counts)


def test_installing_prefill_stays_behind_the_switch(tmp_path, reference, monkeypatch):
    """OMLX_MOE_OFFLOAD_STREAM_PREFILL=0 keeps the installing path: the last
    chunk's experts end up resident."""
    monkeypatch.setattr(dsv4, "_STREAM", False)
    wrapped = _wrapped(tmp_path, reference, 0.25)
    warm = list(range(E - 8, E))
    wrapped(_x(1, 1, D), mx.array(warm).reshape(1, 1, 8))
    i = _routes((2, 64, K))
    mx.eval(wrapped(_x(2, 64, D), i))
    assert set(wrapped.cache.slot_of) != set(warm)


def test_decode_keeps_gpu_busy_while_reads_are_pending(
    tmp_path, reference, monkeypatch
):
    """A decode step whose reads are slow keeps submitting GPU work while it
    waits (the common adapter's keepalive); the output stays bit-identical,
    and with the overlap off nothing is submitted."""
    import omlx.patches.moe_expert_offload as meo

    x = _x(1, 1, D)
    warm = mx.arange(K).reshape(1, 1, K)
    idx = mx.array([[[0, 1, 2, 3, 20, 21]]])  # four hits, two misses
    ref = reference(x, idx)
    mx.eval(ref)

    def run(overlap):
        wrapped = _wrapped(tmp_path, reference, 0.25)
        wrapped._overlap = overlap
        mx.eval(wrapped(x, warm))
        submitted = []
        async_eval = mx.async_eval
        with monkeypatch.context() as patch:
            _slow_reads(patch)
            patch.setattr(
                meo.mx,
                "async_eval",
                lambda *a: (submitted.append(1), async_eval(*a))[1],
            )
            out = wrapped(x, idx)
            mx.eval(out)
        return out, len(submitted)

    on, n_on = run(True)
    off, n_off = run(False)
    assert bool(mx.array_equal(ref, on)) and bool(mx.array_equal(ref, off))
    assert n_on > 5  # pulses while the two experts are read
    assert n_off == 0


def test_decode_starts_the_routed_experts_it_returns(tmp_path, reference, monkeypatch):
    """A decode step hands its routed experts to the GPU before returning
    them, so the host builds the next layer while they run. Scheduling only:
    the bits stay the same, and with the overlap off nothing is started."""
    x = _x(1, 1, D)
    idx = mx.array([[[0, 1, 2, 3, 20, 21]]])
    ref = reference(x, idx)
    mx.eval(ref)

    def run(overlap):
        wrapped = _wrapped(tmp_path, reference, 0.25)
        wrapped._overlap = overlap
        assert wrapped.overlaps_decode(idx.size) == overlap
        started = []
        async_eval = mx.async_eval
        with monkeypatch.context() as patch:
            patch.setattr(
                dsv4.mx, "async_eval", lambda *a: (started.extend(a), async_eval(*a))[1]
            )
            out = wrapped(x, idx)
        mx.eval(out)
        return out, any(a is out for a in started)

    on, started_on = run(True)
    off, started_off = run(False)
    assert bool(mx.array_equal(ref, on)) and bool(mx.array_equal(ref, off))
    assert started_on and not started_off


def _decode_steps(n=24, seed=3):
    """Decode and verify blocks of 1-4 tokens over all experts."""
    mx.random.seed(seed)
    steps = []
    for length in ([1, 3, 2, 4] * n)[:n]:
        i = mx.random.randint(0, E, (1, length, K))
        steps.append((_x(1, length, D), i))
    return steps


def _cache_state(c):
    return (
        c.hits,
        c.misses,
        c.fetched_bytes,
        c.slot_expert.tolist(),
        sorted(c.slot_of.items()),
        sorted(c.free),
        c.map.tolist(),
        c.score.tolist(),
    )


def test_decode_reads_misses_into_their_slots(tmp_path, reference, monkeypatch):
    """Decode misses are read straight into the slots the installing path
    would have given them: no bytes turned into arrays, no MLX slot write,
    the same outputs bit for bit and the same cache state, with slow reads
    (the wait keeps the GPU clocked) or fast ones."""
    import omlx.patches.moe_expert_offload as meo

    steps = _decode_steps()
    refs = [reference(x, i) for x, i in steps]
    mx.eval(refs)
    to_mx = meo.CheckpointExpertStore.to_mx

    def run(into, slow):
        wrapped = _wrapped(tmp_path, reference, 0.75)  # 4-token blocks fit
        converted = []
        outs = []
        with monkeypatch.context() as patch:
            patch.setattr(dsv4, "_INTO_SLOT", into)
            patch.setattr(
                meo.CheckpointExpertStore,
                "to_mx",
                staticmethod(lambda *a: (converted.append(1), to_mx(*a))[1]),
            )
            if slow:
                _slow_reads(patch, 0.002)
            for x, i in steps:
                assert wrapped.overlaps_decode(i.size)
                out = wrapped(x, i)
                mx.eval(out)
                outs.append(out)
        return outs, _cache_state(wrapped.cache), len(converted)

    for slow in (False, True):
        into, state_into, converted_into = run(True, slow)
        installed, state_installed, converted_installed = run(False, slow)
        assert all(bool(mx.array_equal(r, o)) for r, o in zip(refs, into))
        assert all(bool(mx.array_equal(r, o)) for r, o in zip(refs, installed))
        assert state_into == state_installed
        assert state_into[1] > 10  # misses
        assert converted_into == 0 and converted_installed > 0


def test_decode_read_failure_gives_the_claimed_slots_back(
    tmp_path, reference, monkeypatch
):
    """A read that fails mid-call raises, waits out the other reads, and
    leaves no expert resident on a slot its bytes never reached: the misses
    after the failed one give their claimed slots back, the cache stays
    consistent, and the next call reads them again and matches the module."""
    x = _x(1, 2, D)
    idx = mx.array([[[0, 1, 2, 3, 20, 21], [4, 5, 0, 1, 22, 2]]])  # misses 20, 21, 22
    ref = reference(x, idx)
    mx.eval(ref)
    wrapped = _wrapped(tmp_path, reference, 0.5)
    c = wrapped.cache
    for first in (0, 6, 10):  # fill all 16 slots, so the misses evict
        mx.eval(wrapped(_x(1, 1, D), mx.arange(first, first + K).reshape(1, 1, K)))
    assert not c.free
    read_into = dsv4._pread_into
    bad = [plan for _, _, plan in c._plans(21)]

    def failing(store_view, plan, view):
        if plan in bad:
            raise OSError("injected")
        return read_into(store_view, plan, view)

    with monkeypatch.context() as patch:
        patch.setattr(dsv4, "_pread_into", failing)
        with pytest.raises(OSError, match="injected"):
            wrapped(x, idx)
    assert 20 in c.slot_of and {21, 22}.isdisjoint(c.slot_of)
    assert len(c.free) == 2
    assert len(c.slot_of) + len(c.free) == c.capacity
    for e, slot in c.slot_of.items():
        assert c.slot_expert[slot] == e
    for slot in c.free:
        assert c.slot_expert[slot] == -1
    mapped = c.map.tolist()
    assert all(mapped[e] == c.slot_of.get(e, -1) for e in range(E))
    out = wrapped(x, idx)
    assert bool(mx.array_equal(ref, out))


def test_route_trace_records_calls_and_changes_nothing(
    tmp_path, reference, monkeypatch
):
    """The routing trace records every call's routes, and the layer's cache
    state before its first call, without changing anything: the same calls
    with the trace off give the same bits, hits, misses and resident experts,
    and with ENABLE absent no file is written."""
    import numpy as np

    from omlx.patches.deepseek_v4 import route_trace

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    monkeypatch.setenv("OMLX_MOE_ROUTE_TRACE", str(trace_dir))
    calls = [
        (_x(1, 1, D), mx.arange(K).reshape(1, 1, K)),  # decode, cold cache
        (_x(1, 3, D), _routes((1, 3, K), e=8, seed=2)),  # verify, 3 rows
        (_x(2, 64, D), _routes((2, 64, K), seed=3)),  # over-capacity prefill
        (_x(1, 16, D), _routes((1, 16, K), e=6, seed=4)),  # in-capacity, sorted
        (_x(1, 1, D), mx.array([[[0, 1, 2, 3, 20, 21]]])),  # decode after it
    ]
    routes, major = route_trace.ROUTES, route_trace.EXPERT_MAJOR
    kinds = [routes, routes, major, routes, routes]

    def run(on):
        monkeypatch.setattr(route_trace, "_TRACER", route_trace._Tracer())
        enable = trace_dir / "ENABLE"
        if on:
            enable.write_text("x=1")
        elif enable.exists():
            enable.unlink()
        wrapped = _wrapped(tmp_path, reference, 0.25)
        wrapped._layer = 7
        outs = [wrapped(x, i) for x, i in calls]
        mx.eval(outs)
        route_trace._TRACER.close()
        c = wrapped.cache
        return outs, (c.hits, c.misses, sorted(c.slot_of), c.score.tolist())

    off_out, off_cache = run(False)
    assert not list(trace_dir.glob("routes-*.bin"))
    on_out, on_cache = run(True)
    assert on_cache == off_cache
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(off_out, on_out))

    (path,) = trace_dir.glob("routes-*.bin")
    header, recs = route_trace.read(path)
    assert header["x"] is True
    state, *rest = recs
    assert state["kind"] == route_trace.STATE and state["layer"] == 7
    assert len(state["slots"]) == 8 and (state["slots"] == -1).all()
    assert state["calls"] == 0 and not state["score"].any()
    assert [r["kind"] for r in rest] == kinds
    for r, (x, i) in zip(rest, calls):
        assert r["layer"] == 7 and r["k"] == K and r["rows"] == i.size // K
        assert r["ids"].tolist() == i.reshape(-1, K).tolist()
    # decode-sized inputs carry their bf16 bits; larger calls do not
    assert [r["x"] is not None for r in rest] == [True, True, False, False, True]
    for r, (x, _) in zip(rest, calls):
        if r["x"] is not None:
            bits = np.array(x.reshape(r["rows"], D).view(mx.uint16))
            assert (r["x"] == bits).all()


def test_uncovered_checkpoint_is_skipped(tmp_path):
    glu = _make_glu()
    tensors = _tensors(glu)
    tensors.pop(f"{PREFIX}.up_proj.scales")
    _write(tmp_path, tensors)
    model = _Model([_copy(glu)])
    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert isinstance(model.model.layers[0].ffn.switch_mlp, SwitchGLU)


def test_kill_switch(tmp_path, reference, monkeypatch):
    monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
    model = _Model([_copy(reference)])
    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 0


def test_common_entry_point_dispatches_dsv4(tmp_path, reference):
    """The engine calls apply_moe_expert_offload; DeepSeek V4 blocks are
    wrapped by their adapter, counted once, and seen by the shared walkers."""
    model = _Model([_copy(reference)])
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 1
    wrapped = model.model.layers[0].ffn.switch_mlp
    assert isinstance(wrapped, dsv4.OffloadedSwitchGLU)
    assert materialize_offload_state(model) == 1
    x = _x(1, 1, D)
    wrapped(x, mx.array([[[3, 5]]]))
    assert moe_offload_stats(model) == {
        "layers": 1,
        "hits": 0,
        "misses": 2,
        "hit_rate": 0.0,
        "fetched_bytes": 2 * wrapped.cache.expert_bytes,
    }


def test_compile_ffn_layers_stay_eager_when_offloaded(tmp_path, reference):
    """glm5_next decoder layers compile their FFN block at decode shapes
    (mlx_vlm language.py ``compile_ffn``). The offloaded block manages
    slots on the host and cannot be traced — ``tolist()`` inside
    ``mx.compile`` dies with "eval during function transformations". The
    wrap must turn that compilation off, and the layer must then run the
    offloaded block eagerly."""

    class _MoEHost(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.switch_mlp = glu

    class _CompilingLayer(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.mlp = _MoEHost(glu)
            self.compile_ffn = True
            self._ffn_c = None

        def __call__(self, x):
            if self.compile_ffn:
                if self._ffn_c is None:
                    self._ffn_c = mx.compile(self._ffn_block)
                return self._ffn_c(x)
            return self._ffn_block(x)

        def _ffn_block(self, x):
            return self.mlp.switch_mlp(x, mx.zeros((1, 1, K), dtype=mx.int32))

    glu = _copy(reference)
    layer = _CompilingLayer(glu)
    model = nn.Module()
    model.layers = [layer]
    prefix = "layers.0.mlp.switch_mlp"
    _write(tmp_path, _tensors(glu, prefix=prefix))

    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 1
    assert layer.compile_ffn is False
    x = _x(1, 1, D)
    got = layer(x)  # would raise the eval-during-trace error if compiled
    ref = reference(x, mx.zeros((1, 1, K), dtype=mx.int32))
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref, got))


def test_admission_estimate_counts_dsv4_experts(tmp_path):
    glu = _make_glu()
    tensors = _tensors(glu)
    tensors["model.embed_tokens.weight"] = mx.zeros((16, D), dtype=mx.float16)
    _write(tmp_path, tensors)
    expert_bytes = sum(
        v.size * v.dtype.size for k, v in tensors.items() if ".switch_mlp." in k
    )
    full = 10**9
    assert estimate_offload_admission_bytes(tmp_path, full, 0.25) == full - int(
        expert_bytes * 0.75
    )


class _MTPHead(nn.Module):
    """glm5_next's ``mtp.<i>.block.mlp.switch_mlp`` subtree shape: the
    draft head is a plain decoder layer, so its routed experts live under
    an ``mtp.`` path the offload wrap must be able to skip."""

    def __init__(self, glu):
        super().__init__()
        self.block = _FFN(glu)


def test_mtp_resident_keeps_draft_head_unwrapped(tmp_path, reference):
    # MTP armed: the head's experts stay fully resident (unwrapped) so
    # every draft step runs from RAM while the backbone streams. MTP off:
    # today's behavior — the head wraps like any other layer.
    backbone = _copy(reference)
    head = _make_glu(seed=7)
    tensors = _tensors(backbone)
    tensors.update(_tensors(head, prefix="mtp.0.block.switch_mlp"))
    _write(tmp_path, tensors)

    model = _Model([backbone])
    model.mtp = [_MTPHead(head)]
    n = dsv4.apply_deepseek_v4_moe_expert_offload(
        model, tmp_path, 0.5, mtp_resident=True
    )
    assert n == 1
    assert isinstance(model.model.layers[0].ffn.switch_mlp, dsv4.OffloadedSwitchGLU)
    assert isinstance(model.mtp[0].block.switch_mlp, SwitchGLU)
    assert not isinstance(
        model.mtp[0].block.switch_mlp, dsv4.OffloadedSwitchGLU
    )

    off = _Model([_copy(reference)])
    off.mtp = [_MTPHead(_copy(head))]
    n = dsv4.apply_deepseek_v4_moe_expert_offload(off, tmp_path, 0.5)
    assert n == 2
    assert isinstance(
        off.mtp[0].block.switch_mlp, dsv4.OffloadedSwitchGLU
    )


def test_admission_estimate_excludes_resident_draft_head(tmp_path):
    # The estimate must not promise savings on the draft head's slab while
    # the adapter keeps it resident, or admission overcommits and the load
    # OOMs. Checkpoint form: the real glm5_next layout stores the head as
    # ``language_model.mtp.<i>.*`` in its own shard.
    glu = _make_glu()
    tensors = _tensors(glu)
    head_prefix = "language_model.mtp.0.block.mlp.switch_mlp"
    tensors.update(_tensors(_make_glu(seed=7), prefix=head_prefix))
    _write(tmp_path, tensors)
    backbone_bytes = sum(
        v.size * v.dtype.size
        for k, v in tensors.items()
        if ".switch_mlp." in k and ".mtp." not in k
    )
    head_bytes = sum(
        v.size * v.dtype.size
        for k, v in tensors.items()
        if ".switch_mlp." in k and ".mtp." in k
    )
    assert head_bytes > 0
    full = 10**9
    assert estimate_offload_admission_bytes(
        tmp_path, full, 0.25, mtp_resident=True
    ) == full - int(backbone_bytes * 0.75)
    # Default (MTP off): the head's experts stream like any other layer.
    assert estimate_offload_admission_bytes(tmp_path, full, 0.25) == full - int(
        (backbone_bytes + head_bytes) * 0.75
    )


@pytest.mark.parametrize("workers", ["1", "4"])
def test_wrap_and_release_return_descriptors_to_baseline(
    tmp_path, reference, monkeypatch, workers
):
    import gc
    import os

    from omlx.patches.moe_expert_offload import _shutdown_io_pool

    monkeypatch.setenv("OMLX_MOE_OFFLOAD_IO_WORKERS", workers)
    _shutdown_io_pool()

    def cycle():
        wrapped = _wrapped(tmp_path, reference, 0.25)
        mx.eval(wrapped(_x(1, 1, D), mx.arange(K).reshape(1, 1, K)))

    try:
        cycle()
        gc.collect()
        baseline = len(os.listdir("/dev/fd"))
        for _ in range(10):
            cycle()
            gc.collect()
        assert len(os.listdir("/dev/fd")) == baseline
    finally:
        _shutdown_io_pool()
