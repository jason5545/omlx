# SPDX-License-Identifier: Apache-2.0
"""MTP chain cycles hand host-held token ids to the thinking processor.

(omlx/patches/mlx_lm_mtp/batch_generator.py ``_HostTokens``,
``_host_token_procs``; omlx/api/thinking.py ``idle_for``)

The verify rows and the first draft step give the processor ids the host
already holds instead of device reads, and the rest of a draft chain skips
the processor calls when it is provably idle (it is rewound after the chain
anyway). Both must leave every decision where it was: these tests run the
real ``_run_verify_cycle_chain`` and ``_chain_next_drafts`` over many cycles
that cross reminders, budget closes and loop verdicts, with the host ids on
and off, and compare every emitted token, draft and processor state.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.api.repetition import RepetitionDetector
from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.patches.mlx_lm_mtp import batch_generator as bg

VOCAB = 64
END, START, NL, EOS = 42, 41, 7, 2
PARTIAL, TAIL = 50, 51  # a UTF-8 lead byte pair and its continuation
WRAP = [60, 61]
NUDGE = [56, 57, 58]
PIECES = {NL: b"\n", PARTIAL: b"\xe4\xb8", TAIL: b"\xad", 52: b"."}
PROMPT = [1, 2, 3, START]


def _peaked(targets):
    rows = []
    for t in targets:
        row = [-50.0] * VOCAB
        row[t] = 0.0
        rows.append(row)
    return mx.array([rows], dtype=mx.float32)


class _Buffer:
    """A TokenBuffer stand-in that, like the real one, appends lazily."""

    def __init__(self, tokens):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, toks):
        toks = toks.astype(mx.int32).reshape(-1)
        self._buffer = mx.concatenate([self._buffer[: self._size], toks])
        self._size = int(self._buffer.shape[0])
        return self._buffer[: self._size]

    @property
    def tokens(self):
        return self._buffer[: self._size]


def _processor(seed):
    rng = random.Random(seed)
    return ThinkingBudgetProcessor(
        think_end_token_ids=[END],
        budget=rng.choice((None, 140, 200)),
        think_start_token_id=START,
        token_to_piece=lambda t: PIECES.get(t, b"x"),
        wrapup_token_ids=WRAP,
        boundary_grace=rng.choice((0, 3)),
        loop_detector=RepetitionDetector(n=6, window=48, min_tokens=24, threshold=0.4),
        stop_token_id=EOS,
        nudge_after=rng.choice((None, 60, 90)),
        nudge_token_ids=NUDGE,
        nudge_window=rng.choice((10, 25)),
    )


def _run(seed, host_tokens, monkeypatch, cycles=120, k=3):
    """``cycles`` real chain cycles; returns per-cycle observations."""
    monkeypatch.setattr(bg, "_PROCESSOR_HOST_TOKENS", host_tokens)
    rng = random.Random(seed)
    proc = _processor(seed)
    calls = {"n": 0}
    real_call = ThinkingBudgetProcessor.__call__

    def counting(self, tokens, logits):
        calls["n"] += 1
        return real_call(self, tokens, logits)

    monkeypatch.setattr(ThinkingBudgetProcessor, "__call__", counting)

    # The model: a loop-prone token stream with line ends, split UTF-8 and
    # the occasional close; the MTP head proposes it with some noise.
    loop = [rng.randrange(8, 40) for _ in range(9)]

    def model_token(pos):
        r = random.Random(seed * 100003 + pos).random()
        if pos > 80 and pos % 90 < 70:
            return loop[pos % len(loop)]
        if r < 0.06:
            return NL
        if r < 0.09:
            return PARTIAL
        if r < 0.095:
            return END
        if r < 0.10:
            return START
        return 8 + int(r * 1000) % 30

    head_calls = {"n": 0}

    def head_token(pos):
        head_calls["n"] += 1
        r = random.Random(seed * 7919 + pos * 31 + head_calls["n"] % 2).random()
        return model_token(pos) if r < 0.7 else 8 + int(r * 997) % 30

    buf = _Buffer(PROMPT)
    cache = SimpleNamespace(offset=len(PROMPT))
    emitted = list(PROMPT)

    def mtp_forward(hidden, ids, mtp_cache, return_hidden=False, logits_keep=None):
        n = int(ids.shape[1])
        pos = len(emitted) + head_state["chain"]
        head_state["chain"] += 1
        logits = _peaked([head_token(pos)] * n)
        hid = mx.zeros((1, n, 8), dtype=mx.float32)
        return (logits, hid) if return_hidden else logits

    head_state = {"chain": 0}
    model = SimpleNamespace(
        _omlx_mtp_commit_align=0,
        _omlx_mtp_head_prenorm=True,
        mtp_forward=mtp_forward,
    )
    batch = SimpleNamespace(
        model=model,
        prompt_cache=[cache],
        tokens=[emitted],
        samplers=[None],
        fallback_sampler=lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32),
        logits_processors=[[proc]],
        _token_context=[buf],
        max_tokens=[100000],
        _num_tokens=[len(PROMPT)],
        _matchers=[SimpleNamespace(advance=lambda token: False)],
    )

    def backbone(_model, inputs, _cache, **_kwargs):
        width = int(inputs.shape[1])
        base = len(emitted)
        targets = [model_token(base + j) for j in range(width)]
        cache.offset += width
        return _peaked(targets), mx.zeros((1, width, 8), dtype=mx.float32), None

    def rollback(_model, _cache, accepted, num_drafts, _gdn):
        cache.offset -= num_drafts - accepted
        return True

    monkeypatch.setattr(bg, "_call_backbone", backbone)
    monkeypatch.setattr(bg, "_chain_rollback", rollback)
    monkeypatch.setattr(bg, "_clear_rollback", lambda _cache: None)

    # Seed: the processor has seen the prompt; the first chain is drafted
    # from the prompt's last token.
    proc(buf.tokens, mx.zeros((1, VOCAB)))
    state = bg._MtpState(uid=1, chain=True, depth=k, mtp_cache=[])
    state.next_main = mx.array([model_token(len(emitted))], dtype=mx.uint32)
    buf.update_and_fetch(state.next_main)
    proc(buf.tokens, mx.zeros((1, VOCAB)))
    emitted.append(int(state.next_main.item()))
    bg._chain_next_drafts(
        batch,
        state,
        mx.zeros((1, 1, 8), dtype=mx.float32),
        state.next_main,
        buf.tokens[:-1],
        committed_ids=[emitted[-1]],
    )
    emitted.pop()
    buf._size -= 1

    seen = []
    for _ in range(cycles):
        head_state["chain"] = 0
        state.queue.clear()
        bg._run_verify_cycle_chain(batch, state)
        out = [t for t, _lp, _kind in state.queue]
        emitted.extend(out)
        batch._num_tokens[0] = len(emitted)
        snap = proc.snapshot_state()
        snap.pop("_loop_detector", None)
        seen.append(
            (
                out,
                state.drafts.tolist(),
                state.next_main.tolist(),
                snap,
                proc._loop_detector.fed,
                proc._loop_detector.looping,
            )
        )
    return seen, calls["n"], proc


@pytest.mark.parametrize("seed", range(8))
def test_host_ids_change_no_decision(seed, monkeypatch):
    off, off_calls, proc_off = _run(seed, False, monkeypatch)
    on, on_calls, proc_on = _run(seed, True, monkeypatch)
    for i, (a, b) in enumerate(zip(off, on)):
        assert a == b, f"cycle {i} differs"
    # The runs cross the processor's decisions (otherwise this proves little)
    # and the host path skips idle draft calls.
    assert on_calls < off_calls


def test_runs_cross_reminders_closes_and_loops(monkeypatch):
    kinds = set()
    for seed in range(8):
        seen, _, proc = _run(seed, True, monkeypatch)
        if proc._nudge_logged:
            kinds.add("nudge")
        if proc._loop_logged:
            kinds.add("loop")
        if any(END in out for out, *_ in seen):
            kinds.add("close")
    assert kinds == {"nudge", "loop", "close"}


def test_host_tokens_reads_host_positions_without_the_device():
    dev = mx.array([5, 6, 7, 8], dtype=mx.int32)
    view = bg._HostTokens(dev, 2, [70, 80])
    assert len(view) == 4
    assert view[2] == 70 and view[3] == 80 and view[-1] == 80
    assert int(view[1]) == 6  # outside the host tail: the device history
