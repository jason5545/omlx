# SPDX-License-Identifier: Apache-2.0
"""Loop guard and graceful thinking-budget close.

(omlx/api/repetition.py, omlx/api/thinking.py ThinkingBudgetProcessor)

A low-bit model can fall into writing the same long span again and again
until max_tokens. The detector flags that; the budget processor then closes a
repeating reasoning (with the same wrap-up note a budget close gets) or stops
a repeating answer. A budget close waits briefly for a line end and says it
is wrapping up, so the model does not keep drafting inside its answer.
"""

import itertools
import random
from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.api.repetition import RepetitionDetector

END, START, NL, EOS = 42, 41, 7, 2
WRAP = [60, 61]


def _block(rng, size, lo=1000, hi=50000):
    return [rng.randrange(lo, hi) for _ in range(size)]


# ---------------------------------------------------------------------------
# RepetitionDetector
# ---------------------------------------------------------------------------


def test_detector_flags_a_rewritten_block_but_not_varied_text():
    rng = random.Random(0)
    det = RepetitionDetector()
    loop = _block(rng, 700) * 30  # a 700-token answer written 30 times
    hits = [i for i, t in enumerate(loop, 1) if det.feed(t)]
    assert hits and hits[0] < len(loop) // 2

    det = RepetitionDetector()
    assert not any(det.feed(t) for t in _block(rng, 30000))

    # Boilerplate: the same line shape over and over, but every line differs
    # (a counter), so no 64-token span ever repeats.
    det = RepetitionDetector()
    lines = []
    for i in range(3000):
        lines += [11, 12, 13, 14, 15, 16, 2000 + i, NL]
    assert not any(det.feed(t) for t in lines)


def test_detector_waits_for_min_tokens():
    det = RepetitionDetector(n=8, window=256, min_tokens=500, threshold=0.4)
    seq = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 60
    flags = [det.feed(t) for t in seq]
    assert not any(flags[:499]) and flags[499]


def test_detector_restore_matches_a_replay():
    rng = random.Random(1)
    seq = _block(rng, 400) + _block(rng, 150) * 6 + _block(rng, 300)
    det = RepetitionDetector(n=16, window=300, min_tokens=50, threshold=0.4)
    for i, tok in enumerate(seq):
        if i % 37 == 0:
            snap = det.snapshot()
            back = rng.randrange(1, 20)
            for t in seq[i:i + back]:
                det.feed(t)
            det.restore(snap)
        det.feed(tok)
        replay = RepetitionDetector(n=16, window=300, min_tokens=50, threshold=0.4)
        if i % 97 == 0:
            for t in seq[: i + 1]:
                replay.feed(t)
            assert (det.fed, det.ratio, det.looping) == (
                replay.fed,
                replay.ratio,
                replay.looping,
            )


def test_detector_restore_past_its_history_starts_over():
    det = RepetitionDetector(n=8, window=64, min_tokens=16, threshold=0.4)
    snap = det.snapshot()
    for t in range(1000):
        det.feed(t % 9)
    assert det.looping
    det.restore(snap)  # 1000 tokens back: beyond the undo depth
    assert det.fed == 0 and not det.looping and det.ratio == 0.0


# ---------------------------------------------------------------------------
# ThinkingBudgetProcessor
# ---------------------------------------------------------------------------


def _forced(logits):
    finite = mx.isfinite(logits[0])
    if int(finite.sum().item()) == 1:
        return int(mx.argmax(logits[0]).item())
    return None


def _drive(proc, script, prompt=(1,), vocab=128, limit=5000):
    """Generate: take the processor's forced token when it forces one,
    otherwise the next scripted token (``script`` may be a function of the
    output so far). Stops at EOS or when the script ends."""
    hist = list(prompt)
    logits = proc(list(hist), mx.zeros((1, vocab)))
    out = []
    nxt = script if callable(script) else (lambda it: lambda _out: next(it, None))(
        iter(script)
    )
    while len(out) < limit:
        tok = _forced(logits)
        if tok is None:
            tok = nxt(out)
            if tok is None:
                break
        hist.append(tok)
        out.append(tok)
        if tok == EOS:
            break
        logits = proc(list(hist), mx.zeros((1, vocab)))
    return out


def _loop_until_closed(block, answer):
    """Repeat ``block`` while reasoning; once it is closed, write ``answer``."""
    loop, rest = itertools.cycle(block), iter(answer)
    return lambda out: next(rest, None) if END in out else next(loop)


def _proc(**kw):
    from omlx.api.thinking import ThinkingBudgetProcessor

    pieces = {NL: b"\n", 8: b" word.", EOS: b""}
    args = dict(
        think_end_token_ids=[END],
        budget=4,
        think_start_token_id=START,
        leading_token_ids=[NL],
        trailing_token_ids=[NL],
        token_to_piece=lambda t: pieces.get(t, b"w"),
        wrapup_token_ids=WRAP,
        boundary_grace=10,
    )
    args.update(kw)
    return ThinkingBudgetProcessor(**args)


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestGracefulBudgetClose:
    CLOSE = WRAP + [NL, END, NL]

    def test_waits_for_a_line_end_then_wraps_up(self):
        out = _drive(_proc(), [10, 11, 12, 13, 14, 15, NL, 16, 17, 18])
        # budget 4 reached at 13; 14 and 15 end nothing; the newline does
        assert out[:7] == [10, 11, 12, 13, 14, 15, NL]
        assert out[7:12] == self.CLOSE
        assert out[12:] == [16, 17, 18]

    def test_sentence_end_counts_as_a_boundary(self):
        out = _drive(_proc(), [10, 11, 12, 13, 14, 8, 16])
        assert out[:6] == [10, 11, 12, 13, 14, 8]
        assert out[6:11] == self.CLOSE

    def test_closes_when_the_grace_runs_out(self):
        out = _drive(_proc(boundary_grace=3), list(range(10, 30)))
        # budget 4 closes at the 4th position; three more go by, no boundary
        assert out[:6] == [10, 11, 12, 13, 14, 15]
        assert out[6:11] == self.CLOSE

    def test_no_grace_and_no_wrapup_is_the_old_close(self):
        out = _drive(
            _proc(boundary_grace=0, wrapup_token_ids=None), list(range(10, 20))
        )
        assert out[:6] == [10, 11, 12, NL, END, NL]


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestLoopGuard:
    def _det(self):
        return RepetitionDetector(n=8, window=64, min_tokens=32, threshold=0.4)

    def test_repeating_reasoning_is_closed_without_a_budget(self):
        proc = _proc(budget=None, loop_detector=self._det(), stop_token_id=EOS)
        block = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        out = _drive(proc, _loop_until_closed(block, range(100, 110)))
        close = WRAP + [NL, END, NL]
        at = next(i for i in range(len(out)) if out[i : i + len(close)] == close)
        assert 32 <= at < 80
        assert out[at + len(close) :] == list(range(100, 110))
        assert EOS not in out and not proc._loop_stop

    def test_repeating_answer_is_stopped(self):
        proc = _proc(
            budget=None,
            start_in_thinking=False,
            loop_detector=self._det(),
            stop_token_id=EOS,
        )
        loop = [30, 31, 32, 33, 34, 35, 36, 37, 38] * 50
        out = _drive(proc, loop)
        assert out[-1] == EOS and 32 <= len(out) < 80

    def test_answer_after_a_closed_loop_is_judged_on_its_own(self):
        # The reasoning's repeats must not count against the answer: a fresh
        # window, and an answer that repeats the reasoning's block once more
        # than min_tokens would allow on a carried-over window.
        proc = _proc(budget=None, loop_detector=self._det(), stop_token_id=EOS)
        block = [50, 51, 52, 53, 54, 55, 56]
        answer = block * 3 + list(range(200, 220))
        out = _drive(proc, _loop_until_closed(block, answer))
        assert out[-len(answer) :] == answer and EOS not in out

    def test_rewind_undoes_a_loop_verdict(self):
        proc = _proc(
            budget=None,
            start_in_thinking=False,
            loop_detector=self._det(),
            stop_token_id=EOS,
        )
        hist = [1]
        proc(list(hist), mx.zeros((1, 128)))
        loop = [30, 31, 32, 33, 34, 35, 36, 37, 38] * 6
        for tok in loop[:30]:
            hist.append(tok)
            proc(list(hist), mx.zeros((1, 128)))
        snap = proc.snapshot_state()
        for tok in loop[30:]:
            hist.append(tok)
            logits = proc(list(hist), mx.zeros((1, 128)))
        assert _forced(logits) == EOS
        proc.restore_state(snap)
        del hist[31:]
        logits = proc(list(hist), mx.zeros((1, 128)))
        assert _forced(logits) is None and not proc._loop_stop


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestSchedulerWiring:
    def _processors(self, monkeypatch, budget, guard="1", grammar=None):
        from omlx.adapter.output_parser import OutputParserFactory
        from omlx.api.thinking import ThinkingBudgetProcessor
        from omlx.request import Request, SamplingParams
        from omlx.scheduler import Scheduler

        monkeypatch.setenv("OMLX_LOOP_GUARD", guard)
        factory = OutputParserFactory(
            kind="legacy",
            create_session=MagicMock(),
            thinking_start_text="<think>",
            thinking_end_text="</think>",
        )
        ids = {"<think>": START, "</think>": END}
        encoded = {"<think>": [START], "</think>": [END]}
        s = MagicMock(spec=Scheduler)
        s._output_parser_factory = factory
        s._xtc_special_tokens = set()
        s._model_suppress_tokens = set()
        for name in (
            "_get_think_token_id",
            "_get_output_parser_thinking_end_text",
            "_encode_thinking_marker",
            "_token_piece_to_bytes",
            "_resolve_output_parser_thinking_trailing_ids",
            "_resolve_think_end_token_ids",
            "_build_sampler_and_processors",
            "_thinking_wrapup_token_ids",
            "_loop_guard_stop_token_id",
        ):
            setattr(s, name, getattr(Scheduler, name).__get__(s, Scheduler))
        s._resolve_think_close_pattern = MagicMock(return_value=(None, None))
        s._thinking_wrapup_cache = None
        tok = MagicMock()
        tok.encode.side_effect = lambda text, add_special_tokens=False: encoded.get(
            text, [70, 71]
        )
        tok.convert_tokens_to_ids.side_effect = ids.get
        tok.think_start_id, tok.think_end_id, tok.eos_token_id = START, END, EOS
        s.tokenizer = tok
        request = Request(
            request_id="loop-guard",
            prompt="test",
            sampling_params=SamplingParams(
                thinking_budget=budget, compiled_grammar=grammar
            ),
            prompt_token_ids=[1, START],
            num_prompt_tokens=2,
        )
        request.needs_think_prefix = True
        _, procs = s._build_sampler_and_processors(request.sampling_params, request)
        return [p for p in procs if isinstance(p, ThinkingBudgetProcessor)]

    def test_guard_attaches_without_a_budget(self, monkeypatch):
        (proc,) = self._processors(monkeypatch, budget=None)
        assert proc._budget is None and proc._loop_detector is not None
        assert proc._stop_token_id == EOS
        assert proc._force_sequence[:2] == [70, 71]  # the wrap-up note
        assert proc._boundary_grace == 0  # no budget, nothing to wait for

    def test_budget_gets_grace_and_guard(self, monkeypatch):
        (proc,) = self._processors(monkeypatch, budget=1024)
        assert proc._budget == 1024 and proc._boundary_grace == 64
        assert proc._loop_detector is not None

    def test_guard_off_leaves_the_budget_alone(self, monkeypatch):
        (proc,) = self._processors(monkeypatch, budget=1024, guard="0")
        assert proc._loop_detector is None and proc._stop_token_id is None

    def test_guard_off_without_a_budget_attaches_nothing(self, monkeypatch):
        assert self._processors(monkeypatch, budget=None, guard="0") == []

    def test_grammar_requests_skip_the_guard(self, monkeypatch):
        import omlx.api.grammar as grammar_mod

        monkeypatch.setattr(grammar_mod, "GrammarConstraintProcessor", MagicMock())
        grammar = MagicMock()
        grammar._omlx_has_thinking_phase = True
        grammar._omlx_thinking_phase_optional = False
        procs = self._processors(monkeypatch, budget=None, grammar=grammar)
        assert procs == []
