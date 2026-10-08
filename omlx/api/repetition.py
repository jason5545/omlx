# SPDX-License-Identifier: Apache-2.0
"""Detect a generation stuck repeating itself.

Low-bit quantized models sometimes fall into a loop: the same long span,
often a whole function or paragraph with small edits, written again and again
until ``max_tokens``. Speculative decoding makes it worse in wall time, since
the copy drafter predicts the repeats almost perfectly.

:class:`RepetitionDetector` watches the generated token ids. Over the last
``window`` tokens it counts the ``n``-grams that already occurred earlier in
the window; the share of such repeats approaches 1 in a loop. Long n-grams
make ordinary repetition (boilerplate code, tables, lists) invisible: they
repeat short spans, not 64-token ones.

Calibrated on two real GLM-5.3 oQ2e loops (both ran to 32768 tokens) and on
normal long outputs: cloud GLM-5.3 reasoning of 14-23k tokens with many code
drafts, local answers, test files full of patch() boilerplate, docs and a
600-row table. At n=64 over 8192 tokens the loops reach 0.80 and 0.98, the
normal outputs stay at or below 0.12 (2026-10-08). The 0.4 threshold fires
at token 9309 and 22250 of those two loops and on none of the normal ones.
"""

from __future__ import annotations

import os
from collections import Counter, deque

DEFAULT_NGRAM = 64
DEFAULT_WINDOW = 8192
DEFAULT_MIN_TOKENS = 2048
DEFAULT_THRESHOLD = 0.4
# Speculative decoding rewinds a few rejected draft tokens at a time.
_UNDO_DEPTH = 512


def loop_guard_enabled() -> bool:
    """``OMLX_LOOP_GUARD=0`` turns the guard off."""
    return os.environ.get("OMLX_LOOP_GUARD", "1").strip() != "0"


class RepetitionDetector:
    """Sliding-window share of repeated n-grams over generated token ids.

    ``feed(token_id)`` is O(n) per token. ``looping`` turns true once at least
    ``min_tokens`` tokens were fed and the share of repeated n-grams in the
    window reaches ``threshold``; it stays true until ``reset()``, which
    starts a new phase (e.g. the answer after the reasoning).

    ``snapshot()`` is O(1) and ``restore()`` undoes the tokens fed since, so a
    speculative decoder can checkpoint every draft position cheaply.
    """

    def __init__(
        self,
        n: int = DEFAULT_NGRAM,
        window: int = DEFAULT_WINDOW,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if n < 1 or window <= n:
            raise ValueError("window must be longer than the n-gram")
        self.n = n
        self.window = window
        self.min_tokens = min_tokens
        self.threshold = threshold
        self._epoch = 0
        self.reset()

    def reset(self) -> None:
        self._epoch += 1
        self._tokens: deque[int] = deque()
        self._grams: deque[int] = deque()
        self._counts: Counter[int] = Counter()
        # (token dropped from the n-token tail, n-gram added, n-gram evicted)
        self._undo: deque[tuple] = deque(maxlen=_UNDO_DEPTH)
        self._fed = 0
        self.looping = False

    @property
    def fed(self) -> int:
        return self._fed

    @property
    def ratio(self) -> float:
        """Share of n-grams in the window that repeat an earlier one."""
        total = len(self._grams)
        if not total:
            return 0.0
        return 1.0 - len(self._counts) / total

    def feed(self, token_id: int) -> bool:
        """Add one generated token; return whether the output is looping."""
        self._fed += 1
        self._tokens.append(int(token_id))
        dropped = self._tokens.popleft() if len(self._tokens) > self.n else None
        gram = evicted = None
        if len(self._tokens) == self.n:
            gram = hash(tuple(self._tokens))
            self._grams.append(gram)
            self._counts[gram] += 1
            # The window holds ``window - n + 1`` n-grams.
            if len(self._grams) > self.window - self.n + 1:
                evicted = self._grams.popleft()
                self._discount(evicted)
        self._undo.append((dropped, gram, evicted))
        if (
            not self.looping
            and self._fed >= self.min_tokens
            and self.ratio >= self.threshold
        ):
            self.looping = True
        return self.looping

    def may_trigger_within(self, n: int) -> bool:
        """Whether feeding ``n`` more tokens, whatever they are, could turn
        ``looping`` on.

        A feed adds at most one repeated n-gram (a new n-gram that matches
        one in the window; an evicted one only removes a repeat or a unique
        one) and never lowers the window's n-gram count, so after ``n``
        feeds the share is at most ``(repeats + n) / grams``. ``reset()``
        only lowers both counts and ``fed``. Exact enough to let a
        speculative decoder skip rewound calls; never says no when a
        trigger is possible.
        """
        if self.looping or n <= 0:
            return False
        if self._fed + n < self.min_tokens:
            return False
        total = len(self._grams)
        if not total:
            return True
        repeats = total - len(self._counts)
        # Margin against float rounding of ``ratio``.
        return (repeats + n) / total + 1e-9 >= self.threshold

    def _discount(self, gram: int) -> None:
        left = self._counts[gram] - 1
        if left:
            self._counts[gram] = left
        else:
            del self._counts[gram]

    def _undo_one(self) -> None:
        dropped, gram, evicted = self._undo.pop()
        self._tokens.pop()
        if dropped is not None:
            self._tokens.appendleft(dropped)
        if gram is not None:
            self._grams.pop()
            self._discount(gram)
        if evicted is not None:
            self._grams.appendleft(evicted)
            self._counts[evicted] += 1
        self._fed -= 1

    # -- speculative decoding rewind ---------------------------------------
    def snapshot(self) -> tuple:
        return (self._epoch, self._fed, self.looping)

    def restore(self, state: tuple) -> None:
        epoch, fed, looping = state
        back = self._fed - fed
        if epoch != self._epoch or back < 0 or back > len(self._undo):
            # Outside the undo history: start over. A loop is then noticed
            # later, never falsely.
            self.reset()
            self._epoch = epoch
            return
        for _ in range(back):
            self._undo_one()
        self.looping = looping


__all__ = [
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_NGRAM",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "RepetitionDetector",
    "loop_guard_enabled",
]
