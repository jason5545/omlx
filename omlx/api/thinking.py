# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.
"""

import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .repetition import RepetitionDetector

logger = logging.getLogger(__name__)

# Token text that ends a line or a sentence: a budget close waits for one.
_BOUNDARY_ENDS = tuple(s.encode() for s in ("\n", ".", "!", "?", "。", "！", "？"))

# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8
_MINIMAX_OPEN_TAG = "<mm:think>"
_MINIMAX_CLOSE_TAG = "</mm:think>"
_HY3_OPEN_TAG = "<think:opensource>"
_HY3_CLOSE_TAG = "</think:opensource>"

# Regex for non-streaming extraction (complete text)
_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)


def _safe_tokenizer_attr(tokenizer, attr: str, default=None):
    if tokenizer is None:
        return default
    try:
        return getattr(tokenizer, attr, default)
    except (AttributeError, TypeError, ValueError):
        return default


def _single_token_id(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _convert_token_to_id(tokenizer, token: str) -> int | None:
    convert = _safe_tokenizer_attr(tokenizer, "convert_tokens_to_ids")
    if not callable(convert):
        return None
    try:
        token_id = convert(token)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if token_id == _safe_tokenizer_attr(tokenizer, "unk_token_id"):
        return None
    return _single_token_id(token_id)


def _encode_prompt_ids(tokenizer, prompt: str) -> list[int] | None:
    encode = _safe_tokenizer_attr(tokenizer, "encode")
    if not callable(encode):
        return None
    try:
        return list(encode(prompt, add_special_tokens=False))
    except TypeError:
        try:
            return list(encode(prompt))
        except Exception:
            return None
    except Exception:
        return None


def _think_end_token_ids(tokenizer) -> list[int] | None:
    think_end_id = _single_token_id(_safe_tokenizer_attr(tokenizer, "think_end_id"))
    if think_end_id is not None:
        return [think_end_id]

    think_end_tag = _safe_tokenizer_attr(tokenizer, "think_end", _CLOSE_TAG)
    encoded = _encode_prompt_ids(tokenizer, think_end_tag or _CLOSE_TAG)
    if encoded:
        return encoded

    token_id = _convert_token_to_id(tokenizer, _CLOSE_TAG)
    if token_id is not None:
        return [token_id]
    return None


def prompt_opens_thinking(
    tokenizer,
    prompt: str,
    prompt_token_ids: Sequence[int] | None = None,
) -> tuple[bool, str]:
    """Return whether a raw prompt would make the engine prepend ``<think>``.

    Presentation-layer stripping must mirror the engine/scheduler decision, not
    just the raw text suffix. Some prompts can contain a literal ``<think>``
    without tokenizing to the model's think-start id, and templates can leave
    the think-start token in the final token tail without the raw string ending
    in the visible tag. When the caller already has prompt ids from the same
    tokenizer path as the scheduler, those ids are authoritative.
    """
    think_tag = (
        _safe_tokenizer_attr(tokenizer, "think_start", _OPEN_TAG) or _OPEN_TAG
    )
    if tokenizer is None:
        return prompt.rstrip().endswith(think_tag), think_tag

    think_start_id = _single_token_id(
        _safe_tokenizer_attr(tokenizer, "think_start_id")
    )
    if think_start_id is None:
        think_start_id = _convert_token_to_id(tokenizer, think_tag)
    if think_start_id is None:
        return False, think_tag

    if prompt_token_ids is None:
        prompt_ids = _encode_prompt_ids(tokenizer, prompt)
    else:
        prompt_ids = list(prompt_token_ids)
    if not prompt_ids or not think_start_id:
        return False, think_tag

    last_tokens = list(prompt_ids[-3:])
    if think_start_id not in last_tokens:
        return False, think_tag

    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1 :]

    if after_start:
        think_end_ids = _think_end_token_ids(tokenizer)
        if think_end_ids and think_end_ids[0] in after_start:
            return False, think_tag

    return True, think_tag


def extract_thinking(text: str, *, truncated: bool = False) -> Tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: ``<think>reasoning</think>answer`` → ``("reasoning", "answer")``
    - No thinking: ``just answer`` → ``("", "just answer")``
    - Partial (no open tag): ``reasoning</think>answer`` → ``("reasoning", "answer")``
    - Empty think: ``<think></think>answer`` → ``("", "answer")``
    - Think only: ``<think>reasoning</think>`` → ``("reasoning", "")``
    - Malformed (open with no close): ``<think>everything…`` →
      ``("", "everything…")`` — recovery for V4-style models that
      occasionally skip the ``</think>`` boundary token. Without this
      fallback the entire body would be classified as thinking and the
      visible answer would be empty.

    With ``truncated=True``, unfinished thinking stays in the thinking channel.

    Tag-free text is always classified as content. Mirrors
    ``ThinkingParser.finish()`` recovery semantics (`_content_emitted`
    fallback): when the model emits no thinking markers, surface the body
    as the answer so the response is never empty.

    Args:
        text: Complete model output text.
        truncated: Keep unfinished thinking in its channel on length termination.

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    text = (
        text.replace(_MINIMAX_OPEN_TAG, _OPEN_TAG)
        .replace(_MINIMAX_CLOSE_TAG, _CLOSE_TAG)
        .replace(_HY3_OPEN_TAG, _OPEN_TAG)
        .replace(_HY3_CLOSE_TAG, _CLOSE_TAG)
    )

    thinking_parts = []
    remaining = text

    # Extract all <think>...</think> blocks
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if truncated and _OPEN_TAG in remaining:
        before, after = remaining.split(_OPEN_TAG, 1)
        thinking_parts.append(after)
        remaining = before

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    # Malformed: <think> opened but never closed. Drop the open tag and
    # treat the remainder as content so the answer body is not empty.
    if '<think>' in text and '</think>' not in text:
        idx = text.index('<think>')
        before = text[:idx]
        after = text[idx + _OPEN_LEN:]
        return ("", (before + after).strip())

    return ("", text)


class ThinkingParser:
    """Stateful streaming parser for separating <think>...</think> from content.

    Handles streaming chunks where tags may span multiple chunks.
    Returns (thinking_delta, content_delta) tuples for each feed() call.

    Example::

        parser = ThinkingParser()

        # Chunk 1: "<think>Let me"
        t, c = parser.feed("<think>Let me")
        # t = "Let me", c = ""

        # Chunk 2: " think</think>Answer"
        t, c = parser.feed(" think</think>Answer")
        # t = " think", c = "Answer"

        # Flush remaining
        t, c = parser.finish()
    """

    def __init__(self, start_in_thinking: bool = False):
        self._in_thinking: bool = start_in_thinking
        self._buffer: str = ""  # Buffer for potential partial tags
        # Recovery state for malformed thinking: when the prompt prepends
        # ``<think>`` and the model never emits ``</think>`` before EOS,
        # everything we streamed went out as thinking. The streamed events
        # cannot be retracted, so finish() emits the accumulated thinking
        # text once more as content — the client will show both panels but
        # the answer body is no longer empty.
        self._close_seen: bool = False
        self._thinking_accumulated: List[str] = []
        self._content_emitted: bool = False

    def feed(self, text: str) -> Tuple[str, str]:
        """Feed a text chunk, return (thinking_delta, content_delta).

        Args:
            text: New text chunk from model output.

        Returns:
            Tuple of (thinking_text, content_text) extracted from this chunk.
        """
        if not text:
            return ("", "")

        # Prepend any buffered partial tag content
        text = self._buffer + text
        self._buffer = ""

        thinking_out = []
        content_out = []

        i = 0
        while i < len(text):
            if text[i] == '<':
                # Check if this could be a tag start
                remaining = text[i:]

                # Try to match <think>
                if remaining.startswith(_OPEN_TAG):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue

                if remaining.startswith(_HY3_OPEN_TAG):
                    self._in_thinking = True
                    i += len(_HY3_OPEN_TAG)
                    continue

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    self._in_thinking = False
                    self._close_seen = True
                    i += _CLOSE_LEN
                    continue

                if remaining.startswith(_HY3_CLOSE_TAG):
                    self._in_thinking = False
                    self._close_seen = True
                    i += len(_HY3_CLOSE_TAG)
                    continue

                # Check if it could be a partial tag (not enough chars yet)
                if self._could_be_tag(remaining):
                    # Buffer the rest and wait for more data
                    self._buffer = remaining
                    break

                # Not a tag, emit the '<' as regular content
                if self._in_thinking:
                    thinking_out.append('<')
                else:
                    content_out.append('<')
                i += 1
            else:
                if self._in_thinking:
                    thinking_out.append(text[i])
                else:
                    content_out.append(text[i])
                i += 1

        thinking_delta = "".join(thinking_out)
        content_delta = "".join(content_out)
        if thinking_delta:
            self._thinking_accumulated.append(thinking_delta)
        if content_delta:
            self._content_emitted = True
        return (thinking_delta, content_delta)

    def finish(self, *, truncated: bool = False) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.
        Unless truncated, recover an unclosed thinking block as content when no answer was emitted.

        Returns:
            Tuple of (thinking_text, content_text) from remaining buffer
            (plus recovered content if applicable).
        """
        partial = self._buffer
        self._buffer = ""

        # Recovery: prompt opened a thinking block (or model echoed
        # ``<think>`` itself), the close tag never arrived, and nothing
        # ever streamed as content. Re-emit the accumulated thinking text
        # as content so the answer body is not empty. The thinking events
        # already streamed live cannot be retracted, so the client sees
        # the same text twice — once in the thinking panel, once as the
        # answer. UX trade-off documented in the chat template plan.
        if (
            not truncated
            and self._in_thinking
            and not self._close_seen
            and not self._content_emitted
            and self._thinking_accumulated
        ):
            recovered = "".join(self._thinking_accumulated) + partial
            self._content_emitted = True
            return ("", recovered)

        if not partial:
            return ("", "")

        # Partial tag never completed — emit it as-is in the current mode.
        if self._in_thinking:
            self._thinking_accumulated.append(partial)
            return (partial, "")
        else:
            self._content_emitted = True
            return ("", partial)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        """Check if text could be the start of a <think> or </think> tag.

        Returns True if text is a proper prefix of either tag but not
        yet a complete match.
        """
        length = len(text)
        if length >= len(_HY3_CLOSE_TAG):
            # Long enough to determine - not a partial tag
            return False

        # Check against all recognised tags
        for tag in (_OPEN_TAG, _CLOSE_TAG, _HY3_OPEN_TAG, _HY3_CLOSE_TAG):
            if length < len(tag) and tag[:length] == text:
                return True

        return False


class ThinkingBudgetProcessor:
    """Logits processor that enforces a thinking token budget.

    Counts tokens generated while in thinking mode.  When the budget is
    exceeded, forces the close-think token(s) one at a time, then becomes
    a no-op for the rest of generation.

    Handles both single-token and multi-token close-think sequences, and
    supports alternative think markers (e.g. ``<longcat_think>``).

    Two optional behaviors make a forced close read as a decision instead of
    a cut. With ``boundary_grace`` the close waits, at most that many tokens
    past the budget, for a token ending a line or sentence. With
    ``wrapup_token_ids`` a short note precedes the close telling the model to
    answer from the reasoning so far; without it a model cut mid-thought
    keeps drafting inside the answer.

    With ``loop_detector`` the processor also stops a generation repeating
    itself: in the reasoning it closes the reasoning the same way (with or
    without a budget), in the answer it forces ``stop_token_id``.

    With ``nudge_after`` a reasoning that runs long without repeating gets a
    reminder instead of a cut: past that many tokens (at a line or sentence
    end, within ``boundary_grace``) ``nudge_token_ids`` are forced and the
    model keeps reasoning, free to close on its own. If it has not closed
    ``nudge_window`` tokens later, the reasoning is closed like a spent
    budget.

    Args:
        think_end_token_ids: Token ID(s) for the close-think tag.
        budget: Maximum number of thinking tokens before forcing close, or
            ``None`` for no budget (loop guard only).
        think_start_token_id: Token ID for the open-think tag (re-entry detection).
        start_in_thinking: False when the model, not the prompt, opens
            thinking. Counting then starts at ``think_start_token_id``.
        wrapup_token_ids: Tokens forced before the close sequence.
        boundary_grace: Tokens a budget close may wait for a line or sentence
            end (0 closes at the budget, as before).
        loop_detector: Detector fed with every generated token of each phase.
        stop_token_id: Forced when the answer loops (an EOS id).
        label: Request id for log lines.
        nudge_after: Reasoning tokens before the reminder, or ``None``.
        nudge_token_ids: The reminder, forced inside the reasoning.
        nudge_window: Tokens the model gets after the reminder to close the
            reasoning itself.
    """

    def __init__(
        self,
        think_end_token_ids: List[int],
        budget: Optional[int],
        think_start_token_id: Optional[int] = None,
        leading_token_ids: Optional[List[int]] = None,
        trailing_token_ids: Optional[List[int]] = None,
        token_to_piece: Optional[Callable[[int], str | bytes | None]] = None,
        start_in_thinking: bool = True,
        wrapup_token_ids: Optional[List[int]] = None,
        boundary_grace: int = 0,
        loop_detector: Optional["RepetitionDetector"] = None,
        stop_token_id: Optional[int] = None,
        label: Optional[str] = None,
        nudge_after: Optional[int] = None,
        nudge_token_ids: Optional[List[int]] = None,
        nudge_window: int = 0,
    ):
        self._think_end_ids = think_end_token_ids
        # Full force sequence: [wrap-up] + \n + </think> + \n\n (the part from
        # \n on matches the training pattern)
        self._force_sequence = (
            list(wrapup_token_ids or [])
            + (leading_token_ids or [])
            + list(think_end_token_ids)
            + (trailing_token_ids or [])
        )
        self._budget = budget
        self._think_start_id = think_start_token_id
        self._token_to_piece = token_to_piece
        self._boundary_grace = max(0, int(boundary_grace))
        self._loop_detector = loop_detector
        self._stop_token_id = stop_token_id
        self._label = label or ""
        self._loop_logged = False  # not rewound: one log line per request
        self._nudge_sequence = list(nudge_token_ids or [])
        self._nudge_after = (
            max(0, int(nudge_after))
            if nudge_after is not None and self._nudge_sequence
            else None
        )
        self._nudge_window = max(0, int(nudge_window))
        self._nudge_logged = False  # not rewound, like _loop_logged
        self._nudge_outcome_logged = False

        # State
        self._thinking_tokens: int = 0
        self._in_thinking: bool = start_in_thinking
        self._forcing: bool = False
        self._waiting_utf8: bool = False
        self._force_idx: int = 0
        self._done: bool = False
        self._first_call: bool = True
        # Sliding window for multi-token end detection
        self._recent_tokens: List[int] = []
        self._last_token_utf8_complete: bool = True
        self._pending_utf8: bytes = b""
        # Tokens generated past the budget while waiting for a boundary.
        self._grace_count: int = 0
        self._at_boundary: bool = False
        # Loop guard verdicts: close the reasoning / stop the answer.
        self._close_requested: bool = False
        self._loop_stop: bool = False
        # What ``_forcing`` forces: "close" or "nudge" (the reminder).
        self._force_kind: str = "close"
        # Reasoning tokens counted when the reminder ended.
        self._nudged_at: Optional[int] = None

    def __call__(self, tokens, logits):
        """mlx-lm logits processor: (tokens, logits) -> logits."""
        import mlx.core as mx

        # In new mlx-lm API, tokens is the full history list.
        # Accept each genuinely generated token exactly once (see grammar.py).
        n = len(tokens)
        if not hasattr(self, "_accepted_up_to"):
            self._accepted_up_to = n  # skip prompt tokens
        elif n > self._accepted_up_to:
            for i in range(self._accepted_up_to, n):
                self._update_state(int(tokens[i]))
            self._accepted_up_to = n

        if self._loop_stop:
            return self._force_stop(logits, mx)

        # If state changed by _update_state, handle immediately
        if self._done:
            return logits

        if self._forcing:
            return self._force_next_token(logits, mx)

        if self._waiting_utf8:
            return logits

        if self._in_thinking:
            self._thinking_tokens += 1
            if self._close_due():
                return self._start_forcing("close", logits, mx)
            if self._nudge_due():
                self._log_nudge_start()
                return self._start_forcing("nudge", logits, mx)

        return logits

    # The MTP loop may pass a token history whose newest entries it already
    # holds on the host (batch_generator._HostTokens): this processor only
    # takes len() and int() of single positions, so no device read is needed.
    host_token_view = True

    def idle_for(self, n: int) -> bool:
        """Whether the next ``n`` calls, each adding one token of any value,
        return the logits unchanged.

        A speculative drafter that rewinds this processor after its chain
        may then skip those calls, and the device reads of their tokens.
        A call changes the logits only while a sequence is forced (or about
        to be, waiting for a UTF-8 boundary), once the answer loops, or when
        a close or a reminder becomes due. Each is bounded from the current
        state: the reasoning count grows by at most ``n`` (a reasoning that
        reopens restarts from zero, at most ``n`` again), and the loop guard
        gains at most ``n`` repeated n-grams.
        """
        if n <= 0:
            return True
        if not hasattr(self, "_accepted_up_to"):
            return False
        if self._loop_stop or self._forcing or self._waiting_utf8 or self._close_requested:
            return False
        reach = self._thinking_tokens + n
        if self._budget is not None and reach >= self._budget:
            return False
        if (
            self._nudged_at is not None
            and reach - self._nudged_at > self._nudge_window
        ):
            return False
        if self._nudge_after is not None and reach >= self._nudge_after:
            # Due without a reminder yet, or after a reopened reasoning.
            if self._nudged_at is None or n >= self._nudge_after:
                return False
        det = self._loop_detector
        return det is None or det.looping or not det.may_trigger_within(n)

    def _start_forcing(self, kind: str, logits, mx):
        self._force_kind = kind
        self._recent_tokens = []
        if self._last_token_utf8_complete:
            self._forcing = True
            self._force_idx = 0
            return self._force_next_token(logits, mx)
        self._waiting_utf8 = True
        return logits

    def _close_due(self) -> bool:
        """Whether the reasoning should be closed at this position."""
        if self._close_requested:
            return True
        budget_spent = self._budget is not None and self._thinking_tokens >= self._budget
        nudge_ignored = (
            self._nudged_at is not None
            and self._thinking_tokens - self._nudged_at > self._nudge_window
        )
        if not (budget_spent or nudge_ignored):
            return False
        if nudge_ignored and not budget_spent and not self._nudge_outcome_logged:
            self._nudge_outcome_logged = True
            logger.info(
                "Thinking nudge%s: still reasoning %d tokens after the reminder; "
                "closing the reasoning",
                self._label_suffix(),
                self._nudge_window,
            )
        return self._at_line_end()

    def _nudge_due(self) -> bool:
        """Whether the reminder should start at this position."""
        if (
            self._nudge_after is None
            or self._nudged_at is not None
            or self._thinking_tokens < self._nudge_after
        ):
            return False
        return self._at_line_end()

    def _at_line_end(self) -> bool:
        """Past a limit: act at the first line or sentence end, or once the
        grace runs out."""
        if not self._boundary_grace:
            return True
        self._grace_count += 1
        return self._at_boundary or self._grace_count > self._boundary_grace

    def _label_suffix(self) -> str:
        return f" [{self._label}]" if self._label else ""

    def _log_nudge_start(self) -> None:
        if self._nudge_logged:
            return
        self._nudge_logged = True
        logger.info(
            "Thinking nudge%s: reasoning reached %d tokens; reminding the model "
            "to wrap up (closed if still reasoning %d tokens later)",
            self._label_suffix(),
            self._thinking_tokens,
            self._nudge_window,
        )

    def _update_state(self, token_id: int) -> None:
        """Update thinking state based on the last generated token."""
        self._last_token_utf8_complete = self._is_utf8_complete(token_id)
        if self._boundary_grace:
            self._at_boundary = self._is_boundary(token_id)

        if self._loop_stop:
            return

        if self._done:
            if self._think_start_id and token_id == self._think_start_id:
                self._in_thinking = True
                self._done = False
                self._thinking_tokens = 0
                self._recent_tokens = []
                self._new_phase()
                return
            self._feed_loop_guard(token_id)
            return

        if self._forcing:
            self._force_idx += 1
            if self._force_idx >= len(self._forced_sequence()):
                self._forcing = False
                if self._force_kind == "nudge":
                    # Back to free reasoning; the close grace starts over.
                    self._nudged_at = self._thinking_tokens
                    self._grace_count = 0
                    return
                self._in_thinking = False
                self._done = True
                self._new_phase()
            return

        # Detect natural close-think via sliding window
        if len(self._think_end_ids) == 1:
            if token_id == self._think_end_ids[0]:
                self._closed_naturally()
                return
        else:
            self._recent_tokens.append(token_id)
            if len(self._recent_tokens) > len(self._think_end_ids):
                self._recent_tokens.pop(0)
            if self._recent_tokens == self._think_end_ids:
                self._closed_naturally()
                return

        if self._waiting_utf8:
            if self._last_token_utf8_complete:
                self._waiting_utf8 = False
                self._forcing = True
                self._force_idx = 0
                self._recent_tokens = []
            return

        # Detect re-entry into thinking (rare but possible)
        if not self._in_thinking and self._think_start_id and token_id == self._think_start_id:
            self._in_thinking = True
            self._done = False
            self._thinking_tokens = 0
            self._recent_tokens = []
            self._new_phase()
            return

        self._feed_loop_guard(token_id)

    def _closed_naturally(self) -> None:
        if self._nudged_at is not None and not self._nudge_outcome_logged:
            self._nudge_outcome_logged = True
            logger.info(
                "Thinking nudge%s: the model closed its reasoning %d tokens "
                "after the reminder",
                self._label_suffix(),
                self._thinking_tokens - self._nudged_at,
            )
        self._in_thinking = False
        self._done = True
        self._new_phase()

    def _new_phase(self) -> None:
        """Reasoning opened or closed: grace, reminder and loop window start
        over."""
        self._close_requested = False
        self._grace_count = 0
        self._nudged_at = None
        if self._loop_detector is not None:
            self._loop_detector.reset()

    def _feed_loop_guard(self, token_id: int) -> None:
        det = self._loop_detector
        if det is None or det.looping or not det.feed(token_id):
            return
        if self._in_thinking:
            self._close_requested = True
            action = "closing the reasoning"
        elif self._stop_token_id is not None:
            self._loop_stop = True
            action = "stopping the answer"
        else:
            action = "no stop token to force, letting it run"
        if not self._loop_logged:
            self._loop_logged = True
            logger.warning(
                "Loop guard%s: the %s repeats itself (%.0f%% of %d-token spans "
                "repeat over the last %d tokens); %s",
                self._label_suffix(),
                "reasoning" if self._in_thinking else "answer",
                100 * det.ratio,
                det.n,
                min(det.fed, det.window),
                action,
            )

    def _is_boundary(self, token_id: int) -> bool:
        """Whether the token's text ends a line or a sentence."""
        if self._token_to_piece is None:
            return False
        try:
            piece = self._token_to_piece(token_id)
        except Exception:
            return False
        if piece is None:
            return False
        if isinstance(piece, str):
            piece = piece.encode("utf-8", "ignore")
        return piece.rstrip(b" \t").endswith(_BOUNDARY_ENDS)

    def _is_utf8_complete(self, token_id: int) -> bool:
        """Best-effort UTF-8 boundary check for accepted token bytes."""
        if self._token_to_piece is None:
            return True
        try:
            piece = self._token_to_piece(token_id)
        except Exception:
            return True
        if piece is None:
            return True
        if isinstance(piece, str):
            self._pending_utf8 = b""
            return True
        self._pending_utf8 += piece
        try:
            self._pending_utf8.decode("utf-8")
            self._pending_utf8 = b""
            return True
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data" or exc.end == len(
                self._pending_utf8
            ):
                return False
            self._pending_utf8 = b""
            return True

    def _forced_sequence(self) -> List[int]:
        if self._force_kind == "nudge":
            return self._nudge_sequence
        return self._force_sequence

    def _force_next_token(self, logits, mx):
        """Force the next token of the close (or reminder) sequence."""
        target_id = self._forced_sequence()[self._force_idx]
        forced = mx.full(logits.shape, float("-inf"))
        forced[..., target_id] = 0.0
        return forced

    def _force_stop(self, logits, mx):
        """Force the stop token once the answer was found looping."""
        if self._stop_token_id is None:
            return logits
        forced = mx.full(logits.shape, float("-inf"))
        forced[..., self._stop_token_id] = 0.0
        return forced

    # -- Speculative-decoding (vlm_mtp) support -----------------------------
    #
    # The vlm_mtp decode path applies this processor inside mlx-vlm's
    # speculative verify walk, where positions past the first draft
    # rejection are discarded and re-sampled on the next round.
    # ``snapshot_state()`` / ``restore_state()`` let the caller checkpoint
    # the processor after each position and rewind it when a draft suffix
    # is rejected, keeping the one-call-per-emitted-token contract intact.
    # See ``omlx/speculative/processing_sampler.py``.

    _SNAPSHOT_ATTRS = (
        "_thinking_tokens",
        "_in_thinking",
        "_forcing",
        "_waiting_utf8",
        "_force_idx",
        "_done",
        "_first_call",
        "_recent_tokens",
        "_last_token_utf8_complete",
        "_pending_utf8",
        "_accepted_up_to",
        "_grace_count",
        "_at_boundary",
        "_close_requested",
        "_loop_stop",
        "_force_kind",
        "_nudged_at",
    )

    def snapshot_state(self) -> dict:
        """Checkpoint mutable state for position-keyed rewind (vlm_mtp)."""
        state: dict = {}
        for name in self._SNAPSHOT_ATTRS:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if isinstance(value, list):
                value = list(value)
            state[name] = value
        if self._loop_detector is not None:
            state["_loop_detector"] = self._loop_detector.snapshot()
        return state

    def restore_state(self, state: dict) -> None:
        """Restore a checkpoint produced by :meth:`snapshot_state`."""
        for name in self._SNAPSHOT_ATTRS:
            if name in state:
                value = state[name]
                if isinstance(value, list):
                    value = list(value)
                setattr(self, name, value)
            elif name == "_accepted_up_to" and hasattr(self, name):
                # Lazily-created attr absent from the snapshot: drop it so
                # the next __call__ re-baselines from the history it sees.
                delattr(self, name)
        if self._loop_detector is not None and "_loop_detector" in state:
            self._loop_detector.restore(state["_loop_detector"])
