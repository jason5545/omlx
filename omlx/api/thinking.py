# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.

Reasoning effort abstraction
----------------------------
``ReasoningEffort`` maps high-level effort levels (``low`` / ``medium`` /
``high`` / ``xhigh`` / ``native`` / ``off``) to concrete token budgets via
per-model-type ``ReasoningEffortProfile`` entries.  The profile also controls
soft-pressure behaviour so the model can close thinking naturally before the
hard budget is reached, rather than being cut off abruptly.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8

# Regex for non-streaming extraction (complete text)
_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)
_LEADING_CLOSE_TAG_PATTERN = re.compile(r'^(?:\s*</think>\s*)+')


def extract_thinking(
    text: str, start_in_thinking: bool = False
) -> Tuple[str, str]:
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
    - Native reasoning (``start_in_thinking=True``): when the prompt
      pre-opened ``<think>`` and generation contains no tags at all,
      the entire output is treated as thinking with empty content.

    Args:
        text: Complete model output text.
        start_in_thinking: If True, treat tag-free text as thinking
            (native-reasoning mode where the prompt pre-opens <think>).

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    thinking_parts = []
    remaining = text

    # Extract all <think>...</think> blocks
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        remaining = _LEADING_CLOSE_TAG_PATTERN.sub("", remaining).strip()
        return (thinking, remaining)

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            remaining = _LEADING_CLOSE_TAG_PATTERN.sub("", remaining).strip()
            return (thinking, remaining)

    # Malformed: <think> opened but never closed. Drop the open tag and
    # treat the remainder as content so the answer body is not empty.
    if '<think>' in text and '</think>' not in text:
        idx = text.index('<think>')
        before = text[:idx]
        after = text[idx + _OPEN_LEN:]
        return ("", (before + after).strip())

    # Native-reasoning fallback: prompt pre-opened <think>, generation
    # contains no tags at all — treat the whole output as thinking.
    if start_in_thinking and '<think>' not in text and '</think>' not in text:
        return (text.strip(), "")

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

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    self._in_thinking = False
                    self._close_seen = True
                    i += _CLOSE_LEN
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

    def finish(self) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.
        Also recovers from malformed thinking — when the model never
        emitted ``</think>`` and no content was ever produced, returns
        the accumulated thinking text as content so the client surfaces
        a non-empty answer body.

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
            self._in_thinking
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
        if length >= _CLOSE_LEN:
            # Long enough to determine - not a partial tag
            return False

        # Check against both tags
        if _OPEN_TAG[:length] == text:
            return True
        if _CLOSE_TAG[:length] == text:
            return True

        return False


class ReasoningEffort(str, Enum):
    """High-level reasoning depth, analogous to OpenAI's ``reasoning_effort``.

    Values:
        OFF:    Thinking disabled entirely (no budget, no ``<think>``).
        LOW:    Shallow thinking (~256 tokens).
        MEDIUM: Moderate thinking (~512 tokens).
        HIGH:   Deep thinking (~1024 tokens).
        XHIGH:  Maximum thinking (~2048 tokens).
        NATIVE: No budget — the model thinks as long as it wants.
    """
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    NATIVE = "native"


@dataclass
class ReasoningEffortProfile:
    """Per-effort token budget and soft-pressure parameters.

    The budget is the hard cap after which ``</think>`` is force-injected.
    The soft-pressure range (``soft_start_ratio`` to 1.0) applies a
    gradually increasing logit bias toward ``</think>`` so the model can
    close thinking naturally before the hard cut.

    Attributes:
        budget: Token budget for this effort level.
        soft_start_ratio: Fraction of budget where soft pressure begins
            (e.g. 0.75 → bias starts at 75 % of budget). ``None`` disables
            soft pressure for this level (hard-close only).
        soft_max_bias: Logit bias strength at budget edge (default 5.0).
    """
    budget: int
    soft_start_ratio: float | None = 0.75
    soft_max_bias: float = 5.0


# ---------------------------------------------------------------------------
# Default per-model-type effort profiles.
# These are conservative for Qwen-style dense models; larger / MoE models
# (DeepSeek, GLM) may benefit from higher budgets.
# ---------------------------------------------------------------------------
DEFAULT_EFFORT_PROFILES: Dict[str, Dict[str, ReasoningEffortProfile]] = {
    # Qwen 3.5 / 3.6 family — dense 27B model
    "qwen3_5": {
        "low":    ReasoningEffortProfile(budget=256),
        "medium": ReasoningEffortProfile(budget=512),
        "high":   ReasoningEffortProfile(budget=1024),
        "xhigh":  ReasoningEffortProfile(budget=2048),
    },
    "qwen3_6": {
        "low":    ReasoningEffortProfile(budget=256),
        "medium": ReasoningEffortProfile(budget=512),
        "high":   ReasoningEffortProfile(budget=1024),
        "xhigh":  ReasoningEffortProfile(budget=2048),
    },
    # Qwen 4 family — larger context, may need more thinking depth
    "qwen4": {
        "low":    ReasoningEffortProfile(budget=512),
        "medium": ReasoningEffortProfile(budget=1024),
        "high":   ReasoningEffortProfile(budget=2048),
        "xhigh":  ReasoningEffortProfile(budget=4096),
    },
    # DeepSeek V3 / V4 family — MoE models
    "deepseek_v3": {
        "low":    ReasoningEffortProfile(budget=512),
        "medium": ReasoningEffortProfile(budget=1024),
        "high":   ReasoningEffortProfile(budget=2048),
        "xhigh":  ReasoningEffortProfile(budget=4096),
    },
    "deepseek_v4": {
        "low":    ReasoningEffortProfile(budget=512),
        "medium": ReasoningEffortProfile(budget=1024),
        "high":   ReasoningEffortProfile(budget=2048),
        "xhigh":  ReasoningEffortProfile(budget=4096),
    },
    # GLM-4 / GLM-5 family
    "glm4": {
        "low":    ReasoningEffortProfile(budget=256),
        "medium": ReasoningEffortProfile(budget=512),
        "high":   ReasoningEffortProfile(budget=1024),
        "xhigh":  ReasoningEffortProfile(budget=2048),
    },
    "glm5": {
        "low":    ReasoningEffortProfile(budget=512),
        "medium": ReasoningEffortProfile(budget=1024),
        "high":   ReasoningEffortProfile(budget=2048),
        "xhigh":  ReasoningEffortProfile(budget=4096),
    },
}

# Catch-all default for unknown model types
_DEFAULT_EFFORT_PROFILE: Dict[str, ReasoningEffortProfile] = {
    "low":    ReasoningEffortProfile(budget=256),
    "medium": ReasoningEffortProfile(budget=512),
    "high":   ReasoningEffortProfile(budget=1024),
    "xhigh":  ReasoningEffortProfile(budget=2048),
}


def resolve_effort_to_budget(
    effort: ReasoningEffort,
    model_type: str | None = None,
    custom_profile: Dict[str, int] | None = None,
) -> int | None:
    """Resolve a reasoning effort level to a concrete token budget.

    Args:
        effort: The effort level.
        model_type: Model type key (e.g. ``"qwen3_6"``) for per-model defaults.
        custom_profile: Per-model override mapping from effort name to budget
            (e.g. ``{"low": 128, "high": 512}``).  When provided, this takes
            precedence over the built-in default profiles.

    Returns:
        Token budget integer, or ``None`` for ``OFF`` / ``NATIVE``.
    """
    if effort in (ReasoningEffort.OFF, ReasoningEffort.NATIVE):
        return None

    effort_key = effort.value  # "low", "medium", "high", "xhigh"

    # 1. Custom per-model profile
    if custom_profile and effort_key in custom_profile:
        return custom_profile[effort_key]

    # 2. Built-in per-model-type profile
    profiles = DEFAULT_EFFORT_PROFILES.get(model_type or "", {})
    if effort_key in profiles:
        return profiles[effort_key].budget

    # 3. Global default
    return _DEFAULT_EFFORT_PROFILE.get(effort_key, ReasoningEffortProfile(budget=512)).budget


def resolve_effort_soft_pressure(
    effort: ReasoningEffort,
    model_type: str | None = None,
) -> tuple[float | None, float]:
    """Return ``(soft_start_ratio, soft_max_bias)`` for an effort level.

    Returns ``(None, 0.0)`` when soft pressure is disabled.
    """
    if effort in (ReasoningEffort.OFF, ReasoningEffort.NATIVE):
        return None, 0.0

    effort_key = effort.value
    profiles = DEFAULT_EFFORT_PROFILES.get(model_type or "", {})
    profile = profiles.get(effort_key, _DEFAULT_EFFORT_PROFILE.get(effort_key))
    if profile is None:
        return None, 0.0
    return profile.soft_start_ratio, profile.soft_max_bias


def parse_reasoning_effort(value: str | None) -> ReasoningEffort | None:
    """Parse a string into a ReasoningEffort enum value.

    Accepts case-insensitive names and common aliases:
    ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"native"``,
    ``"off"``, ``"none"``, ``"disabled"``, ``"minimal"``, ``"maximum"``.

    Returns ``None`` if the value is unrecognised or empty.
    """
    if not value:
        return None
    if isinstance(value, ReasoningEffort):
        return value
    value = str(value).strip().lower()
    # Direct enum match
    try:
        return ReasoningEffort(value)
    except ValueError:
        pass
    # Common aliases
    aliases: Dict[str, ReasoningEffort] = {
        "none":     ReasoningEffort.OFF,
        "disabled": ReasoningEffort.OFF,
        "minimal":  ReasoningEffort.LOW,
        "maximum":  ReasoningEffort.XHIGH,
    }
    return aliases.get(value)


class ThinkingBudgetProcessor:
    """Logits processor that enforces a thinking token budget.

    Counts tokens generated while in thinking mode.  When the budget is
    exceeded, forces the close-think token(s) one at a time, then becomes
    a no-op for the rest of generation.

    When a soft-pressure zone is configured (``soft_start_ratio``), a
    gradually increasing logit bias is applied toward the ``</think>``
    token(s) before the hard budget, giving the model a chance to close
    thinking naturally rather than being abruptly cut off.

    Handles both single-token and multi-token close-think sequences, and
    supports alternative think markers (e.g. ``<longcat_think>``).

    Args:
        think_end_token_ids: Token ID(s) for the close-think tag.
        budget: Maximum number of thinking tokens before forcing close.
        think_start_token_id: Token ID for the open-think tag (re-entry detection).
        soft_start_ratio: Fraction of budget where soft pressure begins
            (e.g. 0.75). ``None`` disables soft pressure.
        soft_max_bias: Maximum logit bias at the budget edge.
    """

    def __init__(
        self,
        think_end_token_ids: List[int],
        budget: int,
        think_start_token_id: Optional[int] = None,
        leading_token_ids: Optional[List[int]] = None,
        trailing_token_ids: Optional[List[int]] = None,
        *,
        soft_start_ratio: float | None = 0.75,
        soft_max_bias: float = 5.0,
    ):
        self._think_end_ids = think_end_token_ids
        # Full force sequence: \n + </think> + \n\n (matches training pattern)
        self._force_sequence = (
            (leading_token_ids or [])
            + list(think_end_token_ids)
            + (trailing_token_ids or [])
        )
        self._budget = budget
        self._think_start_id = think_start_token_id

        # Soft-pressure: apply increasing bias toward </think> in the zone
        # [budget * soft_start_ratio, budget].  None disables soft pressure.
        self._soft_start: int | None
        self._soft_max_bias: float
        if soft_start_ratio is not None and soft_max_bias > 0 and budget > 0:
            self._soft_start = max(1, int(budget * soft_start_ratio))
            self._soft_max_bias = soft_max_bias
        else:
            self._soft_start = None
            self._soft_max_bias = 0.0

        # State
        self._thinking_tokens: int = 0
        self._in_thinking: bool = True  # Starts True (prompt ends with <think>)
        self._forcing: bool = False
        self._force_idx: int = 0
        self._done: bool = False
        self._first_call: bool = True
        # After forced sequence, suppress duplicate </think> tokens
        self._suppress_end: bool = False
        # Sliding window for multi-token end detection
        self._recent_tokens: List[int] = []
        # Flat set for fast single-token suppression check
        self._end_id_set = set(think_end_token_ids)
        # Native MTP calls logits processors on speculative verify rows.  In
        # that path the caller syncs accepted tokens explicitly so draft rows
        # do not advance the budget state.
        self._external_accept_sync: bool = False
        self._accepted_up_to: int | None = None

    def __call__(self, tokens, logits):
        """mlx-lm logits processor: (tokens, logits) -> logits."""
        import mlx.core as mx

        if self._done:
            return logits

        # Post-forcing phase: suppress duplicate </think> tokens
        if self._suppress_end:
            return self._suppress_end_tokens(logits, mx)

        if not self._external_accept_sync:
            self.sync_accepted_tokens(tokens)

        # Re-check flags — _update_state may have set them
        if self._done:
            return logits
        if self._suppress_end:
            return self._suppress_end_tokens(logits, mx)

        if self._forcing:
            return self._force_next_token(logits, mx)

        # Hard budget exceeded → force close sequence
        if self._in_thinking and self._thinking_tokens >= self._budget:
            self._forcing = True
            self._force_idx = 0
            return self._force_next_token(logits, mx)

        # Soft-pressure zone: apply gradual bias toward </think>
        if (
            self._in_thinking
            and self._soft_start is not None
            and self._thinking_tokens >= self._soft_start
        ):
            return self._apply_soft_pressure(logits, mx)

        return logits

    def sync_accepted_tokens(self, tokens, *, external: bool = False) -> None:
        """Advance state from real emitted history, skipping prompt tokens.

        The standard mlx-lm path passes the accepted history directly to
        ``__call__``. Native MTP, however, applies processors to speculative
        verify/draft rows before acceptance is known. The MTP patch calls this
        method with ``gen_batch.tokens[0]`` so only emitted tokens count toward
        the thinking budget.
        """
        n = len(tokens)
        if self._accepted_up_to is None or n < self._accepted_up_to:
            self._accepted_up_to = n
            if external:
                self._external_accept_sync = True
            return
        if n > self._accepted_up_to:
            for i in range(self._accepted_up_to, n):
                self._update_state(int(tokens[i]))
            self._accepted_up_to = n
        if external:
            self._external_accept_sync = True

    def _update_state(self, token_id: int) -> None:
        """Update thinking state based on the last generated token."""
        if self._forcing:
            # Native MTP applies the same stateful logits processor to several
            # speculative verify rows in one cycle.  Rows after the first may
            # contain draft tokens that were never emitted.  Only advance the
            # forced close sequence after the expected forced token actually
            # appears in the accepted token history.
            if (
                self._force_idx < len(self._force_sequence)
                and token_id != self._force_sequence[self._force_idx]
            ):
                return
            self._force_idx += 1
            if self._force_idx >= len(self._force_sequence):
                self._in_thinking = False
                self._forcing = False
                # Don't set _done — enter suppression mode to prevent
                # the model from generating a duplicate </think>.
                self._suppress_end = True
            return

        # Detect natural close-think via sliding window
        if len(self._think_end_ids) == 1:
            if token_id == self._think_end_ids[0]:
                self._in_thinking = False
                self._done = True
                return
        else:
            self._recent_tokens.append(token_id)
            if len(self._recent_tokens) > len(self._think_end_ids):
                self._recent_tokens.pop(0)
            if self._recent_tokens == self._think_end_ids:
                self._in_thinking = False
                self._done = True
                return

        # Detect re-entry into thinking (rare but possible)
        if not self._in_thinking and self._think_start_id and token_id == self._think_start_id:
            self._in_thinking = True
            return

        if self._in_thinking:
            self._thinking_tokens += 1

    def _force_next_token(self, logits, mx):
        """Force the next token in the close-think + trailing sequence."""
        target_id = self._force_sequence[self._force_idx]
        forced = mx.full(logits.shape, float("-inf"))
        forced[..., target_id] = 0.0
        return forced

    def _apply_soft_pressure(self, logits, mx):
        """Apply gradually increasing logit bias toward close-think tokens.

        The bias scales linearly from 0 at ``_soft_start`` to
        ``_soft_max_bias`` at ``_budget``.  This encourages the model to
        close thinking naturally rather than being abruptly cut off.
        For multi-token close tags, bias only the next token in the close
        sequence instead of every token in the sequence.
        """
        if self._soft_start is None or self._budget <= self._soft_start:
            return logits
        # Linear interpolation: 0 → soft_max_bias
        progress = (self._thinking_tokens - self._soft_start) / (
            self._budget - self._soft_start
        )
        bias = self._soft_max_bias * min(progress, 1.0)
        for tid in self._soft_pressure_target_ids():
            logits[..., tid] = logits[..., tid] + bias
        return logits

    def _soft_pressure_target_ids(self) -> List[int]:
        """Return close-token IDs to bias for the next decode step."""
        if len(self._think_end_ids) <= 1:
            return list(self._end_id_set)

        # If the model has already started a multi-token close marker,
        # continue nudging the next expected token in that sequence.
        max_prefix = min(len(self._recent_tokens), len(self._think_end_ids) - 1)
        for prefix_len in range(max_prefix, 0, -1):
            if self._recent_tokens[-prefix_len:] == self._think_end_ids[:prefix_len]:
                return [self._think_end_ids[prefix_len]]

        return [self._think_end_ids[0]]

    def _suppress_end_tokens(self, logits, mx):
        """Suppress duplicate </think> tokens after forced close."""
        # Set logits of all end-token IDs to -inf so the model
        # cannot produce another </think>.
        for tid in self._end_id_set:
            logits[..., tid] = float("-inf")
        return logits
