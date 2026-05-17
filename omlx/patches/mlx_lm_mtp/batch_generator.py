# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — after the standard ``_step()`` has run
  the prompt's last token through the backbone, we add an MTP "post-init"
  step that runs one more 1-token backbone forward (with hidden) and one
  MTP-head forward. Two confirmed tokens are queued for emission and a
  draft is stashed for the first verify cycle.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence we emit from the per-batch queue first; once empty, we run a
  2-token verify forward over ``[next_main, draft]`` with
  ``n_confirmed=1`` and a single MTP-head forward at the bonus position
  (accept) or confirmed position (reject), refilling the queue from the
  verify outputs.

The throughput math (greedy, accept rate p):
  - Cost per *cycle*: 1× backbone (2-token verify) + 1× MTP head ≈ 1.15
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - At p≈1: 0.575 cost/token → ~1.74× throughput
  - At p≈0.5: ~0.77 cost/token → ~1.30× throughput

Greedy identity (sampler is None): the patched dispatch produces the same
tokens as the standard step. PR 990's ``test_mtp_generate_identity``
encodes this contract; the oMLX-side equivalent lives in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft (mirroring PR 990's
``prev_tokens = prev_tokens[:-1]``).
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

from .adaptive import AdaptiveDepthPolicy

logger = logging.getLogger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Wrap ``GenerationBatch.__init__`` + ``GenerationBatch.next``."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.generate import GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate.GenerationBatch not importable")
        return False

    if hasattr(GenerationBatch, "_omlx_mtp_patched"):
        _PATCHED = True
        return True

    original_init = GenerationBatch.__init__
    original_next = GenerationBatch.next
    original_filter = GenerationBatch.filter
    original_extend = GenerationBatch.extend

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _is_mtp_eligible(self):
            try:
                _post_init_mtp(self)
                logger.info(
                    "MTP path activated for uid=%s (model has mtp_forward, batch=1)",
                    getattr(self, "uids", ["?"])[0],
                )
            except _MtpStepFallback as exc:
                logger.warning("MTP post-init fallback: %s", exc)
        else:
            # The empty-batch case is BatchGenerator.__init__ pre-creating
            # ``self._generation_batch = GenerationBatch.empty(...)`` and is
            # always part of normal startup — silence it. Only log when the
            # batch is genuinely populated (e.g. continuous batching with
            # batch>1) so the message points at a real misconfiguration.
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)

    def patched_next(self, *args, **kwargs):
        if _is_mtp_eligible(self):
            state = getattr(self, "_omlx_mtp_state", None)
            if state is not None:
                try:
                    return _mtp_next(self, state)
                except _MtpStepFallback as exc:
                    logger.debug(
                        "MTP next() fallback to standard step: %s", exc
                    )
                    # Best-effort: drop state so subsequent calls don't try
                    # to resume a half-built MTP cycle from a stale snapshot.
                    if hasattr(self, "_omlx_mtp_state"):
                        try:
                            delattr(self, "_omlx_mtp_state")
                        except AttributeError:
                            pass
        return original_next(self, *args, **kwargs)

    def patched_extend(self, batch, *args, **kwargs):
        # ``BatchGenerator._next()`` builds a fresh single-sequence
        # ``GenerationBatch`` via ``prompt_batch.split(...).generate(...)``
        # then merges it into ``self._generation_batch`` via extend(). The
        # MTP post-init runs on the fresh batch (since that's the one whose
        # __init__ fires with uids=[0]); without this transfer the state
        # would die with the donor instance.  If the merge leaves the host
        # with multiple live rows, drop the donor state instead: this mirrors
        # the mixed-mode constraint from sleepy/omlx without trying to graft
        # its old scheduler path onto the current BatchGenerator patch.
        donor_state = getattr(batch, "_omlx_mtp_state", None)
        result = original_extend(self, batch, *args, **kwargs)
        _reconcile_extended_mtp_state(self, batch, donor_state)
        return result

    def patched_filter(self, keep, *args, **kwargs):
        # When the outer scheduler retires this sequence (e.g. EOS detected
        # outside our finish path), it calls filter([]) to drop everything.
        # Surface stats here so the user sees them even when the standard
        # _emit_response finish path doesn't fire.
        state = getattr(self, "_omlx_mtp_state", None)
        result = original_filter(self, keep, *args, **kwargs)
        if state is not None and not getattr(self, "uids", None):
            # Batch is now empty — log + drop state.
            try:
                _log_mtp_stats(
                    "?", state.stats, getattr(state, "_finish_reason", "external")
                )
            except Exception:
                pass
            try:
                delattr(self, "_omlx_mtp_state")
            except AttributeError:
                pass
        return result

    GenerationBatch.__init__ = patched_init
    GenerationBatch.next = patched_next
    GenerationBatch.filter = patched_filter
    GenerationBatch.extend = patched_extend
    GenerationBatch._omlx_mtp_patched = True
    _PATCHED = True
    return True


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", None)
    if inner is None:
        inner = getattr(model, "_language_model", model)
    return (
        hasattr(inner, "mtp_forward")
        and hasattr(inner, "make_mtp_cache")
        and hasattr(inner, "mtp")
        and getattr(inner, "mtp", None) is not None
    )


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward``, has an attached MTP head, and
    the process-wide ``mtp_active`` flag is on.

    The MTP head may be attached unconditionally (e.g. by the mlx-vlm
    runtime patches, which need it for weight-load matching even when
    inference-time MTP is off) — so head presence alone is not enough
    to decide whether to run the draft/verify cycle. ``is_mtp_active``
    reflects the per-load ``model_settings.mtp_enabled`` choice.
    """
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    try:
        from . import is_mtp_active
        if not is_mtp_active():
            return False
    except Exception:
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head"
    try:
        from . import is_mtp_active
        if not is_mtp_active():
            return "mtp_active flag is off (model_settings.mtp_enabled was False at load time)"
    except Exception:
        return "is_mtp_active import failed"
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        return f"batch size {len(uids)} != 1 (continuous batching, MTP off by design)"
    return ""


def _drop_mtp_state(gen_batch: Any) -> Optional["_MtpState"]:
    """Remove and return any speculative MTP state attached to a batch."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None:
        return None
    try:
        delattr(gen_batch, "_omlx_mtp_state")
    except AttributeError:
        pass
    return state


def _reconcile_extended_mtp_state(
    host: Any,
    donor: Any,
    donor_state: Optional["_MtpState"],
) -> None:
    """Keep MTP state only when an extend result is still a solo batch.

    sleepy/omlx's scheduler-native branch keeps an MTP request out of the
    normal batch when other requests arrive.  In this newer integration the
    safe equivalent is stricter: once ``GenerationBatch.extend`` produces a
    multi-row host, no per-row MTP state may remain attached to that host.
    That prevents a stale single-request draft/verify cycle from surviving
    inside a continuous batch and becoming active again after later filters.
    """
    uids = getattr(host, "uids", None) or []
    host_state = getattr(host, "_omlx_mtp_state", None)

    if len(uids) == 1:
        if donor_state is not None and host_state is None:
            host._omlx_mtp_state = donor_state
            _drop_mtp_state(donor)
            logger.debug(
                "MTP state transferred from donor batch to host batch (uid=%s)",
                uids[0],
            )
        return

    dropped_host = _drop_mtp_state(host)
    dropped_donor = _drop_mtp_state(donor)
    if dropped_host is not None or dropped_donor is not None or donor_state is not None:
        logger.debug(
            "MTP state dropped after batch extend because host batch size is %d",
            len(uids),
        )


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # accepted draft tokens across all cycles
    rejects: int = 0  # rejected draft tokens across all cycles
    full_accepts: int = 0  # cycles where every drafted token was accepted
    drafted_tokens: int = 0  # total draft tokens attempted
    max_depth: int = 1  # configured max draft depth for this sequence
    adaptive: bool = False  # True when adaptive depth policy was active
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with the draft block cheaply.
    next_main: Optional[Any] = None

    # Draft block for the next verify cycle. Each position stores the sampled
    # token, raw logprobs, filtered acceptance logprobs, and a host-side token
    # id so the accept walk avoids per-depth GPU→CPU syncs.
    draft_toks: List[Any] = field(default_factory=list)  # each (1,) uint32
    draft_lps: List[Any] = field(default_factory=list)  # each (vocab,) float
    draft_accept_lps: List[Any] = field(default_factory=list)  # each (vocab,) float
    draft_ids: List[int] = field(default_factory=list)
    draft_depth: int = 1

    # Current sequence position (0-indexed token count in main cache).
    # Used to calculate correct position_offset for MTP RoPE.
    position: int = 0

    # Adaptive depth policy (None when fixed depth is used).
    adaptive_policy: Optional[AdaptiveDepthPolicy] = None

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_generation_stream():
    """Return the ``mlx_lm.generate`` module-level generation stream.

    The standard ``GenerationBatch._step`` runs all forward passes inside
    ``mx.stream(generation_stream)``; the MTP cycle does the same so the
    paged cache writes land on the same stream and ordering is preserved.
    The stream lives on the *outer* ``BatchGenerator``, not on
    ``GenerationBatch``, so we read it from the module.

    Note: ``mlx_lm.__init__`` re-exports a ``generate`` *function*, so
    ``import mlx_lm.generate as mlg`` resolves to the function, not the
    module. We use ``sys.modules`` to grab the actual module.
    """
    import sys

    return sys.modules["mlx_lm.generate"].generation_stream


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch: Any) -> bool:
    """Return True when the active sampler is deterministic argmax.

    mlx-lm PR 990 used ``sampler is None`` as the greedy signal. oMLX always
    routes requests through explicit sampler callables so we also inspect the
    metadata attached by ``omlx.utils.sampling.make_sampler``. Without this,
    ``temperature=0`` per-row samplers take the stochastic acceptance path and
    pay for unnecessary filtered-logprob / residual-sampling work.
    """
    sampler = _resolve_sampler(gen_batch)
    try:
        return float(getattr(sampler, "temp", 0.0) or 0.0) == 0.0
    except (TypeError, ValueError):
        return not (gen_batch.samplers and gen_batch.samplers[0] is not None)


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Reads sampling params off the callable as function attributes (set by
    ``omlx.utils.sampling.make_sampler``). For samplers without metadata —
    e.g. mlx-lm stock callables, fallback samplers — returns `lp` unchanged
    so behavior matches the pre-PR-990 raw-lp acceptance.
    """
    import mlx.core as mx

    from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the output is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _resolve_mtp_draft_depth(gen_batch: Any) -> int:
    """Return the requested native-MTP draft depth for this model."""
    import os

    raw = os.environ.get("OMLX_MTP_DRAFT_DEPTH")
    if raw is None:
        raw = getattr(gen_batch.model, "_omlx_mtp_draft_depth", 1)
    try:
        depth = int(raw)
    except (TypeError, ValueError):
        depth = 1
    return max(1, min(depth, 8))


def _adaptive_depth_enabled(gen_batch: Any) -> bool:
    import os

    env = os.environ.get("OMLX_MTP_ADAPTIVE_DEPTH")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return bool(getattr(gen_batch.model, "_omlx_mtp_adaptive_depth", False))


def _make_adaptive_policy(gen_batch: Any) -> AdaptiveDepthPolicy:
    max_d = _resolve_mtp_draft_depth(gen_batch)
    return AdaptiveDepthPolicy(max_depth=max_d, min_depth=1, start_depth=1)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` (mirrors PR 990 ``prev[:-n]``)."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any], trim_count: int = 1) -> bool:
    """Roll back tokens from each layer cache after a draft rejection."""
    trim_count = max(0, int(trim_count))
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            if trim_count:
                c.trim(trim_count)
            continue
        return False
    return True


def _rollback_after_reject(
    model: Any,
    prompt_cache: List[Any],
    gdn_states: Optional[list],
    accepted: int = 0,
    block_size: int = 2,
) -> bool:
    """Roll back per-layer cache state after a rejected MTP draft token.

    Two mechanisms are supported, dispatched on the model's capability:

    1. **mlx-vlm path** — when the model exposes ``rollback_speculative_cache``
       (Qwen3.5 LanguageModel ships with it upstream) AND ``gdn_states`` is
       populated, we delegate to that method. It batches the per-layer SSM
       replay into a single ``gated_delta_update`` call and trims KV
       caches by ``block_size - (accepted + 1)``. The backbone forward was
       run with both confirmed and draft tokens; the rollback replays only
       the accepted prefix through the original pre-update state.

    2. **mlx-lm path** (PR 990) — per-layer ``cache.rollback_state`` snapshot
       written by the patched ``GatedDeltaNet.__call__`` during the
       confirmed/draft split. We restore the snapshot for SSM layers and
       trim KV layers by 1. ``gdn_states`` is None in this path.

    Returns True on success. False means a cache layer in the list supports
    neither mechanism, in which case the caller falls back to the standard
    non-MTP step.
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        model.rollback_speculative_cache(
            prompt_cache, gdn_states, accepted, block_size
        )
        return True
    trim_count = max(0, int(block_size) - (int(accepted) + 1))
    return _restore_or_trim_caches(prompt_cache, trim_count=trim_count)


def _call_backbone(
    model: Any,
    inputs: Any,
    cache: List[Any],
    n_confirmed: int = 0,
) -> Tuple[Any, Any, Optional[list]]:
    """Run the backbone with ``return_hidden=True`` and normalise the result.

    Returns ``(logits, hidden_pre_norm, gdn_states_or_None)``:

    - mlx-lm path returns the 2-tuple ``(logits, hidden)``; ``gdn_states``
      is ``None`` and rollback uses ``cache.rollback_state``.
    - mlx-vlm path returns the 3-tuple ``(logits, hidden, gdn_states)`` so
      a rejected draft can be rolled back via
      ``rollback_speculative_cache``.

    ``n_confirmed`` is forwarded so the mlx-lm path can split its
    GatedDeltaNet forward into confirmed and draft chunks. mlx-vlm
    discards it (irrelevant — rollback is post-hoc, not splitwise).
    """
    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    result = model(inputs, **kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    raise TypeError(
        f"backbone returned unexpected shape: {type(result).__name__}"
    )


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop ``rollback_state`` snapshots after a draft is accepted."""
    for c in prompt_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


def _trim_mtp_cache(mtp_cache: Optional[List[Any]], trim_count: int) -> bool:
    """Trim speculative draft tokens from the MTP-head cache."""
    if not mtp_cache or trim_count <= 0:
        return True
    for cache in mtp_cache:
        if hasattr(cache, "trim"):
            cache.trim(trim_count)
            continue
        return False
    return True


def _call_mtp_head(
    model: Any,
    hidden_at_position: Any,
    next_token: Any,
    mtp_cache: Optional[List[Any]],
    position_offset: int = 0,
) -> Tuple[Any, Optional[Any]]:
    """Run ``mtp_forward`` and request hidden state when the model supports it."""
    next_ids = next_token.reshape(1, 1)
    try:
        result = model.mtp_forward(
            hidden_at_position,
            next_ids,
            mtp_cache,
            return_hidden=True,
            position_offset=position_offset,
        )
    except TypeError:
        try:
            result = model.mtp_forward(
                hidden_at_position,
                next_ids,
                mtp_cache,
                return_hidden=True,
            )
        except TypeError:
            result = model.mtp_forward(hidden_at_position, next_ids, mtp_cache)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None


def _set_state_drafts(
    state: "_MtpState",
    drafts: List[Tuple[Any, Any, Any, int]],
) -> None:
    state.draft_toks = [item[0] for item in drafts]
    state.draft_lps = [item[1] for item in drafts]
    state.draft_accept_lps = [item[2] for item in drafts]
    state.draft_ids = [item[3] for item in drafts]


def _draft_block_from(
    gen_batch: Any,
    state: "_MtpState",
    hidden_at_position: Any,
    next_main_tok: Any,
    prev_buf: Optional[Any],
    *,
    depth: int,
) -> List[Tuple[Any, Any, Any, int]]:
    """Draft up to ``depth`` tokens using native MTP hidden state chaining."""
    import time

    import mlx.core as mx

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    drafts: List[Tuple[Any, Any, Any, int]] = []
    current_hidden = hidden_at_position
    current_token = _ensure_uint32(next_main_tok)
    processor_context = prev_buf
    base_offset = state.position

    for depth_index in range(max(1, int(depth))):
        t0 = time.perf_counter()
        with mx.stream(_get_generation_stream()):
            mtp_logits, mtp_hidden = _call_mtp_head(
                gen_batch.model,
                current_hidden,
                current_token,
                state.mtp_cache,
                position_offset=base_offset + depth_index,
            )
            mtp_logits_2d = mtp_logits[:, -1, :]
        if procs is not None and processor_context is not None:
            prev_with_current = mx.concatenate(
                [processor_context, _ensure_uint32(current_token)]
            )
            mtp_logits_2d = _apply_processors(
                procs, prev_with_current, mtp_logits_2d
            )
        else:
            prev_with_current = None
        draft_lp_2d = _logprobs(mtp_logits_2d)
        draft_tok = sampler(draft_lp_2d)
        draft_accept_lp_2d = _accept_lp_for(sampler, draft_lp_2d)
        draft_id = int(draft_tok.tolist()[0])
        state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
        drafts.append(
            (
                _ensure_uint32(draft_tok),
                draft_lp_2d.squeeze(0),
                draft_accept_lp_2d.squeeze(0),
                draft_id,
            )
        )

        if depth_index >= depth - 1 or mtp_hidden is None:
            break
        current_hidden = mtp_hidden[:, -1:, :]
        current_token = _ensure_uint32(draft_tok)
        processor_context = prev_with_current

    return drafts


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------

def _post_init_mtp(gen_batch: Any) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into PR 990's cycle 1.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state. No draft yet,
    # so no rollback is possible — discard gdn_states.
    with mx.stream(_get_generation_stream()):
        logits, hidden, _ = _call_backbone(
            gen_batch.model, main_tok[:, None], gen_batch.prompt_cache
        )

    next_main_logits = logits[:, -1, :]  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _logprobs(next_main_logits)
    next_main_tok = sampler(next_main_lp)  # (1,)

    state = _MtpState()
    state.mtp_cache = gen_batch.model.make_mtp_cache()
    state.draft_depth = _resolve_mtp_draft_depth(gen_batch)
    state.stats.max_depth = state.draft_depth
    state.position = len(gen_batch.tokens[0])  # main_tok is already in cache
    gen_batch._omlx_mtp_state = state

    if _adaptive_depth_enabled(gen_batch):
        state.adaptive_policy = _make_adaptive_policy(gen_batch)
        state.draft_depth = state.adaptive_policy.current_depth
        state.stats.adaptive = True
        logger.debug(
            "MTP adaptive depth policy initialised: start=%d max=%d",
            state.adaptive_policy.current_depth,
            state.adaptive_policy.max_depth,
        )

    # MTP head sees (hidden_at_main, next_main_tok) and proposes the draft
    # block that the *next* verify cycle will check against.
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    drafts = _draft_block_from(
        gen_batch,
        state,
        hidden_at_main,
        _ensure_uint32(next_main_tok),
        prev_buf,
        depth=state.draft_depth,
    )
    if not drafts:
        delattr(gen_batch, "_omlx_mtp_state")
        raise _MtpStepFallback("MTP post-init produced no draft tokens")

    mx.eval(main_tok, next_main_tok)

    _set_state_drafts(state, drafts)
    state.next_main = _ensure_uint32(next_main_tok)
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------

def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(uid: Any, stats: "_MtpStats", finish_reason: str) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to match PR 990's headline metrics, plus component timings
    that make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> accept=<A>/<C> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    if stats.drafted_tokens > 0:
        rate_str = f"{stats.accepts / stats.drafted_tokens * 100:.1f}%"
    else:
        rate_str = "n/a"
    depth_tag = f"adaptive≤{stats.max_depth}" if stats.adaptive else f"depth≤{stats.max_depth}"
    logger.info(
        "MTP[%s] finish=%s tokens=%d cycles=%d %s "
        "draft_accept=%d/%d (%s) full=%d/%d rejects=%d "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        total_emits,
        stats.cycles,
        depth_tag,
        stats.accepts,
        stats.drafted_tokens,
        rate_str,
        stats.full_accepts,
        stats.cycles,
        stats.rejects,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------

def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Run one multi-depth verify cycle and refill the emit queue."""
    import time

    import mlx.core as mx

    if state.next_main is None or not state.draft_toks:
        raise _MtpStepFallback("verify cycle entered without next_main / draft")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)
    draft_count = len(state.draft_toks)

    inputs = mx.concatenate([state.next_main, *state.draft_toks])

    prev_contexts: List[Any] = []
    if procs is not None:
        prev_contexts.append(
            gen_batch._token_context[0].update_and_fetch(state.next_main)
        )
        for draft_tok in state.draft_toks:
            prev_contexts.append(
                gen_batch._token_context[0].update_and_fetch(draft_tok)
            )

    t0 = time.perf_counter()
    with mx.stream(_get_generation_stream()):
        logits, hidden, gdn_states = _call_backbone(
            gen_batch.model,
            inputs[None, :],
            gen_batch.prompt_cache,
            n_confirmed=1,
        )
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    processed_logits = []
    for idx in range(draft_count + 1):
        logits_2d = logits[:, idx, :]
        if procs is not None:
            logits_2d = _apply_processors(procs, prev_contexts[idx], logits_2d)
        processed_logits.append(logits_2d)
    combined_logits = mx.concatenate(processed_logits, axis=0)
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    target_toks = sampler(combined_lp)
    mx.eval(target_toks)
    target_ids = [int(x) for x in target_toks.tolist()]

    target_accept_lp = _accept_lp_for(sampler, combined_lp[:draft_count])
    accepted_count = 0
    rejection_correction: Optional[int] = None
    rejected_lp_2d = None

    for depth_index, draft_id in enumerate(state.draft_ids):
        target_lp_2d = combined_lp[depth_index : depth_index + 1]
        if is_greedy:
            accepted_now = target_ids[depth_index] == draft_id
            correction = target_ids[depth_index]
        else:
            draft_accept_lp = state.draft_accept_lps[depth_index]
            log_accept = (
                target_accept_lp[depth_index, draft_id].item()
                - draft_accept_lp[draft_id].item()
            )
            accepted_now = log_accept >= 0 or random.random() < math.exp(log_accept)
            correction = (
                draft_id
                if accepted_now
                else _residual_sample(
                    target_accept_lp[depth_index : depth_index + 1],
                    draft_accept_lp,
                )[0]
            )
        if accepted_now:
            accepted_count += 1
            continue
        rejection_correction = int(correction)
        rejected_lp_2d = target_lp_2d
        break
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    state.stats.cycles += 1
    state.stats.drafted_tokens += draft_count
    state.stats.accepts += accepted_count
    if accepted_count < draft_count:
        state.stats.rejects += 1
    else:
        state.stats.full_accepts += 1

    state.position += 1  # the main (confirmed) token advances position

    if state.adaptive_policy is not None:
        decision = state.adaptive_policy.observe(
            attempted_depth=draft_count, accepted_depths=accepted_count
        )
        if decision["action"] != "hold":
            logger.debug(
                "MTP adaptive depth %s: %d→%d (accept=%d/%d)",
                decision["action"],
                decision["previous_depth"],
                decision["next_depth"],
                accepted_count,
                draft_count,
            )
        state.draft_depth = state.adaptive_policy.current_depth
        state.stats.max_depth = max(state.stats.max_depth, state.draft_depth)

    if accepted_count == draft_count:
        t0 = time.perf_counter()
        _clear_rollback(gen_batch.prompt_cache)
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        bonus_tok = mx.array([target_ids[draft_count]], dtype=mx.uint32)
        bonus_lp_1d = combined_lp[draft_count]
        prev_for_bonus = prev_contexts[-1] if procs is not None else None
        new_drafts = _draft_block_from(
            gen_batch,
            state,
            hidden[:, draft_count : draft_count + 1, :],
            bonus_tok,
            prev_for_bonus,
            depth=state.draft_depth,
        )
        for draft_id, draft_lp in zip(state.draft_ids, state.draft_lps):
            state.queue.append((draft_id, draft_lp, "draft"))
        state.queue.append((int(bonus_tok.tolist()[0]), bonus_lp_1d, "bonus"))
        state.next_main = bonus_tok
        _set_state_drafts(state, new_drafts)
        return

    # Reject path
    t0 = time.perf_counter()
    if not _rollback_after_reject(
        gen_batch.model,
        gen_batch.prompt_cache,
        gdn_states,
        accepted=0,
        block_size=draft_count + 1,
    ):
        if procs is not None:
            _trim_token_buffer(gen_batch, draft_count - accepted_count)
        raise _MtpStepFallback("cache layer rejects rollback")
    if procs is not None:
        _trim_token_buffer(gen_batch, draft_count - accepted_count)
    mtp_trim = max(0, draft_count - (accepted_count + 1))
    if not _trim_mtp_cache(state.mtp_cache, mtp_trim):
        raise _MtpStepFallback("MTP cache rejects rollback")

    if accepted_count > 0:
        replay_tokens = mx.concatenate(state.draft_toks[:accepted_count])
        with mx.stream(_get_generation_stream()):
            replay_logits, replay_hidden, _ = _call_backbone(
                gen_batch.model,
                replay_tokens[None, :],
                gen_batch.prompt_cache,
            )
            mx.eval(replay_logits, replay_hidden)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    if rejection_correction is None or rejected_lp_2d is None:
        raise _MtpStepFallback("reject path missing correction token")
    emit_tok = mx.array([rejection_correction], dtype=mx.uint32)
    prev_for_emit = prev_contexts[accepted_count] if procs is not None else None
    new_drafts = _draft_block_from(
        gen_batch,
        state,
        hidden[:, accepted_count : accepted_count + 1, :],
        emit_tok,
        prev_for_emit,
        depth=state.draft_depth,
    )

    for draft_id, draft_lp in zip(
        state.draft_ids[:accepted_count],
        state.draft_lps[:accepted_count],
    ):
        state.queue.append((draft_id, draft_lp, "draft"))
    state.queue.append((rejection_correction, rejected_lp_2d.squeeze(0), "verify"))
    state.next_main = emit_tok
    _set_state_drafts(state, new_drafts)


def _residual_sample(verify_lp_2d: Any, draft_lp_1d: Any) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    Mirrors mlx-lm PR 990 commit 6594348.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    import mlx.core as mx

    p_target = mx.exp(verify_lp_2d.squeeze(0))
    p_draft = mx.exp(draft_lp_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    # Keep z in graph; mx.where switches to the target distribution when
    # the residual mass is zero. ``categorical`` treats log(0) = -inf as
    # p=0 so no safety epsilon is needed.
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual, p_target)
    sample = mx.random.categorical(mx.log(dist).reshape(1, -1))
    return int(sample.item()), verify_lp_2d.squeeze(0)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------

def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if stats is not None:
            _log_mtp_stats(gen_batch.uids[0], stats, finish_reason)
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]
