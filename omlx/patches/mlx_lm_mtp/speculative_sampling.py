# SPDX-License-Identifier: Apache-2.0
"""NumPy-based speculative sampling primitives for the native-MTP verify cycle.

Adapted from MTPLX (https://github.com/youssofal/MTPLX, Apache-2.0).
These functions run purely on CPU after the initial top-k logit extraction,
avoiding the per-depth GPU→CPU ``.item()`` round-trips that dominate the
full-vocab acceptance path.
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Sparse distribution (NumPy-backed; replaces the Python-list version)
# ---------------------------------------------------------------------------

class SparseDistribution:
    """Top-k probability distribution stored as NumPy arrays.

    Mirrors ``mtplx.sampling.SparseDistribution`` so the acceptance-walk
    probability lookups are O(k) NumPy vectorised comparisons instead of
    per-element Python ``for`` loops.
    """

    __slots__ = ("token_ids", "probs", "vocab_size")

    def __init__(
        self,
        token_ids: np.ndarray,
        probs: np.ndarray,
        vocab_size: int,
    ) -> None:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        if token_ids.ndim != 1 or probs.ndim != 1:
            raise ValueError("SparseDistribution expects 1D arrays")
        if token_ids.shape[0] != probs.shape[0]:
            raise ValueError("token_ids/probs length mismatch")
        if token_ids.shape[0] == 0:
            raise ValueError("SparseDistribution cannot be empty")
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0:
            token_ids = token_ids[:1].copy()
            probs = np.array([1.0], dtype=np.float64)
            total = 1.0
        self.token_ids = token_ids
        self.probs = probs / total
        self.vocab_size = int(vocab_size)

    def probability(self, token_id: int) -> float:
        hits = np.nonzero(self.token_ids == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(hits[0])])

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        keep = self.probs > 0
        return int(rng.choice(self.token_ids[keep], p=self.probs[keep]))


# ---------------------------------------------------------------------------
# Leviathan-Chen acceptance / residual correction (NumPy)
# ---------------------------------------------------------------------------

def acceptance_probability(
    target: SparseDistribution,
    draft: SparseDistribution,
    token_id: int,
) -> float:
    """Return ``min(1, p_target / q_draft)`` for token *token_id*."""
    p = target.probability(token_id)
    q = draft.probability(token_id)
    if q <= 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)


def residual_distribution(
    target: SparseDistribution,
    draft: SparseDistribution,
) -> SparseDistribution:
    """Build ``max(p_target - p_draft, 0)`` and renormalise."""
    all_ids = np.union1d(target.token_ids, draft.token_ids).astype(np.int64)
    residual = np.array(
        [
            max(target.probability(int(tid)) - draft.probability(int(tid)), 0.0)
            for tid in all_ids
        ],
        dtype=np.float64,
    )
    keep = residual > 0
    total = float(residual[keep].sum())
    if total <= 0.0:
        return target
    return SparseDistribution(
        all_ids[keep],
        residual[keep] / total,
        target.vocab_size,
    )


def residual_sample(
    target: SparseDistribution,
    draft: SparseDistribution,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from ``max(p_target - p_draft, 0)``."""
    dist = residual_distribution(target, draft)
    return dist.sample(rng)


# ---------------------------------------------------------------------------
# Batch logits → sparse distributions (NumPy-backed)
# ---------------------------------------------------------------------------

def _batched_top_k_from_logits(
    logits: mx.array,
    top_k: int,
    top_p: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract top-k token ids + probabilities *without* per-row Python loops.

    Returns ``(token_ids [rows,k], probs [rows,k], vocab_size)`` where rows
    with zero mass have their first entry forced to 1.0.
    """
    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    vocab_size = int(rows.shape[-1])
    k = min(int(top_k), vocab_size)
    if k <= 0:
        raise ValueError("top_k must be > 0")

    # Temperature-scale before argpartition so the top-k reflects final probs.
    if temperature > 0:
        rows = rows * (1.0 / float(temperature))

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[:, :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    log_total = mx.logsumexp(rows, axis=-1)
    top_probs = mx.exp(top_vals - log_total[:, None])
    mx.eval(top_idx, top_probs)

    token_rows = np.asarray(top_idx, dtype=np.int64)
    prob_rows = np.asarray(top_probs, dtype=np.float64)

    # Apply top-p filter (same semantics as mlx_lm / MTPLX)
    if 0.0 < top_p < 1.0:
        cum_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        keep = cum_before < float(top_p)
        keep[:, 0] = True
        prob_rows = np.where(keep, prob_rows, 0.0)

    # Ensure every row has positive mass
    row_sums = prob_rows.sum(axis=1)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
    if np.any(bad):
        prob_rows[bad, :] = 0.0
        prob_rows[bad, 0] = 1.0
        row_sums = prob_rows.sum(axis=1)

    prob_rows = prob_rows / row_sums[:, None]
    return token_rows, prob_rows, vocab_size


def sparse_distributions_from_logits(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[SparseDistribution] | None:
    """Build a list of :class:`SparseDistribution` from a batch of logits.

    Returns ``None`` when the sampler config is greedy or top-k is disabled,
    signalling the caller to use the full-vocab path.
    """
    if temperature <= 0.0 or top_k <= 0:
        return None

    token_rows, prob_rows, vocab_size = _batched_top_k_from_logits(
        logits, top_k, top_p, temperature
    )

    distributions: list[SparseDistribution] = []
    for row_idx in range(token_rows.shape[0]):
        keep = prob_rows[row_idx] > 0
        distributions.append(
            SparseDistribution(
                token_rows[row_idx, keep],
                prob_rows[row_idx, keep],
                vocab_size,
            )
        )
    return distributions


def sparse_distribution_from_logits(
    logits_1d: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
) -> SparseDistribution | None:
    """Build a single :class:`SparseDistribution` from 1-D logits."""
    if temperature <= 0.0 or top_k <= 0:
        return None
    dists = sparse_distributions_from_logits(
        logits_1d.reshape(1, -1),
        temperature,
        top_k,
        top_p,
    )
    if dists is None or not dists:
        return None
    return dists[0]
