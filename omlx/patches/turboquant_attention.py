# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Decode-shaped multi-row (1 < L <= 15, causal; MTP verify): a fused
    2-pass kernel that unpacks each KV token once per chunk of two rows and
    scores the chunk against it (MSE codecs, issue #2215); outside its
    envelope the L rows are folded into the GQA repeat dimension so the
    codecs' decode kernels apply, with the causal tail mask injected between
    key scoring and the value weighted sum — one lazy pass over the KV, no
    dequantize
  - Prefill (L>1): tiled quantized attention first for long contexts;
    cache.prefill_attention() first for short contexts; then dequantized SDPA

mlx-vlm's qwen3_5 left-padded helper (B>1 decode and MTP verify) is patched
separately onto the same multi-row routes, with each row's left padding
applied (``_patch_vlm_target_verify_attention``).
"""

import logging
from functools import cache, lru_cache
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_LONG_PREFILL_QUANTIZED_THRESHOLD = 8192
_LONG_PREFILL_QUERY_BLOCK_SIZE = 256
_LONG_PREFILL_KEY_CHUNK_SIZE = 16384
# MTP verify is a decode-shaped multi-row call (q_len = 1 + draft depth <= 9).
# Above this floor a multi-row call is genuine (chunked) prefill.
_DECODE_MULTIROW_MAX_Q_LEN = 15
# The repeat kernels unroll per-repeat register arrays, so folding is only a
# win while n_repeats * q_len stays under the register-pressure knee
# (measured: 24 fine, 30+ loses to single-chunk quantized_attention).
_MAX_FOLDED_REPEATS = 24
# Softmax-denominator floor, matching turboquant's quantized_attention.
_STATS_EPS = 1e-6
# Fused multi-row verify kernel envelope. Below the token floor the fold path
# is already sub-0.2ms and the 2-pass block split has too few tokens per block.
# Above it the fused kernel takes every verify width up to
# _DECODE_MULTIROW_MAX_Q_LEN: with two-row chunks it beat the fold and one-shot
# routes at every L from 4 to 15 (4k tokens: 0.33-0.76 ms vs 0.93-1.15 ms;
# 131k: 4.1-7.7 ms at L=4-8 vs 23-27 ms one-shot).
_FUSED_MULTIROW_MIN_TOKENS = 2048
# Rows one simdgroup scores per KV unpack. Wider verify calls split their rows
# into chunks of this many across the grid: each chunk re-reads the KV, but
# the q/o register arrays stay at the two-row size. Holding all rows in one
# simdgroup spilled past two (131k tokens, 24q/4kv, D=256, fp32 queries:
# L=3 5.0 ms and L=4 8.3 ms per layer against 2.2 ms at L=2).
_FUSED_MULTIROW_ROWS_PER_SIMDGROUP = 2


@cache
def _fused_mse_multirow_2pass1_kernel(
    key_bits: int, val_bits: int, dim: int, padded: bool = False
):
    """Pass 1 of the fused multi-row MSE verify attention.

    Derived from turboquant's ``_fused_mse_decode_2pass_1_kernel`` with one
    structural change: each simdgroup unpacks a token's K/V codebook entries
    once and reuses them across its chunk of RowsPer query rows (per-row
    online softmax stats, causal tail applied inline). The upstream decode
    kernels re-unpack the KV per query row, so MTP verify paid the unpack ALU
    L times over (issue #2215). QRows rows split into RowChunks chunks along
    the grid's y axis; QRows must be a multiple of RowsPer.

    ``padded`` builds the left-padded batch variant (a separate kernel; the
    unpadded source is unchanged). Each batch row starts its KV loop at its
    ``pads`` entry, so padding columns are never unpacked, and an optional
    bool ``mask`` (HasMask; shape (MaskB, 1, MaskT, token_count)) further
    hides columns per row. The running max starts at a finite floor: a block
    that sees no token for a row (short rows leave whole blocks empty) must
    hand pass 2 a finite max, or pass 2 computes exp(-inf - -inf) = NaN.
    """
    from mlx_vlm import turboquant as _tq

    if not _tq._metal_available() or key_bits <= 0 or val_bits <= 0:
        return None
    if dim < 32 or dim % 32 != 0:
        return None

    elems_per_lane = dim // 32
    k_misaligned = (elems_per_lane * key_bits) % 8 != 0
    v_misaligned = (elems_per_lane * val_bits) % 8 != 0
    k_exprs = _tq._gen_unrolled_extract(
        key_bits, elems_per_lane, "key_codebook", "k_bit_off" if k_misaligned else ""
    )
    v_exprs = _tq._gen_unrolled_extract(
        val_bits, elems_per_lane, "val_codebook", "v_bit_off" if v_misaligned else ""
    )
    v_exprs = [e.replace("kb[", "vb[") for e in v_exprs]
    k_lines = "\n            ".join(
        f"k_el[{i}] = {expr};" for i, expr in enumerate(k_exprs)
    )
    v_lines = "\n            ".join(
        f"v_el[{i}] = {expr};" for i, expr in enumerate(v_exprs)
    )

    if padded:
        pad_setup = """

        // Left-padded batch row: jump to this block's first column >= pad
        int pad = pads[batch_idx];
        int t_start = block_idx;
        if (pad > t_start)
            t_start += ((pad - t_start + Blocks - 1) / Blocks) * Blocks;"""
        max_init = "-3.402823466e+38f"
        t_start = "t_start"
        row_visibility = """
            // Chunk row r is query row row_base + r, at global position
            // token_count - QRows + row_base + r; token t is invisible to
            // rows before it and to rows the caller's mask hides. Skip the
            // unpack if none sees t.
            int first_row = t - (int)token_count + QRows - row_base;
            bool vis[RowsPer];
            bool any_vis = false;
            for (int r = 0; r < RowsPer; r++) {
                vis[r] = r >= first_row;
                if constexpr (HasMask)
                    vis[r] = vis[r] && mask[
                        ((MaskB == 1 ? 0 : (int)batch_idx) * MaskT
                         + (MaskT == 1 ? 0 : row_base + r)) * (int)token_count + t];
                any_vis = any_vis || vis[r];
            }
            if (!any_vis)
                continue;
"""
        first_row_block = "\n"
        row_visible = "vis[r]"
        name = f"omlx_tq_mse_multirow_padded_2pass1_k{key_bits}_v{val_bits}_d{dim}"
        extra_inputs = ["pads", "mask"]
    else:
        pad_setup = ""
        max_init = "-INFINITY"
        t_start = "block_idx"
        row_visibility = ""
        first_row_block = """
            // Chunk row r is query row row_base + r, at global position
            // token_count - QRows + row_base + r; token t is invisible to
            // chunk rows r < first_row.
            int first_row = t - (int)token_count + QRows - row_base;
"""
        row_visible = "r >= first_row"
        name = f"omlx_tq_mse_multirow_2pass1_k{key_bits}_v{val_bits}_d{dim}"
        extra_inputs = []

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        // Thread identity — matches turboquant's mse_sdpa_2pass_1 layout
        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y / RowChunks;
        auto block_idx = threadgroup_position_in_grid.z;
        // This threadgroup's chunk of RowsPer query rows (QRows is a
        // multiple of RowsPer; the caller pads short calls).
        int row_base = (threadgroup_position_in_grid.y % RowChunks) * RowsPer;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = batch_idx * kv_heads + kv_head_idx;
        auto bqh = batch_idx * kv_heads * RepeatCount
            + kv_head_idx * RepeatCount + gqa_idx;

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;{pad_setup}

        // This chunk's pre-rotated queries for the (kv_head, repeat) pair
        thread U q[RowsPer][qk_per_thread];
        for (int r = 0; r < RowsPer; r++) {{
            auto qr = queries + (bqh * QRows + row_base + r) * Dim
                + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++)
                q[r][i] = static_cast<U>(qr[i]);
        }}

        thread U o[RowsPer][v_per_thread] = {{}};
        U max_score[RowsPer];
        U sum_exp_score[RowsPer];
        for (int r = 0; r < RowsPer; r++) {{
            max_score[r] = {max_init};
            sum_exp_score[r] = 0;
        }}

        // Byte/bit offset for this lane's first element
        int k_bit_start = simd_lid * qk_per_thread * {key_bits};
        int v_bit_start = simd_lid * v_per_thread * {val_bits};
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        {"int k_bit_off = k_bit_start & 7;" if k_misaligned else ""}
        {"int v_bit_off = v_bit_start & 7;" if v_misaligned else ""}

        // KV loop: unpack each token once, score the chunk's rows against it
        for (int t = {t_start}; t < (int)token_count; t += Blocks) {{{row_visibility}
            U kn = static_cast<U>(k_nm[t]);
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth)
                + k_byte_base;
            U k_el[qk_per_thread];
            {k_lines}

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth)
                + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);
            U v_el[v_per_thread];
            {v_lines}
{first_row_block}            for (int r = 0; r < RowsPer; r++) {{
                U dot = 0;
                for (int i = 0; i < qk_per_thread; i++)
                    dot += q[r][i] * k_el[i];
                U score = simd_sum(dot) * kn;
                if ({row_visible}) {{
                    U new_max = max(max_score[r], score);
                    U factor = fast::exp(max_score[r] - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score[r] = new_max;
                    sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;
                    for (int i = 0; i < v_per_thread; i++)
                        o[r][i] = o[r][i] * factor + exp_score * v_el[i] * vn;
                }}
            }}
        }}

        // Write per-row partial results for this block
        for (int r = 0; r < RowsPer; r++) {{
            auto row_out = bqh * QRows + row_base + r;
            if (simd_lid == 0) {{
                out_sums[row_out * Blocks + block_idx] = sum_exp_score[r];
                out_maxs[row_out * Blocks + block_idx] = max_score[r];
            }}
            for (int i = 0; i < v_per_thread; i++)
                out_acc[(row_out * Blocks + block_idx) * Dim
                    + simd_lid * v_per_thread + i] = static_cast<U>(o[r][i]);
        }}
    """

    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            *extra_inputs,
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


def _fused_multirow_mse_attention(
    real_cache, queries, keys_state, values_state, scale, total, pads=None, mask=None
):
    """Run MTP verify attention through the fused multi-row kernel.

    ``pads`` (int32, one entry per batch row) and ``mask`` (bool,
    (1|B, 1, 1|L, total)) select the left-padded kernel variant; with both
    None the unpadded kernel runs exactly as before.

    Returns None when the states/codecs are outside the kernel envelope
    (non-MSE codecs, fractional bits, mismatched dims); the caller falls
    back to the fold / one-shot paths.
    """
    from mlx_vlm import turboquant as _tq

    key_codec = getattr(real_cache, "key_codec", None)
    value_codec = getattr(real_cache, "value_codec", None)
    if not (
        isinstance(key_codec, _tq._TurboQuantMSECodec)
        and isinstance(value_codec, _tq._TurboQuantMSECodec)
    ):
        return None
    if not (
        isinstance(keys_state, _tq.TurboQuantMSEState)
        and isinstance(values_state, _tq.TurboQuantMSEState)
    ):
        return None
    if key_codec.bits != int(key_codec.bits) or value_codec.bits != int(
        value_codec.bits
    ):
        return None

    B, n_q_heads, L, D = queries.shape
    if key_codec.dim != D or value_codec.dim != D:
        return None
    if keys_state.norms.shape[0] != B:
        return None
    n_kv_heads = keys_state.norms.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    padded = pads is not None or mask is not None
    pass1 = _fused_mse_multirow_2pass1_kernel(
        int(key_codec.bits), int(value_codec.bits), D, padded
    )
    pass2 = _tq._fused_mse_decode_2pass_2_kernel()
    if pass1 is None or pass2 is None:
        return None

    # Each simdgroup scores a chunk of rows_per rows. A row count that does
    # not split evenly gets copies of the first row prepended: they sit just
    # before it in the causal order, so the real rows keep their positions,
    # and their outputs are dropped. A short chunk with a runtime row count
    # cost as much as a full one and more (131k tokens: L=3 4.5 ms, L=4
    # 3.9 ms).
    rows_per = min(L, _FUSED_MULTIROW_ROWS_PER_SIMDGROUP)
    row_chunks = -(-L // rows_per)
    lead = row_chunks * rows_per - L
    if lead > total - L:
        return None
    if lead:
        queries = mx.concatenate([queries[:, :, :1]] * lead + [queries], axis=2)
        if mask is not None and mask.shape[2] == L:
            mask = mx.concatenate([mask[:, :, :1]] * lead + [mask], axis=2)
    rows = L + lead

    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, rows, D)
    q_rot = key_codec.prepare_queries(grouped)
    q_rot_flat = q_rot.reshape(B * n_kv_heads * n_repeats * rows, D)

    # Same block split table as turboquant's 2-pass decode dispatch.
    if total <= 8192:
        num_blocks = 64
    elif total <= 32768:
        num_blocks = 128
    elif total <= 65536:
        num_blocks = 256
    else:
        num_blocks = 512

    inputs = [
        q_rot_flat,
        keys_state.norms,
        keys_state.indices,
        key_codec.codebook,
        values_state.norms,
        values_state.indices,
        value_codec.codebook,
    ]
    template = [
        ("Dim", D),
        ("RepeatCount", n_repeats),
        ("QRows", rows),
        ("Blocks", num_blocks),
        ("KPackedWidth", keys_state.indices.shape[-1]),
        ("VPackedWidth", values_state.indices.shape[-1]),
    ]
    if padded:
        if pads is None:
            pads = mx.zeros((B,), dtype=mx.int32)
        has_mask = mask is not None
        inputs += [pads, mask if has_mask else mx.array([True])]
        template += [
            ("HasMask", has_mask),
            ("MaskB", mask.shape[0] if has_mask else 1),
            ("MaskT", mask.shape[2] if has_mask else 1),
        ]

    template += [("RowsPer", rows_per), ("RowChunks", row_chunks)]

    n_rows = B * n_q_heads * rows
    out_acc, out_sums, out_maxs = pass1(
        inputs=inputs,
        template=template,
        grid=(n_kv_heads * 32, B * row_chunks * n_repeats, num_blocks),
        threadgroup=(32, n_repeats, 1),
        output_shapes=[
            (n_rows * num_blocks, D),
            (n_rows * num_blocks,),
            (n_rows * num_blocks,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = pass2(
        inputs=[out_acc, out_sums, out_maxs],
        template=[("Dim", D), ("Blocks", num_blocks)],
        grid=(n_rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_rows, D)],
        output_dtypes=[mx.float32],
    )[0]

    out_rotated = out.reshape(B, n_kv_heads, n_repeats, rows, D)
    output = value_codec._rotate_inverse(out_rotated)
    output = output.reshape(B, n_q_heads, rows, D)
    if lead:
        output = output[:, :, lead:]
    return output.astype(queries.dtype)


def _decode_multirow_quantized_attention(
    real_cache, queries, keys, values, scale, mask="causal"
):
    """Wider verify rows: one-shot quantized_attention over the whole KV.

    A single query block and a single key chunk turn quantized_attention's
    chunked online softmax into one pass — its einsum path amortizes the
    key unpack across rows, staying flat in q_len where the folded decode
    kernels hit register spill.
    """
    if not hasattr(real_cache, "quantized_attention"):
        return None
    old_query_block_size = getattr(real_cache, "prefill_query_block_size", None)
    old_key_chunk_size = getattr(real_cache, "prefill_key_chunk_size", None)
    try:
        real_cache.prefill_query_block_size = queries.shape[-2]
        real_cache.prefill_key_chunk_size = real_cache.decode_key_chunk_size
        return real_cache.quantized_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=scale,
            mask=mask,
        )
    finally:
        if old_query_block_size is not None:
            real_cache.prefill_query_block_size = old_query_block_size
        if old_key_chunk_size is not None:
            real_cache.prefill_key_chunk_size = old_key_chunk_size


def _padded_causal_visibility(pads, mask, total, q_len):
    """Bool (1|B, 1, q_len, total): row i of batch b sees key t iff
    pads[b] <= t <= total - q_len + i, and ``mask`` (if any) allows it."""
    cols = mx.arange(total)
    visible = (cols[None, :] <= mx.arange(total - q_len, total)[:, None])[
        None, None
    ]
    if pads is not None:
        visible = visible & (cols[None, None, None, :] >= pads[:, None, None, None])
    if mask is not None:
        visible = visible & mask
    return visible


def _decode_multirow_attention(
    real_cache, queries, keys, values, scale, pads=None, mask=None
):
    """Causal multi-row attention over TurboQuant states in one lazy pass.

    MTP verify would otherwise fall into the prefill fallbacks, which
    re-dequantize or chunk-scan the whole cache with per-chunk eval syncs
    on every verify cycle (issue #2127 class). MSE-codec states take the
    fused multi-row kernel (one KV unpack shared across the L rows, issue
    #2215); other codecs fold the L rows into the repeat dimension so the
    L==1 decode kernels stay applicable (repeat count is a kernel template
    parameter), with the causal tail mask applied on the raw scores before
    the value weighted sum. Returns None when the states don't fit; the
    caller falls back to the generic paths.

    ``pads`` (int32 per batch row) and ``mask`` (bool, (1|B, 1, 1|L,
    total)) add left-padded batch visibility on top of the causal tail;
    each path honors them (padded fused kernel, masked fold, masked
    one-shot). With both None the behavior is the plain causal one.
    """
    from mlx_vlm.turboquant import TurboQuantSplitState

    from ..turboquant_kv import _state_length

    keys_state = real_cache._unwrap(keys)
    values_state = real_cache._unwrap(values)
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = (
        keys_state.low.norms.shape[1]
        if isinstance(keys_state, TurboQuantSplitState)
        else keys_state.norms.shape[1]
    )
    n_repeats = n_q_heads // n_kv_heads
    total = _state_length(keys_state)
    if total < L:
        return None
    padded = pads is not None or mask is not None
    if total > _FUSED_MULTIROW_MIN_TOKENS:
        try:
            result = _fused_multirow_mse_attention(
                real_cache,
                queries,
                keys_state,
                values_state,
                scale,
                total,
                pads=pads,
                mask=mask,
            )
        except Exception:
            logger.debug(
                "TurboQuant fused multi-row kernel failed; using fold path",
                exc_info=True,
            )
            result = None
        if result is not None:
            return result
    visible = _padded_causal_visibility(pads, mask, total, L) if padded else None
    if n_repeats * L > _MAX_FOLDED_REPEATS:
        return _decode_multirow_quantized_attention(
            real_cache,
            queries,
            keys,
            values,
            scale,
            mask="causal" if visible is None else visible,
        )

    folded = (queries * scale).reshape(B, n_kv_heads, n_repeats * L, 1, D)
    prepared = real_cache.key_codec.prepare_queries(folded)
    scores = real_cache.key_codec.score_prepared(prepared, keys_state)

    # (B, H, R*L, 1, T): fold index r*L + i is the row at global position
    # total - L + i; mask the keys after it (and, for left-padded batches,
    # the row's padding columns).
    scores = scores.reshape(B, n_kv_heads, n_repeats, L, total)
    if visible is None:
        q_pos = mx.arange(total - L, total)
        causal = mx.arange(total)[None, :] <= q_pos[:, None]
    else:
        causal = visible[:, :, None]
    scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
    scores = scores.reshape(B, n_kv_heads, n_repeats * L, 1, total)

    out, denom, _ = real_cache.value_codec.weighted_sum_stats_from_scores(
        scores, values_state
    )
    out = out / mx.maximum(denom[..., None], _STATS_EPS)
    out = out.reshape(B, n_q_heads, L, real_cache.value_codec.dim)
    return out.astype(queries.dtype)


def _patch_update_eval_policy() -> None:
    """Skip the per-layer eval for decode-shaped multi-row cache appends.

    Upstream ``update_and_fetch`` forces ``mx.eval`` whenever more than one
    token is appended — a graph-bounding measure sized for prefill chunks.
    MTP verify appends 2..9 rows per layer, so that policy serializes every
    layer of every verify cycle (~15 forced syncs/cycle). Raise the eval
    floor to prefill-sized appends; verify rows stay lazy and materialize
    at the cycle's sampling sync like the rest of the forward.
    """
    from mlx_vlm import turboquant as _tq

    cls = _tq.TurboQuantKVCache
    if getattr(cls, "_omlx_multirow_eval_patched", False):
        return

    def update_and_fetch(self, keys, values):
        # Mirror of upstream TurboQuantKVCache.update_and_fetch; the only
        # change is the eval gate (n_new > 1 -> prefill-sized appends).
        self._ensure_codecs(keys, values)

        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is None:
            new_keys = self.key_codec.quantize(keys)
            new_values = self.value_codec.quantize(values)

        new_end = self.offset + keys.shape[2]
        if self.keys is None:
            self.keys = _tq._allocate_state_like(new_keys, new_end)
            self.values = _tq._allocate_state_like(new_values, new_end)
        else:
            self.keys = _tq._reserve_state_capacity(
                self.keys, self.offset, new_end, self.cache_step
            )
            self.values = _tq._reserve_state_capacity(
                self.values, self.offset, new_end, self.cache_step
            )

        _tq._write_state(self.keys, new_keys, self.offset)
        _tq._write_state(self.values, new_values, self.offset)

        n_heads = keys.shape[1]
        n_new = keys.shape[2]

        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        if n_new > _DECODE_MULTIROW_MAX_Q_LEN or (self.offset % 50 == 0):
            mx.eval(self.keys, self.values)
        ks, vs = self.state
        return (
            _tq._QuantizedStateProxy(ks, self.offset, n_heads),
            _tq._QuantizedStateProxy(vs, self.offset, n_heads),
        )

    cls.update_and_fetch = update_and_fetch
    cls._omlx_multirow_eval_patched = True


@lru_cache(maxsize=128)
def _pads_array(pads: tuple) -> mx.array:
    return mx.array(pads, dtype=mx.int32)


def _qwen35_row_pads(q35_lang, cache, real_cache, batch_size):
    """Per-row left padding, read the way mlx-vlm's helper reads it.

    ``left_padded_decode`` forwards stash the host-side pads on each
    full-attention cache; other forwards derive them from ``left_padding``.
    Returns None when the metadata does not describe this batch.
    """
    pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
    if pads is None:
        info_fn = getattr(q35_lang, "_qwen3_5_left_padding_info", None)
        for source in (cache, real_cache):
            info = info_fn(source) if info_fn is not None else None
            if info is not None:
                pads = info[0]
                break
    if pads is None:
        return (0,) * batch_size
    pads = tuple(int(p) for p in pads)
    return pads if len(pads) == batch_size else None


def _bool_visibility_mask(mask, batch_size, q_len, total):
    """Normalize the caller's mask to (1|B, 1, 1|L, total) bool.

    Returns (ok, mask): mask is None for "no extra mask" (None / "causal");
    ok is False for masks the quantized paths do not take (additive, per
    head, other widths) — the caller then uses the dequantize fallback.
    """
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return True, None
    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_:
        return False, None
    if not 2 <= mask.ndim <= 4:
        return False, None
    mask = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
    mask_b, mask_h, mask_t, mask_s = mask.shape
    if (
        mask_b not in (1, batch_size)
        or mask_h != 1
        or mask_t not in (1, q_len)
        or mask_s != total
    ):
        return False, None
    return True, mask


def _left_padded_quantized_attention(
    real_cache, queries, keys, values, scale, pads, mask
):
    """Left-padded batch attention straight from the TurboQuant states.

    Replaces the dequantize-everything fallback for B>1 decode (L=1,
    ``left_padded_decode``) and MTP verify (L>1, array mask): those
    materialized the whole batch cache as float32 in every TurboQuant
    layer on every step, and the growing sizes defeated MLX's buffer
    reuse. Returns None when the call is outside what the quantized paths
    take; the caller keeps the dequantize fallback for those.
    """
    from ..turboquant_kv import _state_length

    B, _, L, _ = queries.shape
    if pads is None or L > _DECODE_MULTIROW_MAX_Q_LEN:
        return None
    total = _state_length(real_cache._unwrap(keys))
    if any(p < 0 or p > total - L for p in pads):
        return None
    ok, bool_mask = _bool_visibility_mask(mask, B, L, total)
    if not ok:
        return None
    pads_arr = _pads_array(pads) if any(pads) else None
    return _decode_multirow_attention(
        real_cache,
        queries,
        keys,
        values,
        scale,
        pads=pads_arr,
        mask=bool_mask,
    )


def _patch_vlm_target_verify_attention() -> None:
    """Make mlx-vlm's qwen3_5 MTP verify attention TurboQuant-safe.

    The upstream verify path slices ``keys[:, :, : prefix + i + 1, :]`` per
    draft row before calling SDPA. With TurboQuant the fetched keys/values
    are packed ``_QuantizedStateProxy`` objects that are not subscriptable,
    so every verify forward crashes (issue #2139). Route TurboQuant caches
    through one causal SDPA call instead — the TurboQuant-patched dispatcher
    handles decode-shaped multi-row natively with identical semantics (row i
    attends the first ``prefix + i + 1`` positions).

    Left-padded batches (B>1 decode with ``mask=None`` from
    ``left_padded_decode``, and B>1 verify with an array mask) run on the
    quantized states with each row's padding and causal tail applied
    (``_left_padded_quantized_attention``). The first version of this patch
    (a915729b) dequantized the whole batch cache per layer per step there,
    and for ``left_padded_decode`` it also let short rows attend to their
    padding columns (mask=None carried no padding).
    """
    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return
    if getattr(q35_lang, "_omlx_tq_target_verify_patched", False):
        return
    original = getattr(q35_lang, "_qwen3_5_left_padded_attention", None)
    if original is None:
        return

    def patched(queries, keys, values, *, cache, scale, mask):
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache

        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache
        if not isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            return original(queries, keys, values, cache=cache, scale=scale, mask=mask)

        sdpa = q35_lang.scaled_dot_product_attention
        if queries.shape[0] == 1 and not isinstance(mask, mx.array):
            return sdpa(
                queries, keys, values, cache=cache, scale=scale, mask="causal"
            )
        pads = _qwen35_row_pads(q35_lang, cache, real_cache, queries.shape[0])
        try:
            result = _left_padded_quantized_attention(
                real_cache, queries, keys, values, scale, pads, mask
            )
        except Exception:
            logger.debug(
                "TurboQuant left-padded attention failed; using dequantize",
                exc_info=True,
            )
            result = None
        if result is not None:
            return result
        # Outside the quantized envelope: dequantize once and replicate the
        # caller's per-row causal slicing on dense arrays.
        dk, dv = real_cache.dequantize(keys_state=keys, values_state=values)
        dk = dk.astype(queries.dtype)
        dv = dv.astype(queries.dtype)
        L = queries.shape[2]
        prefix_len = dk.shape[-2] - L
        if not isinstance(mask, mx.array) and pads is not None and any(pads):
            # left_padded_decode passes mask=None; the padding lives only in
            # the cache metadata, so rebuild it or short rows see padding.
            mask = _padded_causal_visibility(
                _pads_array(pads), None, dk.shape[-2], L
            )
        return mx.concatenate(
            [
                sdpa(
                    queries[:, :, i : i + 1, :],
                    dk[:, :, : prefix_len + i + 1, :],
                    dv[:, :, : prefix_len + i + 1, :],
                    cache=None,
                    scale=scale,
                    mask=(
                        mask[..., i : i + 1, : prefix_len + i + 1]
                        if isinstance(mask, mx.array) and mask.ndim >= 4
                        else None
                    ),
                )
                for i in range(L)
            ],
            axis=2,
        )

    q35_lang._qwen3_5_left_padded_attention = patched
    q35_lang._omlx_tq_target_verify_attention = patched
    q35_lang._omlx_tq_target_verify_original = original
    q35_lang._omlx_tq_target_verify_patched = True


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    try:
        _patch_update_eval_policy()
    except Exception:
        logger.debug("TurboQuant update eval-policy patch skipped", exc_info=True)

    try:
        _patch_vlm_target_verify_attention()
    except Exception:
        logger.debug(
            "TurboQuant VLM target-verify attention patch skipped", exc_info=True
        )

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache, _state_length

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if sinks is not None:
                # TurboQuant's quantized kernels do not implement attention
                # sinks. Preserve correctness by falling back to MLX's
                # sink-aware SDPA over dequantized states.
                dequantized_keys, dequantized_values = real_cache.dequantize(
                    keys_state=keys,
                    values_state=values,
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    dequantized_keys.astype(queries.dtype),
                    dequantized_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )
            if queries.shape[-2] == 1:
                # Decode (B=1 and B>1). Continuous-batching decode passes a
                # per-request left-padding array mask; the masked decode_attention
                # path runs the quantized kernels directly (no full-batch
                # dequantize per step). The RHT masked-decode fix landed upstream
                # in mlx-vlm (Blaizzy/mlx-vlm#1244, in the pinned commit).
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            if (
                queries.shape[-2] <= _DECODE_MULTIROW_MAX_Q_LEN
                and isinstance(mask, str)
                and mask == "causal"
            ):
                # Decode-shaped multi-row (MTP verify) — see helper docstring.
                try:
                    result = _decode_multirow_attention(
                        real_cache, queries, keys, values, scale
                    )
                    if result is not None:
                        return result
                except Exception:
                    logger.debug(
                        "TurboQuant multi-row decode attention failed; "
                        "falling back to prefill paths",
                        exc_info=True,
                    )
            keys_state = getattr(keys, "_state", keys)
            try:
                total_tokens = _state_length(keys_state)
            except Exception:
                total_tokens = 0
            use_tiled_first = (
                total_tokens > _LONG_PREFILL_QUANTIZED_THRESHOLD
                and hasattr(real_cache, "quantized_attention")
            )
            if use_tiled_first:
                old_query_block_size = getattr(
                    real_cache, "prefill_query_block_size", None
                )
                old_key_chunk_size = getattr(
                    real_cache, "prefill_key_chunk_size", None
                )
                try:
                    real_cache.prefill_query_block_size = (
                        _LONG_PREFILL_QUERY_BLOCK_SIZE
                    )
                    real_cache.prefill_key_chunk_size = _LONG_PREFILL_KEY_CHUNK_SIZE
                    return real_cache.quantized_attention(
                        queries,
                        keys_state=keys,
                        values_state=values,
                        scale=scale,
                        mask=mask,
                    )
                except Exception:
                    logger.debug(
                        "TurboQuant quantized prefill attention failed; "
                        "falling back to prefill_attention / dequantize+SDPA",
                        exc_info=True,
                    )
                finally:
                    if old_query_block_size is not None:
                        real_cache.prefill_query_block_size = old_query_block_size
                    if old_key_chunk_size is not None:
                        real_cache.prefill_key_chunk_size = old_key_chunk_size
            result = real_cache.prefill_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=scale,
                mask=mask,
            )
            if result is not None:
                return result
            dequantized_keys, dequantized_values = real_cache.dequantize(
                keys_state=keys,
                values_state=values,
            )
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
