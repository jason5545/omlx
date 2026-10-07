# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Decode-shaped multi-row (1 < L <= 15, causal; MTP verify): a fused
    2-pass kernel that unpacks each KV token once per chunk of two rows and
    scores the chunk against it (MSE codecs, issue #2215); on GPUs with
    matrix units (M5 and later) a pass 1 that scores every row at once with
    MetalPerformancePrimitives matmul2d replaces it for 4-bit, D=256 states
    up to 48 rows; outside its
    envelope the L rows are folded into the GQA repeat dimension so the
    codecs' decode kernels apply, with the causal tail mask injected between
    key scoring and the value weighted sum — one lazy pass over the KV, no
    dequantize
  - Prefill (L>1): on M5 and later, chunks of at least 128 rows dequantize
    the cache once and run MLX's fused SDPA (``_dequantized_prefill_attention``);
    otherwise tiled quantized attention first for long contexts,
    cache.prefill_attention() first for short contexts, then dequantized SDPA

mlx-vlm's qwen3_5 left-padded helper (B>1 decode and MTP verify) is patched
separately onto the same multi-row routes, with each row's left padding
applied (``_patch_vlm_target_verify_attention``).
"""

import logging
import re
from functools import cache, lru_cache
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_LONG_PREFILL_QUANTIZED_THRESHOLD = 8192
_LONG_PREFILL_QUERY_BLOCK_SIZE = 256
_LONG_PREFILL_KEY_CHUNK_SIZE = 16384
# Prefill chunks of at least this many query rows dequantize the cache once
# and run MLX's fused SDPA on the matrix units instead of the tiled quantized
# scan (Python loop over 256-row x 16384-key blocks with an eval each).
# Measured on M5 Max, SAQ shape (24q/4kv, D=256, 4-bit MSE), per layer:
# 100k keys L=2048 639 -> 164 ms, L=512 156 -> 57, L=128 49 -> 33; 32k
# L=2048 221 -> 52. At L=64 both take 32 ms and the dequantized copy is pure
# memory cost, so shorter chunks keep the tiled scan.
_DEQUANT_PREFILL_MIN_Q_LEN = 128
# Ceiling for the per-call dequantized copy (float32 K+V plus the cast to the
# query dtype). B=1 at 262k tokens with 4 kv heads x 256 needs 3 GiB; a wider
# batch above the ceiling keeps the bounded tiled scan.
_DEQUANT_PREFILL_MAX_BYTES = 4 * 1024**3
_DEQUANT_PREFILL_ENABLED = True
# MLX < 0.32.2 has no force_fused=; without it MLX may pick the unfused fp32
# score matrix, so such a runtime keeps the tiled scan. Latched on first use.
_NATIVE_FORCE_FUSED = True
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
# Tokens each KV-loop iteration unpacks and scores together: independent
# unpack/score chains for ILP, and one online-softmax rescale per group. One
# layer at 131k tokens, L=2: 2.33 ms at 1 token, 1.91 at 2, 1.84 at 3, 2.02
# at 4 (registers); 32k: 0.77 -> 0.66 ms at 3.
_FUSED_MULTIROW_TOKENS_PER_ITER = 3
# Matrix-unit pass 1 (_nax_multirow_pass1_kernel). Measured on M5 Max: 8x8
# simdgroup_matrix peaks at ~14 TFLOPS, matmul2d on the matrix units at ~60
# (16x32x32 half/bf16 operands; any float operand drops it to ~15). The
# switch exists so tests can exercise the portable kernel on such GPUs.
_NAX_MULTIROW_ENABLED = True
# Tokens a threadgroup scores per step: one 32-token tile per simdgroup and
# row fragment. The step's score rows live in threadgroup memory, so with
# three 16-row fragments (48 rows: 6 repeats x L <= 8) the kernel sits near
# the 32 KB limit; wider calls keep the portable kernel.
_NAX_MULTIROW_CHUNK = 128
_NAX_MULTIROW_MAX_ROW_FRAGS = 3
# Split half operands into hi + lo halves (QK: q_hi.k_hi + q_lo.k_hi +
# q_hi.k_lo; PV: p.v_hi + p.v_lo) so scores keep float precision. Plain half
# drifts with peaked attention: queries x16 (logits ~50) put the output 1.2e-2
# off a CPU float32 reference, the split 5e-5 (portable kernel 4e-5). Costs
# 131k tokens, L=2: 0.35 -> 0.53 ms per layer (batched).
_NAX_MULTIROW_SPLIT_HALF = True
# Also split the softmax weights P (adds p_lo.v_hi). Without it the half
# rounding of P left the kernel 3e-6 off the reference at 131k (portable
# 4e-8, with it 2.3e-7), and on SAQ's real verify path (131k) the next-token
# probabilities moved up to 1.5% from the portable kernel's, 5-8x the 0.2-0.3%
# that reordering the portable kernel's sums moves them; with it 0.35%.
# Costs 0-6% of the verify forward (131k: L=2 52.6 -> 52.7 ms, L=5 64.3 ->
# 68.1).
_NAX_MULTIROW_SPLIT_P = True


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
    the grid's y axis; QRows must be a multiple of RowsPer. The KV loop takes
    ``_FUSED_MULTIROW_TOKENS_PER_ITER`` tokens per iteration and finishes the
    remainder one token at a time.

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
    group = _FUSED_MULTIROW_TOKENS_PER_ITER
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
        name = f"omlx_tq_mse_multirow_padded_2pass1_strided_g{group}_k{key_bits}_v{val_bits}_d{dim}"
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
        name = f"omlx_tq_mse_multirow_2pass1_strided_g{group}_k{key_bits}_v{val_bits}_d{dim}"
        extra_inputs = []

    # The main KV loop takes `group` tokens (t, t + Blocks, ...) per iteration;
    # the per-token loop after it finishes the remainder.
    nl = "\n            "
    tokens = range(group)
    group_vis = "".join(
        f"""
            bool vis{j}[RowsPer];
            for (int r = 0; r < RowsPer; r++) {{
                vis{j}[r] = r >= t{j} - (int)token_count + QRows - row_base;
                if constexpr (HasMask)
                    vis{j}[r] = vis{j}[r] && mask[
                        ((MaskB == 1 ? 0 : (int)batch_idx) * MaskT
                         + (MaskT == 1 ? 0 : row_base + r)) * (int)token_count + t{j}];
                any_vis = any_vis || vis{j}[r];
            }}"""
        if padded
        else f"""
            int first{j} = t{j} - (int)token_count + QRows - row_base;"""
        for j in tokens
    )
    group_unpack = "".join(
        f"""
            U kn{j} = static_cast<U>(k_nm[t{j} * k_nm_step]);
            auto kb{j} = (const device uint8_t*)(k_pk + t{j} * k_pk_step)
                + k_byte_base;
            U k_el{j}[qk_per_thread];
            {nl.join(f"k_el{j}[{i}] = {e.replace('kb[', f'kb{j}[')};" for i, e in enumerate(k_exprs))}
            U vn{j} = static_cast<U>(v_nm[t{j} * v_nm_step]);
            auto vb{j} = (const device uint8_t*)(v_pk + t{j} * v_pk_step)
                + v_byte_base;
            U v_el{j}[v_per_thread];
            {nl.join(f"v_el{j}[{i}] = {e.replace('vb[', f'vb{j}[')};" for i, e in enumerate(v_exprs))}"""
        for j in tokens
    )
    row_nl = "\n                "
    group_score = row_nl.join(
        f"U s{j} = simd_sum(d{j}) * kn{j};"
        f" bool see{j} = {f'vis{j}[r]' if padded else f'r >= first{j}'};"
        f" if (see{j}) m = max(m, s{j});"
        for j in tokens
    )
    group_weight = row_nl.join(
        f"U e{j} = see{j} ? fast::exp(s{j} - m) : 0.0f; U w{j} = e{j} * vn{j};"
        for j in tokens
    )
    group_loop = f"""
        // KV loop, {group} tokens per iteration: unpack each token once, score
        // the chunk's rows against all of them, one rescale per group.
        int t = {t_start};
        for (; t + {group - 1} * Blocks < (int)token_count; t += {group} * Blocks) {{
            {nl.join(f"int t{j} = t + {j} * Blocks;" for j in tokens)}{"" if not padded else f"{nl}bool any_vis = false;"}{group_vis}{f"{nl}if (!any_vis){nl}    continue;" if padded else ""}{group_unpack}
            for (int r = 0; r < RowsPer; r++) {{
                {" ".join(f"U d{j} = 0;" for j in tokens)}
                for (int i = 0; i < qk_per_thread; i++) {{
                    {" ".join(f"d{j} += q[r][i] * k_el{j}[i];" for j in tokens)}
                }}
                U m = max_score[r];
                {group_score}
                U factor = fast::exp(max_score[r] - m);
                {group_weight}
                max_score[r] = m;
                sum_exp_score[r] = sum_exp_score[r] * factor + {" + ".join(f"e{j}" for j in tokens)};
                for (int i = 0; i < v_per_thread; i++)
                    o[r][i] = o[r][i] * factor + {" + ".join(f"w{j} * v_el{j}[i]" for j in tokens)};
            }}
        }}"""

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
        auto bqh = batch_idx * key_norms_shape[1] * RepeatCount
            + kv_head_idx * RepeatCount + gqa_idx;

        // The cache hands attention its states as views sliced to the
        // written length of a larger preallocated buffer. Walk them by
        // stride rather than have MLX copy them contiguous on every call;
        // each token's packed words stay contiguous (token-axis slices).
        auto k_nm = key_norms + batch_idx * key_norms_strides[0]
            + kv_head_idx * key_norms_strides[1];
        auto k_pk = key_packed + batch_idx * key_packed_strides[0]
            + kv_head_idx * key_packed_strides[1];
        auto v_nm = val_norms + batch_idx * val_norms_strides[0]
            + kv_head_idx * val_norms_strides[1];
        auto v_pk = val_packed + batch_idx * val_packed_strides[0]
            + kv_head_idx * val_packed_strides[1];
        auto k_nm_step = key_norms_strides[2];
        auto k_pk_step = key_packed_strides[2];
        auto v_nm_step = val_norms_strides[2];
        auto v_pk_step = val_packed_strides[2];{pad_setup}

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

{group_loop}

        // Remaining tokens, one per iteration
        for (; t < (int)token_count; t += Blocks) {{{row_visibility}
            U kn = static_cast<U>(k_nm[t * k_nm_step]);
            auto kb = (const device uint8_t*)(k_pk + t * k_pk_step)
                + k_byte_base;
            U k_el[qk_per_thread];
            {k_lines}

            auto vb = (const device uint8_t*)(v_pk + t * v_pk_step)
                + v_byte_base;
            U vn = static_cast<U>(v_nm[t * v_nm_step]);
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
        # The KV states are read by stride (see the kernel); the caller makes
        # every other input contiguous.
        ensure_row_contiguous=False,
    )


@cache
def _nax_available() -> bool:
    """Whether the GPU has the matrix units MetalPerformancePrimitives'
    matmul2d runs on (Apple GPU family 17, M5, and later). MLX checks the
    same thing internally (``is_nax_available``) but does not expose it."""
    try:
        if not mx.metal.is_available():
            return False
        device_info = getattr(mx, "device_info", None) or mx.metal.device_info
        arch = str(device_info().get("architecture", ""))
    except Exception:
        return False
    match = re.match(r"applegpu_g(\d+)", arch)
    return bool(match) and int(match.group(1)) >= 17


def _nax_multirow_blocks(total: int) -> int:
    """Pass-1 block count for the matrix-unit kernel: a power of two giving
    at most 2048 tokens per block, clamped to [32, 512] (few distinct
    template values). Its blocks are contiguous token ranges with a per-
    threadgroup setup, so it wants fewer, longer blocks than the portable
    kernel's table; 16..128 blocks measured within noise at 4k, 32k and
    131k tokens."""
    needed = -(-max(total, 1) // 2048)
    return min(512, max(32, 1 << (needed - 1).bit_length()))


@cache
def _nax_multirow_pass1_kernel(row_frags: int, padded: bool = False):
    """Pass 1 of the fused multi-row MSE verify attention on matrix units.

    Same outputs as ``_fused_mse_multirow_2pass1_kernel`` (per row, per
    block: out_acc, out_sums, out_maxs), so mlx-vlm's pass 2 is reused. 4-bit
    keys and values, D = 256. All RepeatCount * QRows rows of a kv head are
    scored together, padded to ``row_frags`` 16-row fragments.

    A threadgroup (4 * row_frags simdgroups) owns one contiguous token range
    of one (batch row, kv head) and walks it _NAX_MULTIROW_CHUNK tokens at a
    time in three phases:
      1. scores: each simdgroup takes one 32-token tile for one fragment,
         S = Q K^T as 16x32x32 matmul2d ops, the packed key codes unpacked
         straight into the right-operand cooperative tensor; key norms,
         causal tail and padding mask applied, rows written to threadgroup
         memory.
      2. online softmax per row over the step; P' = exp(S - max) * v_norm
         written back in place, rescale factor per row.
      3. each simdgroup owns two 16x32 output blocks: O = O * alpha + P' V,
         value codes unpacked straight into the right operand.
    A matmul2d cooperative-tensor lane holds rows fm and fm + 8 and four
    consecutive columns 4 * fc + e of each 16-wide fragment (MLX's
    BaseNAXFrag layout, checked against get_multidimensional_index). Logical
    k index 32 * jp + 16 * g + 4 * fc + e maps to packed dim 64 * fc + 8 * jp
    + 4 * g + e and output column 32 * J + 16 * f + 4 * fc + e to dim
    64 * fc + 8 * J + 4 * f + e, so a lane reads one contiguous 32-byte run
    (8 words) of each token's packed row.

    ``padded`` builds the left-padded batch variant, as in the portable
    kernel: each block starts at its row's ``pads`` entry and an optional
    bool ``mask`` (HasMask; (MaskB, 1, MaskT, token_count)) hides columns.
    Every variant starts the running max at a finite floor: blocks with no
    visible token for a row are normal here (contiguous ranges, padding).
    """
    if not 1 <= row_frags <= _NAX_MULTIROW_MAX_ROW_FRAGS or not _nax_available():
        return None
    from mlx_vlm import turboquant as _tq

    if not _tq._metal_available():
        return None

    frags = row_frags
    rows = 16 * frags
    simdgroups = 4 * frags
    threads = 32 * simdgroups
    chunk = _NAX_MULTIROW_CHUNK
    split = _NAX_MULTIROW_SPLIT_HALF
    split_p = split and _NAX_MULTIROW_SPLIT_P
    # Phase-2 threads per score row (TPR * rows <= threads, within a simdgroup)
    tpr = 1
    while tpr * 2 * rows <= threads and tpr < 32:
        tpr *= 2

    nl = "\n                "
    # Phase 1: unpack 32 key dims of the four token slots this lane holds
    # (tokens fm, fm + 8, fm + 16, fm + 24 of the tile) into kb[16g + 4s + e].
    k_unpack = []
    for s in range(4):
        k_unpack.append(f"{{ uint w = kw[{s}][jp >> 2][jp & 3];")
        for g in range(2):
            base = 16 * g + 4 * s
            for tensor, lut in [("kb", "k_lut")] + ([("kb_lo", "k_lut_lo")] if split else []):
                k_unpack.append(
                    f"  {{ half2 a = {lut}[(w >> {16 * g}) & 0xFF]; half2 b = {lut}[(w >> {16 * g + 8}) & 0xFF];"
                    f" {tensor}[{base}] = a.x; {tensor}[{base + 1}] = a.y; {tensor}[{base + 2}] = b.x; {tensor}[{base + 3}] = b.y; }}"
                )
        k_unpack.append("}")
    # Queries (float) -> qa[8g + 4ii + e] (+ the half rounding residue)
    q_load = []
    for g in range(2):
        for ii in range(2):
            for e in range(4):
                i = 8 * g + 4 * ii + e
                if split:
                    q_load.append(
                        f"{{ float x = q{ii}[8 * jp + {4 * g + e}]; half h = half(x);"
                        f" qa[{i}] = h; qa_lo[{i}] = half(x - float(h)); }}"
                    )
                else:
                    q_load.append(f"qa[{i}] = half(q{ii}[8 * jp + {4 * g + e}]);")
    qk_runs = ["qk_op.run(qa, kb, S);"]
    if split:
        qk_runs += ["qk_op.run(qa_lo, kb, S);", "qk_op.run(qa, kb_lo, S);"]

    # Phase 3: each simdgroup's two output blocks are items 2 * sg and
    # 2 * sg + 1 over (column block J, row fragment); with an even fragment
    # count both share J, so the value codes are unpacked once. All value
    # words of the step are loaded before any unpack.
    groups = [[0, 1]] if frags % 2 == 0 else [[0], [1]]
    pv_lines = []
    for gi, items in enumerate(groups):
        pv_lines.append(f"uint vw{gi}[4];")
        pv_lines.append(
            f"for (int q = 0; q < 4; q++) vw{gi}[q] = v_pk[tt[q] * v_pk_step + 8 * fc + col{items[0]}];"
        )
    for gi, items in enumerate(groups):
        pv_lines.append("{")
        for g in range(2):
            for ii in range(2):
                pv_lines.append(f"  {{ uint w = vw{gi}[{2 * g + ii}];")
                for f in range(2):
                    base = 16 * g + 8 * f + 4 * ii
                    for tensor, lut in [("pb", "v_lut")] + ([("pb_lo", "v_lut_lo")] if split else []):
                        pv_lines.append(
                            f"    {{ half2 a = {lut}[(w >> {16 * f}) & 0xFF]; half2 b = {lut}[(w >> {16 * f + 8}) & 0xFF];"
                            f" {tensor}[{base}] = a.x; {tensor}[{base + 1}] = a.y; {tensor}[{base + 2}] = b.x; {tensor}[{base + 3}] = b.y; }}"
                        )
                pv_lines.append("  }")
        for k in items:
            for g in range(2):
                for ii in range(2):
                    base = 8 * g + 4 * ii
                    pv_lines.append(
                        f"  {{ float4 p = *((threadgroup float4*)(scores + (frag{k} * 16 + fm + {8 * ii}) * SST"
                        f" + kc * 32 + {16 * g} + 4 * fc));"
                    )
                    for e, comp in enumerate("xyzw"):
                        if split_p:
                            pv_lines.append(
                                f"    {{ half h = half(p.{comp}); pa[{base + e}] = h;"
                                f" pa_lo[{base + e}] = half(p.{comp} - float(h)); }}"
                            )
                        else:
                            pv_lines.append(f"    pa[{base + e}] = half(p.{comp});")
                    pv_lines.append("  }")
            pv_lines.append(f"  pv_op.run(pa, pb, o{k});")
            if split_p:
                pv_lines.append(f"  pv_op.run(pa_lo, pb, o{k});")
            if split:
                pv_lines.append(f"  pv_op.run(pa, pb_lo, o{k});")
        pv_lines.append("}")

    out_lines = []
    for k in range(2):
        for f in range(2):
            for ii in range(2):
                base = 8 * f + 4 * ii
                out_lines.append(
                    f"{{ int m = frag{k} * 16 + fm + {8 * ii};"
                    f" if (m < RC) *((device float4*)(out_acc + ((q_row0 + m) * Blocks + block) * D"
                    f" + 64 * fc + 8 * col{k} + {4 * f})) = float4(o{k}[{base}], o{k}[{base + 1}],"
                    f" o{k}[{base + 2}], o{k}[{base + 3}]); }}"
                )

    if padded:
        # Left-padded batch row: never touch its padding columns
        t_begin = "max(block * span, (int)pads[batch_idx])"
        mask_check = """
                    if constexpr (HasMask)
                        visible = visible && mask[
                            ((MaskB == 1 ? 0 : batch_idx) * MaskT
                             + (MaskT == 1 ? 0 : m % QRows)) * T + t];"""
        name_kind = "padded_"
        extra_inputs = ["pads", "mask"]
    else:
        t_begin = "block * span"
        mask_check = ""
        name_kind = ""
        extra_inputs = []
    lo_luts = (
        "threadgroup half2 k_lut_lo[256];\n        threadgroup half2 v_lut_lo[256];"
        if split
        else ""
    )
    lo_lut_fill = (
        """
            float k0 = key_codebook[i & 15], k1 = key_codebook[i >> 4];
            float v0 = val_codebook[i & 15], v1 = val_codebook[i >> 4];
            k_lut_lo[i] = half2(half(k0 - float(half(k0))), half(k1 - float(half(k1))));
            v_lut_lo[i] = half2(half(v0 - float(half(v0))), half(v1 - float(half(v1))));"""
        if split
        else ""
    )
    lo_tensors_qk = (
        "QA qa_lo = qk_op.template get_left_input_cooperative_tensor<half, half, float>();\n"
        "                QB kb_lo = qk_op.template get_right_input_cooperative_tensor<half, half, float>();"
        if split
        else ""
    )
    lo_tensors_pv = (
        "PB pb_lo = pv_op.template get_right_input_cooperative_tensor<half, half, float>();"
        if split
        else ""
    ) + (
        "\n                PA pa_lo = pv_op.template get_left_input_cooperative_tensor<half, half, float>();"
        if split_p
        else ""
    )

    source = f"""
        constexpr int D = Dim;
        constexpr int RC = RepeatCount * QRows;
        constexpr int CH = {chunk};
        constexpr int SST = CH + 4;
        constexpr int TPR = {tpr};
        static_assert(D == 256, "matrix-unit verify kernel: D == 256");
        static_assert(RC <= {rows}, "rows exceed the kernel's row fragments");

        threadgroup float scores[{rows} * SST];
        threadgroup float v_norm_sh[CH];
        threadgroup half2 k_lut[256];
        threadgroup half2 v_lut[256];
        {lo_luts}
        threadgroup float alpha_sh[{rows}];

        const int tid = thread_position_in_threadgroup.x;
        const int sg = tid >> 5;
        const int lane = tid & 31;
        const int block = threadgroup_position_in_grid.x;
        const int kv_head = threadgroup_position_in_grid.y;
        const int batch_idx = threadgroup_position_in_grid.z;
        const int n_kv = key_norms_shape[1];
        const int T = key_norms_shape[2];

        // matmul2d cooperative-tensor lane coordinates (see the docstring)
        const int fm = ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
        const int fc = ((lane >> 3) & 1) * 2 + (lane & 1);

        // This block's contiguous token range (32-token aligned split)
        const int span = ((T + Blocks - 1) / Blocks + 31) & ~31;
        const int t_begin = {t_begin};
        const int t_end = min(T, block * span + span);

        // KV states are views sliced out of the cache's preallocated buffer:
        // walk them by stride (each token's packed words stay contiguous).
        auto k_nm = key_norms + batch_idx * key_norms_strides[0]
            + kv_head * key_norms_strides[1];
        auto k_pk = key_packed + batch_idx * key_packed_strides[0]
            + kv_head * key_packed_strides[1];
        auto v_nm = val_norms + batch_idx * val_norms_strides[0]
            + kv_head * val_norms_strides[1];
        auto v_pk = val_packed + batch_idx * val_packed_strides[0]
            + kv_head * val_packed_strides[1];
        const int k_nm_step = key_norms_strides[2];
        const int k_pk_step = key_packed_strides[2];
        const int v_nm_step = val_norms_strides[2];
        const int v_pk_step = val_packed_strides[2];

        // Packed byte -> its two codebook entries
        for (int i = tid; i < 256; i += {threads}) {{
            k_lut[i] = half2(half(key_codebook[i & 15]), half(key_codebook[i >> 4]));
            v_lut[i] = half2(half(val_codebook[i & 15]), half(val_codebook[i >> 4]));{lo_lut_fill}
        }}
        const int q_row0 = (batch_idx * n_kv + kv_head) * RC;

        constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 32, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
        constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 32, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
        using QA = decltype(qk_op.template get_left_input_cooperative_tensor<half, half, float>());
        using QB = decltype(qk_op.template get_right_input_cooperative_tensor<half, half, float>());
        using PA = decltype(pv_op.template get_left_input_cooperative_tensor<half, half, float>());
        using PB = decltype(pv_op.template get_right_input_cooperative_tensor<half, half, float>());

        // Phase-1 item: tile qk_tile of each step, fragment qk_frag
        const int qk_tile = sg / {frags};
        const int qk_frag = sg % {frags};
        // Phase-3 items: (column block col, fragment frag) for 2sg, 2sg + 1
        const int col0 = (2 * sg) / {frags};
        const int frag0 = (2 * sg) % {frags};
        const int col1 = (2 * sg + 1) / {frags};
        const int frag1 = (2 * sg + 1) % {frags};
        auto o0 = pv_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<PA>, metal::remove_addrspace_t<PB>, float>();
        auto o1 = pv_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<PA>, metal::remove_addrspace_t<PB>, float>();
        // Element loops over cooperative tensors: one tensor per loop,
        // unrolled. With o0 and o1 zeroed in one plain loop the kernel ran
        // 9% slower at 131k tokens (same output).
        #pragma unroll
        for (int i = 0; i < 16; i++)
            o0[i] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            o1[i] = 0.0f;

        // Phase-2 row ownership and running softmax stats
        const int p2_row = tid / TPR;
        const int p2_part = tid % TPR;
        const bool p2_active = p2_row < {rows};
        float m_run = -3.402823466e+38f;
        float l_run = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int cb = t_begin; cb < t_end; cb += CH) {{
            const int n_valid = min(CH, t_end - cb);
            const int n_cols = (n_valid + 31) & ~31;

            // Phase 1: scores
            for (int c = tid; c < CH; c += {threads}) {{
                int t = cb + c;
                v_norm_sh[c] = t < t_end ? float(v_nm[t * v_nm_step]) : 0.0f;
            }}
            const int tb = cb + qk_tile * 32;
            if (tb < t_end) {{
                uint4 kw[4][2];
                for (int s = 0; s < 4; s++) {{
                    int t = min(tb + 16 * (s >> 1) + 8 * (s & 1) + fm, T - 1);
                    auto p = (const device uint4*)(k_pk + t * k_pk_step) + 2 * fc;
                    kw[s][0] = p[0];
                    kw[s][1] = p[1];
                }}
                auto S = qk_op.template get_destination_cooperative_tensor<
                    metal::remove_addrspace_t<QA>, metal::remove_addrspace_t<QB>, float>();
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    S[i] = 0.0f;
                const int m0 = qk_frag * 16 + fm;
                auto q0 = queries + (q_row0 + min(m0, RC - 1)) * D + 64 * fc;
                auto q1 = queries + (q_row0 + min(m0 + 8, RC - 1)) * D + 64 * fc;
                QB kb = qk_op.template get_right_input_cooperative_tensor<half, half, float>();
                QA qa = qk_op.template get_left_input_cooperative_tensor<half, half, float>();
                {lo_tensors_qk}
                #pragma unroll
                for (int jp = 0; jp < 8; jp++) {{
                {nl.join(k_unpack)}
                {nl.join(q_load)}
                {nl.join(qk_runs)}
                }}
                // Key norms, causal tail (and padding mask), then store
                float kn[8];
                for (int i = 0; i < 8; i++)
                    kn[i] = float(k_nm[min(tb + 16 * (i >> 2) + 4 * fc + (i & 3), T - 1) * k_nm_step]);
                #pragma unroll
                for (int i = 0; i < 16; i++) {{
                    int m = m0 + 8 * ((i >> 2) & 1);
                    int t = tb + 16 * (i >> 3) + 4 * fc + (i & 3);
                    bool visible = t < t_end && m < RC && t <= T - QRows + (m % QRows);{mask_check}
                    scores[m * SST + (t - cb)] =
                        visible ? S[i] * kn[4 * (i >> 3) + (i & 3)] : -INFINITY;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: online softmax per row over this step
            if (p2_active) {{
                float step_max = -INFINITY;
                for (int c = p2_part; c < n_cols; c += TPR)
                    step_max = max(step_max, scores[p2_row * SST + c]);
                for (int o = TPR / 2; o > 0; o >>= 1)
                    step_max = max(step_max, simd_shuffle_xor(step_max, o));
                float m_new = max(m_run, step_max);
                float alpha = fast::exp(m_run - m_new);
                float sum = 0.0f;
                for (int c = p2_part; c < n_cols; c += TPR) {{
                    float p = fast::exp(scores[p2_row * SST + c] - m_new);
                    sum += p;
                    scores[p2_row * SST + c] = p * v_norm_sh[c];
                }}
                for (int o = TPR / 2; o > 0; o >>= 1)
                    sum += simd_shuffle_xor(sum, o);
                l_run = l_run * alpha + sum;
                m_run = m_new;
                if (p2_part == 0)
                    alpha_sh[p2_row] = alpha;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: O = O * alpha + P' V
            {{
                float a00 = alpha_sh[frag0 * 16 + fm], a01 = alpha_sh[frag0 * 16 + fm + 8];
                float a10 = alpha_sh[frag1 * 16 + fm], a11 = alpha_sh[frag1 * 16 + fm + 8];
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    o0[i] *= (i & 4) ? a01 : a00;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    o1[i] *= (i & 4) ? a11 : a10;
                PA pa = pv_op.template get_left_input_cooperative_tensor<half, half, float>();
                PB pb = pv_op.template get_right_input_cooperative_tensor<half, half, float>();
                {lo_tensors_pv}
                #pragma unroll
                for (int kc = 0; kc < CH / 32; kc++) {{
                    if (kc * 32 < n_cols) {{
                        int tt[4];
                        for (int q = 0; q < 4; q++)
                            tt[q] = min(cb + kc * 32 + 16 * (q >> 1) + fm + 8 * (q & 1), T - 1);
                        {(nl + "        ").join(pv_lines)}
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Per-row partial results for this block
        if (p2_active && p2_part == 0 && p2_row < RC) {{
            out_sums[(q_row0 + p2_row) * Blocks + block] = l_run;
            out_maxs[(q_row0 + p2_row) * Blocks + block] = m_run;
        }}
        {(nl[:-8]).join(out_lines)}
    """

    return mx.fast.metal_kernel(
        name=f"omlx_tq_mse_multirow_nax_{name_kind}2pass1_f{frags}_s{int(split)}{int(split_p)}",
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
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensure_row_contiguous=False,
    )


def _fused_multirow_mse_attention(
    real_cache, queries, keys_state, values_state, scale, total, pads=None, mask=None
):
    """Run MTP verify attention through the fused multi-row kernel.

    Pass 1 is the matrix-unit kernel (``_nax_multirow_pass1_kernel``) when
    the GPU has matrix units and the call fits it (4-bit K/V, D=256, at most
    48 rows per kv head); otherwise the portable kernel
    (``_fused_mse_multirow_2pass1_kernel``). Both feed mlx-vlm's pass 2.

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
    pass2 = _tq._fused_mse_decode_2pass_2_kernel()
    if pass2 is None:
        return None

    nax_pass1 = None
    row_frags = -(-(n_repeats * L) // 16)
    if (
        _NAX_MULTIROW_ENABLED
        and key_codec.bits == 4
        and value_codec.bits == 4
        and D == 256
    ):
        nax_pass1 = _nax_multirow_pass1_kernel(row_frags, padded)

    if nax_pass1 is not None:
        # All rows in one pass: no row chunks, no prepended rows.
        lead = 0
    else:
        pass1 = _fused_mse_multirow_2pass1_kernel(
            int(key_codec.bits), int(value_codec.bits), D, padded
        )
        if pass1 is None:
            return None
        # Each simdgroup scores a chunk of rows_per rows. A row count that
        # does not split evenly gets copies of the first row prepended: they
        # sit just before it in the causal order, so the real rows keep their
        # positions, and their outputs are dropped. A short chunk with a
        # runtime row count cost as much as a full one and more (131k
        # tokens: L=3 4.5 ms, L=4 3.9 ms).
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

    if nax_pass1 is not None:
        num_blocks = _nax_multirow_blocks(total)
    # Portable kernel: same block split table as turboquant's 2-pass decode
    # dispatch.
    elif total <= 8192:
        num_blocks = 64
    elif total <= 32768:
        num_blocks = 128
    elif total <= 65536:
        num_blocks = 256
    else:
        num_blocks = 512

    # The pass-1 kernel walks the KV states by stride (they are views sliced
    # out of the cache's preallocated buffer; a contiguous copy cost 0.57 ms
    # per layer per call at 131k tokens). Everything else must be contiguous.
    inputs = [
        mx.contiguous(q_rot_flat),
        keys_state.norms,
        keys_state.indices,
        mx.contiguous(key_codec.codebook),
        values_state.norms,
        values_state.indices,
        mx.contiguous(value_codec.codebook),
    ]
    template = [
        ("Dim", D),
        ("RepeatCount", n_repeats),
        ("QRows", rows),
        ("Blocks", num_blocks),
    ]
    if padded:
        if pads is None:
            pads = mx.zeros((B,), dtype=mx.int32)
        has_mask = mask is not None
        inputs += [
            mx.contiguous(pads),
            mx.contiguous(mask) if has_mask else mx.array([True]),
        ]
        template += [
            ("HasMask", has_mask),
            ("MaskB", mask.shape[0] if has_mask else 1),
            ("MaskT", mask.shape[2] if has_mask else 1),
        ]

    n_rows = B * n_q_heads * rows
    outputs = dict(
        output_shapes=[
            (n_rows * num_blocks, D),
            (n_rows * num_blocks,),
            (n_rows * num_blocks,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    if nax_pass1 is not None:
        # One threadgroup of 4 simdgroups per row fragment for each (block,
        # kv head, batch row).
        threads = 32 * 4 * row_frags
        out_acc, out_sums, out_maxs = nax_pass1(
            inputs=inputs,
            template=template,
            grid=(num_blocks * threads, n_kv_heads, B),
            threadgroup=(threads, 1, 1),
            **outputs,
        )
    else:
        template += [("RowsPer", rows_per), ("RowChunks", row_chunks)]
        out_acc, out_sums, out_maxs = pass1(
            inputs=inputs,
            template=template,
            grid=(n_kv_heads * 32, B * row_chunks * n_repeats, num_blocks),
            threadgroup=(32, n_repeats, 1),
            **outputs,
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


def _dequantized_prefill_attention(real_cache, queries, keys, values, scale, mask):
    """Dequantize the cache once and run MLX's fused SDPA, or return None.

    TurboQuant's own long-prefill route scans the packed states in Python
    blocks and reaches about 4 TFLOPS on M5 Max; the fused kernel over the
    dequantized states runs on the matrix units (about 24-31 TFLOPS). The
    states are the same quantized values either way: against an unquantized
    cache both routes sit about 21.8% off (random 32k-token states), within
    0.02 points of each other; the bf16 operands add about 0.2% mean
    relative error over the float32 scan, the precision class of any
    unquantized bf16 cache.
    """
    global _NATIVE_FORCE_FUSED

    if not (_DEQUANT_PREFILL_ENABLED and _NATIVE_FORCE_FUSED):
        return None
    if queries.shape[-2] < _DEQUANT_PREFILL_MIN_Q_LEN or not _nax_available():
        return None
    # Lazy: nothing is materialized unless the fused call below is evaluated.
    dequantized_keys, dequantized_values = real_cache.dequantize(
        keys_state=keys,
        values_state=values,
    )
    elements = dequantized_keys.size + dequantized_values.size
    if elements * (4 + queries.dtype.size) > _DEQUANT_PREFILL_MAX_BYTES:
        return None
    try:
        return mx.fast.scaled_dot_product_attention(
            queries,
            dequantized_keys.astype(queries.dtype),
            dequantized_values.astype(queries.dtype),
            scale=scale,
            mask=mask,
            force_fused=True,
        )
    except TypeError:
        _NATIVE_FORCE_FUSED = False
        logger.warning(
            "TurboQuant: mlx %s has no force_fused= (0.32.2+); long prefill "
            "keeps the tiled quantized scan",
            getattr(mx, "__version__", "?"),
        )
    except ValueError:
        # A layout the fused kernel rejects; the tiled scan covers it.
        pass
    return None


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
            result = _dequantized_prefill_attention(
                real_cache, queries, keys, values, scale, mask
            )
            if result is not None:
                return result
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
