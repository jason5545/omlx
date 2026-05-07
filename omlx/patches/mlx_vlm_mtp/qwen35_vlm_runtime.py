# SPDX-License-Identifier: Apache-2.0
"""Runtime Native MTP hooks for mlx-vlm's Qwen3.5/Qwen3.6 model.

mlx-vlm carries a separate Qwen3.5 language implementation from mlx-lm.
The server-side BatchGenerator MTP dispatch can drive VLM requests safely
once that language model exposes the same contract:

- TextConfig preserves ``mtp_num_hidden_layers`` from config.json.
- LanguageModel attaches ``self.mtp`` when the oMLX MTP active flag is set.
- ``__call__(..., return_hidden=True, n_confirmed=...)`` returns logits plus
  pre-final-norm hidden states.
- ``mtp_forward`` and ``make_mtp_cache`` are available.

The patch keeps mRoPE handling in mlx-vlm's LanguageModel and only changes
decode internals enough for the two-token draft verify pass to support SSM
rollback on rejection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    """Apply mlx-vlm Qwen3.5 runtime MTP hooks. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import config as q35cfg
        from mlx_vlm.models.qwen3_5 import language as q35lang
        from mlx_vlm.models.qwen3_5 import qwen3_5 as q35vlm
    except Exception as e:
        logger.debug("mlx_vlm qwen3_5 runtime not importable: %s", e)
        return False

    if hasattr(q35lang.LanguageModel, "mtp_forward") and not hasattr(
        q35lang.LanguageModel, "_omlx_mtp_runtime_patched"
    ):
        _APPLIED = True
        q35lang.LanguageModel._omlx_mtp_runtime_patched = "upstream"
        return True

    _patch_text_config(q35cfg)
    _register_mtp_classes(q35lang)
    _patch_gated_delta_net(q35lang)
    _patch_decoder_layer(q35lang)
    _patch_qwen_model(q35lang)
    _patch_language_model(q35lang)
    _patch_outer_model(q35vlm)

    _APPLIED = True
    logger.info("Patched mlx_vlm Qwen3.5/3.6 runtime for Native MTP")
    return True


def _patch_text_config(q35cfg: Any) -> None:
    cls = q35cfg.TextConfig
    if hasattr(cls, "_omlx_mtp_from_dict_patched"):
        return

    original_from_dict = cls.from_dict.__func__

    def patched_from_dict(config_cls, params):
        instance = original_from_dict(config_cls, params)
        instance.mtp_num_hidden_layers = int(
            params.get("mtp_num_hidden_layers", 0) or 0
        )
        instance.num_experts = int(params.get("num_experts", 0) or 0)
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._omlx_mtp_from_dict_patched = True


def _register_mtp_classes(q35lang: Any) -> None:
    if hasattr(q35lang, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Attention = q35lang.Qwen3_5Attention
    MLP = q35lang.Qwen3_5MLP
    KVCache = q35lang.KVCache
    create_attention_mask = q35lang.create_attention_mask

    class MTPDecoderLayer(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [KVCache() for _ in self.layers]

            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    q35lang.MTPDecoderLayer = MTPDecoderLayer
    q35lang.MTPModule = MTPModule


def _patch_gated_delta_net(q35lang: Any) -> None:
    cls = q35lang.Qwen3_5GatedDeltaNet
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    def _process_chunk(
        self,
        qkv_chunk,
        a_chunk,
        b_chunk,
        conv_state,
        ssm_state,
        ssm_mask=None,
        lengths=None,
    ):
        B, S_chunk = qkv_chunk.shape[:2]
        conv_in = mx.concatenate([conv_state, qkv_chunk], axis=1)
        n_keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, S_chunk)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_in, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_in[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_in))
        q, k, v = [
            t.reshape(B, S_chunk, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, new_ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_chunk,
            b_chunk,
            self.A_log,
            self.dt_bias,
            ssm_state,
            ssm_mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state, q, k, v, conv_in

    def __call__(
        self,
        inputs: Any,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        B, S, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, -1, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        ssm_state = cache[1] if cache else None

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        if n_confirmed > 0 and n_confirmed < S:
            mask_c = mask[:, :n_confirmed] if mask is not None else None
            mask_d = mask[:, n_confirmed:] if mask is not None else None
            out_c, conv_c, ssm_c, *_ = self._process_chunk(
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                mask_c,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)
            out_d, conv_f, ssm_f, *_ = self._process_chunk(
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                mask_d,
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = cache.lengths if cache is not None else None
            out, conv_f, ssm_f, q, k, v, conv_in = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )
            if gdn_sink is not None:
                gdn_sink.append(
                    (
                        q,
                        k,
                        v,
                        a,
                        b,
                        self.A_log,
                        self.dt_bias,
                        ssm_state,
                        mask,
                        conv_in,
                        self.conv_kernel_size,
                    )
                )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    cls._process_chunk = _process_chunk
    cls.__call__ = __call__
    cls._omlx_mtp_runtime_patched = True


def _patch_decoder_layer(q35lang: Any) -> None:
    cls = q35lang.Qwen3_5DecoderLayer
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        gdn_sink=None,
        n_confirmed: int = 0,
    ):
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
                gdn_sink=gdn_sink,
                n_confirmed=n_confirmed,
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    cls.__call__ = __call__
    cls._omlx_mtp_runtime_patched = True


def _patch_qwen_model(q35lang: Any) -> None:
    cls = q35lang.Qwen3_5Model
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    create_attention_mask = q35lang.create_attention_mask
    create_ssm_mask = q35lang.create_ssm_mask

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
        gdn_sink=None,
        n_confirmed: int = 0,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            call_code = getattr(getattr(layer, "__call__", None), "__code__", None)
            call_args = set(call_code.co_varnames) if call_code is not None else set()
            layer_kwargs = {}
            if "position_ids" in call_args:
                layer_kwargs["position_ids"] = position_ids
            if "gdn_sink" in call_args:
                layer_kwargs["gdn_sink"] = gdn_sink
            if "n_confirmed" in call_args:
                layer_kwargs["n_confirmed"] = n_confirmed
            h = layer(h, layer_mask, c, **layer_kwargs)
            if hidden_sink is not None and i in capture_set:
                hidden_sink.append(h)

        return h

    cls.__call__ = __call__
    cls._omlx_mtp_runtime_patched = True


def _patch_language_model(q35lang: Any) -> None:
    cls = q35lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    import mlx.core as mx

    LanguageModelOutput = q35lang.LanguageModelOutput
    original_init = cls.__init__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        from ..mlx_lm_mtp import is_mtp_active

        if n_mtp > 0 and is_mtp_active():
            self.mtp = q35lang.MTPModule(args)

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        cache_offset = 0
        cache_offsets = None
        if cache and cache[self.model.fa_idx] is not None:
            c0 = cache[self.model.fa_idx]
            cache_offset = c0._idx if hasattr(c0, "_idx") else c0.offset
            if (
                isinstance(c0.offset, mx.array)
                and c0.offset.ndim > 0
                and c0.offset.size > 1
            ):
                cache_offsets = mx.maximum(c0.offset, 0)

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            batch_size, seq_length = inputs.shape

            if (
                (
                    cache is not None
                    and cache[self.model.fa_idx] is not None
                    and (cache_offset == 0)
                )
                or self._rope_deltas is None
                or cache is None
            ):
                if (
                    self._position_ids is not None
                    and self._position_ids.shape[1] == batch_size
                    and self._position_ids.shape[-1] >= cache_offset + seq_length
                ):
                    position_ids = self._position_ids[
                        :, :, cache_offset : cache_offset + seq_length
                    ]
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                if cache_offsets is not None and cache_offsets.size >= batch_size:
                    offsets = cache_offsets[:batch_size]
                    rope_deltas = (
                        rope_deltas_kw
                        if rope_deltas_kw is not None
                        else self._rope_deltas
                    )
                    if rope_deltas.shape[0] > batch_size:
                        rope_deltas = rope_deltas[:batch_size]
                    delta = (offsets + rope_deltas.squeeze(-1))[:, None]
                else:
                    delta = mx.array(
                        cache_offset + self._rope_deltas if cache is not None else 0
                    )
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        hidden_sink = [] if capture_layer_ids is not None else None
        gdn_sink = [] if capture_layer_ids is not None else None

        hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            gdn_sink=gdn_sink,
            n_confirmed=n_confirmed,
        )
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        if return_hidden:
            return logits, hidden
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(mtp_out)
        return self.lm_head(mtp_out)

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [q35lang.KVCache() for _ in self.mtp.layers]
        return []

    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mtp.fc"):
                return False
            return True

        if int(getattr(self.args, "mtp_num_hidden_layers", 0) or 0) <= 0:
            return None
        return predicate

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.quant_predicate = property(quant_predicate)
    cls._omlx_mtp_runtime_patched = True


def _patch_outer_model(q35vlm: Any) -> None:
    cls = q35vlm.Model
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_runtime_patched = True
