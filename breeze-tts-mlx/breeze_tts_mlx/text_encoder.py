from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx import nn

from .layers import GemmaRMSNorm, RotaryEmbedding


class ScaledTextEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        eoi_token_index: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.eoi_embedding = mx.zeros((hidden_size,))
        self.scale = hidden_size**0.5
        self.eoi_token_index = int(eoi_token_index)

    def __call__(self, input_ids: mx.array) -> mx.array:
        embedded = self.embedding(input_ids) * self.scale
        return mx.where(
            (input_ids == self.eoi_token_index)[..., None],
            self.eoi_embedding,
            embedded,
        )


class T5GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class T5GemmaAttention(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        layer_type: str,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.scale = float(config["query_pre_attn_scalar"]) ** -0.5
        bias = bool(config.get("attention_bias", False))
        self.q_proj = nn.Linear(hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden_size, bias=bias)
        eps = float(config["rms_norm_eps"])
        self.q_norm = GemmaRMSNorm(self.head_dim, eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps)
        self.layer_type = layer_type
        self._rope = rope

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        q = self.q_norm(
            self.q_proj(hidden_states).reshape(
                batch, length, self.n_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, length, self.n_kv_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q, k = self._rope.apply(q, k, position_embeddings)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class T5GemmaLayer(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        layer_type: str,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        eps = float(config["rms_norm_eps"])
        self.self_attn = T5GemmaAttention(config, layer_type=layer_type, rope=rope)
        self.pre_self_attn_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.mlp = T5GemmaMLP(hidden_size, int(config["intermediate_size"]))
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.attention_type = layer_type
        self.compute_dtype = mx.float16

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
    ) -> mx.array:
        # T5Gemma was trained in BF16. Its residual stream can exceed FP16's
        # range before the final normalization. Keep quantized matmuls and
        # attention in FP16, but accumulate residual and post-norm paths in FP32.
        residual = hidden_states.astype(mx.float32)
        attended = self.self_attn(
            self.pre_self_attn_layernorm(hidden_states).astype(self.compute_dtype),
            position_embeddings=position_embeddings,
            mask=mask,
        )
        hidden_states = residual + self.post_self_attn_layernorm(
            attended.astype(mx.float32)
        )
        residual = hidden_states
        feed_forward = self.mlp(
            self.pre_feedforward_layernorm(hidden_states).astype(self.compute_dtype)
        )
        return residual + self.post_feedforward_layernorm(
            feed_forward.astype(mx.float32)
        )


class T5GemmaTextEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if config.get("attn_logit_softcapping") is not None:
            raise ValueError("text attention soft-capping is not implemented")
        hidden_size = int(config["hidden_size"])
        self.embed_tokens = ScaledTextEmbedding(
            int(config["vocab_size"]),
            hidden_size,
            eoi_token_index=int(config["eoi_token_index"]),
        )
        ropes: dict[str, RotaryEmbedding] = {}
        for layer_type, rope_config in config["rope_parameters"].items():
            linear_factor = (
                float(rope_config["factor"])
                if rope_config.get("rope_type", "default") == "linear"
                else None
            )
            ropes[layer_type] = RotaryEmbedding(
                int(config["head_dim"]),
                base=float(rope_config["rope_theta"]),
                linear_factor=linear_factor,
            )
        layer_types = list(config["layer_types"])
        self.layers = [
            T5GemmaLayer(config, layer_type=layer_type, rope=ropes[layer_type])
            for layer_type in layer_types
        ]
        self.norm = GemmaRMSNorm(hidden_size, float(config["rms_norm_eps"]))
        self.sliding_window = int(config["sliding_window"])
        self.compute_dtype = mx.float16

    def set_compute_dtype(self, dtype: mx.Dtype) -> None:
        self.compute_dtype = dtype
        for layer in self.layers:
            layer.compute_dtype = dtype

    def _mask(self, attention_mask: mx.array, layer_type: str) -> mx.array | None:
        batch, length = attention_mask.shape
        valid_keys = attention_mask.astype(mx.bool_)[:, None, None, :]
        if layer_type == "full_attention":
            if bool(mx.all(valid_keys).item()):
                return None
            return mx.broadcast_to(valid_keys, (batch, 1, length, length))
        if layer_type != "sliding_attention":
            raise ValueError(f"Unknown text attention type: {layer_type}")
        query = mx.arange(length)[:, None]
        key = mx.arange(length)[None, :]
        distance = query - key
        left = (self.sliding_window + 1) // 2
        right = self.sliding_window // 2 + 1
        local = ((distance >= 0) & (distance < left)) | (
            (distance < 0) & (-distance < right)
        )
        return valid_keys & local[None, None, :, :]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array,
        position_ids: mx.array,
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        masks: dict[str, mx.array | None] = {}
        position_embeddings: dict[str, tuple[mx.array, mx.array]] = {}
        for layer in self.layers:
            if layer.attention_type not in masks:
                masks[layer.attention_type] = self._mask(
                    attention_mask, layer.attention_type
                )
                position_embeddings[layer.attention_type] = (
                    layer.self_attn._rope.embeddings(position_ids, self.compute_dtype)
                )
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings[layer.attention_type],
                mask=masks[layer.attention_type],
            )
        return self.norm(hidden_states)
