from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx import nn

from .layers import (
    DecoderLayer,
    DecoderSpec,
    KVCache,
    RotaryEmbedding,
    make_causal_padding_mask,
)


class Qwen3Backbone(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        spec = DecoderSpec(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(config["rope_theta"]),
            attention_bias=bool(config.get("attention_bias", False)),
            qk_norm=True,
        )
        rope = RotaryEmbedding(spec.head_dim, base=spec.rope_theta)
        self._rope = rope
        self.layers = [
            DecoderLayer(spec) for _ in range(int(config["num_hidden_layers"]))
        ]
        self.norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        inputs_embeds: mx.array,
        *,
        attention_mask: mx.array,
        position_ids: mx.array,
        cache: list[KVCache],
    ) -> mx.array:
        if len(cache) != len(self.layers):
            raise ValueError("backbone cache layer count does not match the model")
        offset = cache[0].offset if cache else 0
        query_length = int(inputs_embeds.shape[1])
        mask = make_causal_padding_mask(
            attention_mask, query_length=query_length, cache_offset=offset
        )
        hidden_states = inputs_embeds
        position_embeddings = self._rope.embeddings(position_ids, hidden_states.dtype)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(hidden_states, position_embeddings, layer_cache, mask)
        return self.norm(hidden_states)
