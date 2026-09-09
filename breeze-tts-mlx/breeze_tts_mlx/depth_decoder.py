from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx import nn

from .layers import DecoderLayer, DecoderSpec, KVCache, RotaryEmbedding


class CodebookHeads(nn.Module):
    def __init__(self, hidden_size: int, num_codebooks: int, vocab_size: int) -> None:
        super().__init__()
        self.heads = [
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_codebooks - 1)
        ]

    def __call__(self, hidden_states: mx.array, head_index: int) -> mx.array:
        return self.heads[head_index](hidden_states)


class BreezeDepthDecoder(nn.Module):
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
            qk_norm=False,
            rope_scaling=config.get("rope_scaling"),
        )
        rope = RotaryEmbedding(
            spec.head_dim,
            base=spec.rope_theta,
            rope_scaling=spec.rope_scaling,
        )
        self._rope = rope
        self.layers = [
            DecoderLayer(spec) for _ in range(int(config["num_hidden_layers"]))
        ]
        self.norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.input_projection = nn.Linear(
            int(config["audio_embed_size"]), spec.hidden_size, bias=False
        )
        self.codebooks_head = CodebookHeads(
            spec.hidden_size,
            int(config["num_codebooks"]),
            int(config["vocab_size"]),
        )

    def make_cache(self) -> list[KVCache]:
        return [KVCache(step=32) for _ in self.layers]

    def _forward(
        self,
        inputs_embeds: mx.array,
        *,
        position_ids: mx.array,
        cache: list[KVCache],
    ) -> mx.array:
        hidden_states = self.input_projection(inputs_embeds)
        offset = cache[0].offset if cache else 0
        mask: str | None = (
            "causal" if hidden_states.shape[1] > 1 and offset == 0 else None
        )
        position_embeddings = self._rope.embeddings(position_ids, hidden_states.dtype)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(hidden_states, position_embeddings, layer_cache, mask)
        return self.norm(hidden_states)

    def begin_frame(
        self,
        backbone_hidden: mx.array,
        first_code_embedding: mx.array,
        cache: list[KVCache],
    ) -> mx.array:
        inputs = mx.stack([backbone_hidden, first_code_embedding], axis=1)
        positions = mx.broadcast_to(
            mx.array([[0, 1]], dtype=mx.int32),
            (inputs.shape[0], 2),
        )
        hidden = self._forward(inputs, position_ids=positions, cache=cache)
        return self.codebooks_head(hidden[:, -1, :], 0)

    def step_frame(
        self,
        code_embedding: mx.array,
        *,
        codebook_index: int,
        cache: list[KVCache],
    ) -> mx.array:
        # Feeding codebook N predicts codebook N+1 with head N.
        position = codebook_index + 1
        inputs = code_embedding[:, None, :]
        positions = mx.full((inputs.shape[0], 1), position, dtype=mx.int32)
        hidden = self._forward(inputs, position_ids=positions, cache=cache)
        return self.codebooks_head(hidden[:, -1, :], codebook_index)
