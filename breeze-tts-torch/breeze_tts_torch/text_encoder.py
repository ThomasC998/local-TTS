"""The T5Gemma text encoder that turns the prompt into conditioning vectors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import GemmaRMSNorm, RotaryEmbedding, attend


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
        self.eoi_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.scale = hidden_size**0.5
        self.eoi_token_index = int(eoi_token_index)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids) * self.scale
        return torch.where(
            (input_ids == self.eoi_token_index)[..., None],
            self.eoi_embedding.to(embedded.dtype),
            embedded,
        )


class T5GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gated * self.up_proj(x))


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
        # Gemma scales by a configured scalar, not by 1/sqrt(head_dim).
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        q = self.q_norm(
            self.q_proj(hidden_states).reshape(
                batch, length, self.n_heads, self.head_dim
            )
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, length, self.n_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        v = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = self._rope.apply(q, k, position_embeddings)
        output = attend(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(1, 2).reshape(batch, length, -1)
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
        self.compute_dtype = torch.float16

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # T5Gemma was trained in BF16 and its residual stream can exceed FP16's
        # range before the final normalization. Keep the matmuls and attention
        # in the compute dtype, but accumulate residual and post-norm in FP32.
        # This is not caution copied from the MLX port: drop it and long prompts
        # come back as inf.
        residual = hidden_states.float()
        attended = self.self_attn(
            self.pre_self_attn_layernorm(hidden_states).to(self.compute_dtype),
            position_embeddings=position_embeddings,
            mask=mask,
        )
        hidden_states = residual + self.post_self_attn_layernorm(attended.float())
        residual = hidden_states
        feed_forward = self.mlp(
            self.pre_feedforward_layernorm(hidden_states).to(self.compute_dtype)
        )
        return residual + self.post_feedforward_layernorm(feed_forward.float())


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
        self.layers = nn.ModuleList(
            [
                T5GemmaLayer(config, layer_type=layer_type, rope=ropes[layer_type])
                for layer_type in layer_types
            ]
        )
        self.norm = GemmaRMSNorm(hidden_size, float(config["rms_norm_eps"]))
        self.sliding_window = int(config["sliding_window"])
        self.compute_dtype = torch.float16

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.compute_dtype = dtype
        for layer in self.layers:
            layer.compute_dtype = dtype

    def _mask(
        self, attention_mask: torch.Tensor, layer_type: str
    ) -> torch.Tensor | None:
        batch, length = attention_mask.shape
        valid_keys = attention_mask.bool()[:, None, None, :]
        if layer_type == "full_attention":
            if bool(valid_keys.all()):
                return None
            return valid_keys.expand(batch, 1, length, length)
        if layer_type != "sliding_attention":
            raise ValueError(f"Unknown text attention type: {layer_type}")
        device = attention_mask.device
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        distance = query - key
        left = (self.sliding_window + 1) // 2
        right = self.sliding_window // 2 + 1
        local = ((distance >= 0) & (distance < left)) | (
            (distance < 0) & (-distance < right)
        )
        return valid_keys & local[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        masks: dict[str, torch.Tensor | None] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            if layer.attention_type not in masks:
                masks[layer.attention_type] = self._mask(
                    attention_mask, layer.attention_type
                )
                position_embeddings[layer.attention_type] = layer.self_attn._rope.embeddings(
                    position_ids, self.compute_dtype
                )
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings[layer.attention_type],
                mask=masks[layer.attention_type],
            )
        return self.norm(hidden_states)
