from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm, whose checkpoint weight is an offset from one."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class RotaryEmbedding:
    """HF-compatible half-split RoPE with optional Llama 3 scaling."""

    def __init__(
        self,
        head_dim: int,
        *,
        base: float,
        rope_scaling: dict[str, Any] | None = None,
        linear_factor: float | None = None,
    ) -> None:
        inv_freq = [
            1.0 / (float(base) ** (index / head_dim)) for index in range(0, head_dim, 2)
        ]
        if linear_factor is not None:
            inv_freq = [value / float(linear_factor) for value in inv_freq]
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type != "llama3":
                raise ValueError(f"Unsupported RoPE scaling type: {rope_type!r}")
            factor = float(rope_scaling["factor"])
            low_factor = float(rope_scaling["low_freq_factor"])
            high_factor = float(rope_scaling["high_freq_factor"])
            old_context = float(rope_scaling["original_max_position_embeddings"])
            low_wavelength = old_context / low_factor
            high_wavelength = old_context / high_factor
            scaled: list[float] = []
            for value in inv_freq:
                wavelength = 2.0 * math.pi / value
                low_value = value / factor if wavelength > low_wavelength else value
                is_medium = not (wavelength < high_wavelength) and not (
                    wavelength > low_wavelength
                )
                if is_medium:
                    smooth = (old_context / wavelength - low_factor) / (
                        high_factor - low_factor
                    )
                    low_value = (1.0 - smooth) * low_value / factor + smooth * low_value
                scaled.append(low_value)
            inv_freq = scaled
        # Store constants outside the MLX parameter tree.
        self._inv_freq = tuple(inv_freq)

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        midpoint = x.shape[-1] // 2
        return mx.concatenate([-x[..., midpoint:], x[..., :midpoint]], axis=-1)

    def embeddings(
        self, position_ids: mx.array, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        freqs = position_ids.astype(mx.float32)[..., None] * inv_freq[None, None, :]
        angles = mx.concatenate([freqs, freqs], axis=-1)
        return (
            mx.cos(angles).astype(dtype)[:, None, :, :],
            mx.sin(angles).astype(dtype)[:, None, :, :],
        )

    def apply(
        self,
        q: mx.array,
        k: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        cos, sin = position_embeddings
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )


class KVCache:
    """Append-only KV cache using the active main-model compute dtype."""

    def __init__(self, step: int = 256) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = int(step)

    def reset(self) -> None:
        # Existing storage is overwritten before it becomes visible through a
        # slice, so resetting the logical length is sufficient.
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        previous = self.offset
        new_tokens = int(keys.shape[2])
        required = previous + new_tokens
        capacity = 0 if self.keys is None else int(self.keys.shape[2])
        if required > capacity:
            blocks = max(1, math.ceil((required - capacity) / self.step))
            key_growth = mx.zeros(
                (keys.shape[0], keys.shape[1], blocks * self.step, keys.shape[3]),
                dtype=keys.dtype,
            )
            value_growth = mx.zeros(
                (values.shape[0], values.shape[1], blocks * self.step, values.shape[3]),
                dtype=values.dtype,
            )
            if self.keys is None:
                self.keys, self.values = key_growth, value_growth
            else:
                self.keys = mx.concatenate([self.keys, key_growth], axis=2)
                self.values = mx.concatenate([self.values, value_growth], axis=2)
        self.offset = required
        assert self.keys is not None and self.values is not None
        self.keys[..., previous:required, :] = keys
        self.values[..., previous:required, :] = values
        return (
            self.keys[..., :required, :],
            self.values[..., :required, :],
        )


def make_causal_padding_mask(
    attention_mask: mx.array, *, query_length: int, cache_offset: int
) -> mx.array:
    total_length = cache_offset + query_length
    if attention_mask.shape[-1] != total_length:
        raise ValueError(
            "attention mask width must equal cached plus current sequence length: "
            f"{attention_mask.shape[-1]} != {cache_offset} + {query_length}"
        )
    query_positions = mx.arange(cache_offset, total_length)[:, None]
    key_positions = mx.arange(total_length)[None, :]
    causal = key_positions <= query_positions
    valid_keys = attention_mask.astype(mx.bool_)[:, None, None, :]
    return valid_keys & causal[None, None, :, :]


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class DecoderSpec:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool = False
    qk_norm: bool = False
    rope_scaling: dict[str, Any] | None = None


class DecoderAttention(nn.Module):
    def __init__(self, spec: DecoderSpec) -> None:
        super().__init__()
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.scale = spec.head_dim**-0.5
        self.q_proj = nn.Linear(
            spec.hidden_size,
            spec.num_attention_heads * spec.head_dim,
            bias=spec.attention_bias,
        )
        self.k_proj = nn.Linear(
            spec.hidden_size,
            spec.num_key_value_heads * spec.head_dim,
            bias=spec.attention_bias,
        )
        self.v_proj = nn.Linear(
            spec.hidden_size,
            spec.num_key_value_heads * spec.head_dim,
            bias=spec.attention_bias,
        )
        self.o_proj = nn.Linear(
            spec.num_attention_heads * spec.head_dim,
            spec.hidden_size,
            bias=spec.attention_bias,
        )
        self.q_norm = (
            nn.RMSNorm(spec.head_dim, eps=spec.rms_norm_eps) if spec.qk_norm else None
        )
        self.k_norm = (
            nn.RMSNorm(spec.head_dim, eps=spec.rms_norm_eps) if spec.qk_norm else None
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        cache: KVCache,
        mask: mx.array | str | None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(
            batch, length, self.n_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        cos, sin = position_embeddings
        q = q * cos + RotaryEmbedding._rotate_half(q) * sin
        k = k * cos + RotaryEmbedding._rotate_half(k) * sin
        k, v = cache.update_and_fetch(k, v)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, spec: DecoderSpec) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(spec)
        self.mlp = SwiGLU(spec.hidden_size, spec.intermediate_size)
        self.input_layernorm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            spec.hidden_size, eps=spec.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        cache: KVCache,
        mask: mx.array | str | None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, cache, mask
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
