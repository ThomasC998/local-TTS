"""Attention, norms and RoPE, ported one-for-one from the MLX layers.

Every shape, every transpose order and every place a value is cast is the same
as in ``breeze_tts_mlx/layers.py``. Where PyTorch offers a shortcut that is not
numerically identical -- ``torch.nn.RMSNorm`` normalizes in the input dtype --
the long form is written out instead, because a BF16 mean of squares over 2048
channels is not the FP32 one MLX computes, and the difference shows up as a
drifting timbre rather than as an obvious failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``mx.fast.rms_norm``: normalize in FP32, scale, return the input dtype."""
    dtype = x.dtype
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(dtype)


class RMSNorm(nn.Module):
    """MLX's ``nn.RMSNorm``: the checkpoint weight is the scale itself."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm, whose checkpoint weight is an offset from one."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dims))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, 1.0 + self.weight, self.eps)


class RotaryEmbedding:
    """HF-compatible half-split RoPE with optional Llama 3 scaling.

    Not an ``nn.Module``: ``inv_freq`` is a constant derived from the config,
    never a checkpoint tensor, and registering it as a buffer would put a key in
    the state dict that the original checkpoint has nothing to load into.
    """

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
        self._inv_freq = tuple(inv_freq)
        self._cached: torch.Tensor | None = None

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        if self._cached is None or self._cached.device != device:
            self._cached = torch.tensor(
                self._inv_freq, dtype=torch.float32, device=device
            )
        return self._cached

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        midpoint = x.shape[-1] // 2
        return torch.cat([-x[..., midpoint:], x[..., :midpoint]], dim=-1)

    def embeddings(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._frequencies(position_ids.device)
        freqs = position_ids.float()[..., None] * inv_freq[None, None, :]
        angles = torch.cat([freqs, freqs], dim=-1)
        return (
            angles.cos().to(dtype)[:, None, :, :],
            angles.sin().to(dtype)[:, None, :, :],
        )

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = position_embeddings
        return (
            q * cos + self.rotate_half(q) * sin,
            k * cos + self.rotate_half(k) * sin,
        )


class KVCache:
    """Append-only KV cache that grows in blocks, as the MLX one does.

    Growing in ``step``-sized blocks rather than concatenating per token is what
    keeps a 1500-token generation from doing 1500 reallocations of an ever
    larger tensor -- the single biggest avoidable cost in a decode loop.
    """

    def __init__(self, step: int = 256) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.offset = 0
        self.step = int(step)

    def reset(self) -> None:
        # Storage is overwritten before it becomes visible through a slice, so
        # resetting the logical length is enough.
        self.offset = 0

    def update_and_fetch(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self.offset
        new_tokens = int(keys.shape[2])
        required = previous + new_tokens
        capacity = 0 if self.keys is None else int(self.keys.shape[2])
        if required > capacity:
            blocks = max(1, math.ceil((required - capacity) / self.step))
            shape = (keys.shape[0], keys.shape[1], blocks * self.step, keys.shape[3])
            key_growth = torch.zeros(shape, dtype=keys.dtype, device=keys.device)
            value_growth = torch.zeros(
                (values.shape[0], values.shape[1], blocks * self.step, values.shape[3]),
                dtype=values.dtype,
                device=values.device,
            )
            if self.keys is None:
                self.keys, self.values = key_growth, value_growth
            else:
                self.keys = torch.cat([self.keys, key_growth], dim=2)
                self.values = torch.cat([self.values, value_growth], dim=2)
        self.offset = required
        assert self.keys is not None and self.values is not None
        self.keys[..., previous:required, :] = keys
        self.values[..., previous:required, :] = values
        return self.keys[..., :required, :], self.values[..., :required, :]


def make_causal_padding_mask(
    attention_mask: torch.Tensor, *, query_length: int, cache_offset: int
) -> torch.Tensor:
    total_length = cache_offset + query_length
    if attention_mask.shape[-1] != total_length:
        raise ValueError(
            "attention mask width must equal cached plus current sequence length: "
            f"{attention_mask.shape[-1]} != {cache_offset} + {query_length}"
        )
    device = attention_mask.device
    query_positions = torch.arange(cache_offset, total_length, device=device)[:, None]
    key_positions = torch.arange(total_length, device=device)[None, :]
    causal = key_positions <= query_positions
    valid_keys = attention_mask.bool()[:, None, None, :]
    return valid_keys & causal[None, None, :, :]


def expand_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Grouped-query attention, spelled out rather than left to the kernel.

    ``enable_gqa`` is new enough that pinning it would narrow which PyTorch
    builds run this, and the repeat costs a copy of a tensor that is already in
    cache. MLX broadcasts internally; this is the same arithmetic.
    """
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    mask: torch.Tensor | str | None,
) -> torch.Tensor:
    """``mx.fast.scaled_dot_product_attention`` with the same mask vocabulary.

    A boolean mask means "True participates", which is PyTorch's convention as
    well as MLX's, so it passes straight through.

    One case the two backends genuinely answer differently, and it is fine: a
    query row with *every* key masked. That happens only at a padded position,
    where softmax over an all-excluded row has no defined value -- PyTorch
    returns zeros, MLX returns whatever was in the accumulator. Nothing reads
    those positions: the text encoder slices padding off before projecting, and
    the backbone only ever takes the last position, which left padding never
    occupies. ``test_torch_parity.py`` excludes them for the same reason.
    """
    repeats = q.shape[1] // k.shape[1]
    k = expand_kv(k, repeats)
    v = expand_kv(v, repeats)
    if mask == "causal":
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=True
        )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale
    )


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


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
        self.q_norm = RMSNorm(spec.head_dim, spec.rms_norm_eps) if spec.qk_norm else None
        self.k_norm = RMSNorm(spec.head_dim, spec.rms_norm_eps) if spec.qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache,
        mask: torch.Tensor | str | None,
    ) -> torch.Tensor:
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q = q * cos + RotaryEmbedding.rotate_half(q) * sin
        k = k * cos + RotaryEmbedding.rotate_half(k) * sin
        k, v = cache.update_and_fetch(k, v)
        output = attend(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, spec: DecoderSpec) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(spec)
        self.mlp = SwiGLU(spec.hidden_size, spec.intermediate_size)
        self.input_layernorm = RMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(spec.hidden_size, spec.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache,
        mask: torch.Tensor | str | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, cache, mask
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
