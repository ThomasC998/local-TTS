#!/usr/bin/env python3
"""Do the CUDA and Apple-Silicon backends compute the same thing?

``breeze_tts_torch`` is a hand port of ``breeze_tts_mlx``, and the failure mode
of a hand port is not a crash -- it is a model that runs, produces speech, and
sounds subtly unlike the same model on the other machine. A transposed RoPE
half, a norm that scales before it normalizes, a grouped-query mapping that
pairs the wrong heads: all of those generate audio, and all of them generate
the wrong audio.

So this feeds identical random weights and identical inputs through both
implementations and compares the numbers. It needs no GPU, no checkpoint and
no network -- a shrunken configuration with the real one's *shape* (the same
layer types, the same RoPE scaling, the same attention pattern) in FP32 on the
CPU is enough to catch every structural mistake. Run it on either machine:

    python test_torch_parity.py

Skipped, not failed, when only one backend is installed -- which is the normal
case on a machine that is only ever going to run one of them.

What is deliberately not compared
---------------------------------
Positions the attention mask excludes entirely. A padded row whose every key is
masked has no defined answer: PyTorch returns zeros, MLX returns whatever was
in the accumulator. Neither is read -- the text encoder slices padding off
before projecting, and the backbone only ever takes the last position -- so
forcing them to agree would be inventing a contract that nothing relies on.
Everything else is compared to a relative tolerance of 2e-4.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
for extra in ("breeze-tts-torch", "breeze-tts-mlx"):
    if str(PROJECT / extra) not in sys.path:
        sys.path.insert(0, str(PROJECT / extra))

TOLERANCE = 2e-4

# The real checkpoint's config, reduced to something a laptop CPU runs in a
# second. Every field that changes *behaviour* -- layer types, RoPE scaling,
# the sliding window, the codebook count -- is kept; only the sizes shrink.
BASE_CONFIG: dict = {
    "backbone_model_type": "qwen3",
    "text_encoder_proj_type": "linear",
    "text_encoder_feature_layer_idx": -1,
    "num_codebooks": 16,
    "vocab_size": 2051,
    "hidden_size": 64,
    "audio_embed_size": 64,
    "audio_token_id": 262144,
    "audio_eos_token_id": 262145,
    "dtype": "float32",
    "codec_config": {"codebook_size": 2048},
    "backbone_config": {
        "hidden_size": 64,
        "head_dim": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "attention_bias": False,
    },
    "depth_decoder_config": {
        "hidden_size": 48,
        "head_dim": 16,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "intermediate_size": 96,
        "audio_embed_size": 64,
        "num_codebooks": 16,
        "vocab_size": 2051,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "attention_bias": False,
        # The llama3 scaling schedule, kept exactly: it is the fiddliest
        # arithmetic in the whole port.
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 0.0078125,
            "low_freq_factor": 0.001953125,
            "original_max_position_embeddings": 16,
            "rope_type": "llama3",
        },
    },
    "text_encoder_config": {
        "hidden_size": 64,
        "head_dim": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 512,
        "eoi_token_index": 500,
        "sliding_window": 4,
        "rms_norm_eps": 1e-6,
        "query_pre_attn_scalar": 256,
        "attention_bias": False,
        "attn_logit_softcapping": None,
        # One of each, so both mask shapes are exercised.
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "linear",
                "factor": 8.0,
            },
        },
    },
}


class Results:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def compare(self, label: str, mlx_out, torch_out, *, keep=None) -> None:
        """``keep`` is a boolean array selecting the positions that are read."""
        import mlx.core as mx

        left = np.asarray(mlx_out.astype(mx.float32))
        right = torch_out.detach().float().numpy()
        if left.shape != right.shape:
            self.failures.append(f"{label}: shape {left.shape} != {right.shape}")
            print(f"  FAIL  {label:<22} shape {left.shape} != {right.shape}")
            return
        if keep is not None:
            left, right = left[keep], right[keep]
        self.checks += 1
        difference = float(np.abs(left - right).max())
        scale = max(float(np.abs(left).max()), 1e-9)
        relative = difference / scale
        if relative < TOLERANCE:
            print(f"  ok    {label:<22} rel {relative:.2e}")
        else:
            self.failures.append(f"{label}: relative difference {relative:.2e}")
            print(f"  FAIL  {label:<22} rel {relative:.2e}  (abs {difference:.3e})")


def random_weights(module, rng) -> dict[str, np.ndarray]:
    """One set of weights, keyed by the name both module trees use."""
    return {
        name: (
            rng.standard_normal(tuple(parameter.shape)).astype(np.float32)
            * (0.05 if parameter.ndim > 1 else 0.02)
        )
        for name, parameter in module.state_dict().items()
    }


def load_both(mlx_module, torch_module, weights: dict[str, np.ndarray]) -> None:
    import mlx.core as mx
    import torch
    from mlx.utils import tree_flatten

    torch_module.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in weights.items()},
        strict=True,
    )
    mlx_names = {name for name, _ in tree_flatten(mlx_module.parameters())}
    if mlx_names != set(weights):
        raise AssertionError(
            "the two module trees name their parameters differently\n"
            f"  only in mlx:   {sorted(mlx_names - set(weights))[:6]}\n"
            f"  only in torch: {sorted(set(weights) - mlx_names)[:6]}"
        )
    mlx_module.load_weights(
        [(name, mx.array(value)) for name, value in weights.items()], strict=True
    )
    mx.eval(mlx_module.parameters())


def main() -> int:
    try:
        import mlx.core as mx  # noqa: F401
        import torch
    except ImportError as exc:
        print(f"skipped: both backends must be installed to compare them ({exc})")
        return 0

    import mlx.core as mx
    from breeze_tts_mlx.backbone import Qwen3Backbone as MlxBackbone
    from breeze_tts_mlx.config import BreezeMLXConfig
    from breeze_tts_mlx.depth_decoder import BreezeDepthDecoder as MlxDepth
    from breeze_tts_mlx.text_encoder import T5GemmaTextEncoder as MlxText
    from breeze_tts_torch.backbone import Qwen3Backbone as TorchBackbone
    from breeze_tts_torch.depth_decoder import BreezeDepthDecoder as TorchDepth
    from breeze_tts_torch.text_encoder import T5GemmaTextEncoder as TorchText

    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(BASE_CONFIG))
        config = BreezeMLXConfig.from_file(path)

    rng = np.random.default_rng(0)
    results = Results()

    # -- text encoder ----------------------------------------------------
    print("\ntext encoder  (sliding + full attention, linear RoPE, FP32 residual)")
    mlx_text, torch_text = MlxText(config.text), TorchText(config.text)
    torch_text.eval()
    torch_text.set_compute_dtype(torch.float32)
    mlx_text.set_compute_dtype(mx.float32)
    load_both(mlx_text, torch_text, random_weights(torch_text, rng))

    ids = rng.integers(0, 500, size=(2, 9)).astype(np.int64)
    ids[0, 4] = 500  # the end-of-image token, which takes its own embedding path
    mask = np.ones((2, 9), dtype=np.int64)
    mask[1, 6:] = 0  # right padding, as _encode_text_segments produces
    positions = np.tile(np.arange(9), (2, 1)).astype(np.int64)
    results.compare(
        "hidden states",
        mlx_text(mx.array(ids), attention_mask=mx.array(mask),
                 position_ids=mx.array(positions)),
        torch_text(torch.from_numpy(ids), attention_mask=torch.from_numpy(mask),
                   position_ids=torch.from_numpy(positions)),
        keep=mask.astype(bool),
    )

    # -- backbone --------------------------------------------------------
    print("\nbackbone  (grouped-query attention, qk-norm, KV cache, padding mask)")
    mlx_backbone, torch_backbone = MlxBackbone(config.backbone), TorchBackbone(config.backbone)
    torch_backbone.eval()
    load_both(mlx_backbone, torch_backbone, random_weights(torch_backbone, rng))

    embeds = (rng.standard_normal((2, 7, 64)) * 0.5).astype(np.float32)
    mask = np.ones((2, 7), dtype=np.int64)
    mask[1, :2] = 0  # left padding, as the CFG branch builder produces
    positions = np.tile(np.arange(7), (2, 1)).astype(np.int64)
    mlx_cache, torch_cache = mlx_backbone.make_cache(), torch_backbone.make_cache()
    results.compare(
        "prefill",
        mlx_backbone(mx.array(embeds), attention_mask=mx.array(mask),
                     position_ids=mx.array(positions), cache=mlx_cache),
        torch_backbone(torch.from_numpy(embeds), attention_mask=torch.from_numpy(mask),
                       position_ids=torch.from_numpy(positions), cache=torch_cache),
        keep=mask.astype(bool),
    )
    # Two decode steps, so the cache is exercised past its first append -- the
    # step where an off-by-one in the growth logic would first show.
    for step in range(2):
        one = (rng.standard_normal((2, 1, 64)) * 0.5).astype(np.float32)
        mask = np.concatenate([mask, np.ones((2, 1), dtype=np.int64)], axis=1)
        step_positions = np.full((2, 1), 7 + step, dtype=np.int64)
        results.compare(
            f"decode step {step}",
            mlx_backbone(mx.array(one), attention_mask=mx.array(mask),
                         position_ids=mx.array(step_positions), cache=mlx_cache),
            torch_backbone(torch.from_numpy(one), attention_mask=torch.from_numpy(mask),
                           position_ids=torch.from_numpy(step_positions), cache=torch_cache),
        )

    # -- depth decoder ---------------------------------------------------
    print("\ndepth decoder  (llama3 RoPE scaling, 15 codebook heads)")
    mlx_depth, torch_depth = MlxDepth(config.depth), TorchDepth(config.depth)
    torch_depth.eval()
    load_both(mlx_depth, torch_depth, random_weights(torch_depth, rng))

    hidden = (rng.standard_normal((1, 64)) * 0.5).astype(np.float32)
    first = (rng.standard_normal((1, 64)) * 0.5).astype(np.float32)
    mlx_cache, torch_cache = mlx_depth.make_cache(), torch_depth.make_cache()
    results.compare(
        "begin_frame",
        mlx_depth.begin_frame(mx.array(hidden), mx.array(first), mlx_cache),
        torch_depth.begin_frame(torch.from_numpy(hidden), torch.from_numpy(first), torch_cache),
    )
    # 16 codebooks, so heads 0..14; head 14 predicts the last one.
    for index in (1, 2, 14):
        embedding = (rng.standard_normal((1, 64)) * 0.5).astype(np.float32)
        results.compare(
            f"step_frame {index}",
            mlx_depth.step_frame(mx.array(embedding), codebook_index=index, cache=mlx_cache),
            torch_depth.step_frame(torch.from_numpy(embedding), codebook_index=index,
                                   cache=torch_cache),
        )

    print()
    if results.failures:
        print(f"{len(results.failures)} of {results.checks} comparisons differ:")
        for failure in results.failures:
            print(f"  - {failure}")
        return 1
    print(f"All {results.checks} comparisons match to within {TOLERANCE:g} relative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
