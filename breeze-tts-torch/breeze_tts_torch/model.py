"""Assemble the Breeze main model and fill it from the original checkpoint.

The module tree here is built so that every parameter name matches the target
names ``breeze_tts_mlx.checkpoint`` already maps the Hugging Face tensors onto.
That is not a coincidence to preserve casually: it is what lets one mapping
table serve both backends, so a checkpoint that loads on the Mac cannot quietly
load *differently* here.

What this does not do is read MLX's INT8 artifact. Those weights are packed in
MLX's affine quantization format -- scales and biases per group of 64, in MLX's
own layout -- and nothing in PyTorch reads it. Windows loads the untouched
Hugging Face shards instead; MODELS.md says which ones and how to fetch them.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from ._shared import BreezeConfig, load_weight_map, map_source_tensor
from .backbone import Qwen3Backbone
from .depth_decoder import BreezeDepthDecoder
from .text_encoder import T5GemmaTextEncoder

DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in DTYPES:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose one of {sorted(set(DTYPES))}"
        )
    return DTYPES[key]


class SharedAudioEmbedding(nn.Module):
    """One physical table for all sixteen codebooks, offset per codebook."""

    def __init__(self, num_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class BreezeTorchModel(nn.Module):
    """Inference-only active Breeze main model on PyTorch.

    As in the MLX port, the legacy direct text embedding and the embedded Mimi
    codec are absent: neither is reachable from this project's pipeline, and
    together they are most of the checkpoint's size.
    """

    def __init__(self, config: BreezeConfig) -> None:
        super().__init__()
        self.breeze_config = config
        raw = config.model
        self.text_encoder = T5GemmaTextEncoder(config.text)
        self.text_encoder_proj = nn.Linear(
            int(config.text["hidden_size"]), int(raw["hidden_size"]), bias=False
        )
        self.backbone = Qwen3Backbone(config.backbone)
        self.depth_decoder = BreezeDepthDecoder(config.depth)
        self.audio_embedding = SharedAudioEmbedding(
            int(raw["num_codebooks"]) * int(raw["vocab_size"]),
            int(raw["hidden_size"]),
        )
        self.lm_head = nn.Linear(
            int(raw["hidden_size"]), int(raw["vocab_size"]) + 1, bias=False
        )
        self.compute_dtype = torch.bfloat16

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.compute_dtype = dtype
        self.text_encoder.set_compute_dtype(dtype)

    def embed_audio_frames(self, code_ids: torch.Tensor) -> torch.Tensor:
        if code_ids.shape[-1] != int(self.breeze_config.model["num_codebooks"]):
            raise ValueError("audio frames must contain exactly 16 codebook IDs")
        offsets = torch.arange(
            code_ids.shape[-1], dtype=code_ids.dtype, device=code_ids.device
        ) * int(self.breeze_config.model["vocab_size"])
        return self.audio_embedding(code_ids + offsets).sum(dim=-2)

    def embed_depth_code(
        self, code_ids: torch.Tensor, *, codebook_index: int
    ) -> torch.Tensor:
        return self.audio_embedding(
            code_ids + codebook_index * int(self.breeze_config.model["vocab_size"])
        )

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> BreezeTorchModel:
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {config_path}")
        if not (checkpoint_dir / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(
                f"{checkpoint_dir} is not a complete Hugging Face checkpoint: no "
                "model.safetensors.index.json. The MLX INT8 artifact cannot be "
                "loaded by the CUDA backend -- see MODELS.md."
            )

        # Built on the meta device so the 6 GB of randomly initialized weights
        # PyTorch would otherwise allocate, and immediately overwrite, are never
        # allocated at all. On a 8 GB card that difference is load-or-fail.
        with torch.device("meta"):
            model = cls(BreezeConfig.from_file(config_path))
        model.set_compute_dtype(dtype)

        weight_map = load_weight_map(checkpoint_dir)
        by_shard: dict[str, list[str]] = defaultdict(list)
        for source_name in weight_map:
            if map_source_tensor(source_name) is not None:
                by_shard[weight_map[source_name]].append(source_name)

        state: dict[str, torch.Tensor] = {}
        prefixes = {
            "text_encoder": "text_encoder.",
            "text_encoder_proj": "text_encoder_proj.",
            "backbone": "backbone.",
            "depth_decoder": "depth_decoder.",
            "audio_embedding": "audio_embedding.",
            "lm_head": "lm_head.",
        }
        for shard_name, names in sorted(by_shard.items()):
            shard_path = checkpoint_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint shard {shard_path}. Download the complete "
                    "Hugging Face snapshot -- see MODELS.md."
                )
            # One shard open at a time, and each tensor materialized on the
            # target device as it is read: peak host memory stays at one shard
            # rather than at the whole checkpoint.
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for source_name in sorted(names):
                    target = map_source_tensor(source_name)
                    assert target is not None
                    tensor = handle.get_tensor(source_name)
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    prefix = prefixes[target.component]
                    if target.transform == "split_codebook_heads":
                        if tensor.ndim != 3 or tensor.shape[0] != 15:
                            raise ValueError(
                                "depth codebook head must have shape "
                                f"[15, hidden, vocab], got {tuple(tensor.shape)}"
                            )
                        for index in range(tensor.shape[0]):
                            key = f"{prefix}{target.target_name}.{index}.weight"
                            state[key] = (
                                tensor[index].T.contiguous().to(device, non_blocking=True)
                            )
                    else:
                        key = f"{prefix}{target.target_name}"
                        state[key] = tensor.to(device, non_blocking=True)
                    del tensor

        # assign=True is what makes the meta construction work: parameters are
        # replaced by the loaded tensors rather than copied into storage that
        # does not exist.
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            # A parameter left on the meta device would fail much later, inside
            # a matmul, with an error naming neither the tensor nor the file.
            raise RuntimeError(
                "Checkpoint does not match the model.\n"
                f"  missing: {list(missing)[:8]}\n"
                f"  unexpected: {list(unexpected)[:8]}"
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
