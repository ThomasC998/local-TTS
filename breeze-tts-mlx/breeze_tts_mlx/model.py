from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from .backbone import Qwen3Backbone
from .checkpoint import (
    COMPONENT_FILES,
    classify_checkpoint,
    load_component_weights,
    load_weight_map,
)
from .config import BreezeMLXConfig
from .depth_decoder import BreezeDepthDecoder
from .text_encoder import T5GemmaTextEncoder

PRECISION_CHOICES = ("original", "int8", "int4")
QUANTIZATION_BY_PRECISION = {
    "int8": {"group_size": 64, "bits": 8, "mode": "affine"},
    "int4": {"group_size": 64, "bits": 4, "mode": "affine"},
}
QUANTIZED_PRECISION_CHOICES = tuple(QUANTIZATION_BY_PRECISION)
# Backward-compatible name for callers that expect the default INT8 policy.
QUANTIZATION = QUANTIZATION_BY_PRECISION["int8"]


def _original_compute_dtype(config: BreezeMLXConfig) -> mx.Dtype:
    name = str(
        config.model.get("dtype", config.model.get("torch_dtype", "bfloat16"))
    ).lower()
    dtypes = {
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float16": mx.float16,
        "fp16": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported original checkpoint dtype: {name!r}")
    return dtypes[name]


def quantize_module(
    module: nn.Module, quantization: dict[str, Any] | None = None
) -> None:
    quantization = quantization or QUANTIZATION
    nn.quantize(
        module,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=str(quantization["mode"]),
        class_predicate=lambda _path, child: isinstance(
            child, (nn.Linear, nn.Embedding)
        ),
    )


def artifact_precision(manifest: dict[str, Any]) -> str:
    """Return one of the three supported model policies."""
    precision = manifest.get("main_model_precision")
    if precision is not None:
        if precision not in PRECISION_CHOICES:
            raise ValueError(f"Unsupported MLX artifact precision: {precision!r}")
        return str(precision)

    quantization = manifest.get("main_model_quantization")
    for candidate, expected in QUANTIZATION_BY_PRECISION.items():
        if quantization == expected:
            return candidate
    raise ValueError("MLX manifest does not declare a supported main-model precision")


class SharedAudioEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.embedding(token_ids)


class BreezeMLXModel(nn.Module):
    """Inference-only active Breeze main model.

    The legacy direct text embedding and embedded Mimi codec are intentionally
    absent. The tied audio embedding has one physical MLX module.
    """

    def __init__(self, config: BreezeMLXConfig) -> None:
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
        self.main_model_precision = "original"
        self.compute_dtype = _original_compute_dtype(config)
        self._quantization: dict[str, Any] | None = None

    def quantize(self, quantization: dict[str, Any] | None = None) -> None:
        quantization = dict(quantization or QUANTIZATION)
        if self._quantization == quantization:
            return
        if self._quantization is not None:
            raise RuntimeError("model is already quantized with a different policy")
        quantize_module(self.text_encoder, quantization)
        quantize_module(self.backbone, quantization)
        quantize_module(self.depth_decoder, quantization)
        quantize_module(self.audio_embedding, quantization)
        # These two projections total only about 13 MB in BF16 and remain FP16.
        self._quantization = quantization

    def set_precision_policy(self, precision: str) -> None:
        if precision not in PRECISION_CHOICES:
            raise ValueError(f"Unsupported precision policy: {precision!r}")
        self.main_model_precision = precision
        self.compute_dtype = (
            _original_compute_dtype(self.breeze_config)
            if precision == "original"
            else mx.float16
        )
        self.text_encoder.set_compute_dtype(self.compute_dtype)

    def embed_audio_frames(self, code_ids: mx.array) -> mx.array:
        if code_ids.shape[-1] != int(self.breeze_config.model["num_codebooks"]):
            raise ValueError("audio frames must contain exactly 16 codebook IDs")
        offsets = mx.arange(code_ids.shape[-1], dtype=code_ids.dtype) * int(
            self.breeze_config.model["vocab_size"]
        )
        return self.audio_embedding(code_ids + offsets).sum(axis=-2)

    def embed_depth_code(self, code_ids: mx.array, *, codebook_index: int) -> mx.array:
        return self.audio_embedding(
            code_ids + codebook_index * int(self.breeze_config.model["vocab_size"])
        )

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> BreezeMLXModel:
        artifact_dir = Path(artifact_dir)
        manifest_path = artifact_dir / "mlx_config.json"
        if not manifest_path.is_file():
            return cls.from_original_checkpoint(artifact_dir)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
        precision = artifact_precision(manifest)
        expected_quant = QUANTIZATION_BY_PRECISION.get(precision)
        actual_quant = manifest.get("main_model_quantization")
        if expected_quant is not None and actual_quant != expected_quant:
            raise ValueError(
                f"Invalid {precision} quantization metadata: {actual_quant!r}; "
                f"expected {expected_quant!r}"
            )
        if expected_quant is None and actual_quant not in (None, {}):
            raise ValueError(
                f"Floating-point artifact {precision} cannot declare quantization "
                f"metadata: {actual_quant!r}"
            )
        if precision == "original":
            raise ValueError(
                "Original must be the untouched Hugging Face checkpoint, not a "
                "converted MLX artifact"
            )
        expected_codec_dtype = "float32"
        if manifest.get("audio_tokenizer_dtype") != expected_codec_dtype:
            raise ValueError(
                f"{precision} requires an {expected_codec_dtype} audio tokenizer"
            )

        model = cls(BreezeMLXConfig.from_file(artifact_dir / "config.json"))
        if expected_quant is not None:
            model.quantize(expected_quant)
        model.set_precision_policy(precision)
        modules = {
            "text_encoder": model.text_encoder,
            "text_encoder_proj": model.text_encoder_proj,
            "backbone": model.backbone,
            "depth_decoder": model.depth_decoder,
            "audio_embedding": model.audio_embedding,
            "lm_head": model.lm_head,
        }
        for component, module in modules.items():
            weight_path = artifact_dir / COMPONENT_FILES[component]
            if not weight_path.is_file():
                raise FileNotFoundError(f"Missing MLX component weights: {weight_path}")
            module.load_weights(str(weight_path), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model

    @classmethod
    def from_original_checkpoint(cls, checkpoint_dir: str | Path) -> BreezeMLXModel:
        """Load the raw ``chkpt-full`` shards without casting or rewriting them."""

        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"{checkpoint_dir} is neither an MLX INT8/INT4 artifact nor a "
                "complete original checkpoint"
            )

        model = cls(BreezeMLXConfig.from_file(config_path))
        model.set_precision_policy("original")
        weight_map = load_weight_map(checkpoint_dir)
        components = classify_checkpoint(checkpoint_dir)
        modules = {
            "text_encoder": model.text_encoder,
            "text_encoder_proj": model.text_encoder_proj,
            "backbone": model.backbone,
            "depth_decoder": model.depth_decoder,
            "audio_embedding": model.audio_embedding,
            "lm_head": model.lm_head,
        }
        for component, module in modules.items():
            weights = load_component_weights(
                checkpoint_dir,
                component,
                components[component],
                weight_map,
            )
            module.load_weights(weights, strict=True)
            mx.eval(module.parameters())
            del weights
            mx.clear_cache()
        model.eval()
        return model
