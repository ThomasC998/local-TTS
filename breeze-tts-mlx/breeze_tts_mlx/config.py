from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigView(dict[str, Any]):
    """Dictionary with attribute access for the existing template helpers."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, Mapping) and not isinstance(value, ConfigView):
            value = ConfigView(value)
            self[name] = value
        return value


@dataclass(frozen=True)
class BreezeMLXConfig:
    """Validated view of a Breeze TTS 2 Hugging Face configuration."""

    raw: ConfigView

    @classmethod
    def from_file(cls, path: str | Path) -> BreezeMLXConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = ConfigView(json.load(handle))
        config = cls(raw=raw)
        config.validate()
        return config

    def validate(self) -> None:
        errors: list[str] = []
        if self.raw.get("backbone_model_type") != "qwen3":
            errors.append("backbone_model_type must be 'qwen3'")
        if self.raw.get("text_encoder_proj_type", "linear") != "linear":
            errors.append("text_encoder_proj_type must be 'linear'")
        feature_idx = self.raw.get("text_encoder_feature_layer_idx", -1)
        if feature_idx not in (-1, [-1], (-1,)):
            errors.append("text_encoder_feature_layer_idx must select the final layer")
        if self.raw.get("text_encoder_dimfusion_fuse_first_layer", False):
            errors.append("text-encoder DimFusion is not supported by this MLX backend")
        if int(self.raw.get("num_codebooks", 0)) != 16:
            errors.append("this checkpoint must have exactly 16 codebooks")
        if int(self.raw.get("vocab_size", 0)) != 2051:
            errors.append("this checkpoint must have audio vocab_size=2051")
        if "text_encoder_config" not in self.raw:
            errors.append(
                "text_encoder_config is required; fallback text embeddings are omitted"
            )
        if "backbone_config" not in self.raw:
            errors.append("bundled Qwen3 backbone_config is required")
        if errors:
            formatted = "\n  - ".join(errors)
            raise ValueError(
                f"Unsupported Breeze checkpoint configuration:\n  - {formatted}"
            )

    @property
    def model(self) -> ConfigView:
        return self.raw

    @property
    def text(self) -> ConfigView:
        return ConfigView(self.raw["text_encoder_config"])

    @property
    def backbone(self) -> ConfigView:
        return ConfigView(self.raw["backbone_config"])

    @property
    def depth(self) -> ConfigView:
        return ConfigView(self.raw["depth_decoder_config"])
