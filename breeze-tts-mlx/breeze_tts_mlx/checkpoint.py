from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mlx.core as mx

# MLX is imported inside the two functions that load arrays, not here. The
# name mapping above them is pure Python, and the CUDA backend reuses it so the
# two platforms cannot disagree about which checkpoint tensor is which.

COMPONENT_FILES = {
    "text_encoder": "text_encoder.safetensors",
    "text_encoder_proj": "text_encoder_proj.safetensors",
    "backbone": "backbone.safetensors",
    "depth_decoder": "depth_decoder.safetensors",
    "audio_embedding": "audio_embedding.safetensors",
    "lm_head": "lm_head.safetensors",
}

INACTIVE_PREFIXES = ("embed_text_tokens.", "codec_model.")


@dataclass(frozen=True)
class TensorTarget:
    component: str
    target_name: str
    transform: str | None = None


def map_source_tensor(name: str) -> TensorTarget | None:
    """Map one original state-dict name to the inference-only MLX layout."""

    if name.startswith(INACTIVE_PREFIXES):
        return None
    if name == "text_encoder.embed_tokens.weight":
        return TensorTarget("text_encoder", "embed_tokens.embedding.weight")
    if name.startswith("text_encoder."):
        return TensorTarget("text_encoder", name.removeprefix("text_encoder."))
    if name == "text_encoder_proj.weight":
        return TensorTarget("text_encoder_proj", "weight")
    if name.startswith("backbone_model."):
        return TensorTarget("backbone", name.removeprefix("backbone_model."))
    if name == "depth_decoder.model.embed_tokens.weight":
        return TensorTarget("audio_embedding", "embedding.weight")
    if name == "depth_decoder.codebooks_head.weight":
        return TensorTarget(
            "depth_decoder", "codebooks_head.heads", "split_codebook_heads"
        )
    if name == "depth_decoder.model.inputs_embeds_projector.weight":
        return TensorTarget("depth_decoder", "input_projection.weight")
    if name.startswith("depth_decoder.model."):
        return TensorTarget("depth_decoder", name.removeprefix("depth_decoder.model."))
    if name == "lm_head.weight":
        return TensorTarget("lm_head", "weight")
    raise KeyError(f"Unrecognized checkpoint tensor: {name}")


def load_weight_map(source_dir: str | Path) -> dict[str, str]:
    source_dir = Path(source_dir)
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing {index_path}. Download the complete Hugging Face snapshot first."
        )
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def classify_checkpoint(source_dir: str | Path) -> dict[str, list[str]]:
    components = {name: [] for name in COMPONENT_FILES}
    for source_name in load_weight_map(source_dir):
        target = map_source_tensor(source_name)
        if target is not None:
            components[target.component].append(source_name)
    for names in components.values():
        names.sort()
    return components


def to_weight_dtype(array: "mx.array", dtype: "mx.Dtype | None") -> "mx.array":
    """Cast a floating tensor when conversion requests a storage dtype."""

    import mlx.core as mx

    if dtype is not None and mx.issubdtype(array.dtype, mx.floating):
        return array.astype(dtype)
    return array


def load_component_weights(
    source_dir: str | Path,
    component: str,
    source_names: list[str],
    weight_map: dict[str, str],
    *,
    weight_dtype: "mx.Dtype | None" = None,
) -> list[tuple[str, Any]]:
    """Load and remap one component from the original Hugging Face shards.

    A ``None`` dtype preserves each source tensor's dtype. This is used when
    loading ``chkpt-full`` as the original model; quantized conversion passes
    FP16 explicitly before packing weights as INT8 or INT4.
    """

    import mlx.core as mx

    source_dir = Path(source_dir)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for source_name in source_names:
        by_shard[weight_map[source_name]].append(source_name)

    weights: list[tuple[str, Any]] = []
    for shard_name, names in sorted(by_shard.items()):
        shard_path = source_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint shard {shard_path}. The files in chkpt/ are "
                "metadata only; download the complete Hugging Face snapshot."
            )
        shard = mx.load(str(shard_path))
        for source_name in names:
            if source_name not in shard:
                raise KeyError(
                    f"{source_name} is absent from declared shard {shard_path}"
                )
            target = map_source_tensor(source_name)
            if target is None or target.component != component:
                raise RuntimeError(f"Internal component-map mismatch for {source_name}")
            array = to_weight_dtype(shard[source_name], weight_dtype)
            if target.transform == "split_codebook_heads":
                if array.ndim != 3 or array.shape[0] != 15:
                    raise ValueError(
                        "depth codebook head must have shape [15, hidden, vocab], "
                        f"got {array.shape}"
                    )
                for index in range(array.shape[0]):
                    weights.append(
                        (f"{target.target_name}.{index}.weight", array[index].T)
                    )
            else:
                weights.append((target.target_name, array))
        del shard
    weights.sort(key=lambda item: item[0])
    return weights
