from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from .backbone import Qwen3Backbone
from .checkpoint import (
    COMPONENT_FILES,
    classify_checkpoint,
    load_component_weights,
    load_weight_map,
    to_weight_dtype,
)
from .config import BreezeMLXConfig
from .depth_decoder import BreezeDepthDecoder
from .model import (
    QUANTIZATION_BY_PRECISION,
    QUANTIZED_PRECISION_CHOICES,
    SharedAudioEmbedding,
    quantize_module,
)
from .text_encoder import T5GemmaTextEncoder

ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "LICENSE",
    "MODEL_LICENSE",
)


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX conversion requires an Apple-Silicon Mac")


def _component_module(
    component: str, config: BreezeMLXConfig
) -> tuple[nn.Module, bool]:
    raw = config.model
    if component == "text_encoder":
        return T5GemmaTextEncoder(config.text), True
    if component == "text_encoder_proj":
        return (
            nn.Linear(
                int(config.text["hidden_size"]), int(raw["hidden_size"]), bias=False
            ),
            False,
        )
    if component == "backbone":
        return Qwen3Backbone(config.backbone), True
    if component == "depth_decoder":
        return BreezeDepthDecoder(config.depth), True
    if component == "audio_embedding":
        return (
            SharedAudioEmbedding(
                int(raw["num_codebooks"]) * int(raw["vocab_size"]),
                int(raw["hidden_size"]),
            ),
            True,
        )
    if component == "lm_head":
        return (
            nn.Linear(int(raw["hidden_size"]), int(raw["vocab_size"]) + 1, bias=False),
            False,
        )
    raise KeyError(component)


def _to_weight_dtype(array: mx.array, dtype: mx.Dtype | None) -> mx.array:
    """Compatibility wrapper retained for callers and unit tests."""

    return to_weight_dtype(array, dtype)


def _audio_tokenizer_is_fp32(audio_dir: Path) -> bool:
    model_file = audio_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing FP32 audio tokenizer weights: {model_file}")
    floating_dtypes: set[str] = set()
    with safe_open(model_file, framework="numpy") as handle:
        for name in tuple(handle.keys()):
            dtype = handle.get_slice(name).get_dtype()
            if dtype.startswith(("F", "BF")):
                floating_dtypes.add(dtype)
    if floating_dtypes != {"F32"}:
        raise ValueError(
            "The requested artifact must preserve the current audio tokenizer in "
            f"FP32, but found floating dtypes {sorted(floating_dtypes)}"
        )
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_runtime_assets(source_dir: Path, output_dir: Path) -> None:
    for name in ASSET_FILES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def _copy_audio_tokenizer(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    audio_source = source_dir / "audio_tokenizer"
    audio_output = output_dir / "audio_tokenizer"
    if audio_output.exists() and not overwrite:
        raise FileExistsError(
            f"{audio_output} already exists; pass --overwrite to replace it"
        )
    shutil.copytree(audio_source, audio_output, dirs_exist_ok=overwrite)
    model_file = audio_output / "model.safetensors"
    _audio_tokenizer_is_fp32(audio_output)
    return {
        "file": "audio_tokenizer/model.safetensors",
        "bytes": model_file.stat().st_size,
        "sha256": _sha256(model_file),
        "dtype": "float32",
    }


def convert_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_revision: str | None = None,
    precision: str = "int8",
) -> dict[str, Any]:
    _require_apple_silicon()
    if precision not in QUANTIZED_PRECISION_CHOICES:
        raise ValueError(
            "Only INT8 and INT4 create converted artifacts; use the source "
            f"checkpoint directly for original. Got {precision!r}"
        )
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config = BreezeMLXConfig.from_file(source_dir / "config.json")
    _audio_tokenizer_is_fp32(source_dir / "audio_tokenizer")
    weight_map = load_weight_map(source_dir)
    components = classify_checkpoint(source_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to update it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    quantization = QUANTIZATION_BY_PRECISION[precision]
    # Quantized kernels use FP16 scales and compute. The original checkpoint is
    # loaded directly elsewhere and never passes through this conversion path.
    weight_dtype = mx.float16
    artifact_files: dict[str, dict[str, Any]] = {}
    for component in COMPONENT_FILES:
        print(f"converting {component} ({precision}) ...", flush=True)
        module, supports_quantization = _component_module(component, config)
        weights = load_component_weights(
            source_dir,
            component,
            components[component],
            weight_map,
            weight_dtype=weight_dtype,
        )
        module.load_weights(weights, strict=True)
        mx.eval(module.parameters())
        quantize_this_component = supports_quantization
        if quantize_this_component:
            quantize_module(module, quantization)
            mx.eval(module.parameters())
        output_path = output_dir / COMPONENT_FILES[component]
        module.save_weights(str(output_path))
        artifact_files[component] = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "quantized": quantize_this_component,
        }
        del weights, module
        gc.collect()
        mx.clear_cache()

    _copy_runtime_assets(source_dir, output_dir)
    codec_dtype = "float32"
    codec_file = _copy_audio_tokenizer(
        source_dir,
        output_dir,
        overwrite=overwrite,
    )
    manifest: dict[str, Any] = {
        "format": "breeze-tts-2-mlx",
        "format_version": 3,
        "source_model": "BreezeBlue/Breeze-TTS-2",
        "source_revision": source_revision,
        "main_model_precision": precision,
        "main_model_weight_dtype": f"packed_{precision}",
        "main_model_quantization": quantization,
        "activation_dtype": "mixed",
        "text_encoder_compute_dtype": "float16",
        "text_encoder_residual_dtype": "float32",
        "backbone_depth_activation_dtype": "float16",
        "kv_cache_dtype": "float16",
        "audio_tokenizer_dtype": codec_dtype,
        "audio_tokenizer": codec_file,
        "omitted_source_prefixes": ["embed_text_tokens.", "codec_model."],
        "components": artifact_files,
    }
    manifest_path = output_dir / "mlx_config.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Breeze TTS 2 active main weights to INT8 or INT4"
    )
    parser.add_argument("source", type=Path, help="Complete Hugging Face checkpoint")
    parser.add_argument("output", type=Path, help="Self-contained MLX artifact")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--precision",
        choices=QUANTIZED_PRECISION_CHOICES,
        default="int8",
        help="Main-model quantization (default: int8)",
    )
    args = parser.parse_args()
    manifest = convert_checkpoint(
        args.source,
        args.output,
        overwrite=args.overwrite,
        source_revision=args.source_revision,
        precision=args.precision,
    )
    total = sum(item["bytes"] for item in manifest["components"].values())
    print(
        f"saved {manifest['main_model_precision']} MLX main weights: "
        f"{total / 1_000_000_000:.3f} GB"
    )
    print(
        "audio tokenizer: "
        f"{manifest['audio_tokenizer_dtype']} "
        f"({manifest['audio_tokenizer']['bytes'] / 1_000_000_000:.3f} GB)"
    )


if __name__ == "__main__":
    main()
