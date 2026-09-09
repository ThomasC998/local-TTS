"""Standalone Breeze TTS 2 inference on Apple Silicon with MLX."""

from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path

import soundfile as sf
from mlx.utils import tree_flatten

from breeze_tts_mlx.runtime import BreezeMLXRuntime, MLXRuntimeConfig
from breeze_tts_mlx.sampling import SamplingConfig
from breeze_tts_mlx.templates import get_template, prepare_inputs

DEFAULT_CFG_SCALE = 1.0
DEFAULT_MAX_NEW_TOKENS = 1500
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_REPETITION_PENALTY = 1.1


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("infer.py requires an Apple-Silicon Mac")


def _loaded_weight_bytes(runtime: BreezeMLXRuntime) -> int:
    main_bytes = sum(
        int(array.nbytes) for _name, array in tree_flatten(runtime.model.parameters())
    )
    audio_model = runtime.audio_tokenizer.model
    audio_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in audio_model.parameters()
    ) + sum(
        tensor.numel() * tensor.element_size() for tensor in audio_model.buffers()
    )
    return main_bytes + audio_bytes


def _generation_status(
    frames: int,
    *,
    audio_seconds: float = 0.0,
    elapsed: float = 0.0,
    weight_bytes: int,
    finished: bool = False,
) -> str:
    realtime_speed = audio_seconds / elapsed if elapsed > 0 else 0.0
    state = "generated" if finished else "generating"
    return (
        f"{state} | frames {frames} | audio {audio_seconds:.2f}s | "
        f"elapsed {elapsed:.1f}s | weights {weight_bytes / 1024**3:.2f} GiB | "
        f"{realtime_speed:.2f}x realtime"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate one WAV with the original, INT8, or INT4 MLX model")
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Original chkpt-full directory or a converted INT8/INT4 artifact",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument(
        "--mode",
        choices=("auto", "plain", "guided", "clone", "edit"),
        default="auto",
        help=(
            "Prompt mode. auto selects edit for reference+instruction, clone for "
            "reference only, guided for instruction only, otherwise plain"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("output_mlx.wav"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax for both backbone and depth-token selection",
    )
    parser.add_argument(
        "--audio-device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="FP32 audio-tokenizer device (auto selects MPS when available)",
    )
    parser.add_argument(
        "--codec-chunk-frames",
        type=int,
        default=2,
        help="Stateful codec frames per launch",
    )
    args = parser.parse_args()

    _require_apple_silicon()
    if not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError("--cfg-scale must be finite and greater than zero")
    has_ref_audio = args.ref_audio is not None
    has_ref_text = bool(args.ref_text and args.ref_text.strip())
    if has_ref_audio != has_ref_text:
        raise ValueError("--ref-audio and --ref-text must be provided together")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )
    runtime = BreezeMLXRuntime(
        args.model,
        audio_device=args.audio_device,
        seed=args.seed,
        config=MLXRuntimeConfig(
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            repetition_penalty=args.repetition_penalty,
            codec_chunk_frames=args.codec_chunk_frames,
            backbone_sampling=sampling,
            depth_sampling=sampling,
        ),
    )
    weight_bytes = _loaded_weight_bytes(runtime)

    mode = args.mode
    if mode == "auto":
        mode = (
            "edit"
            if has_ref_audio and args.instruction
            else "clone"
            if has_ref_audio
            else "guided"
            if args.instruction
            else "plain"
        )
    if mode in {"clone", "edit"} and not has_ref_audio:
        raise ValueError(f"--mode {mode} requires --ref-audio and --ref-text")
    if mode in {"guided", "edit"} and not args.instruction:
        raise ValueError(f"--mode {mode} requires --instruction")
    if mode in {"plain", "clone"} and args.cfg_scale != 1.0:
        raise ValueError(f"--mode {mode} does not support CFG; use --cfg-scale 1")

    request = {
        "id": "single-request-mlx",
        "text": args.text,
        "speaker": "S0",
    }
    if args.instruction:
        request["instruction"] = args.instruction
    if args.ref_audio is not None:
        request["ref_audio_path"] = str(args.ref_audio)
        request["ref_text"] = args.ref_text.strip()
    template_name = {
        "plain": "tts_plain",
        "guided": "tts_instruction",
        "clone": "ref_clone_tata",
        "edit": "ref_edit_tata",
    }[mode]

    inputs = prepare_inputs(
        runtime.tokenizer,
        runtime.audio_tokenizer,
        runtime,
        [request],
        get_template(template_name),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    total_codec_frames = 0
    generation_elapsed = 0.0
    status_width = 0
    live_progress = sys.stdout.isatty()
    with sf.SoundFile(
        args.output,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output_file:
        try:
            for chunk in runtime.iter_audio_chunks(
                inputs, request_id="single-request-mlx"
            ):
                output_file.write(chunk.audio)
                total_samples += chunk.audio.size
                total_codec_frames = int(chunk.timing["total_frames"])
                generation_elapsed = float(chunk.timing["elapsed_ms"]) / 1000.0
                audio_seconds = total_samples / runtime.sample_rate
                if live_progress:
                    status = _generation_status(
                        total_codec_frames,
                        audio_seconds=audio_seconds,
                        elapsed=generation_elapsed,
                        weight_bytes=weight_bytes,
                    )
                    print(f"\r{status:<{status_width}}", end="", flush=True)
                    status_width = max(status_width, len(status))
        except Exception:
            if live_progress:
                print()
            raise

    if live_progress:
        duration = total_samples / runtime.sample_rate
        status = _generation_status(
            total_codec_frames,
            audio_seconds=duration,
            elapsed=generation_elapsed,
            weight_bytes=weight_bytes,
            finished=True,
        )
        print(f"\r{status:<{status_width}}")

    duration = total_samples / runtime.sample_rate
    summary = _generation_status(
        total_codec_frames,
        audio_seconds=duration,
        elapsed=generation_elapsed,
        weight_bytes=weight_bytes,
        finished=True,
    )
    print(f"saved {args.output} | {summary.removeprefix('generated | ')}")


if __name__ == "__main__":
    main()
