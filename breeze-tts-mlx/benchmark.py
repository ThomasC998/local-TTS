"""Benchmark the original, INT8, and INT4 Breeze MLX models."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRECISIONS = ("original", "int8", "int4")
DEFAULT_TEXT = "Hello Rishikesh, welcome to the world of AI TTS model listen carefull because quality is subjective."
DEFAULT_INSTRUCTION = "Speak clearly and naturally."


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("benchmark.py requires an Apple-Silicon Mac")


def _parse_artifact_overrides(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        precision, separator, path = value.partition("=")
        if not separator or precision not in PRECISIONS or not path:
            raise ValueError(
                "--artifact must use PRECISION=PATH, where PRECISION is one of "
                f"{PRECISIONS}: {value!r}"
            )
        artifacts[precision] = Path(path).resolve()
    return artifacts


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _artifact_sizes(
    artifact_dir: Path, manifest: dict[str, Any] | None
) -> dict[str, int]:
    if manifest is None:
        index = json.loads(
            (artifact_dir / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        shard_names = set(index["weight_map"].values())
        main_bytes = sum((artifact_dir / name).stat().st_size for name in shard_names)
    else:
        main_bytes = sum(
            int(component["bytes"])
            for component in manifest["components"].values()
        )
    audio_dir = artifact_dir / "audio_tokenizer"
    return {
        "main_weights_bytes": main_bytes,
        "audio_tokenizer_bytes": _directory_bytes(audio_dir),
        "artifact_bytes": _directory_bytes(artifact_dir),
    }


def _peak_rss_bytes() -> int:
    # ru_maxrss is bytes on macOS (and KiB on Linux, which this project rejects).
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _worker(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import soundfile as sf

    from breeze_tts_mlx.runtime import BreezeMLXRuntime, MLXRuntimeConfig
    from breeze_tts_mlx.sampling import NumpySampler, SamplingConfig
    from breeze_tts_mlx.templates import get_template, prepare_inputs

    artifact_dir = args.worker_artifact.resolve()
    manifest_path = artifact_dir / "mlx_config.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )
    runtime_config = MLXRuntimeConfig(
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        repetition_penalty=args.repetition_penalty,
        codec_chunk_frames=args.codec_chunk_frames,
        backbone_sampling=sampling,
        depth_sampling=sampling,
    )

    baseline_rss = _peak_rss_bytes()
    load_started = time.perf_counter()
    runtime = BreezeMLXRuntime(
        artifact_dir,
        audio_device=args.audio_device,
        config=runtime_config,
        seed=args.seed,
    )
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    actual_precision = runtime.model.main_model_precision
    if actual_precision != args.worker_precision:
        raise ValueError(
            f"Expected {args.worker_precision} model at {artifact_dir}, "
            f"found {actual_precision}"
        )
    loaded_rss = _peak_rss_bytes()
    loaded_metal = int(mx.get_active_memory())

    request = {
        "id": "mlx-benchmark",
        "text": args.text,
        "instruction": args.instruction,
        "speaker": "S0",
    }
    inputs = prepare_inputs(
        runtime.tokenizer,
        runtime.audio_tokenizer,
        runtime,
        [request],
        get_template("tts_instruction"),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    def generate(run_index: int) -> dict[str, float | int]:
        runtime.sampler = NumpySampler(args.seed)
        mx.synchronize()
        started = time.perf_counter()
        frames = 0
        samples = 0
        chunks = 0
        rendered_audio = [] if run_index == 0 else None
        for chunk in runtime.iter_audio_chunks(
            inputs, request_id=f"benchmark-{actual_precision}-{run_index}"
        ):
            frames += chunk.codec_frames
            samples += int(chunk.audio.size)
            chunks += 1
            if rendered_audio is not None:
                rendered_audio.append(chunk.audio)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if rendered_audio is not None:
            audio_output = args.worker_audio_output.resolve()
            audio_output.parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(
                audio_output,
                mode="w",
                samplerate=runtime.sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as output_file:
                for audio in rendered_audio:
                    output_file.write(audio)
        audio_seconds = samples / runtime.sample_rate
        return {
            "generation_seconds": elapsed,
            "codec_frames": frames,
            "chunks": chunks,
            "audio_samples": samples,
            "audio_seconds": audio_seconds,
            "frames_per_second": frames / max(elapsed, 1e-9),
            "realtime_speed": audio_seconds / max(elapsed, 1e-9),
        }

    for warmup_index in range(args.warmup_runs):
        generate(-(warmup_index + 1))
    mx.reset_peak_memory()
    generation = generate(0)
    generation_metal_peak = int(mx.get_peak_memory())
    peak_rss = _peak_rss_bytes()

    result: dict[str, Any] = {
        "precision": actual_precision,
        "artifact": str(artifact_dir),
        **_artifact_sizes(artifact_dir, manifest),
        "load_seconds": load_seconds,
        "baseline_peak_rss_bytes": baseline_rss,
        "loaded_peak_rss_bytes": loaded_rss,
        "load_peak_rss_delta_bytes": max(0, loaded_rss - baseline_rss),
        "loaded_metal_bytes": loaded_metal,
        "generation_peak_rss_bytes": peak_rss,
        "generation_metal_peak_bytes": generation_metal_peak,
        "sample_rate": runtime.sample_rate,
        "audio_output": str(args.worker_audio_output.resolve()),
        "warmup_runs": args.warmup_runs,
        **generation,
    }
    args.worker_result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _worker_command(
    args: argparse.Namespace,
    precision: str,
    artifact: Path,
    result: Path,
    audio_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-artifact",
        str(artifact),
        "--worker-precision",
        precision,
        "--worker-result",
        str(result),
        "--worker-audio-output",
        str(audio_output),
        "--text",
        args.text,
        "--instruction",
        args.instruction,
        "--cfg-scale",
        str(args.cfg_scale),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--warmup-runs",
        str(args.warmup_runs),
        "--audio-device",
        args.audio_device,
        "--codec-chunk-frames",
        str(args.codec_chunk_frames),
    ]
    if args.greedy:
        command.append("--greedy")
    return command


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f}"


def _print_table(results: list[dict[str, Any]]) -> None:
    print()
    print(
        "| precision | main GiB | artifact GiB | load s | load RSS Δ GiB | "
        "Metal load GiB | frame/s | realtime | gen peak RSS GiB |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['precision']} "
            f"| {_gib(result['main_weights_bytes'])} "
            f"| {_gib(result['artifact_bytes'])} "
            f"| {result['load_seconds']:.2f} "
            f"| {_gib(result['load_peak_rss_delta_bytes'])} "
            f"| {_gib(result['loaded_metal_bytes'])} "
            f"| {result['frames_per_second']:.2f} "
            f"| {result['realtime_speed']:.2f}x "
            f"| {_gib(result['generation_peak_rss_bytes'])} |"
        )


def _parent(args: argparse.Namespace) -> None:
    from breeze_tts_mlx.convert import convert_checkpoint

    overrides = _parse_artifact_overrides(args.artifact)
    artifact_root = args.artifacts_dir.resolve()
    artifact_paths: dict[str, Path] = {}
    for precision in args.precisions:
        if precision == "original":
            original = overrides.get("original", args.source)
            if original is None:
                raise ValueError(
                    "--source chkpt-full or --artifact original=PATH is required "
                    "when benchmarking original"
                )
            artifact_paths[precision] = original.resolve()
        else:
            artifact_paths[precision] = overrides.get(
                precision, artifact_root / precision
            )

    for precision, artifact in artifact_paths.items():
        if precision == "original":
            index_path = artifact / "model.safetensors.index.json"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"Original checkpoint is incomplete: missing {index_path}"
                )
            continue
        manifest = artifact / "mlx_config.json"
        rebuild = args.convert_missing and args.overwrite_converted
        if manifest.is_file() and not rebuild:
            continue
        if not args.convert_missing:
            raise FileNotFoundError(
                f"Missing {precision} artifact: {artifact}. Pass --convert-missing "
                "with --source CHECKPOINT, or provide --artifact PRECISION=PATH."
            )
        if args.source is None:
            raise ValueError("--source is required with --convert-missing")
        print(f"Converting {precision} artifact -> {artifact}", flush=True)
        convert_checkpoint(
            args.source,
            artifact,
            overwrite=args.overwrite_converted,
            source_revision=args.source_revision,
            precision=precision,
        )

    results: list[dict[str, Any]] = []
    audio_output_dir = args.audio_output_dir.resolve()
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="breeze-mlx-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for precision in args.precisions:
            print(f"Benchmarking {precision} ...", flush=True)
            result_path = temp_root / f"{precision}.json"
            audio_output = audio_output_dir / f"{precision}.wav"
            subprocess.run(
                _worker_command(
                    args,
                    precision,
                    artifact_paths[precision],
                    result_path,
                    audio_output,
                ),
                check=True,
            )
            results.append(json.loads(result_path.read_text(encoding="utf-8")))

    report = {
        "format": "breeze-tts-mlx-benchmark",
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "settings": {
            "text": args.text,
            "instruction": args.instruction,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "greedy": args.greedy,
            "audio_device": args.audio_device,
            "audio_output_dir": str(audio_output_dir),
        },
        "results": results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_table(results)
    print(f"\nSaved JSON report: {args.json_output}")
    print(f"Saved listening samples: {audio_output_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original, INT8, and INT4 weight size, load memory, "
            "and generation speed"
        )
    )
    parser.add_argument("--source", type=Path, help="Complete Hugging Face checkpoint")
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("mlx-benchmark-artifacts")
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="PRECISION=PATH",
        help="Override one artifact path; may be repeated",
    )
    parser.add_argument(
        "--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS)
    )
    parser.add_argument("--convert-missing", action="store_true")
    parser.add_argument("--overwrite-converted", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--json-output", type=Path, default=Path("mlx_benchmark.json"))
    parser.add_argument(
        "--audio-output-dir",
        type=Path,
        default=Path("mlx_benchmark_audio"),
        help="Directory for measured-run WAV files (default: mlx_benchmark_audio)",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--audio-device", choices=("auto", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--codec-chunk-frames", type=int, default=2)
    parser.add_argument("--worker-artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-precision", choices=PRECISIONS, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-audio-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    _require_apple_silicon()
    args = _parser().parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    worker_values = (
        args.worker_artifact,
        args.worker_precision,
        args.worker_result,
        args.worker_audio_output,
    )
    if any(value is not None for value in worker_values):
        if not all(value is not None for value in worker_values):
            raise ValueError("internal worker arguments must be provided together")
        _worker(args)
        return
    _parent(args)


if __name__ == "__main__":
    main()
