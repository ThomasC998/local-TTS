#!/usr/bin/env python3
"""Fetch the speech model this machine needs, into the directory it expects.

Different checkpoints, because the two backends genuinely need different files
-- see MODELS.md for the whole story, and for how to point this at another one:

    macOS    an MLX INT8 artifact, quantized for Apple Silicon     ~3.5 GB
    Windows  a torchao INT8 quantization of the original, for CUDA ~5.9 GB
             or the original BF16 weights: --variant torch-bf16    ~7 GB

    python download_model.py                     # whatever this machine needs
    python download_model.py --check             # is it already here?
    python download_model.py --variant torch-bf16
    python download_model.py --repo OWNER/NAME --dest ./somewhere

Resumable: interrupted downloads pick up where they stopped, so a dropped
connection costs the current file rather than the whole checkpoint.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# What to download, per backend.
#
# These are the two identifiers to change when moving to another Breeze TTS 2
# variant -- a newer release, or a differently quantized artifact. MODELS.md
# explains how to find a replacement and what a valid one has to contain.
# BREEZE_MODEL_REPO and BREEZE_MODEL_REVISION override them without editing
# this file.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, object]] = {
    "mlx": {
        "backend": "mlx",
        "repo": "rishikksh20/Breeze-TTS-2-mlx",
        "revision": None,  # whatever is current: this artifact is not versioned
        "dest": "chkpt-mlx-int8",
        "size": "~3.5 GB",
        "what": "MLX INT8 artifact for Apple Silicon",
        # Self-contained already: per-component safetensors plus the FP32 Qwen
        # audio tokenizer, and nothing else in the repository.
        "allow": None,
        "required": ("mlx_config.json", "config.json", "backbone.safetensors"),
    },
    "torch-int8": {
        "backend": "torch",
        "repo": "smcleod/Breeze-TTS-2-int8",
        "revision": None,
        "dest": "chkpt-breeze-tts-2-int8",
        "size": "~5.9 GB",
        "what": "torchao INT8 quantization of the original, for CUDA",
        # Pickle shards, not safetensors -- transformers cannot currently
        # round-trip an INT8 torchao checkpoint through safetensors, so the
        # .bin route is what keeps it loadable. The bundled inference code
        # (breeze_infer/, models/, infer.py) comes along because it is small
        # and it is the reference implementation to compare against if our own
        # runtime disagrees with it.
        "allow": [
            "config.json",
            "generation_config.json",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "audio_tokenizer/*",
            "breeze_infer/*",
            "models/*",
            "configs/*",
            "infer.py",
            "*LICENSE*",
        ],
        "required": (
            "config.json",
            "pytorch_model.bin.index.json",
            "audio_tokenizer",
        ),
        "needs": ("torchao",),
    },
    "torch-bf16": {
        "backend": "torch",
        "repo": "BreezeBlue/Breeze-TTS-2",
        # Pinned: this is the revision the MLX port was converted from, so both
        # platforms are running the same weights rather than merely the same
        # model name. Unpin it deliberately, not by accident.
        "revision": "c1c8ca18b70b30822735633991d9ebf4898e47d4",
        "dest": "chkpt-breeze-tts-2",
        "size": "~7 GB",
        "what": "original Hugging Face checkpoint, BF16, for CUDA",
        # The repository also carries assets and licences. These are what
        # inference reads, and skipping the rest is download that would never
        # be opened.
        "allow": [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "audio_tokenizer/*",
        ],
        "required": (
            "config.json",
            "model.safetensors.index.json",
            "audio_tokenizer",
        ),
    },
}

# What each backend downloads when nothing is asked for. Windows takes the INT8
# conversion: it is 1.8 GB smaller than BF16 and about 5.2 GB of weights rather
# than 7, which is the difference between fitting an 8 GB card comfortably and
# not. Accuracy cost is roughly 1% mean relative error per layer, which is well
# under what changing the seed does.
DEFAULT_VARIANT = {"mlx": "mlx", "torch": "torch-int8"}


def variants_for(backend: str) -> list[str]:
    return [name for name, source in SOURCES.items() if source["backend"] == backend]


def backend_name() -> str:
    """Which backend this machine will use, without importing the runtime.

    Deliberately not ``tts_backends.load_backend()``: this script has to run
    *before* the model is there, on a machine where torch may not be installed
    yet, and asking that question should not require either to be true.
    """
    explicit = (os.getenv("BREEZE_BACKEND") or "").strip().lower()
    if explicit in SOURCES:
        return explicit
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"


def _warn_about_extra_packages(source: dict[str, object]) -> None:
    """Say now if the checkpoint will need a package that is not installed.

    Better here than after six gigabytes: a torchao checkpoint that loads
    without torchao fails inside pickle, with a message about an unsupported
    global that names neither the package nor the checkpoint.
    """
    from importlib.util import find_spec

    missing = [
        name for name in source.get("needs", ()) if find_spec(name) is None
    ]
    if missing:
        print(
            f"  note     this checkpoint needs {', '.join(missing)}, which is not "
            f"installed.\n           Run: pip install {' '.join(missing)}"
        )


def describe(source: dict[str, object], dest: Path) -> None:
    print(f"  model    {source['repo']}")
    if source.get("revision"):
        print(f"  revision {source['revision']}")
    print(f"  into     {dest}")
    print(f"  size     {source['size']}  ({source['what']})")


def is_present(source: dict[str, object], dest: Path) -> bool:
    """Whether the checkpoint looks complete enough to load.

    A file-count check, not a checksum: the point is to catch "you have not
    downloaded it yet" and "the download stopped half way", both of which show
    up as a missing marker file. Corruption is the loader's problem, and it
    reports it with the tensor name.
    """
    if not dest.is_dir():
        return False
    return all((dest / name).exists() for name in source["required"])  # type: ignore[index]


def download(source: dict[str, object], dest: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is not installed. Run:\n"
            "    pip install huggingface_hub[hf_transfer]",
            file=sys.stderr,
        )
        return 1

    print("Downloading. This is large, and resumable — interrupting it and")
    print("running this again picks up where it stopped.\n")
    try:
        snapshot_download(
            repo_id=str(source["repo"]),
            revision=source["revision"],  # type: ignore[arg-type]
            local_dir=str(dest),
            allow_patterns=source["allow"],  # type: ignore[arg-type]
            max_workers=4,
        )
    except Exception as exc:  # noqa: BLE001 - one readable message, not a traceback
        print(f"\nerror: download failed: {exc}", file=sys.stderr)
        print(
            "\nIf the model is gated, log in first with `hf auth login`. "
            "If the repository has moved, see MODELS.md for how to point this "
            "at another one.",
            file=sys.stderr,
        )
        return 1

    if not is_present(source, dest):
        missing = [
            name for name in source["required"] if not (dest / name).exists()  # type: ignore[index]
        ]
        print(
            f"\nerror: the download finished but {dest} is missing {missing}. "
            "The repository layout may have changed — see MODELS.md.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument(
        "--variant",
        choices=sorted(SOURCES),
        default=None,
        help="which checkpoint; default is this machine's backend's own",
    )
    parser.add_argument("--repo", help="override the Hugging Face repository")
    parser.add_argument("--revision", help="override the revision")
    parser.add_argument("--dest", help="override the destination directory")
    parser.add_argument(
        "--force", action="store_true", help="download even if it is already here"
    )
    args = parser.parse_args(argv)

    backend = backend_name()
    variant = args.variant or DEFAULT_VARIANT[backend]
    source = dict(SOURCES[variant])
    backend = str(source["backend"])
    source["repo"] = args.repo or os.getenv("BREEZE_MODEL_REPO") or source["repo"]
    if args.revision or os.getenv("BREEZE_MODEL_REVISION"):
        source["revision"] = args.revision or os.getenv("BREEZE_MODEL_REVISION")
    if args.repo and not args.revision:
        # A pinned revision belongs to the repository it was taken from.
        source["revision"] = None

    # BREEZE_MODEL names *this* machine's checkpoint, so it only applies when
    # downloading for this machine's own backend -- otherwise asking a Mac to
    # fetch the CUDA checkpoint would aim it at the MLX directory.
    configured = (
        os.getenv("BREEZE_MODEL")
        if variant == DEFAULT_VARIANT[backend_name()]
        else None
    )
    dest = Path(args.dest or configured or str(source["dest"]))
    if not dest.is_absolute():
        dest = PROJECT / dest

    print(f"Breeze TTS 2 model — {variant}")
    describe(source, dest)
    _warn_about_extra_packages(source)
    print()

    if is_present(source, dest) and not args.force:
        print("Already here. Nothing to do. (--force re-downloads.)")
        return 0
    if args.check:
        print("Not downloaded yet. Run this without --check to fetch it.")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    status = download(source, dest)
    if status == 0:
        print(f"\nDone. {dest} is ready.")
        if dest.name != str(SOURCES[variant]["dest"]):
            print(f"Set BREEZE_MODEL={dest} in .env so the server finds it.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
