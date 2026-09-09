"""Windows and Linux: the PyTorch runtime on an NVIDIA GPU.

The counterpart to ``mlx_backend``, over ``breeze_tts_torch``. The engine above
cannot tell the two apart; the differences that remain are the ones that are
genuinely different -- a discrete card with its own VRAM instead of unified
memory, and the original Hugging Face checkpoint instead of MLX's INT8 artifact.

There is no CPU fallback here on purpose. See ``resolve_device`` in
``breeze_tts_torch.runtime``: on the CPU this model generates far slower than
real time, so the streaming player it feeds would underrun continuously. A
server that refuses to start is easier to diagnose than one that stutters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# The torch runtime lives in a sibling checkout, not a pip install.
_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "breeze-tts-torch"
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

NAME = "torch"
DEFAULT_MODEL_PATH = "./chkpt-breeze-tts-2"

# BF16 by default: the checkpoint was trained in it, and every NVIDIA card from
# Ampere (RTX 30-series) on runs it natively. Set BREEZE_TORCH_DTYPE=float16 on
# an older card -- Turing and Pascal emulate BF16 slowly.
DTYPE_ENV = "BREEZE_TORCH_DTYPE"
DEVICE_ENV = "BREEZE_TORCH_DEVICE"


def unavailable_reason() -> str | None:
    """None when this backend can run, otherwise why it cannot."""
    try:
        import torch
    except ImportError as exc:
        return f"torch is not installed ({exc})"
    if os.getenv(DEVICE_ENV, "auto").strip().lower() == "cpu":
        return None  # explicitly asked for, for loading tests only
    if not torch.cuda.is_available():
        return (
            "no CUDA device is visible to PyTorch. Check `nvidia-smi`, and that "
            "torch was installed from the CUDA index rather than the CPU one"
        )
    return None


def describe() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "device": "cpu",
        "precision": os.getenv(DTYPE_ENV, "bfloat16"),
        "model_path": DEFAULT_MODEL_PATH,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        info.update(
            {
                "device": torch.cuda.get_device_name(index),
                "cuda_version": torch.version.cuda,
                "memory_total": int(total),
                "memory_free": int(free),
            }
        )
    return info


def build(
    model_path: str | Path,
    *,
    audio_device: str,
    seed: int,
    sampling: Any,
    max_new_tokens: int,
    max_seq_len: int,
    repetition_penalty: float,
    codec_chunk_frames: int,
) -> Any:
    from breeze_tts_torch.runtime import BreezeTorchRuntime, TorchRuntimeConfig

    # "auto" means the same thing to both backends -- put the codec wherever the
    # main model went -- so the server's --audio-device flag keeps working
    # unchanged, and only the resolved answer differs.
    codec_device = None if audio_device in ("", "auto") else audio_device
    return BreezeTorchRuntime(
        model_path,
        device=os.getenv(DEVICE_ENV, "auto"),
        dtype=os.getenv(DTYPE_ENV, "bfloat16"),
        audio_device=codec_device,
        seed=seed,
        config=TorchRuntimeConfig(
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            repetition_penalty=repetition_penalty,
            codec_chunk_frames=codec_chunk_frames,
            backbone_sampling=sampling,
            depth_sampling=sampling,
        ),
    )


def memory_stats(runtime: Any) -> dict[str, int]:
    """VRAM accounting, in bytes.

    ``allocated`` is live tensors -- weights plus whatever the current
    generation holds. ``reserved`` is what the caching allocator has claimed
    from the driver and keeps for reuse, which is the number that only ever
    seems to grow. ``free``/``total`` come from the driver, so they include
    whatever else on the machine is using the card.
    """
    import torch

    if not torch.cuda.is_available():
        return {}
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {
        "cuda_allocated": int(torch.cuda.memory_allocated(index)),
        "cuda_reserved": int(torch.cuda.memory_reserved(index)),
        "cuda_peak": int(torch.cuda.max_memory_allocated(index)),
        "cuda_free": int(free),
        "cuda_total": int(total),
    }


def trim_memory(runtime: Any) -> dict[str, int]:
    """Return the allocator's cached-but-free blocks to the driver.

    Weights are live allocations and are untouched, so the checkpoint stays
    resident -- this only releases the pool that a burst of long generations
    left claimed.
    """
    import torch

    if not torch.cuda.is_available():
        return {}
    index = torch.cuda.current_device()
    before = int(torch.cuda.memory_reserved(index))
    torch.cuda.empty_cache()
    return {"cuda_reserved": before - int(torch.cuda.memory_reserved(index))}


# The pool this backend cannot bound itself, and so has to watch.
GROWTH_KEY = "cuda_reserved"
GROWTH_BUDGET_ENV = "BREEZE_CUDA_CACHE_BUDGET_GB"
