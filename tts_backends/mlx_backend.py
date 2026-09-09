"""Apple Silicon: the MLX runtime over the INT8 artifact.

This is the path this project has always taken on the Mac. Nothing about its
behaviour changed when the Windows backend arrived -- the code simply moved
here from ``breeze_pipeline`` so that the engine above it stopped naming MLX.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

# The MLX runtime lives in the cloned repo, which is not pip-installed.
_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "breeze-tts-mlx"
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

NAME = "mlx"
CANDIDATE_MODEL_PATHS = ("./chkpt-mlx-int8",)
DEFAULT_MODEL_PATH = CANDIDATE_MODEL_PATHS[0]


def resolve_model_path() -> str:
    """The checkpoint directory to use. Only one shape of artifact runs here."""
    root = Path(__file__).resolve().parent.parent
    for candidate in CANDIDATE_MODEL_PATHS:
        if (root / candidate).is_dir():
            return candidate
    return DEFAULT_MODEL_PATH


def unavailable_reason() -> str | None:
    """None when this backend can run, otherwise why it cannot."""
    if platform.system() != "Darwin":
        return "MLX runs only on macOS"
    if platform.machine() != "arm64":
        return "MLX needs Apple Silicon; this is an Intel Mac"
    try:
        import mlx.core  # noqa: F401
    except ImportError as exc:
        return f"mlx is not installed ({exc})"
    return None


def describe() -> dict[str, Any]:
    import mlx.core as mx

    return {
        "device": f"Apple Silicon (Metal), {platform.machine()}",
        "precision": "int8",
        "model_path": resolve_model_path(),
        "cache_limit": int(mx.get_cache_memory()),
    }


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
    from breeze_tts_mlx.runtime import BreezeMLXRuntime, MLXRuntimeConfig

    runtime = BreezeMLXRuntime(
        model_path,
        audio_device=audio_device,
        seed=seed,
        config=MLXRuntimeConfig(
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            repetition_penalty=repetition_penalty,
            codec_chunk_frames=codec_chunk_frames,
            backbone_sampling=sampling,
            depth_sampling=sampling,
        ),
    )

    # Both allocators keep freed blocks for reuse rather than returning them, so
    # a burst of long generations leaves the process holding several spare
    # gigabytes. MLX can bound its own pool, which costs nothing per request.
    import mlx.core as mx

    cache_limit_gb = float(os.getenv("BREEZE_MLX_CACHE_LIMIT_GB", "1.0"))
    if cache_limit_gb > 0:
        mx.set_cache_limit(int(cache_limit_gb * 1024**3))
    return runtime


def memory_stats(runtime: Any) -> dict[str, int]:
    """Unified-memory accounting, in bytes.

    Two allocators are in play and both keep freed blocks for reuse rather than
    handing them back, which is why "VRAM" only ever appears to climb:

    * MLX, for the backbone and depth decoder. ``active`` is live data (model
      weights plus whatever a generation is holding); ``cache`` is free blocks
      kept for the next allocation.
    * PyTorch on MPS, for the audio tokenizer -- the part that encodes reference
      recordings and decodes codec frames.
    """
    import mlx.core as mx

    stats = {
        "mlx_active": int(mx.get_active_memory()),
        "mlx_cache": int(mx.get_cache_memory()),
        "mlx_peak": int(mx.get_peak_memory()),
    }
    try:
        import torch

        if torch.backends.mps.is_available():
            stats["torch_mps_allocated"] = int(torch.mps.current_allocated_memory())
            stats["torch_mps_driver"] = int(torch.mps.driver_allocated_memory())
    except Exception:  # noqa: BLE001 - reporting must never break a request
        pass
    return stats


def trim_memory(runtime: Any) -> dict[str, int]:
    """Hand cached-but-free blocks back. Weights are live and stay resident."""
    import mlx.core as mx

    before_cache = int(mx.get_cache_memory())
    mx.clear_cache()
    released = {"mlx_cache": before_cache - int(mx.get_cache_memory())}
    try:
        import torch

        if torch.backends.mps.is_available():
            before_driver = int(torch.mps.driver_allocated_memory())
            torch.mps.empty_cache()
            released["torch_mps"] = before_driver - int(
                torch.mps.driver_allocated_memory()
            )
    except Exception:  # noqa: BLE001
        pass
    return released


# The pool this backend cannot bound itself, and so has to watch.
GROWTH_KEY = "torch_mps_driver"
GROWTH_BUDGET_ENV = "BREEZE_MPS_CACHE_BUDGET_GB"
