"""The parts of Breeze TTS 2 that are identical on Apple Silicon and CUDA.

Prompt collation, the sampler, reference-audio loading and the whole Qwen codec
streaming runtime are plain PyTorch and NumPy. They contain no MLX at all --
they were only ever *stored* in ``breeze-tts-mlx/`` because that is the vendored
upstream port this project started from. Running them twice, once per platform,
would mean maintaining two copies of the prompt format, which is exactly the
kind of drift that makes a cloned voice sound different on the other machine.

So the CUDA backend imports them from there. This module exists so that the
answer to "why is the CUDA runtime importing a package called mlx" is written
down once, here, instead of being re-asked at every import site.

``breeze_tts_mlx/__init__.py`` imports only ``config``, which is pure JSON
handling, so importing the package on a machine with no MLX installed is
harmless -- which is the whole reason this indirection works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The MLX package is a sibling checkout, not a pip install.
_MLX_DIR = Path(__file__).resolve().parents[2] / "breeze-tts-mlx"
if _MLX_DIR.is_dir() and str(_MLX_DIR) not in sys.path:
    sys.path.insert(0, str(_MLX_DIR))

from breeze_tts_mlx.audio_codec import AudioTokenizer  # noqa: E402
from breeze_tts_mlx.checkpoint import (  # noqa: E402
    classify_checkpoint,
    load_weight_map,
    map_source_tensor,
)
from breeze_tts_mlx.config import BreezeMLXConfig as BreezeConfig  # noqa: E402
from breeze_tts_mlx.sampling import NumpySampler, SamplingConfig  # noqa: E402
from breeze_tts_mlx.templates import get_template, prepare_inputs  # noqa: E402

__all__ = [
    "AudioTokenizer",
    "BreezeConfig",
    "NumpySampler",
    "SamplingConfig",
    "classify_checkpoint",
    "get_template",
    "load_weight_map",
    "map_source_tensor",
    "prepare_inputs",
]
