"""CUDA inference for Breeze TTS 2, for the Windows side of this project.

A line-for-line port of ``breeze_tts_mlx`` onto PyTorch. Only the six modules
that actually touched MLX were rewritten -- the attention layers, the three
sub-models, the weight loader and the generation loop. Prompt collation,
sampling, reference-audio loading and the Qwen codec are shared with the Mac
backend unchanged; see ``_shared.py`` for why that is safe.

The port is deliberate about two things, because they are what makes the two
platforms sound the same rather than merely both work:

*Sampling stays on the CPU in NumPy.* ``NumpySampler`` is the same object the
MLX runtime uses, seeded the same way, fed logits cast to FP32 at the same
point in the loop. Same seed and same prompt therefore give the same token
sequence on both machines -- the audio differs only by the floating-point noise
of two different matrix-multiply kernels, not by a different sampling RNG.

*The original checkpoint is what loads here.* MLX's INT8 artifact is packed in
MLX's own affine quantization format, which PyTorch cannot read. So Windows
loads the untouched Hugging Face shards -- see MODELS.md -- in BF16 or FP16.
"""

from .model import BreezeTorchModel
from .runtime import BreezeTorchRuntime, TorchAudioChunk, TorchRuntimeConfig

__all__ = [
    "BreezeTorchModel",
    "BreezeTorchRuntime",
    "TorchAudioChunk",
    "TorchRuntimeConfig",
]
