from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Installing SoX is the one native step neither pip nor conda covers, and the
# error qwen_tts raises without it names a temporary file rather than the
# missing program -- so the instruction is spelled out per platform here.
_SOX_HINT = {
    "Darwin": (
        "Install it with `brew install sox`, then confirm `sox --version` works. "
        "On Apple Silicon, ensure /opt/homebrew/bin is on PATH."
    ),
    "Windows": (
        "Install it with `winget install --id ChrisBagwell.SoX`, then open a new "
        "terminal and confirm `sox --version` works. If winget put it somewhere "
        "off PATH, add the SoX folder (usually "
        r"C:\Program Files (x86)\sox-14-4-2) to PATH."
    ),
}


def require_sox() -> str:
    """Return the SoX executable or fail before qwen_tts prints a cryptic error."""
    executable = shutil.which("sox")
    if executable is None:
        hint = _SOX_HINT.get(
            platform.system(), "Install SoX with your system package manager."
        )
        raise RuntimeError(
            f"The Qwen audio tokenizer requires the native SoX executable. {hint}"
        )
    return executable


def resolve_audio_device(requested: str, *, dtype: torch.dtype) -> torch.device:
    """Choose where the codec runs: Metal on a Mac, CUDA on Windows, else CPU.

    The codec is small next to the main model, but it decodes every chunk on the
    critical path between the sampler and the speaker, so it belongs on the same
    accelerator the main model is using rather than on the CPU beside it.
    """
    if requested in ("", "auto"):
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("audio device must be one of: auto, cpu, mps, cuda")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested for the audio tokenizer but is unavailable"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for the audio tokenizer but is unavailable"
        )
    if dtype == torch.float16 and device.type == "cpu":
        raise RuntimeError(
            "An FP16 codec needs an accelerator; on the CPU use float32"
        )
    return device


def _extract_audio(decoded: Any) -> np.ndarray:
    if hasattr(decoded, "audio_values"):
        decoded = decoded.audio_values
    while isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, np.ndarray):
        audio = decoded
    elif isinstance(decoded, torch.Tensor):
        audio = decoded.detach().float().cpu().numpy()
    else:
        raise TypeError(f"Unsupported decoded audio type: {type(decoded)!r}")
    while audio.ndim > 1:
        audio = audio[0]
    return np.ascontiguousarray(audio, dtype=np.float32)


class AudioTokenizer:
    """Qwen codec with an explicit FP32 or FP16 execution policy."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "float32",
    ) -> None:
        require_sox()
        from qwen_tts import Qwen3TTSTokenizer

        if dtype not in {"float16", "float32"}:
            raise ValueError("audio tokenizer dtype must be float16 or float32")
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.device = resolve_audio_device(device, dtype=self.dtype)
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Missing bundled audio tokenizer: {self.model_dir}"
            )
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(self.model_dir), device_map=str(self.device), dtype=self.dtype
        )
        if self.tokenizer.model is None:
            raise RuntimeError("Qwen audio tokenizer loaded without its neural model")
        self.tokenizer.model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.tokenizer.model.parameters():
            parameter.requires_grad_(False)
            if parameter.is_floating_point() and parameter.dtype != self.dtype:
                raise RuntimeError(
                    f"audio tokenizer must remain {self.dtype}, found "
                    f"parameter dtype {parameter.dtype}"
                )
        self._stream_runtimes: dict[int, Any] = {}

    @property
    def model(self) -> Any:
        return self.tokenizer.model

    @property
    def sample_rate(self) -> int:
        config = self.tokenizer.model.decoder.config
        return int(getattr(config, "sampling_rate", 24000))

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode():
            return self.tokenizer.encode(*args, **kwargs)

    def stream_runtime(self, chunk_frames: int) -> Any:
        runtime = self._stream_runtimes.get(chunk_frames)
        if runtime is not None:
            return runtime
        from .codec_stream.stream.runtime import (
            MultiRequestStreamRuntime,
            QwenStreamRuntimeConfig,
        )

        runtime = MultiRequestStreamRuntime(
            self.tokenizer,
            QwenStreamRuntimeConfig(
                chunk_frames=chunk_frames,
                non_integer_chunk_strategy="eager",
                num_lanes=1,
                max_active_reqs=1,
                fast=False,
                lifecycle_assert_mode="raise",
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self._stream_runtimes[chunk_frames] = runtime
        return runtime

    def decode_chunk(
        self,
        runtime: Any,
        request_id: str,
        frames: list[np.ndarray],
        *,
        reset: bool,
    ) -> np.ndarray:
        frame_array = np.stack(frames, axis=0).astype(np.int64, copy=False)
        codes = torch.from_numpy(frame_array.T[None]).to(
            device=self.device, dtype=torch.long
        )
        with torch.inference_mode():
            decoded = runtime.decode_request_chunk(request_id, codes, reset=reset)
        return _extract_audio(decoded)
