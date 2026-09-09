"""Which Breeze TTS 2 runtime this machine can run, and how to build it.

There is one engine in this project -- ``BreezeEngine`` in ``breeze_pipeline``
-- and it holds all the behaviour worth arguing about: sentence chunking, the
voice lock, paragraph gaps, request serialization, cancellation. None of that
is platform-specific, and none of it is duplicated.

What *is* platform-specific is small and lives here: constructing the model
runtime, and asking the allocator how much memory it is sitting on. Apple
Silicon runs MLX with an INT8 artifact; Windows runs PyTorch on CUDA with the
original Hugging Face checkpoint. Both then expose the identical surface the
engine drives -- ``tokenizer``, ``audio_tokenizer``, ``runtime_config``,
``sampler``, ``config``, ``device``, ``sample_rate`` and ``iter_audio_chunks``
-- so above this line nothing knows which machine it is on.

Choosing
--------
``BREEZE_BACKEND`` picks one explicitly (``mlx`` or ``torch``). Left unset, the
first backend that imports on this machine wins, in platform order. An explicit
choice that cannot load raises rather than falling back: silently running the
CPU path would produce a server that starts fine and then never keeps up with
the player, which is a much worse failure than not starting.
"""

from __future__ import annotations

import os
import platform
from importlib import import_module
from typing import Any

# Import order per platform. Apple Silicon has no CUDA and Windows has no MLX,
# so in practice exactly one of these ever imports.
_PREFERENCE = {
    "Darwin": ("mlx", "torch"),
    "Windows": ("torch",),
    "Linux": ("torch",),
}

_MODULES = {"mlx": ".mlx_backend", "torch": ".torch_backend"}


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run here, with the reason attached."""


def _load(name: str) -> Any:
    if name not in _MODULES:
        raise BackendUnavailable(
            f"Unknown backend {name!r}; expected one of {sorted(_MODULES)}"
        )
    return import_module(_MODULES[name], package=__name__)


def preferred_order() -> tuple[str, ...]:
    return _PREFERENCE.get(platform.system(), ("torch", "mlx"))


def load_backend(name: str | None = None) -> Any:
    """The backend module to build the runtime with.

    Pass a name, or set ``BREEZE_BACKEND``, to pin one. Otherwise the first
    importable backend for this platform is used.
    """
    requested = (name or os.getenv("BREEZE_BACKEND") or "").strip().lower()
    if requested:
        backend = _load(requested)
        reason = backend.unavailable_reason()
        if reason:
            raise BackendUnavailable(
                f"BREEZE_BACKEND={requested} was requested but cannot run: {reason}"
            )
        return backend

    reasons: list[str] = []
    for candidate in preferred_order():
        try:
            backend = _load(candidate)
        except Exception as exc:  # noqa: BLE001 - report, then try the next one
            reasons.append(f"{candidate}: {exc}")
            continue
        reason = backend.unavailable_reason()
        if reason is None:
            return backend
        reasons.append(f"{candidate}: {reason}")
    detail = "\n  ".join(reasons) or "no backend modules for this platform"
    raise BackendUnavailable(
        "No Breeze TTS backend can run on this machine:\n  " + detail
    )


def survey() -> list[dict[str, Any]]:
    """Every backend and whether it could run, for /v1/capabilities."""
    rows: list[dict[str, Any]] = []
    for name in _MODULES:
        try:
            backend = _load(name)
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "available": False, "reason": str(exc)})
            continue
        reason = backend.unavailable_reason()
        row = {"name": name, "available": reason is None}
        if reason is not None:
            row["reason"] = reason
        else:
            row.update(backend.describe())
        rows.append(row)
    return rows


def default_model_path() -> str:
    """Where each backend's checkpoint is expected to sit, unconfigured.

    The two are different directories on purpose: they hold different files --
    MLX's INT8 artifact and the original Hugging Face shards -- and a machine
    that has both should not have to rename one to switch.
    """
    configured = os.getenv("BREEZE_MODEL")
    if configured:
        return configured
    try:
        return load_backend().resolve_model_path()
    except BackendUnavailable:
        return "./chkpt-mlx-int8"
