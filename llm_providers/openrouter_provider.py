"""OpenRouter as two providers: the free tier, and a fixed paid model.

They are listed separately rather than as one provider with a flag because
choosing between them is a real decision with different consequences -- free
means an unpredictable model and per-key rate limits, paid means a model you
picked and a bill -- and a setting that hides that behind a checkbox invites
picking it by accident.

The client itself is in ``openrouter.py``, which imports nothing from this
project so it can be lifted into another one. This file is only the adapter.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from typing import Any

from .base import Provider, ProviderUnavailable
from .openrouter import FREE_ROUTER, OpenRouterClient, OpenRouterError, pick_free_model

# A small, cheap, widely available instruct model. Stated rather than
# discovered, which is the whole point of the paid path.
DEFAULT_PAID_MODEL = "google/gemini-2.0-flash-lite-001"

# Sent so usage shows up attributed on OpenRouter's dashboard. Harmless if the
# project is not listed there.
_REFERER = "https://github.com/breeze-tts"
_TITLE = "Breeze TTS"

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"client": None, "free_model": None}


def _client() -> OpenRouterClient:
    with _LOCK:
        if _STATE["client"] is not None:
            return _STATE["client"]
        try:
            client = OpenRouterClient(referer=_REFERER, title=_TITLE)
        except OpenRouterError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        _STATE["client"] = client
        return client


class _OpenRouterBase(Provider):
    key_env = "OPENROUTER_API_KEY"

    def check(self) -> str:
        _client()  # raises with an actionable message when the key is missing
        return self.default_model()

    def stream(
        self,
        text: str,
        *,
        system: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
    ) -> Iterator[str]:
        client = _client()
        resolved = model or self.default_model()
        try:
            yield from client.stream(
                text,
                model=resolved,
                system=system,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except OpenRouterError as exc:
            raise ProviderUnavailable(str(exc)) from exc


class OpenRouterFreeProvider(_OpenRouterBase):
    """Whichever model is free right now, rediscovered rather than hard-coded."""

    name = "openrouter_free"
    label = "OpenRouter (free models)"

    def default_model(self) -> str:
        """Look up the best free model, and hold onto it for the session.

        Rediscovering per request would put an HTTP round trip in front of every
        hotkey press. Caching for the process means a model going paid is
        noticed at the next restart, which for a desktop tool is soon enough --
        and a stale choice fails loudly with a 402, not silently.
        """
        configured = (os.getenv("BREEZE_OPENROUTER_MODEL") or "").strip()
        if configured:
            return configured
        with _LOCK:
            if _STATE["free_model"]:
                return _STATE["free_model"]
        prefer = [
            part.strip()
            for part in (os.getenv("BREEZE_OPENROUTER_PREFER") or "").split(",")
            if part.strip()
        ]
        chosen = pick_free_model(prefer=prefer)
        with _LOCK:
            _STATE["free_model"] = chosen
        return chosen

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["note"] = (
            "The model is chosen at start-up from whatever OpenRouter lists at "
            f"$0. Falls back to {FREE_ROUTER} if the catalogue is unreachable. "
            "Free models are rate limited per key."
        )
        return base


class OpenRouterPaidProvider(_OpenRouterBase):
    """One model, named in configuration, billed to the account."""

    name = "openrouter_paid"
    label = "OpenRouter (fixed model)"

    def default_model(self) -> str:
        return (
            os.getenv("BREEZE_OPENROUTER_MODEL") or DEFAULT_PAID_MODEL
        ).strip()

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["note"] = (
            "Set BREEZE_OPENROUTER_MODEL in .env to any model id from "
            "https://openrouter.ai/models. Requests are billed to the key."
        )
        return base
