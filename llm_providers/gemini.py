"""Google's Gemini API, authenticated with an API key from AI Studio.

The same models as Vertex, reached the other way. This is the default on a
fresh machine because it needs one line in ``.env`` and no CLI login, where
Vertex needs the gcloud SDK installed and a project configured.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from typing import Any

from .base import Provider, ProviderUnavailable

# Cheap, fast, and quite good enough for laying out prose to be read aloud --
# which is the only thing this project asks a language model to do.
DEFAULT_MODEL = "gemini-2.5-flash-lite"

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"client": None, "model": None}


def _thinking_config(model: str) -> Any:
    """Turn thinking off. The knob differs across model generations.

    Worth the branch: with output going straight to a speaker, a model that
    stops to think adds seconds of silence before the first word.
    """
    from google.genai import types

    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_budget=0)


class GeminiProvider(Provider):
    name = "gemini"
    label = "Google Gemini (API key)"
    key_env = "GEMINI_API_KEY"

    def api_key(self) -> str:
        for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = (os.getenv(env) or "").strip().strip('"')
            if value:
                return value
        raise ProviderUnavailable(
            "No Gemini API key. Put GEMINI_API_KEY in .env -- create one at "
            "https://aistudio.google.com/apikey -- or set BREEZE_LLM_PROVIDER "
            "to another provider."
        )

    def default_model(self) -> str:
        return (os.getenv("BREEZE_GEMINI_MODEL") or DEFAULT_MODEL).strip()

    def _client(self) -> Any:
        with _LOCK:
            if _STATE["client"] is not None:
                return _STATE["client"]
            key = self.api_key()
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderUnavailable(
                    "The google-genai package is not installed. Run "
                    "`pip install google-genai`."
                ) from exc
            try:
                client = genai.Client(api_key=key)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                raise ProviderUnavailable(
                    f"Could not create a Gemini client: {exc}"
                ) from exc
            _STATE["client"] = client
            return client

    def check(self) -> str:
        model = self.default_model()
        with _LOCK:
            if _STATE["model"] == model:
                return model
        from google.genai import types

        client = self._client()
        try:
            client.models.generate_content(
                model=model,
                contents="ok",
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.0,
                    thinking_config=_thinking_config(model),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - reported as data, not a traceback
            raise ProviderUnavailable(f"{model} did not answer: {str(exc)[:200]}") from exc
        with _LOCK:
            _STATE["model"] = model
        return model

    def stream(
        self,
        text: str,
        *,
        system: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
    ) -> Iterator[str]:
        from google.genai import types

        client = self._client()
        resolved = model or self.default_model()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_config(resolved),
        )
        stream = client.models.generate_content_stream(
            model=resolved, contents=text, config=config
        )
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if delta:
                yield delta
