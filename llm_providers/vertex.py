"""Vertex AI, authenticated with Application Default Credentials.

No API key: ``gcloud auth application-default login`` once, and the credentials
live in a file under the user's home directory. That is why this stays the
default on a machine that already has the gcloud SDK, and why it is not the
default in the repository -- on a fresh Windows install it means downloading
and configuring an SDK before anything works.

The project id is found rather than configured: the environment first, then the
gcloud CLI's active config, then the ADC file itself. A machine that can already
run ``gcloud`` needs nothing set here.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import Provider, ProviderUnavailable

logger = logging.getLogger("breeze.llm.vertex")

# Vertex's ``global`` endpoint, which is where the 3.x Flash Lite models live.
LOCATION = os.getenv("BREEZE_VERTEX_LOCATION", "global")

# Tried in order; the first one the project can actually call is cached and
# reused. Newer models are not available on every project at the same time, so
# this degrades rather than failing.
MODEL_CHAIN = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"client": None, "model": None, "project": None}


def resolve_project() -> str:
    """Find the Google Cloud project without asking the user to configure one."""
    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT"):
        value = (os.getenv(name) or "").strip().strip('"')
        if value:
            return value

    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            # Without this a gcloud.cmd shim opens a console window on Windows
            # every time the settings page is refreshed.
            **_no_window(),
        )
        project = result.stdout.strip()
        if project and project != "(unset)":
            return project
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("gcloud is not callable: %s", exc)

    adc = _adc_path()
    if adc.is_file():
        try:
            data = json.loads(adc.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        project = str(data.get("quota_project_id") or "").strip()
        if project:
            return project

    raise ProviderUnavailable(
        "No Google Cloud project. Run `gcloud auth application-default login`, "
        "or set GOOGLE_CLOUD_PROJECT in .env -- or switch BREEZE_LLM_PROVIDER "
        "to gemini and use an API key instead."
    )


def _adc_path() -> Path:
    """Where gcloud stores Application Default Credentials, per platform."""
    override = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "gcloud" / "application_default_credentials.json"
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def _no_window() -> dict[str, Any]:
    """Keep a console window from flashing when gcloud is shelled out to."""
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def _thinking_config(model: str) -> Any:
    """Turn thinking off. The knob differs across model generations."""
    from google.genai import types

    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_budget=0)


class VertexProvider(Provider):
    name = "vertex"
    label = "Google Vertex AI (gcloud login)"
    key_env = None

    def default_model(self) -> str:
        preferred = (os.getenv("BREEZE_LLM_MODEL") or "").strip()
        return preferred or MODEL_CHAIN[0]

    def _client(self) -> tuple[Any, str]:
        """A Vertex client plus the model id that answered, cached after the first call."""
        with _LOCK:
            if _STATE["client"] is not None and _STATE["model"]:
                return _STATE["client"], _STATE["model"]

            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise ProviderUnavailable(
                    "The google-genai package is not installed. Run "
                    "`pip install google-genai`."
                ) from exc

            project = resolve_project()
            try:
                client = genai.Client(vertexai=True, project=project, location=LOCATION)
            except Exception as exc:  # noqa: BLE001 - reported to the caller as data
                raise ProviderUnavailable(
                    f"Could not create a Vertex client: {exc}"
                ) from exc

            preferred = (os.getenv("BREEZE_LLM_MODEL") or "").strip()
            chain = (preferred, *MODEL_CHAIN) if preferred else MODEL_CHAIN

            errors: list[str] = []
            for model in chain:
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
                except Exception as exc:  # noqa: BLE001 - try the next model
                    errors.append(f"{model}: {str(exc)[:120]}")
                    continue
                logger.info(
                    "Vertex LLM ready: model=%s project=%s location=%s",
                    model, project, LOCATION,
                )
                _STATE.update(client=client, model=model, project=project)
                return client, model

            raise ProviderUnavailable(
                "No model in the chain was callable -- " + "; ".join(errors)
            )

    def check(self) -> str:
        _client, model = self._client()
        return model

    def status(self) -> dict[str, Any]:
        base = super().status()
        base["project"] = _STATE.get("project")
        base["location"] = LOCATION
        return base

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

        client, resolved = self._client()
        resolved = model or resolved
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
