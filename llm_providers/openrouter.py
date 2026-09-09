"""OpenRouter: free-model discovery and OpenAI-compatible chat, standalone.

This module deliberately imports nothing from the rest of this project. Drop
the file into another codebase, give it an API key, and it works -- that is
what it is for. Its only dependency is ``requests``.

Two ways to use OpenRouter, and both are here
---------------------------------------------
*Free.* OpenRouter's catalogue always contains some models priced at zero, but
which ones changes week to week -- a model that was free in March is a paid
endpoint by June, and hard-coding one is how a working integration quietly
starts returning 402. So the catalogue is fetched and filtered at run time, and
the best free model is chosen from what is actually free right now. There is
also ``openrouter/free``, a router slug that spreads requests across whatever
free capacity exists; it is the fallback when the catalogue cannot be reached
at all, since it needs no discovery to work.

*Paid.* A fixed model id, chosen once and stated in configuration. This is what
you want when output quality has to be predictable, because with the free tier
you do not control which model answers or how heavily it is rate-limited.

Choosing among free models
--------------------------
Ranked by context length, longest first. For this project's job -- reformatting
a page of prose for reading aloud -- context is the constraint that actually
bites, and quality differences between the free instruct models are small next
to the difference between a model that fits the input and one that truncates
it. ``prefer`` overrides that with substring matches, in order, for a caller
that knows better.

Before ranking, anything that does not answer text with text is dropped. Free
does not mean chat: image and music models are listed at zero price too, and
Google's Lyria has the longest context of anything currently free -- so ranking
without this filter picks a music generator to reformat your prose.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

API_BASE = "https://openrouter.ai/api/v1"

# The router slug: always callable, costs nothing, and picks a free model
# server-side. Used when discovery fails, so a network hiccup degrades to a
# working call rather than to an error.
FREE_ROUTER = "openrouter/free"

# How long a fetched catalogue is trusted. Long enough that a burst of requests
# costs one HTTP call, short enough that a model going paid is noticed the same
# day.
CATALOGUE_TTL_SECONDS = 3600.0


class OpenRouterError(RuntimeError):
    """Anything that stops a request: no key, bad key, or the API said no."""


@dataclass(frozen=True)
class Model:
    """One entry from the catalogue, reduced to what choosing needs."""

    id: str
    name: str
    context_length: int
    prompt_price: float
    completion_price: float
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)

    @property
    def is_free(self) -> bool:
        return self.prompt_price == 0.0 and self.completion_price == 0.0

    @property
    def is_text_chat(self) -> bool:
        """Text in, text and nothing else out.

        The stricter half is the output check. A model that also emits audio or
        images is not a drop-in for a chat completion -- it may ignore the
        system prompt, bill differently, or return a payload with no text in it
        at all.
        """
        return "text" in self.input_modalities and set(self.output_modalities) == {
            "text"
        }


@dataclass
class _Catalogue:
    models: list[Model] = field(default_factory=list)
    fetched_at: float = 0.0


_CACHE = _Catalogue()
_CACHE_LOCK = threading.Lock()


def _as_price(value: Any) -> float:
    """Prices arrive as strings, and occasionally as ``null`` or ``"-1"``.

    A negative price means "variable" on some routed entries, which is not free.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def fetch_models(*, timeout: float = 10.0, force: bool = False) -> list[Model]:
    """The whole catalogue, cached for an hour.

    No API key needed: the model list is public. That matters, because it means
    a caller can show the user which models are free before asking them to go
    and create a key.
    """
    with _CACHE_LOCK:
        fresh = time.monotonic() - _CACHE.fetched_at < CATALOGUE_TTL_SECONDS
        if _CACHE.models and fresh and not force:
            return list(_CACHE.models)

    response = requests.get(f"{API_BASE}/models", timeout=timeout)
    response.raise_for_status()
    entries = response.json().get("data", [])

    models: list[Model] = []
    for entry in entries:
        identifier = str(entry.get("id") or "")
        if not identifier:
            continue
        pricing = entry.get("pricing") or {}
        architecture = entry.get("architecture") or {}
        models.append(
            Model(
                id=identifier,
                name=str(entry.get("name") or identifier),
                context_length=int(entry.get("context_length") or 0),
                prompt_price=_as_price(pricing.get("prompt", 0)),
                completion_price=_as_price(pricing.get("completion", 0)),
                input_modalities=tuple(architecture.get("input_modalities") or ("text",)),
                output_modalities=tuple(
                    architecture.get("output_modalities") or ("text",)
                ),
            )
        )

    with _CACHE_LOCK:
        _CACHE.models = models
        _CACHE.fetched_at = time.monotonic()
    return list(models)


def free_models(
    *, timeout: float = 10.0, force: bool = False, text_only: bool = True
) -> list[Model]:
    """Every free text model, longest context first.

    The ``:free`` suffix is checked as well as the price, because OpenRouter
    marks some free variants that way while listing a nominal price for the
    paid endpoint of the same model.
    """
    models = fetch_models(timeout=timeout, force=force)
    free = [
        model
        for model in models
        if (model.is_free or model.id.endswith(":free"))
        and (model.is_text_chat or not text_only)
    ]
    return sorted(free, key=lambda model: model.context_length, reverse=True)


def pick_free_model(
    *, prefer: list[str] | None = None, timeout: float = 10.0
) -> str:
    """The model id to use for a free request, right now.

    Never raises: a failure to reach the catalogue returns the router slug,
    which is exactly the situation it exists for.
    """
    try:
        candidates = free_models(timeout=timeout)
    except Exception:  # noqa: BLE001 - discovery is an optimization, not a gate
        return FREE_ROUTER
    if not candidates:
        return FREE_ROUTER
    for wanted in prefer or []:
        needle = wanted.strip().lower()
        if not needle:
            continue
        for model in candidates:
            if needle in model.id.lower():
                return model.id
    return candidates[0].id


class OpenRouterClient:
    """Chat completions over OpenRouter's OpenAI-compatible endpoint.

    Written against the HTTP API with ``requests`` rather than through the
    ``openai`` package, so that this file stays droppable into any project. The
    streaming format is server-sent events carrying OpenAI-shaped deltas; the
    twenty lines that parse it are below.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        referer: str | None = None,
        title: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key:
            raise OpenRouterError(
                "No OpenRouter API key. Put OPENROUTER_API_KEY in .env, or pass "
                "one to OpenRouterClient. Keys are created at "
                "https://openrouter.ai/keys"
            )
        self.timeout = timeout
        # OpenRouter attributes usage to these when they are present. Optional,
        # and only meaningful for apps listed on their leaderboard.
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if referer:
            self._headers["HTTP-Referer"] = referer
        if title:
            self._headers["X-Title"] = title

    def _body(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        return body

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _raise_for_error(self, response: requests.Response) -> None:
        if response.status_code < 400:
            return
        detail = response.text[:500]
        try:
            payload = response.json()
            detail = str(payload.get("error", {}).get("message") or detail)
        except Exception:  # noqa: BLE001 - the raw body is still useful
            pass
        if response.status_code == 401:
            raise OpenRouterError(f"OpenRouter rejected the API key: {detail}")
        if response.status_code == 402:
            raise OpenRouterError(
                f"OpenRouter needs credit for this model: {detail}. Free models "
                "are listed by `python -m llm_providers.openrouter --list-free`."
            )
        if response.status_code == 429:
            raise OpenRouterError(
                f"OpenRouter rate limit reached: {detail}. Free models are rate "
                "limited per key and per model; try a paid model or wait."
            )
        raise OpenRouterError(f"OpenRouter returned {response.status_code}: {detail}")

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """One request, one answer. The whole text at once."""
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=self._headers,
            json=self._body(
                model=model,
                messages=self._messages(prompt, system),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            ),
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise OpenRouterError(f"OpenRouter returned no choices: {payload}")
        return str(choices[0].get("message", {}).get("content") or "")

    def stream(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive.

        Streaming is the point on this project's hotkey path: the first
        paragraph is spoken while the model is still writing the third.
        """
        with requests.post(
            f"{API_BASE}/chat/completions",
            headers=self._headers,
            json=self._body(
                model=model,
                messages=self._messages(prompt, system),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ),
            timeout=self.timeout,
            stream=True,
        ) as response:
            self._raise_for_error(response)
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                # OpenRouter sends ": OPENROUTER PROCESSING" comment lines to
                # hold the connection open while a model cold-starts.
                if line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in payload.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OpenRouter helper.")
    parser.add_argument(
        "--list-free", action="store_true", help="Print the currently free models."
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prompt", help="Send a prompt and print the answer.")
    parser.add_argument("--model", help="Model id; default is the best free one.")
    args = parser.parse_args(argv)

    if args.list_free:
        models = free_models(force=True)
        print(f"{len(models)} free model(s) right now:\n")
        for model in models[: args.limit]:
            print(f"  {model.id:<52} {model.context_length:>9,} tokens  {model.name}")
        if not models:
            print(f"  none listed; {FREE_ROUTER} still works")
        return 0

    if args.prompt:
        model = args.model or pick_free_model()
        print(f"[{model}]\n")
        client = OpenRouterClient()
        for delta in client.stream(args.prompt, model=model):
            print(delta, end="", flush=True)
        print()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
