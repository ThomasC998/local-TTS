"""Which language model prepares the text, and how it is chosen.

The job is small and identical whoever does it: take a page the user copied,
lay it out to be read aloud, stream it back. So the provider is a setting, not
an architecture -- ``BREEZE_LLM_PROVIDER`` in ``.env``, one of the names in
``REGISTRY`` below.

    vertex           Google Vertex AI, via `gcloud auth application-default login`
    gemini           Google Gemini, via an API key from AI Studio
    openrouter_free  OpenRouter, on whatever is free right now
    openrouter_paid  OpenRouter, on a model you name

Adding another
--------------
Write a module beside this one with a ``Provider`` subclass -- see ``base.py``
for the four methods -- and add a line to ``REGISTRY``. The settings UI, the
``.env`` documentation and the status endpoint are all generated from the
registry, so there is nothing else to update.

The default, and why it differs from this machine
-------------------------------------------------
Unset, the provider is chosen by ``auto``: the first one whose credential is
actually present, API keys before Vertex. That ordering is deliberate. A fresh
checkout on a new machine needs one line in ``.env`` to work, where Vertex needs
the gcloud SDK installed, a login, and a project -- so an API key is the right
default *for the repository*. A machine that has already done the gcloud setup
just writes ``BREEZE_LLM_PROVIDER=vertex`` in its own ``.env``, which is not
committed, and nothing about the shared default affects it.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Provider, ProviderUnavailable
from .gemini import GeminiProvider
from .openrouter_provider import OpenRouterFreeProvider, OpenRouterPaidProvider
from .vertex import VertexProvider

__all__ = [
    "Provider",
    "ProviderUnavailable",
    "REGISTRY",
    "available",
    "describe_all",
    "resolve",
    "resolve_name",
]

REGISTRY: dict[str, Provider] = {
    provider.name: provider
    for provider in (
        VertexProvider(),
        GeminiProvider(),
        OpenRouterFreeProvider(),
        OpenRouterPaidProvider(),
    )
}

# Tried in order by ``auto``. Keys first: a key that is present was put there on
# purpose, whereas ADC may be left over from unrelated work on the machine.
AUTO_ORDER = ("gemini", "openrouter_free", "openrouter_paid", "vertex")


def resolve_name(requested: str | None = None) -> str:
    """The provider name to use, resolving ``auto`` against what is configured.

    An explicitly named provider is always honoured, even if its credential is
    missing -- the resulting error names the missing key, which is far more
    useful than silently answering with a different model than the one asked
    for.
    """
    name = (requested or os.getenv("BREEZE_LLM_PROVIDER") or "auto").strip().lower()
    if name and name != "auto":
        if name not in REGISTRY:
            raise ProviderUnavailable(
                f"Unknown BREEZE_LLM_PROVIDER {name!r}. Expected one of: "
                + ", ".join(sorted(REGISTRY))
            )
        return name
    for candidate in AUTO_ORDER:
        provider = REGISTRY[candidate]
        if provider.key_env and (os.getenv(provider.key_env) or "").strip():
            return candidate
    # Nothing has a key. Vertex is the one provider that can still work without
    # one, so it gets the last word -- and if it cannot, its error explains the
    # whole situation better than a generic "no provider configured" would.
    return "vertex"


def resolve(requested: str | None = None) -> Provider:
    return REGISTRY[resolve_name(requested)]


def available() -> list[str]:
    """Providers whose credential is present, without calling any of them."""
    found = []
    for name, provider in REGISTRY.items():
        if provider.key_env is None or (os.getenv(provider.key_env) or "").strip():
            found.append(name)
    return found


def describe_all() -> list[dict[str, Any]]:
    """Static facts about every provider, for the settings UI."""
    rows = []
    for name, provider in REGISTRY.items():
        row = provider.describe()
        row["configured"] = (
            provider.key_env is None
            or bool((os.getenv(provider.key_env) or "").strip())
        )
        rows.append(row)
    return rows
