"""What every language-model provider has to offer, and nothing more.

The text-preparation pass is a small job: one system prompt, one document, text
back as it is generated. Everything else a provider API offers -- tools, images,
multi-turn history -- is irrelevant here, so the interface stays at four
methods and adding a provider stays a fifty-line file.

Adding one
----------
Write a module beside this with a ``Provider`` subclass, then add its name to
``llm_providers.REGISTRY``. Nothing else in the project needs to change:
``llm_stream`` calls providers only through this interface, the settings UI
lists whatever the registry contains, and the ``.env`` key it reads is named by
the provider itself.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any


class ProviderUnavailable(RuntimeError):
    """This provider cannot be used, with the reason attached.

    The reason is shown to the user, so it should say what to do -- which key
    is missing, which command to run -- not merely that something failed.
    """


class Provider(abc.ABC):
    """One language-model backend, from this project's narrow point of view."""

    #: Stable identifier, used in ``.env`` and in the settings UI.
    name: str = ""
    #: Shown in the settings UI beside the name.
    label: str = ""
    #: The environment variable holding this provider's credential, if it has
    #: one. ``None`` means it authenticates some other way -- Vertex uses
    #: Application Default Credentials, which is a file, not a key.
    key_env: str | None = None

    @abc.abstractmethod
    def default_model(self) -> str:
        """The model to use when the config does not name one."""

    @abc.abstractmethod
    def check(self) -> str:
        """Confirm this provider can answer, and return the model that will.

        Raises ``ProviderUnavailable`` when it cannot. Called on start-up and
        whenever the settings page is opened, so it must be cheap -- and it may
        be called while a generation is in flight, so it must not mutate
        anything a generation is using.
        """

    @abc.abstractmethod
    def stream(
        self,
        text: str,
        *,
        system: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
    ) -> Iterator[str]:
        """Yield the model's output as it arrives.

        Deltas are raw. The sentinel filter and the sanitizer run downstream in
        ``llm_stream``, where text has been reassembled into whole chunks --
        which is also why a provider must never buffer to sentence boundaries
        here. The first delta is what starts the speech.
        """

    def status(self) -> dict[str, Any]:
        """What the UI shows: which model is live, or why none is."""
        try:
            model = self.check()
        except Exception as exc:  # noqa: BLE001 - a status call never raises
            return {
                "provider": self.name,
                "label": self.label,
                "available": False,
                "model": None,
                "error": str(exc),
            }
        return {
            "provider": self.name,
            "label": self.label,
            "available": True,
            "model": model,
            "error": None,
        }

    def describe(self) -> dict[str, Any]:
        """Static facts for the settings UI, with no network call."""
        return {
            "name": self.name,
            "label": self.label,
            "key_env": self.key_env,
            "default_model": self.default_model(),
        }
