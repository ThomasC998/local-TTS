"""Make raw text TTS-ready in one pass, before it is synthesized.

Rewrites punctuation and line breaks for natural prosody and inserts only the
vocal-event tags Breeze TTS 2 actually supports. This is the *web UI's*
preparation pass -- one request, one answer, used when a user pastes text into
the browser and asks for it to be tidied.

The hotkey path uses ``llm_stream`` instead, which streams, because there the
first paragraph has to start being spoken while the model is still writing the
third. Both reach the model through ``llm_providers``, so the provider is
configured once in ``.env`` and applies to both.
"""

from __future__ import annotations

import logging
import os
import re

import llm_providers
import platform_support

platform_support.load_env()

logger = logging.getLogger("breeze.textprep")

# Only honoured when the provider has no opinion of its own; each provider has
# a default model that suits it, and PREP_MODEL_NAME predates them.
DEFAULT_MODEL = (os.getenv("PREP_MODEL_NAME") or "").strip() or None

# Documented Breeze TTS 2 vocal events. Anything outside this set would be read
# aloud literally, so the output is filtered against it.
ENGLISH_EVENTS = ("laugh", "cough", "clears throat", "sigh")
CHINESE_EVENTS = ("笑", "咳嗽", "清嗓子", "叹气")

_EN_TAG = re.compile(r"\(([^)]{1,40})\)")
_ZH_TAG = re.compile(r"\[([^\]]{1,40})\]")

SYSTEM_PROMPT = f"""You prepare raw text for a text-to-speech engine (Breeze TTS 2).

Rewrite the user's text so it reads naturally aloud. You may:
- Fix and add punctuation (commas, periods, question marks, ellipses) to control pacing.
- Insert line breaks between distinct thoughts.
- Split run-on sentences into shorter ones.
- Insert vocal-event tags where they genuinely fit the emotion.

Supported vocal events -- use ONLY these, exactly as written:
- English, in parentheses: {", ".join(f"({e})" for e in ENGLISH_EVENTS)}
- Chinese, in square brackets: {", ".join(f"[{e}]" for e in CHINESE_EVENTS)}

Hard rules:
- NEVER invent any other tag. No (excited), no (pause), no [emphasis], no SSML, no markdown.
- Do NOT translate. Keep the original language; use the bracket style matching that language.
- Do NOT add, remove, or reword the substance. Only punctuation, line breaks, and tags.
- Use vocal events sparingly -- at most one per two sentences, and only where clearly warranted.
- Return ONLY the prepared text. No preamble, no quotes, no explanation."""


def _strip_unsupported_tags(text: str) -> str:
    """Remove any bracketed tag the model invented outside the supported set."""

    def keep_en(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).strip().lower() in ENGLISH_EVENTS else ""

    def keep_zh(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).strip() in CHINESE_EVENTS else ""

    text = _EN_TAG.sub(keep_en, text)
    text = _ZH_TAG.sub(keep_zh, text)
    # Collapse whitespace left behind by removed tags, preserving line breaks.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


class TextPreparer:
    """One preparation pass, through whichever provider is configured."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.provider = llm_providers.resolve()
        logger.info(
            "Text prep ready: provider=%s model=%s",
            self.provider.name,
            self.model_name or self.provider.default_model(),
        )

    def prepare(self, text: str, extra_instruction: str | None = None) -> dict:
        """Return ``{"success", "text", "error"}``; never raises on API failure."""
        text = (text or "").strip()
        if not text:
            return {"success": False, "text": text, "error": "empty input"}

        prompt = text
        if extra_instruction:
            prompt = f"Additional direction: {extra_instruction}\n\nText:\n{text}"

        try:
            # Every provider streams; this one just wants the whole answer, so
            # the deltas are joined. Keeping a single code path means a new
            # provider works here the moment it works on the hotkey path.
            prepared = "".join(
                self.provider.stream(
                    prompt,
                    system=SYSTEM_PROMPT,
                    model=self.model_name,
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            ).strip()
            if not prepared:
                return {"success": False, "text": text, "error": "empty model response"}
            return {"success": True, "text": _strip_unsupported_tags(prepared), "error": None}
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as data
            logger.warning("Text prep failed: %s", exc)
            return {"success": False, "text": text, "error": str(exc)}


_preparer: TextPreparer | None = None
_PREPARER_PROVIDER: str | None = None


def get_preparer(model_name: str | None = None) -> TextPreparer:
    """Reuse one preparer so the provider's connection stays warm.

    Rebuilt when the configured provider changes, so editing ``.env`` and
    reloading the settings page takes effect without restarting the server.
    """
    global _preparer, _PREPARER_PROVIDER
    current = llm_providers.resolve_name()
    if _preparer is None or _PREPARER_PROVIDER != current:
        _preparer = TextPreparer(model_name)
        _PREPARER_PROVIDER = current
    return _preparer


def prepare_text(text: str, extra_instruction: str | None = None) -> dict:
    try:
        return get_preparer().prepare(text, extra_instruction)
    except Exception as exc:  # noqa: BLE001 - client construction can fail too
        return {"success": False, "text": text, "error": str(exc)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare text for Breeze TTS 2")
    parser.add_argument("text")
    parser.add_argument("--instruction")
    args = parser.parse_args()

    result = prepare_text(args.text, args.instruction)
    if not result["success"]:
        raise SystemExit(f"Text prep failed: {result['error']}")
    print(result["text"])
