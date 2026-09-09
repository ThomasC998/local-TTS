"""Stream text through a language model, token by token, and clean it up.

Used by the system-speech path: the clipboard text goes to the model, and the
model's output goes to the TTS engine as it arrives, so speech starts long
before the rewrite has finished.

*Which* model is a setting -- Vertex, Gemini, or either OpenRouter tier -- and
that choice lives entirely in ``llm_providers``. What lives here is everything
that is true whichever model answered: the sentinel that discards a preamble,
and the sanitizer that turns model output into text an engine can actually
speak. Those are the parts that took experimenting to get right, so they are
written once and every provider gets them.

    stream_text()    raw deltas from the configured provider
    SentinelFilter   drops everything before <<<SPEAK>>>
    sanitize()       stage directions, citations, markdown, stray slashes
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

import llm_providers
from llm_providers.vertex import MODEL_CHAIN, resolve_project  # noqa: F401

logger = logging.getLogger("breeze.llm")

# Everything the model emits before this marker is discarded. Prompting alone
# does not reliably suppress "Sure, here's the text:" -- and with output going
# straight to a speaker, a preamble is not a cosmetic problem.
SENTINEL = "<<<SPEAK>>>"

# Vocal events the engine understands.
ENGLISH_EVENTS = ("laugh", "cough", "clears throat", "sigh")
CHINESE_EVENTS = ("笑", "咳嗽", "清嗓子", "叹气")

# Brackets are *kept*. The engine speaks a parenthetical correctly, and "(see
# below)" or "(2026)" is content the author wrote -- stripping every bracket to
# be safe cost more than it saved. What is dropped instead is the narrow set of
# stage directions a model invents when it is asked for vocal events: those are
# not content, and they are read out literally.
_INVENTED_TAGS = frozenset(
    """
    pause long pause short pause beat silence break emphasis emphasized stressed
    whisper whispering shouting shouts softly loudly slowly quickly quietly
    music applause laughter chuckle chuckles smiles smiling grins sarcastic
    excited nervous angry sad happy calm serious deadpan singing hums humming
    breath breathes inhales exhales gasp gasps ahem tone voice narrator
    speaking continues cont'd end start stop begin fade in fade out sfx
    sound effect background noise 静音 停顿 强调 语气
    """.split()
)
_TAG_SHAPE = re.compile(r"^[\w' -]{1,40}$")
_EN_TAG = re.compile(r"\(([^)]{1,40})\)")
_ZH_TAG = re.compile(r"\[([^\]]{1,40})\]")
# A bare "[12]" or "[3, 4]" is a footnote marker, never something to say.
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]")
# The engine has no way to voice a slash. When the model leaves one in anyway,
# the overwhelmingly common meaning is "or" -- but only between two bare words,
# where nothing else it could be fits. A neighbouring slash rules out a path
# ("Users/x" inside "/Users/x/y"), a digit rules out a date or a fraction, and
# the unit list rules out "km/h", which means "per" rather than "or".
_OR_SLASH = re.compile(r"(?<![/\w])([A-Za-z]{2,20})\s*/\s*([A-Za-z]{2,20})(?![/\w])")
_PER_UNITS = frozenset(
    "km mi ft in cm mm nm kg lb oz ml cl dl gal hr hrs min mins sec secs "
    "kb mb gb tb bps kbps mbps rpm mph kph kwh wh amp amps volt volts".split()
)


def _slash_to_or(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if left.lower() in _PER_UNITS or right.lower() in _PER_UNITS:
        return match.group(0)
    if {left.lower(), right.lower()} == {"and", "or"}:
        return "or"  # "and/or" spoken as "and or or" is worse than either word
    return f"{left} or {right}"
_MD_FENCE = re.compile(r"^\s*```[^\n]*$", re.MULTILINE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,3}([-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")

# A provider that cannot answer is reported to the caller as data, never as a
# traceback. The name is kept as an alias so that existing handlers, and any
# code outside this project catching it, keep working.
LLMUnavailable = llm_providers.ProviderUnavailable


def provider() -> llm_providers.Provider:
    """The configured provider. Resolved per call, so .env edits take effect."""
    return llm_providers.resolve()


def status() -> dict[str, Any]:
    """What the UI shows: which provider and model are live, or why none is.

    Never raises: this is called to render a settings page, including on a
    machine where nothing is configured yet, which is exactly when the page is
    most worth showing.
    """
    try:
        return provider().status()
    except Exception as exc:  # noqa: BLE001 - resolution itself can fail
        return {
            "provider": None,
            "label": None,
            "available": False,
            "model": None,
            "error": str(exc),
        }


# --------------------------------------------------------------------------
# Sanitizing
# --------------------------------------------------------------------------
def _is_invented_tag(inner: str) -> bool:
    """Whether a bracketed fragment is a stage direction rather than content.

    Deliberately narrow. It has to look like a tag -- a few plain words, no
    sentence punctuation -- *and* be built from the vocabulary models reach for
    when they invent one. "(pause)" goes; "(see the table above)" stays, because
    a listener losing the author's own aside is the worse failure.
    """
    stripped = inner.strip().strip(".!?,;:").lower()
    if not stripped or not _TAG_SHAPE.match(stripped):
        return False
    words = stripped.replace("-", " ").split()
    return len(words) <= 3 and all(word in _INVENTED_TAGS for word in words)


def sanitize(text: str) -> str:
    """Strip what would be read out literally, and keep what would not.

    Applied to whole chunks on their way into the engine, never to a partial
    token: half a ``**`` is not markdown yet, and a regex cannot know that.

    Brackets, dashes and numbers survive untouched -- the engine speaks all
    three. Markdown syntax, footnote markers and invented tags do not.
    """
    text = _MD_FENCE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _CITATION.sub("", text)

    def keep_en(match: re.Match[str]) -> str:
        inner = match.group(1)
        if inner.strip().lower() in ENGLISH_EVENTS:
            return match.group(0)
        return "" if _is_invented_tag(inner) else match.group(0)

    def keep_zh(match: re.Match[str]) -> str:
        inner = match.group(1)
        if inner.strip() in CHINESE_EVENTS:
            return match.group(0)
        return "" if _is_invented_tag(inner) else match.group(0)

    text = _EN_TAG.sub(keep_en, text)
    text = _ZH_TAG.sub(keep_zh, text)
    text = _OR_SLASH.sub(_slash_to_or, text)
    # Removing a tag mid-sentence leaves the space that separated it, and a
    # space before a comma is a visible artefact the engine pauses on.
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


class SentinelFilter:
    """Drops everything a stream emits before the sentinel.

    Streaming means the marker can arrive split across deltas, so a tail short
    enough to be a partial marker is held back rather than passed on.
    """

    def __init__(self, sentinel: str = SENTINEL) -> None:
        self.sentinel = sentinel
        self.open = False
        self._buffer = ""

    def feed(self, delta: str) -> str:
        if self.open:
            return delta

        self._buffer += delta
        index = self._buffer.find(self.sentinel)
        if index >= 0:
            self.open = True
            tail = self._buffer[index + len(self.sentinel):]
            self._buffer = ""
            return tail.lstrip()

        # No marker yet. Hold back only as much as could still become one; if
        # the model ignored the instruction entirely, the text still flows.
        keep = len(self.sentinel) - 1
        if len(self._buffer) > 4096:
            # Far past any plausible preamble: assume there is no marker.
            self.open = True
            text, self._buffer = self._buffer, ""
            return text
        return ""

    def flush(self) -> str:
        """Whatever is left when the stream ends, marker or not."""
        if self.open or not self._buffer:
            return ""
        text, self._buffer = self._buffer, ""
        self.open = True
        return text


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
def stream_text(
    text: str,
    *,
    prompt: str,
    instruction: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> Iterator[str]:
    """Yield the model's spoken-text output as it arrives.

    Deltas are raw: the sentinel filter and the sanitizer run downstream, where
    text has been reassembled into whole chunks. Nothing here waits for a
    sentence boundary, because the first delta is what starts the speech.
    """
    contents = (
        text
        if not instruction
        else f"Additional direction: {instruction}\n\nText:\n{text}"
    )
    yield from provider().stream(
        contents,
        system=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
