"""Settings for system-wide speech: the hotkey path's voice, keys, LLM and archive.

One JSON file, read on every request and written whole. It is a handful of
fields edited by one person from one browser tab, so there is nothing here that
wants a database -- but writes are atomic and locked, because the hotkey path
and the web UI really can touch it at the same moment.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import hotkeys

STATE_DIR = Path(
    os.getenv("BREEZE_STATE_DIR", Path(__file__).resolve().parent / "state")
)
CONFIG_PATH = STATE_DIR / "system_speech.json"

_LOCK = threading.RLock()

# The blueprint the LLM rewrite runs on. Editable in the web UI and stored in
# the config, so iterating on it never means editing Python.
#
# Two things it must get right, or the streaming pipeline downstream breaks:
# the output has to be speakable text and nothing else, and it has to start
# immediately. The sentinel is the cheap insurance -- everything the model emits
# before it is discarded, so a stray "Sure, here you go:" costs nothing.
DEFAULT_LLM_PROMPT = """You prepare text to be read aloud by a text-to-speech engine.

Your job is pacing and pronounceability, not rewriting. Take the user's text and
lay it out so a listener -- who cannot see the page -- can follow it.

PARAGRAPHS
A blank line between two blocks of text is a paragraph break. It becomes an
audible pause, and it is also the unit the listener skips forwards and backwards
through, so getting these right matters more than anything else here.

- One paragraph carries one main concept. When the text moves to a genuinely new
  point, close the paragraph and open the next one.
- A paragraph is usually five to six sentences. Go longer when one concept
  honestly needs it -- a worked example, a chain of reasoning, a list of related
  facts belong together rather than being cut in half.
- Never leave a one-sentence paragraph unless it is a standalone conclusion.
- Put a blank line after a conclusion, a result, or a claim that matters.
- Separate paragraphs with exactly one blank line. Never indent them.

SENTENCES
- Break run-on sentences into shorter ones. Fix and add punctuation -- commas,
  periods, question marks, ellipses -- to control pacing within a sentence.
- Keep sentences speakable in one breath where you can.

WHAT THE ENGINE HANDLES, AND MUST KEEP
These are spoken correctly. Do not strip or rewrite them:
- Parentheses and brackets: keep them, with their text intact.
- Dashes and hyphens: em dashes, en dashes and hyphenated words.
- Numbers, decimals and ordinary punctuation.

WHAT TO CONVERT INTO SPOKEN WORDS
Anything the engine would read out as symbols, or read wrongly, is written out
the way a person would say it.

Abbreviations and acronyms -- expand to what they are said as:
- btw -> by the way; e.g. -> for example; i.e. -> that is; etc. -> and so on;
  vs. -> versus; approx. -> approximately; aka -> also known as;
  FYI -> for your information; ASAP -> as soon as possible; TL;DR -> in short.
- Units: Hz -> hertz; kHz -> kilohertz; GB -> gigabytes; MB -> megabytes;
  ms -> milliseconds; km/h -> kilometres per hour; °C -> degrees Celsius;
  °F -> degrees Fahrenheit; kg -> kilograms; W -> watts; V -> volts.
- Titles: Dr. -> Doctor; Mr. -> Mister; Prof. -> Professor; St. -> Street or
  Saint, whichever the sentence means.
- Initialisms said letter by letter (API, USB, HTTP, CPU) stay as they are --
  the engine spells those correctly. Acronyms said as words (NASA, RAM) also
  stay. Only expand where the written form would be misread.

Symbols -- replace with the word:
- & -> and; @ -> at; % -> percent; # -> number (or "hash" when it means the
  character); + -> plus; = -> equals; < -> less than; > -> greater than;
  ~ -> about; ± -> plus or minus; × -> times; ÷ -> divided by; → -> leads to.
- Currency: $50 -> fifty dollars; €1.5M -> one point five million euros;
  £20 -> twenty pounds.
- Maths: x^2 -> x squared; 1/2 -> a half; 3/4 -> three quarters.

Slashes are NOT supported. Never leave one in the output:
- "and/or", "his/her", "read/write" -> use "or" (or "and", whichever is meant).
- A date like 12/03/2025 -> "the twelfth of March, twenty twenty-five".
- A unit like km/h -> "kilometres per hour".

Numbers, dates and times -- write the way they are said:
- 2026 -> twenty twenty-six; 1,250 -> one thousand two hundred and fifty;
  3.5 -> three point five; 1st -> first; v2.1 -> version two point one.
- 14:30 -> half past two in the afternoon; 09:05 -> five past nine.
- Ranges: 10-20 -> ten to twenty; 2019-2024 -> twenty nineteen to twenty
  twenty-four.
- A phone number or a long identifier: read it in digit groups, or say what it
  is ("a sixteen-digit account number") if reading it adds nothing.

STRUCTURE THAT CANNOT BE READ OUT LITERALLY
Describe it in short sentences instead of transcribing it.
- File paths and URLs: say what it is, not what it spells. "/Users/alex/
  Documents/report.pdf" -> "a report PDF in the user's Documents folder".
  "https://example.com/pricing" -> "the pricing page on example dot com". Never
  read slashes, dots or protocol prefixes character by character.
- Tables: say what the table shows, then its shape, then the contents. "A table
  is presented with columns for device, sample rate and status, and rows for the
  headphones, the built-in speakers and the monitor." List every row when there
  are only a few; summarise the pattern and name the extremes when there are
  many.
- Charts, graphs and diagrams: give the conclusion in one or two sentences --
  what it shows and which way it goes -- and drop the axis labels, legends and
  data labels. "The chart shows memory use climbing steadily until the trim, and
  flat afterwards."
- Code and commands: describe what the code does in a sentence. Do not read
  syntax aloud. A short command a listener needs verbatim may be spoken as
  words, spelling out the symbols.
- Bulleted and numbered lists: turn them into flowing sentences. Keep the
  ordering words ("first", "then", "finally") when the order matters.
- Headings: fold them into the first sentence of the section, or drop them if
  the sentence already says it.
- Footnote markers, citation brackets like [12], figure and table references,
  page numbers, navigation text, cookie banners, "click here", image alt text,
  emoji and decorative characters: remove them entirely.
- Keyboard shortcuts: ⌘K -> "command K"; ctrl-alt-S -> "control alt S".
- Email addresses: "name at example dot com".

HARD RULES
- Do NOT summarize, add, remove, or reword the substance. The listener must hear
  the author's own words, only better paced and pronounceable. Describing a
  table or a chart is the one exception, and only because it cannot be read.
- Do NOT translate. Keep the original language, and use that language's own
  conventions for everything above.
- The only tags you may emit are these vocal events, exactly as written:
  (laugh) (cough) (clears throat) (sigh) in English, and
  [笑] [咳嗽] [清嗓子] [叹气] in Chinese. Use them sparingly, or not at all.
  NEVER invent any other tag: no (pause), no (emphasis), no SSML, no markdown.
- Output ONLY the text to be spoken. No preamble, no commentary, no headings,
  no quotes around it, no explanation of what you changed.

Begin your reply with the marker <<<SPEAK>>> on its own, then the text."""

# Prompts that *were* the default. The prompt is stored in the config -- the
# whole config is written whole on every save, so a default the user never
# touched ends up on disk looking exactly like a deliberate choice. Without
# this, improving the default would only ever reach a fresh install.
#
# Matched by hash, and only a byte-for-byte match migrates: an edited prompt is
# the user's, and is left alone.
SUPERSEDED_PROMPT_HASHES = frozenset({
    # The original, before paragraph shaping and the pronunciation rules.
    "82e837e19c209fb2bbf8907786f1c95c37caca9ef93e6612370976e9a93bca75",
    # The same, with a real home directory in the file-path example. Only the
    # example changed, but a config written before this keeps the old text
    # forever without a line here.
    "1311eda7ef5ba5e274d4ccab8528abe03a2c0cbdaa36cd64ed8ff01ef9dd6753",
})


def _is_superseded_default(prompt: str) -> bool:
    return (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        in SUPERSEDED_PROMPT_HASHES
    )


DEFAULTS: dict[str, Any] = {
    # Which saved voice the hotkeys speak with. None means "the first voice".
    "voice_id": None,
    # Rotation defaults for the system-speech path. Empty values fall through to
    # the voice's own stored rotation settings.
    "rotation": {
        "override": False,
        "enabled": True,
        "mode": "random",
        "every_words": 1000,
        "boundary": "paragraph",
        "tags": [],
    },
    "llm": {
        "model": None,  # None: walk the preference chain and use what answers
        "prompt": DEFAULT_LLM_PROMPT,
        "instruction": "",
        "temperature": 0.3,
        "max_output_tokens": 8192,
    },
    "synthesis": {
        "cfg_scale": None,
        "max_words": None,
        "voice_lock": True,
        "seed": None,
    },
    "archive": {
        "enabled": True,
        "keep_audio": True,
        "max_entries": 500,
    },
    "playback": {
        # Audio is buffered this far ahead before the first sample is played, so
        # a slow first chunk does not stutter the opening word.
        "prebuffer_ms": 400,
        "device": None,
        # Headphones running out of battery should behave like a YouTube tab:
        # stop, do not carry on out of the laptop speakers. Off means the audio
        # follows whatever macOS moves it to.
        "pause_on_device_change": True,
    },
    "controls": {
        # A skip press cancels what is playing and arms the next paragraph; the
        # engine only starts once no further press has arrived for this long, so
        # holding the key scrolls through paragraphs without synthesizing each
        # one on the way past.
        "skip_debounce_ms": 2000,
    },
    # The global shortcuts, shared by both hotkey hosts. Hammerspoon on macOS
    # and the Python daemon on Windows each read these from the server rather
    # than hard-coding their own, so changing one here changes it everywhere --
    # which is the only way five shortcuts stay the same on two machines.
    "hotkeys": dict(hotkeys.DEFAULTS),
}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursive update: a patch names only the leaves it changes."""
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load() -> dict[str, Any]:
    """Current settings, with any key missing from disk filled from defaults."""
    with _LOCK:
        if not CONFIG_PATH.is_file():
            return json.loads(json.dumps(DEFAULTS))
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt config must not make the hotkey silently stop working.
            return json.loads(json.dumps(DEFAULTS))
        if not isinstance(stored, dict):
            stored = {}
        prompt = (stored.get("llm") or {}).get("prompt")
        if isinstance(prompt, str) and _is_superseded_default(prompt):
            stored = dict(stored)
            stored["llm"] = {key: value for key, value in stored["llm"].items()
                             if key != "prompt"}
        return _merge(DEFAULTS, stored)


def save(patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update and return the whole config."""
    with _LOCK:
        merged = _merge(load(), patch)
        _validate(merged)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, CONFIG_PATH)
        return merged


def _validate(config: dict[str, Any]) -> None:
    llm = config.get("llm") or {}
    if not str(llm.get("prompt") or "").strip():
        raise ValueError("llm.prompt must not be empty")
    temperature = float(llm.get("temperature", 0.3))
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.temperature must be between 0 and 2")

    rotation = config.get("rotation") or {}
    if "every_words" in rotation and rotation["every_words"] is not None:
        words = int(rotation["every_words"])
        if not 20 <= words <= 100_000:
            raise ValueError("rotation.every_words must be between 20 and 100000")

    playback = config.get("playback") or {}
    prebuffer = int(playback.get("prebuffer_ms") or 0)
    if not 0 <= prebuffer <= 5000:
        raise ValueError("playback.prebuffer_ms must be between 0 and 5000")

    controls = config.get("controls") or {}
    debounce = int(controls.get("skip_debounce_ms") or 0)
    if not 0 <= debounce <= 10_000:
        raise ValueError("controls.skip_debounce_ms must be between 0 and 10000")

    # Normalized in place, so what is written to disk is the canonical spelling
    # and the two hotkey hosts never have to agree on how to parse "Ctrl+Alt+S".
    config["hotkeys"] = hotkeys.validate(config.get("hotkeys") or {})


def reset_prompt() -> dict[str, Any]:
    return save({"llm": {"prompt": DEFAULT_LLM_PROMPT}})
