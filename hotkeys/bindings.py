"""The five shortcuts, in one notation both hotkey hosts understand.

The shortcuts are the whole product on the hotkey path -- there is no window to
click in -- so the *same* five actions have to exist on both machines, and the
user has to be able to change them in one place. That place is the server
config, edited on the System speech tab; Hammerspoon and the Python daemon each
read it from there and bind whatever it says.

Notation
--------
A binding is written ``ctrl+alt+s``: modifiers first in any order, then exactly
one key, lower case, joined by ``+``. ``cmd`` and ``win`` are the same physical
role on the two platforms and are accepted interchangeably -- a config written
on a Mac loads on Windows, and the daemon maps whichever one it has.

Why the defaults are what they are
----------------------------------
``ctrl+alt`` is free on both systems: macOS reserves ``cmd`` combinations and
Windows reserves ``win`` ones, and neither reserves ctrl+alt broadly. The arrow
keys are used for seeking because they sit in the same place on every keyboard
layout, where a letter does not.

The one known collision is Rectangle on macOS, which claims ctrl+alt with the
arrows in its default shortcut set; the fix is in the README, and now also just
changing these.
"""

from __future__ import annotations

import re
from typing import Any

# Ordered: this is the order the UI lists them in, and it is the order they are
# meant to be learned in -- speak, speak-with-model, seek, seek, stop.
ACTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "toggle",
        "Speak the clipboard",
        "Press once to start, again to silence it. The press after that starts "
        "a new read from whatever is on the clipboard by then.",
    ),
    (
        "toggle_llm",
        "Speak, through the language model",
        "The same, but the text is laid out for reading aloud first.",
    ),
    ("next", "Next paragraph", "Hold to scroll forward without speaking each one."),
    ("previous", "Previous paragraph", "Hold to scroll back the same way."),
    ("stop", "Stop", "Silence the read without starting anything."),
)

DEFAULTS: dict[str, str] = {
    "toggle": "ctrl+alt+s",
    "toggle_llm": "ctrl+alt+a",
    "next": "ctrl+alt+right",
    "previous": "ctrl+alt+left",
    "stop": "ctrl+alt+x",
}

# ``cmd`` and ``win`` name the same key role; both spellings normalize to one so
# that a config moves between machines unchanged.
MODIFIERS = {"ctrl", "control", "alt", "option", "shift", "cmd", "command", "win", "super"}
_CANONICAL_MODIFIER = {
    "control": "ctrl",
    "option": "alt",
    "command": "cmd",
    "super": "cmd",
    "win": "cmd",
}

# Named keys, beyond the printable single characters. Anything not here has to
# be a single character, which is what keeps a typo from binding to nothing.
NAMED_KEYS = frozenset(
    """
    left right up down space tab enter return escape esc backspace delete home
    end pageup pagedown insert
    f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12
    """.split()
)

_TOKEN = re.compile(r"^[a-z0-9]+$")


class InvalidBinding(ValueError):
    """A binding string that no hotkey host could register."""


# Modifiers are emitted in this order, not alphabetically: it is how every
# shortcut is written in every menu on both systems, and the canonical form is
# what the user reads back in the settings UI.
_MODIFIER_ORDER = ("ctrl", "alt", "shift", "cmd")


def parse(binding: str) -> tuple[tuple[str, ...], str]:
    """``"alt+ctrl+S"`` -> ``(("ctrl", "alt"), "s")``.

    Modifiers come back in a fixed order so that two spellings of the same
    shortcut compare equal, which is what makes the duplicate check below
    meaningful.
    """
    if not isinstance(binding, str) or not binding.strip():
        raise InvalidBinding("a shortcut cannot be empty")
    parts = [part.strip().lower() for part in binding.split("+")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        raise InvalidBinding(
            f"{binding!r} needs at least one modifier and a key, like ctrl+alt+s"
        )
    *modifier_parts, key = parts
    modifiers: set[str] = set()
    for part in modifier_parts:
        if part not in MODIFIERS:
            raise InvalidBinding(
                f"{part!r} is not a modifier; use ctrl, alt, shift or cmd"
            )
        modifiers.add(_CANONICAL_MODIFIER.get(part, part))
    if not _TOKEN.match(key):
        raise InvalidBinding(f"{key!r} is not a key name")
    if len(key) > 1 and key not in NAMED_KEYS:
        raise InvalidBinding(
            f"{key!r} is not a known key. Use a single character or one of: "
            + ", ".join(sorted(NAMED_KEYS))
        )
    if not modifiers:
        raise InvalidBinding(
            f"{binding!r} has no modifier. A bare key would fire while typing."
        )
    ordered = tuple(name for name in _MODIFIER_ORDER if name in modifiers)
    return ordered, key


def canonical(binding: str) -> str:
    modifiers, key = parse(binding)
    return "+".join([*modifiers, key])


def validate(bindings: dict[str, Any]) -> dict[str, str]:
    """Check a whole set, and return it normalized.

    Two actions on one shortcut is rejected rather than resolved: whichever one
    won would depend on registration order, and a hotkey that does the wrong
    thing is worse than one that was refused with a message.
    """
    resolved: dict[str, str] = {}
    seen: dict[str, str] = {}
    for action, _label, _help in ACTIONS:
        raw = bindings.get(action, DEFAULTS[action])
        try:
            normalized = canonical(str(raw))
        except InvalidBinding as exc:
            raise InvalidBinding(f"{action}: {exc}") from exc
        if normalized in seen:
            raise InvalidBinding(
                f"{normalized} is bound to both {seen[normalized]} and {action}"
            )
        seen[normalized] = action
        resolved[action] = normalized
    unknown = set(bindings) - set(DEFAULTS)
    if unknown:
        raise InvalidBinding(f"unknown action(s): {', '.join(sorted(unknown))}")
    return resolved


def describe() -> list[dict[str, str]]:
    """The action list for the web UI, defaults included."""
    return [
        {"action": action, "label": label, "help": help_text, "default": DEFAULTS[action]}
        for action, label, help_text in ACTIONS
    ]
