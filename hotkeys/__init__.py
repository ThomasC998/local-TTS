"""Global speak-the-clipboard shortcuts, and where each platform gets them.

Two hosts, one set of bindings. macOS uses Hammerspoon, which already solves
starting at login and surviving display sleep; Windows uses the Python daemon
in ``daemon.py``, which needs nothing installed beyond a pip package. Both read
their shortcuts from the server, so changing one on the System speech tab
changes it everywhere without editing a file.

``bindings`` holds everything they have to agree on: the five actions, the
notation, and the validation. Nothing here imports a hotkey library, so the
server can import this module to serve and validate the config on a machine
where no hotkey host is installed at all.
"""

from __future__ import annotations

import platform

from .bindings import (
    ACTIONS,
    DEFAULTS,
    InvalidBinding,
    canonical,
    describe,
    parse,
    validate,
)

__all__ = [
    "ACTIONS",
    "DEFAULTS",
    "InvalidBinding",
    "canonical",
    "describe",
    "host",
    "parse",
    "validate",
]


def host() -> str:
    """Which hotkey host this platform is expected to run.

    Reported to the web UI so the setup instructions on the System speech tab
    are the ones for the machine the browser is talking to, rather than a page
    that lists both and makes the reader work out which half applies.
    """
    return "hammerspoon" if platform.system() == "Darwin" else "python"
