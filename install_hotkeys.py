#!/usr/bin/env python3
"""Set up the global speak-the-clipboard hotkeys, on whichever system this is.

    python install_hotkeys.py                      # install and report
    python install_hotkeys.py --check              # report only, change nothing
    python install_hotkeys.py --startup            # also start at login
    python install_hotkeys.py --uninstall-startup

The shortcuts themselves are the same five on both systems and are configured
in one place -- the "System speech" tab of the web UI, stored in the server's
config. What differs is only what holds the keyboard hook, and that is what
this script sets up: Hammerspoon on macOS, a small Python daemon on Windows.

The real work is in ``hotkeys/install_macos.py`` and
``hotkeys/install_windows.py``. They are separate files rather than one file
with branches because they share nothing beyond this command line: a
Hammerspoon module hooked into ``init.lua`` and a ``.lnk`` in the Startup
folder have no common shape worth inventing one for.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    system = platform.system()
    if system == "Darwin":
        from hotkeys import install_macos

        return install_macos.main(sys.argv[1:])
    if system == "Windows":
        from hotkeys import install_windows

        return install_windows.main(sys.argv[1:])

    print(
        f"No hotkey installer for {system}. The daemon may still work: run "
        "`python -m hotkeys.daemon` and grant it whatever input permission "
        "your desktop requires.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
