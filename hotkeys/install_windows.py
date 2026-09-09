"""The Windows half of the hotkey installer: the daemon, and Startup shortcuts.

Reached through ``install_hotkeys.py`` at the project root. The macOS half sets
up Hammerspoon and a LaunchAgent; this one sets up the Python hotkey daemon and
a shortcut in the user's Startup folder. Nothing here needs administrator
rights, and nothing is written outside the user's own profile.

Two processes have to be running for a hotkey to do anything, and they are
separate on purpose:

*The server* holds the model and does the speaking. Starting it is slow -- the
checkpoint is several gigabytes -- so it wants to be running already, not
started on the first press.

*The daemon* owns the keyboard hook and posts to the server. It is instant to
start and cheap to restart, which is what makes changing a shortcut in the web
UI a matter of restarting one small process.

``--startup`` installs both as login shortcuts. Without it, nothing is added to
the Startup folder: putting an entry there because a script happened to be run
is exactly the sort of thing an installer should ask about first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import platform_support  # noqa: E402

SERVER_URL = "http://127.0.0.1:7860"
SERVER_SHORTCUT = "Breeze TTS server"
DAEMON_SHORTCUT = "Breeze TTS hotkeys"


def say(message: str = "") -> None:
    print(message)


def ok(message: str) -> None:
    print(f"  [ok]   {message}")


def warn(message: str) -> None:
    print(f"  [!]    {message}")


def bad(message: str) -> None:
    print(f"  [x]    {message}")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def pynput_installed() -> bool:
    try:
        import pynput.keyboard  # noqa: F401
    except Exception:  # noqa: BLE001 - an import error is the answer either way
        return False
    return True


def server_running() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as response:
            return response.status == 200, "ready"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def daemon_running() -> bool:
    """Whether a hotkey daemon is already up.

    Asked of the process list rather than of a lock file, because the thing
    that matters is whether a keyboard hook exists -- and a stale lock file
    would claim one does when it does not.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # Both python.exe and pythonw.exe can be hosting it, and tasklist does not
    # show arguments, so this can only say "some Python is running" -- which is
    # why it is reported as a maybe below rather than as a fact.
    return "python" in result.stdout.lower()


def shortcut_path(name: str) -> Path:
    return platform_support.startup_dir() / f"{name}.lnk"


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
def install_startup_shortcuts(python: str) -> None:
    """Start the server and the hotkey daemon at login.

    ``pythonw.exe`` rather than ``python.exe``: the console window a login
    shortcut would otherwise open stays on screen for the whole session, and
    the server already writes to a log file.
    """
    windowless = platform_support.python_executable()
    try:
        server = platform_support.write_startup_shortcut(
            SERVER_SHORTCUT,
            windowless,
            f'"{PROJECT / "breeze_server.py"}" --host 127.0.0.1 --port 7860',
            PROJECT,
        )
        ok(f"Installed {server.name}")
        # The daemon waits for the server rather than racing it: at login both
        # start at once, and the daemon wants the configured shortcuts, which
        # only the server can give it.
        daemon = platform_support.write_startup_shortcut(
            DAEMON_SHORTCUT,
            windowless,
            f'-m hotkeys.daemon --server {SERVER_URL} --wait-for-server 120',
            PROJECT,
        )
        ok(f"Installed {daemon.name}")
        say(f"      Both live in {platform_support.startup_dir()}")
        say("      Remove them with: python install_hotkeys.py --uninstall-startup")
    except subprocess.CalledProcessError as exc:
        bad(f"PowerShell could not create the shortcut: {exc.stderr or exc}")
    except Exception as exc:  # noqa: BLE001
        bad(f"Could not create the login shortcuts: {exc}")


def uninstall_startup_shortcuts() -> None:
    removed = 0
    for name in (SERVER_SHORTCUT, DAEMON_SHORTCUT):
        try:
            if platform_support.remove_startup_shortcut(name):
                ok(f"Removed {name}.lnk")
                removed += 1
        except Exception as exc:  # noqa: BLE001
            bad(f"Could not remove {name}.lnk: {exc}")
    if not removed:
        warn("Nothing was installed in the Startup folder")


def current_bindings_report() -> str:
    """The live shortcuts, asked of the server, falling back to the defaults."""
    import hotkeys

    bindings = dict(hotkeys.DEFAULTS)
    suffix = "  (defaults; the server is not running to confirm)"
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/v1/hotkeys", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload.get("bindings"), dict):
            bindings.update(payload["bindings"])
            suffix = ""
    except Exception:  # noqa: BLE001
        pass
    width = max(len(value) for value in bindings.values())
    lines = [
        f"  {bindings[action]:<{width}}   {label}"
        for action, label, _help in hotkeys.ACTIONS
    ]
    return "\n".join(lines) + suffix


# --------------------------------------------------------------------------
def report() -> None:
    say("\nState of the hotkey path")
    say("------------------------")

    if pynput_installed():
        ok("pynput is installed, so the global keyboard hook is available")
    else:
        bad(
            "pynput is not installed — no hotkey can fire. Run: "
            "pip install pynput"
        )

    try:
        startup = platform_support.startup_dir()
        ok(f"Startup folder: {startup}")
    except Exception as exc:  # noqa: BLE001
        bad(f"Could not locate the Startup folder: {exc}")
        startup = None

    running, detail = server_running()
    if running:
        ok("The TTS server is answering on 127.0.0.1:7860")
    else:
        warn(
            f"The TTS server is not answering ({detail}). Start it with: "
            "python breeze_server.py"
        )

    say("\nAfter a reboot")
    say("--------------")
    if startup is not None:
        for label, name in (
            ("server", SERVER_SHORTCUT),
            ("hotkey daemon", DAEMON_SHORTCUT),
        ):
            if shortcut_path(name).is_file():
                ok(f"The {label} starts at login ({name}.lnk)")
            else:
                warn(
                    f"The {label} will NOT start at login. Install it with: "
                    "python install_hotkeys.py --startup"
                )

    say("\nHotkeys")
    say(current_bindings_report())
    say(
        """
Start the daemon by hand with:  python -m hotkeys.daemon

If a shortcut does nothing, another program almost certainly owns it — Windows
gives the combination to whoever registered it first, silently. Change it on
the "System speech" tab at http://127.0.0.1:7860 and restart the daemon.

Losing the output device — Bluetooth headphones going flat, say — pauses the
read where it stands rather than moving it to the speakers. Press the
next-paragraph key to pick it up again.
"""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_hotkeys.py", description=__doc__
    )
    parser.add_argument(
        "--check", action="store_true", help="report only; change nothing"
    )
    parser.add_argument(
        "--startup",
        action="store_true",
        help="run the server and the hotkey daemon at login",
    )
    parser.add_argument(
        "--uninstall-startup",
        action="store_true",
        help="stop running them at login, and remove the shortcuts",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter the login shortcuts should use",
    )
    args = parser.parse_args(argv)

    if args.uninstall_startup:
        uninstall_startup_shortcuts()
        report()
        return 0

    if not args.check and args.startup:
        say("Installing")
        install_startup_shortcuts(args.python)

    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
