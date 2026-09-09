"""The handful of things this project has to do differently per operating system.

Four, in total, and they are worth naming because the list being this short is
the point: everything else -- the server, the web UI, the voice library, the
archive, chunking, streaming, the session's transport controls -- is the same
code on both machines.

1. *Reading the clipboard.* The hotkey path has no browser to ask, so it reads
   the system clipboard directly. ``pbpaste`` on macOS, the Win32 clipboard on
   Windows.
2. *Noticing the output device changed.* Bluetooth headphones running out of
   battery must stop a private document being read to the room, and the audio
   library's own device list is a snapshot taken at start-up. Both systems can
   be asked directly; see ``audio_out``.
3. *Starting at login.* A LaunchAgent on macOS, a Startup-folder shortcut on
   Windows. Neither needs administrator rights and neither writes outside the
   user's own directories.
4. *Where the config lives.* Both use the project directory, but the paths are
   built with ``pathlib`` so the separator is never written down.

Nothing here imports a platform library at module scope: this file is imported
on both systems, including by tests, and an import of ``ctypes.wintypes`` on a
Mac is an ImportError, not a graceful degradation.

``.env`` is loaded here too. It is not platform-specific -- ``python-dotenv``
reads LF and CRLF files alike -- but it is the first thing every entry point
has to do, and having one function do it means the server, the hotkey daemon
and the install scripts cannot disagree about which file was read.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("breeze.platform")

IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

PROJECT_DIR = Path(__file__).resolve().parent


ENV_PATH = PROJECT_DIR / ".env"


class ClipboardUnavailable(RuntimeError):
    """The clipboard could not be read, with the reason attached."""


# --------------------------------------------------------------------------
# Configuration file
# --------------------------------------------------------------------------
def load_env(*, override: bool = False) -> Path | None:
    """Read ``.env`` from the project directory, if it is there.

    ``override=False`` means a variable already exported in the shell wins over
    the file, which is what makes ``BREEZE_LLM_PROVIDER=vertex python
    breeze_server.py`` work for a one-off test without editing anything.

    Returns the file that was read, or None. Missing is not an error: every
    setting in it has a default, and a machine that only ever uses the defaults
    should not need the file to exist.
    """
    if not ENV_PATH.is_file():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        # A hand-rolled fallback so a missing package degrades to a warning
        # rather than to a server that starts with no configuration at all.
        logger.warning(
            "python-dotenv is not installed; reading %s with a minimal parser. "
            "Run `pip install python-dotenv` for quoting and export support.",
            ENV_PATH,
        )
        _load_env_minimal(ENV_PATH, override=override)
        return ENV_PATH
    load_dotenv(dotenv_path=ENV_PATH, override=override)
    return ENV_PATH


def _load_env_minimal(path: Path, *, override: bool) -> None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if name and (override or name not in os.environ):
            os.environ[name] = value


# --------------------------------------------------------------------------
# Clipboard
#
# Read, never written. The hotkey copies nothing and replaces nothing; it only
# speaks what the user themselves put there.
# --------------------------------------------------------------------------
def _clipboard_macos() -> str:
    result = subprocess.run(
        ["pbpaste"], capture_output=True, text=True, timeout=5, check=False
    )
    return result.stdout or ""


def _clipboard_windows() -> str:
    """Read CF_UNICODETEXT straight from the Win32 clipboard.

    Done with ctypes rather than a package because it is thirty lines, has no
    wheel to go stale, and the alternatives (``pyperclip``) shell out to
    PowerShell on some paths -- which pops a console window on every hotkey
    press and adds a quarter-second to a path whose whole point is to feel
    instant.
    """
    import ctypes
    from ctypes import wintypes

    CF_UNICODETEXT = 13
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

    # Another process can hold the clipboard open for a few milliseconds after
    # a copy. Retrying briefly is the difference between "press it again" and
    # "it just works".
    for _ in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        raise ClipboardUnavailable(
            "another application is holding the clipboard open"
        )
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""  # something non-textual is on the clipboard
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return ""
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _clipboard_linux() -> str:
    for command in (["wl-paste", "--no-newline"], ["xclip", "-selection", "clipboard", "-o"]):
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return result.stdout or ""
    raise ClipboardUnavailable(
        "install wl-clipboard (Wayland) or xclip (X11) to read the clipboard"
    )


def clipboard_text() -> str:
    """Whatever text is on the clipboard, or "" when there is none."""
    try:
        if IS_MACOS:
            return _clipboard_macos()
        if IS_WINDOWS:
            return _clipboard_windows()
        return _clipboard_linux()
    except ClipboardUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - one message shape for the caller
        raise ClipboardUnavailable(str(exc)) from exc


# --------------------------------------------------------------------------
# Starting at login
# --------------------------------------------------------------------------
def python_executable() -> str:
    """The interpreter to relaunch with.

    ``sys.executable`` is the one running now, which on Windows is
    ``python.exe`` -- and launching that from a login shortcut opens a console
    window that stays for the session. ``pythonw.exe``, beside it, is the same
    interpreter with no console.
    """
    executable = Path(sys.executable)
    if IS_WINDOWS:
        windowless = executable.with_name("pythonw.exe")
        if windowless.is_file():
            return str(windowless)
    return str(executable)


def startup_dir() -> Path:
    """The per-user Startup folder. Windows only."""
    if not IS_WINDOWS:
        raise RuntimeError("the Startup folder exists only on Windows")
    appdata = os.getenv("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set; cannot find the Startup folder")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def write_startup_shortcut(name: str, target: str, arguments: str, workdir: Path) -> Path:
    """Put a .lnk in the user's Startup folder, without any extra packages.

    A ``.lnk`` is a binary format, so this asks Windows to make one through the
    same COM object Explorer uses. PowerShell is the shortest path to that
    object that is guaranteed present on Windows 11.
    """
    destination = startup_dir() / f"{name}.lnk"
    destination.parent.mkdir(parents=True, exist_ok=True)

    def quoted(value: object) -> str:
        """A PowerShell single-quoted literal. Doubling is its only escape."""
        return "'" + str(value).replace("'", "''") + "'"

    script = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({quoted(destination)});"
        f"$s.TargetPath = {quoted(target)};"
        f"$s.Arguments = {quoted(arguments)};"
        f"$s.WorkingDirectory = {quoted(workdir)};"
        # 7 is minimized. pythonw opens no window anyway, but a user who swaps
        # the target for python.exe to see the log gets it out of the way.
        "$s.WindowStyle = 7;"
        "$s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return destination


def remove_startup_shortcut(name: str) -> bool:
    path = startup_dir() / f"{name}.lnk"
    if path.exists():
        path.unlink()
        return True
    return False


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def describe() -> dict[str, str]:
    """What the web UI shows on the System speech tab, and /health reports."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "hotkey_host": "hammerspoon" if IS_MACOS else "python",
    }
