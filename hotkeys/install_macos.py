"""The macOS half of the hotkey installer: Hammerspoon and a LaunchAgent.

Reached through ``install_hotkeys.py`` at the project root, which picks this or
``install_windows`` depending on where it is run. The two files do the same job
with the tools each system actually has, and share nothing but that CLI --
there is no useful abstraction over "a Hammerspoon module" and "a shortcut in
the Startup folder".

Copies the Hammerspoon module into place, hooks it into an existing
``init.lua`` without disturbing what is already there, and reports on the two
things that have to be true for the hotkey to fire: Hammerspoon running, and
Hammerspoon holding Accessibility permission.

Surviving a reboot needs two things running, and they are separate: the server,
which holds the model and does the speaking, and Hammerspoon, which owns the
hotkeys. ``--startup`` handles the first with a LaunchAgent; the second is
handled by ``hs.autoLaunch(true)`` in breeze.lua, which is what this script
installs. Neither needs sudo, and nothing is installed outside the user's own
directories.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "hammerspoon" / "breeze.lua"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
HAMMERSPOON = Path.home() / ".hammerspoon"
TARGET = HAMMERSPOON / "breeze.lua"
INIT = HAMMERSPOON / "init.lua"
REQUIRE_LINE = 'require("breeze")'
MARKER = "-- Breeze TTS clipboard hotkeys"

AGENT_LABEL = "com.breeze.tts.server"
AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{AGENT_LABEL}.plist"

SERVER_URL = "http://127.0.0.1:7860"

# launchd starts an agent with almost no PATH, so anything the server shells out
# to has to be findable without the user's shell profile. Only ``gcloud`` is in
# that category, and only for resolving the Vertex project.
EXTRA_PATHS = (
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    str(Path.home() / "google-cloud-sdk" / "bin"),
)
# Passed through from the environment the installer runs in, when they are set.
FORWARDED_ENV = (
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "BREEZE_VERTEX_LOCATION",
    "BREEZE_LLM_MODEL",
    "BREEZE_MODEL",
    "BREEZE_AUDIO_DEVICE",
)


def say(message: str = "") -> None:
    print(message)


def ok(message: str) -> None:
    print(f"  \033[32m✓\033[0m {message}")


def warn(message: str) -> None:
    print(f"  \033[33m!\033[0m {message}")


def bad(message: str) -> None:
    print(f"  \033[31m✗\033[0m {message}")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def hammerspoon_installed() -> bool:
    return Path("/Applications/Hammerspoon.app").is_dir()


def hammerspoon_running() -> bool:
    result = subprocess.run(["pgrep", "-x", "Hammerspoon"], capture_output=True)
    return result.returncode == 0


def hammerspoon_has_accessibility() -> bool | None:
    """True, False, or None when the TCC database cannot be read.

    Reading the database directly requires Full Disk Access, which is a bigger
    grant than this script deserves -- so an unreadable database is reported as
    unknown rather than treated as a failure.
    """
    database = Path.home() / "Library/Application Support/com.apple.TCC/TCC.db"
    if not database.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "sqlite3", str(database),
                "SELECT auth_value FROM access WHERE service='kTCCServiceAccessibility' "
                "AND client='org.hammerspoon.Hammerspoon';",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value.startswith("2") if value else False


def server_running() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as response:
            return response.status == 200, "ready"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
def install_module(autolaunch: bool = False) -> None:
    """Copy the module into ~/.hammerspoon, optionally turning on autolaunch.

    Autolaunch is a login item, and installing one is exactly the sort of thing
    a script should not do because it happened to be run -- so it rides on
    ``--launch-agent``, which is the flag that means "make this survive a
    reboot" for the server too.
    """
    HAMMERSPOON.mkdir(parents=True, exist_ok=True)
    module = SOURCE.read_text(encoding="utf-8")
    if autolaunch:
        module = module.replace("M.autoLaunch = false", "M.autoLaunch = true", 1)
    TARGET.write_text(module, encoding="utf-8")
    ok(f"Copied the hotkey module to {TARGET}")
    if autolaunch:
        ok("...with Hammerspoon set to start at login")


def hook_into_init() -> None:
    """Append the require line, leaving any existing config untouched."""
    existing = INIT.read_text(encoding="utf-8") if INIT.is_file() else ""
    if REQUIRE_LINE in existing:
        ok("init.lua already loads it")
        return

    if existing:
        backup = INIT.with_suffix(".lua.before-breeze")
        backup.write_text(existing, encoding="utf-8")
        ok(f"Backed up your init.lua to {backup.name}")

    addition = f"\n\n{MARKER}\n{REQUIRE_LINE}\n"
    INIT.write_text(existing + addition, encoding="utf-8")
    ok("Added it to init.lua")


def reload_hammerspoon() -> None:
    if not hammerspoon_running():
        subprocess.run(["open", "-a", "Hammerspoon"], check=False)
        ok("Started Hammerspoon")
        return
    # Handled by the url binding in breeze.lua. On the very first install that
    # binding does not exist yet, which is why the manual reload is mentioned.
    subprocess.run(["open", "-g", "hammerspoon://breeze-reload"], check=False)
    ok("Asked Hammerspoon to reload its config")


def agent_environment() -> dict[str, str]:
    """The environment launchd cannot work out for itself.

    Three things are missing when a process starts from a LaunchAgent rather
    than a shell: a PATH with Homebrew on it, so ``gcloud`` can be found; the
    project's own ``.env``; and whatever Google Cloud variables the user has
    exported. The credentials themselves need nothing -- Application Default
    Credentials live in a file under ``~/.config/gcloud`` that the agent, being
    the same user, can already read.
    """
    environment: dict[str, str] = {}

    env_file = PROJECT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            environment[key.strip()] = value.strip().strip('"').strip("'")

    for name in FORWARDED_ENV:
        value = os.getenv(name)
        if value:
            environment[name] = value

    existing = [part for part in (os.getenv("PATH") or "").split(":") if part]
    ordered: list[str] = []
    for part in (*EXTRA_PATHS, *existing, "/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        if part and part not in ordered and Path(part).is_dir():
            ordered.append(part)
    environment["PATH"] = ":".join(ordered)
    environment["HOME"] = str(Path.home())
    return environment


def _launchctl(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments], capture_output=True, text=True, check=False
    )


def install_launch_agent(python: str) -> None:
    """Start the TTS server at login, so the hotkey always has something to call."""
    AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logs = PROJECT / "state"
    logs.mkdir(exist_ok=True)
    plist = {
        "Label": AGENT_LABEL,
        "ProgramArguments": [
            python, str(PROJECT / "breeze_server.py"),
            "--host", "127.0.0.1", "--port", "7860",
        ],
        "WorkingDirectory": str(PROJECT),
        "EnvironmentVariables": agent_environment(),
        "RunAtLoad": True,
        # Restart on a crash, but not on a clean exit -- so stopping the server
        # by hand stays stopped until the next login.
        "KeepAlive": {"SuccessfulExit": False},
        # The model takes the better part of a minute to load. Without this,
        # launchd's throttle would keep restarting a server it thinks failed.
        "ThrottleInterval": 30,
        "ProcessType": "Interactive",
        "StandardOutPath": str(logs / "server.out.log"),
        "StandardErrorPath": str(logs / "server.err.log"),
    }
    AGENT_PATH.write_bytes(plistlib.dumps(plist))

    # bootout/bootstrap is the supported pair; load/unload still works but is
    # deprecated and silently does nothing on some releases.
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{AGENT_LABEL}")
    result = _launchctl("bootstrap", domain, str(AGENT_PATH))
    if result.returncode != 0:
        _launchctl("unload", str(AGENT_PATH))
        result = _launchctl("load", str(AGENT_PATH))
    if result.returncode == 0:
        ok(f"Installed the login agent ({AGENT_PATH.name})")
        say(f"      Logs: {logs / 'server.err.log'}")
        say("      Remove it with: python install_hotkeys.py --uninstall-startup")
    else:
        bad(f"launchctl refused the agent: {result.stderr.strip()}")


def uninstall_launch_agent() -> None:
    if not AGENT_PATH.is_file():
        warn("No login agent is installed")
        return
    _launchctl("bootout", f"gui/{os.getuid()}/{AGENT_LABEL}")
    _launchctl("unload", str(AGENT_PATH))
    AGENT_PATH.unlink(missing_ok=True)
    ok("Removed the login agent. The server will not start at login any more.")


def agent_loaded() -> bool:
    result = _launchctl("list", AGENT_LABEL)
    return result.returncode == 0


def hammerspoon_autolaunches() -> bool | None:
    """Whether Hammerspoon is set to start at login.

    ``--launch-agent`` turns it on, by installing a breeze.lua that calls
    ``hs.autoLaunch(true)``; it takes effect once Hammerspoon reloads that
    config, which the installer asks it to do.

    Asked of the login-item list rather than a preferences key, because that is
    the thing that actually decides, and the key Hammerspoon writes has changed
    name across releases.
    """
    result = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to get the name of every login item'],
        capture_output=True, text=True, check=False, timeout=15,
    )
    if result.returncode == 0:
        return "hammerspoon" in result.stdout.lower()
    return None


def current_bindings_report() -> str:
    """The live shortcuts, asked of the server, falling back to the defaults.

    Printing the defaults when the user has changed them would be worse than
    printing nothing, so the server is asked first -- it is the only thing that
    knows.
    """
    import hotkeys

    bindings = dict(hotkeys.DEFAULTS)
    suffix = "  (defaults; the server is not running to confirm)"
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/v1/hotkeys", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload.get("bindings"), dict):
            bindings.update(payload["bindings"])
            suffix = ""
    except Exception:  # noqa: BLE001 - reporting must not fail on a dead server
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

    if hammerspoon_installed():
        ok("Hammerspoon is installed")
    else:
        bad("Hammerspoon is not in /Applications. Install it from hammerspoon.org")

    if hammerspoon_running():
        ok("Hammerspoon is running")
    else:
        warn("Hammerspoon is not running — the hotkey cannot fire. "
             "Open it, then run this script again.")

    access = hammerspoon_has_accessibility()
    if access is True:
        ok("Hammerspoon has Accessibility permission")
    elif access is False:
        bad("Hammerspoon does NOT have Accessibility permission — hotkeys will not "
            "fire. Grant it in System Settings → Privacy & Security → Accessibility.")
    else:
        warn("Could not read the permission database (that is normal). If the hotkey "
             "does nothing, check System Settings → Privacy & Security → Accessibility.")

    if TARGET.is_file():
        ok(f"{TARGET} is in place")
    else:
        bad("The hotkey module is not installed yet — run this script without --check")

    if INIT.is_file() and REQUIRE_LINE in INIT.read_text(encoding="utf-8"):
        ok("init.lua loads it")
    else:
        bad("init.lua does not load it yet")

    running, detail = server_running()
    if running:
        ok("The TTS server is answering on 127.0.0.1:7860")
    else:
        warn(f"The TTS server is not answering ({detail}). Start it with: "
             "python breeze_server.py")

    say("\nAfter a reboot")
    say("--------------")

    autolaunch = hammerspoon_autolaunches()
    if autolaunch is True:
        ok("Hammerspoon starts at login, so the hotkeys come back")
    elif autolaunch is False:
        warn("Hammerspoon does NOT start at login, so the hotkeys will not come "
             "back after a reboot. Turn it on with: "
             "python install_hotkeys.py --startup")
    else:
        warn("Could not tell whether Hammerspoon starts at login. Its preferences "
             "have a “Launch Hammerspoon at login” tick box.")

    if AGENT_PATH.is_file():
        ok(f"A login agent is installed for the server ({AGENT_PATH.name})")
        if agent_loaded():
            ok("...and launchd has it loaded")
        else:
            warn("...but launchd does not have it loaded. Re-run with --launch-agent.")
    else:
        warn("The server will NOT start at login. Install the agent with: "
             "python install_hotkeys.py --startup")

    say("\nHotkeys")
    say(current_bindings_report())
    say("""
Losing the output device — Bluetooth headphones going flat, say — pauses the
read where it stands rather than moving it to the laptop speakers. Press the
next-paragraph key to pick it up again.

Everything else -- the shortcuts themselves, which voice, tone rotation, the
model prompt, archiving -- lives on the “System speech” tab at
http://127.0.0.1:7860
""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_hotkeys.py", description=__doc__
    )
    parser.add_argument("--check", action="store_true",
                        help="report only; change nothing")
    parser.add_argument("--startup", "--launch-agent", action="store_true",
                        dest="startup",
                        help="also run the TTS server at login")
    parser.add_argument("--uninstall-startup", "--uninstall-agent",
                        action="store_true", dest="uninstall_startup",
                        help="stop running the TTS server at login, and remove it")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter the login agent should use")
    args = parser.parse_args(argv)

    if args.uninstall_startup:
        uninstall_launch_agent()
        report()
        return 0

    if not args.check:
        if not SOURCE.is_file():
            bad(f"Missing {SOURCE}")
            return 1
        say("Installing")
        install_module(autolaunch=args.startup)
        hook_into_init()
        reload_hammerspoon()
        if args.startup:
            install_launch_agent(args.python)

    report()
    if not args.check and not hammerspoon_running():
        say("Hammerspoon was just started — press the speak key to try it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
