"""The global hotkey listener. Windows' answer to Hammerspoon.

Hammerspoon owns two things on the Mac that a browser cannot do: a system-wide
hotkey, and reading the clipboard while another application is frontmost. On
Windows nothing equivalent is installed by default, so this process does the
same job in about the same amount of code -- a global keyboard hook, and one
HTTP POST per press.

It is deliberately as thin as ``breeze.lua``. Voice selection, the language
model pass, tone rotation, paragraph seeking and playback all happen in the
server, so none of it is configured twice and the two platforms cannot drift.

Run it with::

    python -m hotkeys.daemon

It works on macOS too, where it needs Accessibility permission for the terminal
it runs in, and is the fallback for anyone who would rather not install
Hammerspoon. Hammerspoon remains the default there because it already handles
starting at login and surviving a display sleep.

What a press actually does
--------------------------
Nothing but a POST. The server is the one that knows whether something is
already speaking, so ``toggle`` is one endpoint rather than a decision made
here -- which is also why holding a skip key is safe: the presses arrive as
fast as the keyboard repeats, and the server debounces them into one seek.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from . import bindings as bindings_module

logger = logging.getLogger("breeze.hotkeys")

DEFAULT_SERVER = "http://127.0.0.1:7860"

# Where each action posts. The bodies are empty: everything the server needs to
# know it already has -- the clipboard it reads itself, the voice and the
# language-model settings come from the config.
ENDPOINTS: dict[str, tuple[str, dict[str, Any]]] = {
    "toggle": ("/v1/speak/toggle", {}),
    "toggle_llm": ("/v1/speak/toggle", {"use_llm": True}),
    "next": ("/v1/speak/skip", {"delta": 1}),
    "previous": ("/v1/speak/skip", {"delta": -1}),
    "stop": ("/v1/speak/stop", {}),
}


class HotkeysUnavailable(RuntimeError):
    """No usable global-hotkey backend, with the reason attached."""


def _require_pynput() -> Any:
    try:
        import pynput.keyboard as keyboard
    except ImportError as exc:
        raise HotkeysUnavailable(
            "pynput is not installed. Run `pip install pynput`, or use the "
            "install script, which does it for you."
        ) from exc
    return keyboard


# --------------------------------------------------------------------------
# Talking to the server
# --------------------------------------------------------------------------
class ServerClient:
    """One POST per press, on a thread, so a slow reply cannot wedge the hook.

    A global keyboard hook that blocks is not a slow program, it is a frozen
    keyboard: on Windows the hook has a timeout after which the system stops
    delivering events to it entirely. So nothing here waits on the network.
    """

    def __init__(self, server: str = DEFAULT_SERVER, timeout: float = 10.0) -> None:
        self.server = server.rstrip("/")
        self.timeout = timeout
        self._notify: Callable[[str], None] | None = None
        self._warned_offline = False

    def on_message(self, callback: Callable[[str], None] | None) -> None:
        self._notify = callback

    def _say(self, message: str) -> None:
        logger.info("%s", message)
        if self._notify is not None:
            try:
                self._notify(message)
            except Exception:  # noqa: BLE001 - a toast must not kill the daemon
                logger.debug("Notification failed", exc_info=True)

    def post(self, path: str, body: dict[str, Any]) -> None:
        threading.Thread(
            target=self._post_now, args=(path, body), daemon=True
        ).start()

    def _post_now(self, path: str, body: dict[str, Any]) -> None:
        request = urllib.request.Request(
            f"{self.server}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            self._warned_offline = False
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            self._say(detail or f"Breeze server returned {exc.code}")
        except Exception:  # noqa: BLE001 - the server not running is normal
            if not self._warned_offline:
                # Said once per outage, not once per press: a dead server plus a
                # held skip key would otherwise be twenty identical toasts.
                self._warned_offline = True
                self._say("Breeze server is not running")

    def fetch_bindings(self) -> dict[str, str] | None:
        """The shortcuts as configured in the web UI, or None if unreachable."""
        try:
            with urllib.request.urlopen(
                f"{self.server}/v1/hotkeys", timeout=5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - fall back to the defaults
            return None
        keys = payload.get("bindings")
        return keys if isinstance(keys, dict) else None


# --------------------------------------------------------------------------
# Telling the user something happened
# --------------------------------------------------------------------------
def make_notifier() -> Callable[[str], None]:
    """A best-effort toast, matching ``hs.alert`` on the Mac.

    Only failures are ever announced. A hotkey that worked announces itself by
    speaking, and a toast on every press would be noise.
    """
    if platform.system() == "Windows":
        def notify(message: str) -> None:
            try:
                import ctypes

                # A message box, because this process normally runs under
                # pythonw with no console -- printing would go nowhere. It is
                # modal, and it blocks the short-lived thread that posted the
                # request until it is dismissed; that is acceptable only because
                # ServerClient says each outage once, not once per press.
                # MB_OK | MB_ICONWARNING | MB_SETFOREGROUND, with no parent
                # window, so it cannot end up behind whatever is frontmost.
                ctypes.windll.user32.MessageBoxW(
                    None, message, "Breeze TTS", 0x30 | 0x10000
                )
            except Exception:  # noqa: BLE001
                print(f"Breeze: {message}", file=sys.stderr)

        return notify

    def notify(message: str) -> None:
        print(f"Breeze: {message}", file=sys.stderr)

    return notify


# --------------------------------------------------------------------------
# The listener
# --------------------------------------------------------------------------
def to_pynput(binding: str) -> str:
    """Our notation to pynput's ``<ctrl>+<alt>+s``.

    The one translation with a decision in it is ``cmd``: it is Command on a
    Mac and the Windows key on Windows, which is the mapping the notation
    promises -- the same physical key role, whatever it is called locally.
    """
    modifiers, key = bindings_module.parse(binding)
    aliases = {"esc": "escape", "return": "enter"}
    key = aliases.get(key, key)
    parts = [f"<{modifier}>" for modifier in modifiers]
    parts.append(f"<{key}>" if len(key) > 1 else key)
    return "+".join(parts)


class HotkeyDaemon:
    """Binds the configured shortcuts and posts to the server on each press."""

    def __init__(
        self,
        client: ServerClient,
        bindings: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.bindings = bindings_module.validate(bindings or {})
        self._listener: Any = None

    def _handler(self, action: str) -> Callable[[], None]:
        path, body = ENDPOINTS[action]

        def fire() -> None:
            logger.debug("hotkey %s -> %s", action, path)
            self.client.post(path, body)

        return fire

    def start(self) -> None:
        keyboard = _require_pynput()
        mapping = {
            to_pynput(binding): self._handler(action)
            for action, binding in self.bindings.items()
        }
        try:
            self._listener = keyboard.GlobalHotKeys(mapping)
            self._listener.start()
        except Exception as exc:  # noqa: BLE001
            raise HotkeysUnavailable(
                f"Could not register the global hotkeys: {exc}. On macOS, grant "
                "Accessibility permission to the app running this. On Windows, "
                "another program may already own one of these combinations -- "
                "change it on the System speech tab."
            ) from exc

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()

    def wait(self) -> None:
        """Block until interrupted. The daemon's whole main loop."""
        try:
            while self._listener is not None and self._listener.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def summary(bindings: dict[str, str]) -> str:
    width = max(len(binding) for binding in bindings.values())
    lines = []
    for action, label, _help in bindings_module.ACTIONS:
        lines.append(f"  {bindings[action]:<{width}}  {label}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Global speak-the-clipboard hotkeys for Breeze TTS."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument(
        "--wait-for-server",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Poll for the server before binding. Used by the login shortcut, "
            "where this process and the server start at the same moment."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    client = ServerClient(args.server)
    client.on_message(make_notifier())

    deadline = time.monotonic() + args.wait_for_server
    configured = client.fetch_bindings()
    while configured is None and time.monotonic() < deadline:
        time.sleep(2.0)
        configured = client.fetch_bindings()
    if configured is None:
        print(
            f"Could not reach {args.server}; binding the default shortcuts. "
            "They will work as soon as the server starts."
        )

    try:
        daemon = HotkeyDaemon(client, configured)
        daemon.start()
    except HotkeysUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except bindings_module.InvalidBinding as exc:
        print(f"error: the configured shortcuts are not usable: {exc}", file=sys.stderr)
        return 1

    print("Breeze hotkeys are live:")
    print(summary(daemon.bindings))
    print("\nPress Ctrl+C to stop.")
    daemon.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
