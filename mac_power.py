"""Keeping this MacBook awake while the phone is listening, and letting it sleep after.

The point of the phone path is that the Mac is *not* on all the time. It sleeps
in a bag with the lid shut, a tap on the phone wakes it, it reads, and it goes
back to sleep. That needs three separate things, and only one of them is
obvious.

*Waking it* is not done from here -- a sleeping machine runs no code. It is done
by the phone, with a magic packet and a connection attempt to the port, both of
which this module can only prepare for by reporting the addresses to send them
to and whether the system is configured to listen. See ``wake_readiness``.

*Keeping it awake with the lid closed* cannot be done with ``caffeinate`` or any
other power assertion. Those hold off *idle* sleep; closing the lid with no
display attached triggers clamshell sleep, which sits below them and ignores
them. The only thing that stops it is ``pmset disablesleep``, which is a
system-wide setting and needs root -- hence the one-time helper that grants this
user exactly two ``pmset`` invocations and nothing else.

*Letting it sleep again* is ``pmset sleepnow`` once the read is done and nothing
else is going on.

Because ``disablesleep`` is global and persists after the process that set it
dies, every path out of this module releases it: the normal one, the exception,
the signal, and a hard timer in case something else goes wrong. A Mac left
unable to sleep in a bag is a flat battery and a hot bag, and that is the one
failure here that actually costs something.
"""

from __future__ import annotations

import atexit
import logging
import re
import subprocess
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("breeze.power")

MODES = ("off", "keep_on", "sleep_when_done")

# Below this, "keep on" gives up and lets the machine sleep. Ignored while the
# charger is connected.
DEFAULT_BATTERY_FLOOR = 25

# How long after the last read before "sleep when done" acts. Long enough to
# start another one without the Mac dropping out from under you.
DEFAULT_IDLE_SECONDS = 120

# However long a read lasts, the lid has been shut for hours by now and nobody
# is listening. A backstop against a mode left armed by mistake.
MAX_AWAKE_SECONDS = 8 * 60 * 60

_PMSET = "/usr/bin/pmset"
_SUDO = "/usr/bin/sudo"


class PowerUnavailable(RuntimeError):
    """The power helper is not installed, with the fix in the message."""


def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def _sudo_pmset(*args: str) -> None:
    """One privileged ``pmset`` call, without ever prompting for a password.

    ``-n`` is what makes this safe to call from a web request: if the helper is
    not installed, it fails immediately instead of hanging on a password prompt
    nobody is there to answer.
    """
    result = _run([_SUDO, "-n", _PMSET, *args])
    if result.returncode != 0:
        raise PowerUnavailable(
            "This needs the power helper. Run:  sudo ./install_power_helper.sh"
            + (f"  ({result.stderr.strip()})" if result.stderr.strip() else "")
        )


# ---------------------------------------------------------------------------
# Reading the machine's state
# ---------------------------------------------------------------------------
def battery() -> dict[str, Any]:
    """Charge, and whether the charger is in. Empty for a desktop."""
    result = _run([_PMSET, "-g", "batt"])
    if result.returncode != 0:
        return {}
    text = result.stdout
    percent = re.search(r"(\d+)%", text)
    return {
        "percent": int(percent.group(1)) if percent else None,
        "charging": "AC Power" in text,
        "present": "InternalBattery" in text,
    }


def settings() -> dict[str, Any]:
    """The bits of ``pmset -g`` that decide whether any of this works."""
    result = _run([_PMSET, "-g"])
    found: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"womp", "SleepDisabled", "powernap",
                                            "tcpkeepalive", "sleep", "standby"}:
            try:
                found[parts[0]] = int(parts[1])
            except ValueError:
                found[parts[0]] = parts[1]
    return found


def interfaces() -> list[dict[str, Any]]:
    """The network interfaces that could carry a wake packet, and their addresses.

    A Mac reports a dozen of these -- Thunderbolt bridges, the Wi-Fi Direct
    interface, the low-latency one -- and a magic packet sent to the wrong one
    does nothing at all. The one that matters is whichever currently holds this
    machine's LAN address, so that one is marked and the rest are only listed.
    """
    result = _run(["/sbin/ifconfig"])
    found: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            name = line.split(":", 1)[0]
            current = {"interface": name, "mac": None, "inet": None, "active": False}
            if name.startswith("en"):
                found.append(current)
            else:
                current = None
            continue
        if current is None:
            continue
        ether = re.search(r"\bether\s+([0-9a-f:]{17})", line)
        if ether:
            current["mac"] = ether.group(1)
        inet = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", line)
        if inet:
            current["inet"] = inet.group(1)
        if "status: active" in line:
            current["active"] = True
    return [item for item in found if item["mac"]]


def wake_targets() -> list[dict[str, Any]]:
    """Where to send a magic packet, the likeliest first.

    "Likeliest" is the interface holding the address the phone is already
    talking to: if this Mac is reachable there while awake, that is the radio
    that will be listening while it sleeps.
    """
    here = _lan_address()
    ranked = sorted(
        interfaces(),
        key=lambda item: (
            item["inet"] != here,
            not item["active"],
            item["interface"],
        ),
    )
    return ranked


def _lan_address() -> str | None:
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def mac_addresses() -> list[str]:
    """Just the hardware addresses, likeliest first."""
    return [item["mac"] for item in wake_targets() if item["mac"]]


def wake_readiness() -> dict[str, Any]:
    """Whether a sleeping Mac would answer the phone, and what to send it.

    ``womp`` is "Wake for network access". It is documented for Ethernet magic
    packets; Apple Silicon notebooks also commonly wake for a connection
    attempt to a port they were listening on, and commonly do not while running
    on battery. Reported rather than asserted, because it varies by machine and
    the honest answer is "measure it on yours".
    """
    current = settings()
    charge = battery()
    return {
        "womp": current.get("womp"),
        "sleep_disabled": bool(current.get("SleepDisabled")),
        "interfaces": wake_targets(),
        "mac_addresses": mac_addresses(),
        "battery": charge,
        "likely": bool(current.get("womp")) and bool(charge.get("charging", True)),
        "note": (
            "Wake for network access must be on, and a notebook on battery may "
            "not wake at all. Keep-on mode is the reliable answer for a Mac "
            "that lives in a bag."
        ),
    }


# ---------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------
class PowerManager:
    """Which of the three sleep behaviours is armed, and the timers behind them.

    ``busy`` is asked before sleeping the machine: it is the server's answer to
    "is anybody still listening", and it is the difference between a Mac that
    sleeps after a read and a Mac that sleeps in the middle of one.
    """

    def __init__(
        self,
        busy: Callable[[], bool] | None = None,
        *,
        battery_floor: int = DEFAULT_BATTERY_FLOOR,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
    ) -> None:
        self._busy = busy or (lambda: False)
        self._battery_floor = battery_floor
        self._idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._mode = "off"
        self._held = False
        self._held_since = 0.0
        self._timer: threading.Timer | None = None
        self._guard: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_error: str | None = None
        atexit.register(self.release)

    # -- state ------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode,
                "holding_awake": self._held,
                "held_for_seconds": round(time.time() - self._held_since, 1)
                if self._held else 0.0,
                "battery_floor": self._battery_floor,
                "idle_seconds": self._idle_seconds,
                "error": self._last_error,
                **wake_readiness(),
            }

    def set_mode(self, mode: str) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self._lock:
            self._mode = mode
            self._last_error = None
        if mode == "keep_on":
            self.hold()
        else:
            self.release()
            if mode == "sleep_when_done":
                self.arm_sleep()
        return self.state()

    # -- staying awake ----------------------------------------------------
    def hold(self) -> bool:
        """Stop the Mac sleeping, lid or no lid. False means it declined."""
        charge = battery()
        percent = charge.get("percent")
        if (
            charge.get("present")
            and not charge.get("charging")
            and percent is not None
            and percent < self._battery_floor
        ):
            with self._lock:
                self._last_error = (
                    f"Battery is at {percent}%, below the {self._battery_floor}% "
                    "floor, so the Mac is being allowed to sleep"
                )
            logger.warning("%s", self._last_error)
            self.release()
            return False
        try:
            _sudo_pmset("-a", "disablesleep", "1")
        except PowerUnavailable as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning("Could not hold the Mac awake: %s", exc)
            return False
        with self._lock:
            if not self._held:
                self._held_since = time.time()
            self._held = True
            self._last_error = None
        self._start_guard()
        logger.info("Holding this Mac awake; the lid can be closed")
        return True

    def release(self) -> None:
        """Let the Mac sleep normally again. Safe to call at any time, twice."""
        self._stop.set()
        with self._lock:
            held = self._held
            self._held = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if not held:
            return
        try:
            _sudo_pmset("-a", "disablesleep", "0")
            logger.info("Released the sleep hold")
        except PowerUnavailable as exc:  # pragma: no cover - helper vanished
            logger.error("COULD NOT RELEASE THE SLEEP HOLD: %s", exc)

    def _start_guard(self) -> None:
        """A thread that re-checks the battery and enforces the hard limit."""
        with self._lock:
            if self._guard is not None and self._guard.is_alive():
                return
            self._stop.clear()
            self._guard = threading.Thread(
                target=self._watch, name="power-guard", daemon=True
            )
            self._guard.start()

    def _watch(self) -> None:
        while not self._stop.wait(60.0):
            with self._lock:
                if not self._held:
                    return
                held_for = time.time() - self._held_since
            if held_for > MAX_AWAKE_SECONDS:
                logger.warning("Held awake for %.1f hours; letting it sleep",
                               held_for / 3600)
                self.release()
                return
            charge = battery()
            percent = charge.get("percent")
            if (
                charge.get("present")
                and not charge.get("charging")
                and percent is not None
                and percent < self._battery_floor
            ):
                logger.warning("Battery down to %d%%; letting the Mac sleep", percent)
                with self._lock:
                    self._last_error = f"Released at {percent}% battery"
                self.release()
                return

    # -- going back to sleep ----------------------------------------------
    def arm_sleep(self) -> None:
        """In ``sleep_when_done``, start the countdown to sleeping the Mac."""
        with self._lock:
            if self._mode != "sleep_when_done":
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._idle_seconds, self._sleep_if_idle)
            self._timer.daemon = True
            self._timer.start()

    def cancel_sleep(self) -> None:
        """Something started; the Mac is needed again."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _sleep_if_idle(self) -> None:
        if self._busy():
            logger.debug("Still reading; not sleeping")
            self.arm_sleep()
            return
        with self._lock:
            if self._mode != "sleep_when_done":
                return
        logger.info("Nothing to read; going to sleep")
        self.release()
        try:
            _sudo_pmset("sleepnow")
        except PowerUnavailable as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning("Could not sleep the Mac: %s", exc)

    def sleep_now(self) -> None:
        """Sleep immediately, whatever the mode. What the phone's button does."""
        self.release()
        _sudo_pmset("sleepnow")


MANAGER = PowerManager()
