"""Play streamed audio on the machine's own speakers.

The hotkey path has no client: Hammerspoon sends the clipboard text and forgets
about it, so the audio has to come out of the server process. That keeps the
trigger side down to one HTTP call with no audio code in it at all.

One utterance plays at a time. Starting a new one cancels whatever was playing,
which is what a second press of the hotkey should do.

Sample rate, and why this does not just hand PortAudio 24 kHz
-------------------------------------------------------------
The engine produces 24 kHz. Almost no macOS output device runs at 24 kHz --
this machine's headphones run at 44.1 kHz -- so *something* has to resample.
Opening the stream at 24 kHz and letting PortAudio deal with it is the obvious
move, and it crackles: the conversion then happens per callback buffer, and a
resampler that cannot see across a buffer edge leaves a discontinuity at every
one of them. Measured on this machine, block-wise conversion of a signal
peaking at 0.4 differs from a continuous conversion of the same signal by up to
0.28 -- a click hundreds of times a second, which is heard as constant
crackling. The saved WAV is clean because it is converted in one pass, which is
exactly what makes the streaming path sound worse than the file it wrote.

So: the stream is opened at the *device's own rate*, leaving PortAudio nothing
to convert, and the 24 kHz audio goes through one ``soxr`` resampler that lives
for the whole utterance and carries its state across blocks. Fed the same audio
in one piece or in a hundred, that resampler returns bit-identical output.

The same bug, in its browser form, is documented in the web UI's playback
worklet: an AudioContext left at the device rate resamples every scheduled
block independently. Different API, identical mistake.

Losing the device mid-sentence
------------------------------
Bluetooth headphones run out of battery and the system moves to the laptop
speakers, which is how a private document ends up being read to the room.
``DeviceWatcher`` exists to stop that: it asks the operating system directly --
not PortAudio, whose device list is a snapshot taken at initialisation -- which
device is currently the default, and reports a change within half a second. The
session treats that as a pause, exactly as a video player would.

Both systems can answer that question cheaply and safely from another thread
while a stream is open, which is the only reason polling is acceptable here.
macOS answers with a four-byte device id from CoreAudio; Windows answers with
an endpoint id string from the multimedia device enumerator. The value is
opaque either way -- all the watcher does is compare it with the last one -- so
the two paths meet at ``default_output_id`` and nothing above it is
platform-specific.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("breeze.audio")

# How long the ramp is when audio starts, stops, or the engine falls behind.
# A hard jump between a non-zero sample and silence is itself a click, so
# every discontinuity this player can create is smoothed over instead.
FADE_MS = 8.0


class PlaybackUnavailable(RuntimeError):
    """No output device, or sounddevice is not installed."""


def output_rate(device: Any = None) -> tuple[int, Any]:
    """The rate the device actually runs at, and the resolved device id.

    Opening a stream at anything else makes PortAudio resample per callback
    buffer, which is the crackle this module exists to avoid.
    """
    import sounddevice as sd

    if device is None or device == "":
        device = sd.default.device[1]
    try:
        info = sd.query_devices(device)
        return int(round(float(info["default_samplerate"]))), device
    except Exception:  # noqa: BLE001 - fall back to a rate every device takes
        logger.warning("Could not read the output device rate; assuming 48 kHz")
        return 48000, device


# --------------------------------------------------------------------------
# Which device is actually the default, right now
#
# PortAudio answers this from a list it built when it was initialised, so a
# device that disconnected two seconds ago is still in it. The OS answers from
# the live system, with no allocation worth speaking of, and is safe to call
# while a stream is open -- which is the whole reason the watcher can poll
# during playback.
# --------------------------------------------------------------------------
_K_AUDIO_OBJECT_SYSTEM_OBJECT = 1


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


def _core_audio() -> Any:
    if platform.system() != "Darwin":
        return None
    if _CORE_AUDIO["loaded"]:
        return _CORE_AUDIO["library"]
    _CORE_AUDIO["loaded"] = True
    try:
        path = ctypes.util.find_library("CoreAudio")
        _CORE_AUDIO["library"] = ctypes.CDLL(path) if path else None
    except Exception:  # noqa: BLE001 - the watcher degrades, it does not fail
        logger.debug("CoreAudio is not loadable; device changes go unnoticed")
        _CORE_AUDIO["library"] = None
    return _CORE_AUDIO["library"]


_CORE_AUDIO: dict[str, Any] = {"loaded": False, "library": None}


def _default_output_macos() -> int | None:
    """CoreAudio's id for the current default output device, or None.

    The id changes the moment the system switches -- headphones disconnecting,
    a monitor waking, the user picking another device in Control Centre -- so
    comparing it against the last one is all the change detection needed.
    """
    library = _core_audio()
    if library is None:
        return None
    address = _PropertyAddress(
        _fourcc("dOut"),  # kAudioHardwarePropertyDefaultOutputDevice
        _fourcc("glob"),  # kAudioObjectPropertyScopeGlobal
        0,
    )
    device = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device))
    try:
        status = library.AudioObjectGetPropertyData(
            ctypes.c_uint32(_K_AUDIO_OBJECT_SYSTEM_OBJECT),
            ctypes.byref(address),
            ctypes.c_uint32(0),
            None,
            ctypes.byref(size),
            ctypes.byref(device),
        )
    except Exception:  # noqa: BLE001
        return None
    if status != 0 or device.value == 0:
        return None
    return int(device.value)


# --------------------------------------------------------------------------
# The same question on Windows
#
# ``IMMDeviceEnumerator::GetDefaultAudioEndpoint`` answers it, and the endpoint
# id it returns changes when the default changes -- including the case this
# exists for, where a Bluetooth headset drops and Windows moves everything to
# the speakers. It is reached through raw COM here rather than through pycaw or
# comtypes: three vtable slots and two GUIDs is less to carry than a dependency
# whose wheels have to keep matching the Python version, and this file already
# does exactly this much ctypes for CoreAudio.
# --------------------------------------------------------------------------
_CLSID_MM_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMM_DEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_E_RENDER = 0
_E_CONSOLE = 0
_CLSCTX_ALL = 23
# COM must be initialised per thread. The watcher polls from its own thread and
# callers may ask from others, so the flag is thread-local rather than global.
_COM_READY = threading.local()


def _windows_guid(text: str) -> Any:
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    guid = GUID()
    ole32 = ctypes.WinDLL("ole32")
    if ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(guid)) != 0:
        raise OSError(f"Could not parse GUID {text}")
    return guid


def _default_output_windows() -> str | None:
    """The current default render endpoint's id string, or None."""
    try:
        import ctypes
        from ctypes import wintypes

        ole32 = ctypes.WinDLL("ole32")
        if not getattr(_COM_READY, "done", False):
            # COINIT_APARTMENTTHREADED. S_FALSE means already initialised on
            # this thread, which is a success for our purposes.
            result = ole32.CoInitializeEx(None, 2)
            if result not in (0, 1):
                return None
            _COM_READY.done = True

        enumerator = ctypes.c_void_p()
        status = ole32.CoCreateInstance(
            ctypes.byref(_windows_guid(_CLSID_MM_DEVICE_ENUMERATOR)),
            None,
            _CLSCTX_ALL,
            ctypes.byref(_windows_guid(_IID_IMM_DEVICE_ENUMERATOR)),
            ctypes.byref(enumerator),
        )
        if status != 0 or not enumerator:
            return None
        try:
            # IMMDeviceEnumerator: 0-2 are IUnknown, 3 EnumAudioEndpoints,
            # 4 GetDefaultAudioEndpoint.
            vtable = ctypes.cast(
                enumerator, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            )[0]
            get_default = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            )(vtable[4])
            device = ctypes.c_void_p()
            if get_default(enumerator, _E_RENDER, _E_CONSOLE, ctypes.byref(device)) != 0:
                return None  # no output device at all: unplugged, or none installed
            try:
                # IMMDevice: 3 Activate, 4 OpenPropertyStore, 5 GetId.
                device_vtable = ctypes.cast(
                    device, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                )[0]
                get_id = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)
                )(device_vtable[5])
                identifier = wintypes.LPWSTR()
                if get_id(device, ctypes.byref(identifier)) != 0:
                    return None
                try:
                    return str(identifier.value)
                finally:
                    # The string was allocated by the callee; nothing else frees it.
                    ole32.CoTaskMemFree(identifier)
            finally:
                _com_release(device)
        finally:
            _com_release(enumerator)
    except Exception:  # noqa: BLE001 - the watcher degrades, it does not fail
        logger.debug("Could not query the default Windows endpoint", exc_info=True)
        return None


def _com_release(interface: Any) -> None:
    import ctypes

    vtable = ctypes.cast(
        interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    )[0]
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    release(interface)


def default_output_id() -> Any:
    """An opaque identity for the current default output device, or None.

    An int on macOS and a string on Windows. Callers only ever compare it with
    the previous value, so the difference does not leak upwards -- and treating
    it as opaque is what lets a third platform be added by writing one more
    function here.
    """
    system = platform.system()
    if system == "Darwin":
        return _default_output_macos()
    if system == "Windows":
        return _default_output_windows()
    return None


_REFRESH_NEEDED = threading.Event()


def request_refresh() -> None:
    """Ask for PortAudio's device list to be rebuilt before the next stream.

    Not done where the change is noticed. Re-initialising PortAudio while
    another thread is opening a stream is a crash, and the watcher has no way
    to know it is not -- whereas ``start`` has just torn the old stream down
    and holds the only path to a new one. So the request is left here and
    collected there.
    """
    _REFRESH_NEEDED.set()


def refresh_devices() -> bool:
    """Rebuild PortAudio's device list so a new device becomes visible.

    Only safe with no stream open: re-initialising underneath a live stream
    tears it down anyway, and not in a way that ends cleanly.
    """
    try:
        import sounddevice as sd

        sd._terminate()
        sd._initialize()
    except Exception:  # noqa: BLE001 - a stale list beats a crashed watcher
        logger.exception("Could not re-initialise PortAudio")
        return False
    logger.info("Rebuilt PortAudio's device list")
    return True


class DeviceWatcher:
    """Calls back when the default output device changes, or a stream dies.

    Two signals, because they are two different failures. The default changing
    is the Bluetooth case -- CoreAudio has already moved the system elsewhere,
    and audio would keep playing out of the new device if nothing stopped it.
    A stream ending before the producer said it was finished is the pinned-device
    case, where the device the stream was opened on simply went away.
    """

    def __init__(self, player: "SpeechPlayer", interval: float = 0.5) -> None:
        self._player = player
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._callback: Callable[[str], None] | None = None
        self._baseline = default_output_id()

    def on_change(self, callback: Callable[[str], None] | None) -> None:
        self._callback = callback

    def start(self) -> None:
        if self._thread is not None:
            return
        self._baseline = default_output_id()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="audio-device-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _fire(self, reason: str) -> None:
        callback = self._callback
        logger.info("Output device event: %s", reason)
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:  # noqa: BLE001 - a bad listener must not kill the watcher
            logger.exception("Device-change listener failed")

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            current = default_output_id()
            if current is not None and current != self._baseline:
                self._baseline = current
                self._fire("default-output-changed")
                continue
            if self._player.ended_early():
                self._fire("output-stream-ended")


class SpeechPlayer:
    """A single-utterance streaming player over one PortAudio output stream.

    Audio arrives in blocks from the synthesis thread and leaves in fixed frames
    on PortAudio's callback thread, so everything shared between them sits under
    one lock. A block is never split across the boundary of an utterance: when
    an utterance is cancelled the queue is dropped whole.

    Blocks are resampled to the device rate on the way in, by one resampler per
    utterance -- see the module docstring for why that matters more than it
    looks like it should.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocks: deque[np.ndarray] = deque()
        self._offset = 0
        self._stream: Any = None
        self._sample_rate = 0        # what the engine produces
        self._device_rate = 0        # what the stream actually runs at
        self._resampler: Any = None
        self._device: Any = None
        self._utterance: str | None = None
        self._finished = False       # producer said "no more audio"
        self._started = False        # prebuffer satisfied, stream opened
        self._stopping = False       # fade out, then end the stream
        self._prebuffer_frames = 0
        self._buffered = 0           # frames queued but not yet played
        self._played = 0
        self._underruns = 0
        self._gain = 0.0             # current playback gain, for the ramps
        self._fade_frames = 1
        # PortAudio ended the stream while the producer still had audio to give.
        # On this machine that means one thing: the device went away.
        self._ended_early = False
        self._closing = False        # inside stop(); an end now is one we asked for
        self._drained = threading.Event()
        self._drained.set()

    # -- lifecycle --------------------------------------------------------
    def start(self, utterance_id: str, sample_rate: int, prebuffer_ms: int = 400,
              device: Any = None) -> None:
        """Claim the speaker for a new utterance, cancelling any current one."""
        self.stop()
        if _REFRESH_NEEDED.is_set():
            # A device came or went since the last stream. Nothing is open now,
            # so this is the one safe moment to let PortAudio look again.
            _REFRESH_NEEDED.clear()
            refresh_devices()
        device_rate, resolved = output_rate(device)

        resampler = None
        if device_rate != int(sample_rate):
            try:
                import soxr

                resampler = soxr.ResampleStream(
                    int(sample_rate), device_rate, 1, dtype="float32", quality="VHQ"
                )
            except Exception:  # noqa: BLE001 - better a resample than a failure
                logger.warning(
                    "soxr unavailable; letting PortAudio convert %d -> %d Hz, "
                    "which is audibly worse", sample_rate, device_rate,
                )
                device_rate = int(sample_rate)

        with self._lock:
            self._blocks.clear()
            self._offset = 0
            self._sample_rate = int(sample_rate)
            self._device_rate = device_rate
            self._resampler = resampler
            self._device = resolved
            self._utterance = utterance_id
            self._finished = False
            self._started = False
            self._stopping = False
            self._buffered = 0
            self._played = 0
            self._underruns = 0
            self._gain = 0.0
            self._ended_early = False
            self._fade_frames = max(1, int(device_rate * (FADE_MS / 1000.0)))
            # Measured at the device rate, since that is what the callback pulls.
            self._prebuffer_frames = int(device_rate * (max(0, prebuffer_ms) / 1000.0))
            self._drained.clear()

    def _convert(self, block: np.ndarray, last: bool = False) -> np.ndarray:
        """One block, at the device's rate, continuous with its neighbours."""
        # Matches what soundfile does when it writes the archived WAV, so the
        # two paths cannot diverge on a loud sample.
        block = np.clip(block, -1.0, 1.0)
        if self._resampler is None:
            return block
        converted = self._resampler.resample_chunk(block, last=last)
        return np.asarray(converted, dtype=np.float32).reshape(-1)

    def write(self, utterance_id: str, audio: np.ndarray) -> bool:
        """Queue one block. False means this utterance was cancelled."""
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not block.size:
            return self._utterance == utterance_id
        with self._lock:
            if self._utterance != utterance_id:
                return False
            converted = self._convert(block)
            if converted.size:
                self._blocks.append(converted)
                self._buffered += converted.size
            ready = self._buffered >= self._prebuffer_frames
        if ready:
            self._ensure_stream(utterance_id)
        return True

    def finish(self, utterance_id: str) -> None:
        """No more audio is coming; play out what is queued, then stop."""
        with self._lock:
            if self._utterance != utterance_id:
                return
            # Flush the resampler's tail, or the last few milliseconds of the
            # utterance are left inside it and never heard.
            tail = self._convert(np.zeros(0, dtype=np.float32), last=True)
            if tail.size:
                self._blocks.append(tail)
                self._buffered += tail.size
            self._finished = True
            empty = not self._blocks
        # A short utterance can finish before the prebuffer target is reached,
        # which would otherwise leave it queued and silent forever.
        self._ensure_stream(utterance_id)
        if empty:
            self.stop()

    def wait_drained(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)

    def stop(self, fade: bool = True, utterance_id: str | None = None) -> None:
        """Stop playback and release the device.

        With ``fade`` the callback ramps down first, so replacing an utterance
        mid-word -- which is what a second press of the hotkey does -- does not
        click. The wait is short and bounded; a stream that does not stop in
        time is torn down regardless.

        ``utterance_id`` makes the stop conditional on still owning the speaker,
        and every stop belonging to one utterance passes it. Cancelling a read
        takes time to unwind -- the worker has to finish the chunk it is inside
        -- and by then the read that replaced it may already be playing. Without
        the check, the outgoing utterance silences the incoming one. It is
        re-checked after the fade, because the fade is where that overlap fits.
        """
        with self._lock:
            if utterance_id is not None and self._utterance != utterance_id:
                return
            playing = self._stream is not None and self._utterance is not None
            self._closing = True
            if playing and fade:
                self._stopping = True
        if playing and fade:
            self._drained.wait(0.2)

        with self._lock:
            if utterance_id is not None and self._utterance != utterance_id:
                self._closing = False
                self._stopping = False
                return
            stream, self._stream = self._stream, None
            self._blocks.clear()
            self._offset = 0
            self._buffered = 0
            self._utterance = None
            self._started = False
            self._stopping = False
            self._ended_early = False
            self._resampler = None
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.exception("Failed to close the output stream")
        with self._lock:
            # Only now: aborting the stream fires the finished-callback, and
            # until this clears, that callback knows the end was requested.
            self._closing = False
            self._ended_early = False
        self._drained.set()

    # -- internals --------------------------------------------------------
    def _ensure_stream(self, utterance_id: str) -> None:
        with self._lock:
            if self._started or self._utterance != utterance_id:
                return
            self._started = True
            device_rate = self._device_rate
            device = self._device

        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001 - reported as data upstream
            raise PlaybackUnavailable(
                "sounddevice is not installed; run 'pip install sounddevice'"
            ) from exc

        try:
            stream = sd.OutputStream(
                samplerate=device_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._callback,
                finished_callback=self._on_finished,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001
            self.stop(fade=False)
            raise PlaybackUnavailable(f"Could not open an audio output: {exc}") from exc

        with self._lock:
            self._stream = stream

    def _on_finished(self) -> None:
        """PortAudio has played the last frame.

        The stream object is left for the next ``start``/``stop`` to close:
        closing a stream from inside its own finished-callback thread is not
        safe, and every path that opens a new one closes the old one first.

        Reaching here with neither ``finish`` nor ``stop`` having been called is
        not a normal end: nobody asked for it, so the device did. That is
        recorded rather than acted on here, because this runs on PortAudio's own
        thread and stopping the session from it would deadlock.
        """
        with self._lock:
            self._ended_early = not (self._finished or self._stopping or self._closing)
            self._utterance = None
            self._started = False
        self._drained.set()

    def ended_early(self) -> bool:
        """Whether the last stream stopped without anyone asking it to.

        Read once: it is a report of an event, and reporting it twice would
        pause the session a second time after it had already recovered.
        """
        with self._lock:
            ended, self._ended_early = self._ended_early, False
        return ended

    def is_playing(self) -> bool:
        with self._lock:
            return self._utterance is not None and self._stream is not None

    def _ramp(self, target: float, frames: int) -> np.ndarray:
        """A gain envelope from the current gain toward ``target``."""
        steps = min(frames, self._fade_frames)
        envelope = np.full(frames, target, dtype=np.float32)
        if steps:
            envelope[:steps] = np.linspace(
                self._gain, target, steps, endpoint=True, dtype=np.float32
            )
        self._gain = target
        return envelope

    def _callback(self, outdata, frames, _time, status) -> None:
        if status:
            logger.debug("PortAudio status: %s", status)

        written = 0
        with self._lock:
            stopping = self._stopping
            while written < frames and self._blocks:
                block = self._blocks[0]
                take = min(frames - written, block.size - self._offset)
                outdata[written : written + take, 0] = block[
                    self._offset : self._offset + take
                ]
                written += take
                self._offset += take
                self._buffered -= take
                self._played += take
                if self._offset >= block.size:
                    self._blocks.popleft()
                    self._offset = 0
            finished = self._finished and not self._blocks

        if stopping:
            # Ramp what we have down to silence, then end the stream.
            outdata[written:, 0] = 0.0
            outdata[:, 0] *= self._ramp(0.0, frames)
            raise _StopPlayback

        if written < frames:
            outdata[written:, 0] = 0.0
            if written:
                # Fade into the silence rather than cutting to it: the jump from
                # a mid-waveform sample to zero is a click in its own right.
                outdata[:written, 0] *= self._ramp(0.0, written)
            else:
                self._gain = 0.0
            if not finished:
                # The engine fell behind. Silence is the only honest option; the
                # utterance continues when the next block lands.
                self._underruns += 1
        elif self._gain < 1.0:
            outdata[:, 0] *= self._ramp(1.0, frames)

        if finished:
            raise _StopPlayback

    # -- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            rate = self._device_rate or self._sample_rate
            return {
                "playing": self._utterance is not None and self._stream is not None,
                "utterance_id": self._utterance,
                "buffered_seconds": round(self._buffered / rate, 3) if rate else 0.0,
                "played_seconds": round(self._played / rate, 3) if rate else 0.0,
                "underruns": self._underruns,
                "sample_rate": self._sample_rate or None,
                "device_rate": self._device_rate or None,
                "resampling": self._resampler is not None,
                "device": self._device,
            }


class _StopPlayback(Exception):
    """Raised inside the callback to end the stream once the queue is empty."""


# sounddevice turns this into a clean stop rather than an error.
try:  # pragma: no cover - depends on the installed sounddevice
    import sounddevice as _sd

    _StopPlayback = _sd.CallbackStop  # type: ignore[misc,assignment]
except Exception:  # noqa: BLE001 - the fallback above still ends the stream
    pass


def devices() -> list[dict[str, Any]]:
    """Output devices, for the settings page."""
    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001
        return []
    try:
        default = sd.default.device[1]
    except Exception:  # noqa: BLE001
        default = None
    found = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_output_channels", 0) > 0:
            found.append(
                {
                    "index": index,
                    "name": info.get("name"),
                    "is_default": index == default,
                }
            )
    return found


PLAYER = SpeechPlayer()
WATCHER = DeviceWatcher(PLAYER)
