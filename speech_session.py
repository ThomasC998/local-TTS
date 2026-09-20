"""One spoken document on this machine's speakers, seekable a paragraph at a time.

The hotkey path used to be a straight line: text in, one long generation, audio
out, and the only control was "stop". That is fine until you want to skip the
paragraph you are half way through.

Everything about *making* the audio now lives in ``paragraph_engine``: the text
as it becomes known, the generator running a few paragraphs in front of the ear,
and the cache that means a paragraph is made once and can be heard any number of
times. The phone serves itself from the same engine over HTTP. What is left here
is the part that is specific to speaking out loud on this machine.

*A press that means "and the next one too".* Skips are debounced. Each press
silences what is playing and moves a target index; only once the presses stop
for a moment does the session land. Holding the key scrolls through the document
instead of speaking every paragraph on the way past. Landing on audio that
already exists -- forwards into what was generated ahead, backwards into what
was heard before -- costs the debounce and nothing else.

*A position that means what the listener hears.* The engine is seconds in front
of the speaker, and a press is aimed at the sound, not at the engine. So the
player's queue is held short and a paragraph is not called done until that queue
has nearly drained. Without that, a paragraph shorter than the queue would be
"finished" before anybody had heard a word of it.

Pausing is the same machinery under another name. When the output device
disappears -- headphones out of battery, the usual case -- playback stops where
it is and the session sits paused, exactly like a video player, until a skip
press picks it back up.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

import archive
import audio_out
from paragraph_engine import (  # noqa: F401 - re-exported for callers and tests
    GENERATE_AHEAD_PARAGRAPHS,
    AnyEvent,
    AudioCache,
    ParagraphBook,
    ParagraphEngine,
)

logger = logging.getLogger("breeze.session")

# How long the session waits after the last skip press before it lands. Long
# enough to press again without hearing anything start up, short enough that a
# single press does not feel like a stall.
SKIP_DEBOUNCE_SECONDS = 2.0

# The largest piece handed to the player at once. The queue is how the session
# measures its distance from the ear, and a piece is only counted once it is
# whole, so anything longer than this would overshoot the lead by the difference.
WRITE_SLICE_SECONDS = 0.25

# How much audio may sit in the player's queue. The queue is unbounded and
# cached audio would fill it instantly, which would put the sound seconds behind
# the paragraph the session thinks it is on -- and a skip press is aimed at what
# is being heard.
PLAYER_LEAD_SECONDS = 1.0

# ...and how little of it may be left before the session calls a paragraph done
# and moves to the next. A paragraph can be shorter than the lead, so handing
# the last block over is not evidence anybody has heard it. What is left covers
# the gap while the next paragraph's first block is found.
PARAGRAPH_HANDOVER_SECONDS = 0.25

# States a caller can see. Only the first four are live.
LIVE_STATES = frozenset({"starting", "speaking", "seeking", "paused"})

_AnyEvent = AnyEvent  # the old private name, kept for callers that used it


def _paced(blocks: Iterator[np.ndarray], sample_rate: int) -> Iterator[np.ndarray]:
    """Engine blocks, cut down to something the queue can be measured in.

    Slices are views, not copies, and a block already short enough passes
    straight through -- which is every block the engine produces live. It is
    cached audio replayed at once that this exists for.
    """
    limit = max(1, int(sample_rate * WRITE_SLICE_SECONDS))
    for block in blocks:
        if block.size <= limit:
            yield block
            continue
        for start in range(0, block.size, limit):
            yield block[start : start + limit]


class SpeechSession:
    """One utterance on the machine's speakers, with transport controls.

    Owns a ``ParagraphEngine`` and one thread of its own, which takes paragraphs
    out of the engine's cache and puts them on the speaker. Everything a hotkey
    does -- skip, pause, stop -- only sets flags; the playback thread is what
    acts on them, because a generator may only be closed from the thread
    iterating it.
    """

    def __init__(
        self,
        prepared: dict[str, Any],
        text: str,
        *,
        debounce: float = SKIP_DEBOUNCE_SECONDS,
    ) -> None:
        self._prepared = prepared
        self._text = text
        self._job = prepared["job"]
        self._engine = self._job["engine"]
        self._recording: archive.Recording = prepared["recording"]
        self._config = prepared["config"]
        self._debounce = max(0.0, debounce)

        self.utterance_id: str = prepared["utterance_id"]
        self.started_at = time.time()

        self._paragraphs = ParagraphEngine(prepared, text)
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._done = threading.Event()

        self._state = "starting"
        self._error: str | None = None
        self._index = 0              # what the listener is hearing
        self._seek_target: int | None = None
        self._seek_deadline = 0.0
        self._paused = False
        self._pause_reason: str | None = None
        self._player_open = False
        self._spoken: list[int] = []

        self._player: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin producing text and speaking it. Returns immediately."""
        self._paragraphs.start()
        with self._lock:
            self._state = "speaking"
        self._player = threading.Thread(
            target=self._run, name="speak-player", daemon=True
        )
        self._player.start()

    # -- playback ---------------------------------------------------------
    def _run(self) -> None:
        error: str | None = None
        try:
            while not self._stop.is_set():
                if not self._settle():
                    break
                index = self._index
                if not self._paragraphs.wait_for(index, self._stop):
                    break  # nothing at this index, and nothing more is coming
                self._interrupt.clear()
                interrupted = self._play_paragraph(index)
                if self._stop.is_set():
                    break
                if interrupted:
                    continue  # a skip, a pause or a lost paragraph decides next
                with self._lock:
                    if self._seek_target is None and not self._paused:
                        self._index = index + 1
                        self._wake.notify_all()
        except Exception as exc:  # noqa: BLE001 - reported through status()
            error = str(exc)
            logger.exception("Speaking failed")
        finally:
            self._finish(error)

    def _settle(self) -> bool:
        """Wait out a pause or a run of skip presses. False means give up.

        The deadline is re-read every time round: another press while we are
        waiting pushes it out, which is what makes holding the key cheap.
        """
        while not self._stop.is_set():
            with self._lock:
                if self._paused and self._seek_target is None:
                    self._wake.wait(0.25)
                    continue
                if self._seek_target is not None:
                    remaining = self._seek_deadline - time.monotonic()
                    if remaining > 0:
                        self._wake.wait(min(remaining, 0.25))
                        continue
                    self._index = self._paragraphs.clamp(self._seek_target)
                    self._seek_target = None
                    self._paused = False
                    self._pause_reason = None
                    self._interrupt.clear()
                    self._state = "speaking"
                    self._wake.notify_all()
                return True
        return False

    def _play_paragraph(self, index: int) -> bool:
        """Put paragraph ``index`` on the speaker. True if it was cut short.

        Nothing is synthesized here. The engine either has this paragraph
        already, is part way through it, or is pointed at it and its blocks are
        taken as they land. All three look the same from here, which is why a
        skip backwards and a skip onto a paragraph the engine is half way
        through both start speaking the instant the presses stop.
        """
        player = audio_out.PLAYER
        playback = self._config.get("playback") or {}
        self._paragraphs.position = index

        with self._lock:
            open_now = not self._player_open
            self._player_open = True
        if open_now:
            player.start(
                self.utterance_id,
                self._engine.sample_rate,
                prebuffer_ms=int(playback.get("prebuffer_ms") or 400),
                device=playback.get("device"),
            )
        else:
            # The conceptual gap between paragraphs, written before the next
            # one starts so it also covers the engine spinning up.
            player.write(self.utterance_id, self._engine.paragraph_pause_buffer)

        # Never hold less than the player is waiting for before it opens the
        # stream, or nothing would ever start playing and nothing would drain.
        lead = max(
            PLAYER_LEAD_SECONDS,
            int(playback.get("prebuffer_ms") or 400) / 1000.0 + WRITE_SLICE_SECONDS,
        )

        interrupted = False
        cached = self._paragraphs.blocks(index, AnyEvent(self._stop, self._interrupt))
        for block in _paced(cached, self._engine.sample_rate):
            if self._interrupt.is_set() or self._stop.is_set():
                interrupted = True
                break
            if not player.write(self.utterance_id, block):
                interrupted = True
                break
            if not self._wait_until(player, lead):
                interrupted = True
                break

        if self._interrupt.is_set() or self._stop.is_set():
            interrupted = True
        if interrupted:
            return True
        if not self._paragraphs.complete(index):
            # The cache lost this paragraph underneath us -- generation was
            # aimed elsewhere, or it was trimmed. Stay put and have it made.
            return True

        with self._lock:
            if not self._spoken or self._spoken[-1] != index:
                self._spoken.append(index)
        # Heard, not merely handed over: a paragraph can be shorter than the
        # queue, and moving on now would put the position a paragraph in front
        # of the listener with no way for a skip press to mean what it says.
        return not self._wait_until(player, PARAGRAPH_HANDOVER_SECONDS)

    def _wait_until(self, player: Any, buffered_seconds: float) -> bool:
        """Wait for the queue to fall to ``buffered_seconds``. False: interrupted.

        A player that cannot say how much it holds is not waited on, and neither
        is one that is not playing yet -- before the prebuffer is reached
        nothing is draining, so waiting would be waiting for good.
        """
        while not (self._stop.is_set() or self._interrupt.is_set()):
            try:
                if not player.is_playing():
                    return True
                buffered = float((player.status() or {}).get("buffered_seconds") or 0.0)
            except Exception:  # noqa: BLE001 - a player that cannot say never waits
                return True
            if buffered <= buffered_seconds:
                return True
            time.sleep(0.05)
        return False

    def _finish(self, error: str | None) -> None:
        player = audio_out.PLAYER
        cancelled = self._stop.is_set()

        # Wind the engine up first: it writes into the recording, and the
        # archive is about to be written from it.
        self._stop.set()
        self._paragraphs.stop()
        self._paragraphs.join(30.0)

        with self._lock:
            open_player = self._player_open
            self._player_open = False
        if error or cancelled:
            player.stop(utterance_id=self.utterance_id)
        elif open_player:
            # Play out what is queued rather than cutting the last words off.
            player.finish(self.utterance_id)
            player.wait_drained(30.0)

        book_error = self._paragraphs.error
        if error is None and book_error is not None:
            error = str(book_error)

        with self._lock:
            self._error = error
            self._state = (
                "error" if error else "cancelled" if cancelled else "done"
            )
            spoken = list(self._spoken)
            rotator = self._paragraphs.rotator
            if rotator is not None:
                self._recording.meta["reference_segments"] = rotator.segments

        try:
            self._job["cleanup"]()
        except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
            logger.exception("Job cleanup failed")

        entry = self._recording.write(
            input_text=self._text,
            sample_rate=self._engine.sample_rate,
            meta={
                "error": error,
                "cancelled": cancelled,
                "output": "device",
                "paragraphs": self._paragraphs.count(),
                "paragraphs_spoken": spoken,
            },
        )
        if entry:
            archive.prune(self._prepared["archive_max"])
        # The cache is the read's, not the document's: the archive keeps what
        # was said, and nothing here outlives the read that made it.
        self._paragraphs.cache.clear()
        self._done.set()

    # -- controls ---------------------------------------------------------
    def _silence(self) -> None:
        """Stop the sound, but only if this session still owns the speaker."""
        audio_out.PLAYER.stop(utterance_id=self.utterance_id)
        with self._lock:
            self._player_open = False

    def skip(self, delta: int) -> dict[str, Any]:
        """Move ``delta`` paragraphs from what is being heard and arm the debounce.

        Counted from the speaker, not the engine, which is usually a paragraph
        or two further on. The engine is left alone: if the paragraph landed on
        has been made already -- forwards into what was generated ahead, or
        backwards into what was heard before -- it is replayed from the cache
        and generation carries on undisturbed. Only a landing past everything
        made so far moves the engine, and then it starts there.

        A press while paused resumes rather than moves. The paragraph was cut
        off part way through, so it has to be spoken again from its start
        whatever happens; landing on it is what the listener meant. Press twice
        and the second press does move, because by then the first one has
        already resumed.
        """
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            resuming = self._paused and self._seek_target is None
            base = self._seek_target if self._seek_target is not None else self._index
            self._seek_target = self._paragraphs.clamp(
                base if resuming else base + delta
            )
            self._seek_deadline = time.monotonic() + self._debounce
            self._paused = False
            self._pause_reason = None
            self._state = "seeking"
            self._wake.notify_all()
        self._interrupt.set()
        self._silence()
        return self.status()

    def pause(self, reason: str = "paused") -> dict[str, Any]:
        """Stop the sound but keep the place. Resumed with a skip press."""
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            self._paused = True
            self._pause_reason = reason
            self._seek_target = None
            self._state = "paused"
            self._wake.notify_all()
        self._interrupt.set()
        self._silence()
        return self.status()

    def resume(self) -> dict[str, Any]:
        """Pick up at the paragraph that was interrupted, without waiting."""
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            self._seek_target = self._paragraphs.clamp(self._index)
            self._seek_deadline = time.monotonic()
            self._paused = False
            self._pause_reason = None
            self._state = "seeking"
            self._wake.notify_all()
        self._interrupt.set()
        return self.status()

    def cancel(self, reason: str = "cancelled") -> None:
        """End the session. Safe to call more than once, and from any thread."""
        with self._lock:
            if self._state in LIVE_STATES:
                self._state = reason
            self._stop.set()
            self._interrupt.set()
            self._wake.notify_all()
        self._paragraphs.stop()
        self._silence()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    # -- reporting --------------------------------------------------------
    @property
    def active(self) -> bool:
        with self._lock:
            return self._state in LIVE_STATES

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "utterance_id": self.utterance_id,
                "state": self._state,
                "error": self._error,
                "started_at": self.started_at,
                "paragraph": self._index,
                "paragraph_target": self._seek_target,
                "paragraph_generating": self._paragraphs.generating,
                "paragraphs": self._paragraphs.count(),
                "paragraphs_final": self._paragraphs.final,
                "paragraphs_cached": len(self._paragraphs.cache.cached()),
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "paragraph_preview": self._paragraphs.text(self._index)[:200],
            }

    # -- test seam --------------------------------------------------------
    @property
    def _book(self) -> ParagraphBook:
        """The engine's book. Tests drive a half-written document through it."""
        return self._paragraphs.book

    @property
    def _cache(self) -> AudioCache:
        return self._paragraphs.cache
