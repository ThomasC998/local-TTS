"""The same read, served to a phone instead of to the speakers.

A ``SpeechSession`` takes paragraphs out of a ``ParagraphEngine`` and puts them
on this Mac's output device. A ``RemoteRead`` takes the same paragraphs out of
the same kind of engine and puts them on the wire, one HTTP response per
paragraph, and lets the phone decide which one it wants next.

Splitting it per paragraph rather than streaming one long response is what makes
the transport controls work. Each paragraph is a separate URL, so on the phone
it is a separate item in the player's playlist, so "next" and "previous" on the
lock screen are paragraph skips without a line of code on either side arranging
it. It also means the phone can ask for paragraph nine without having listened
to one through eight, and the engine will oblige.

Which paragraph was asked for is also the only position signal there is: the
phone does not report where it is, it simply fetches what it is about to play.
So a request sets ``engine.position``, and everything the engine does with that
-- staying a few paragraphs ahead, trimming the furthest away when the cache
grows -- follows the listener without being told about it.

One shape of response, and the reason is worth writing down because the other
one looked so much better. A paragraph goes out only once it is finished, with
its true length in the header.

The tempting alternative is to send it as it is made, so the first one starts
playing seconds sooner. But a WAV says its length in its first 44 bytes, before
a single sample exists, and audio still being generated has no length to
declare. The convention for that is to write 0xFFFFFFFF and mean "read to the
end of the stream", which is a lie players are free to believe: ExoPlayer
believes it, works out a duration of twenty-four hours, and when the audio runs
out it does not end the paragraph -- it sits there with the clock running,
never reaching the next one, holding the playback service open for the rest of
the day. Measured, not guessed: twelve seconds of speech, then two minutes of
silence with paragraphs two and three never fetched.

So the wait is real, and it is paid once. Only the paragraph being listened to
now can ever be unfinished; the generator runs faster than speech, so by the
time the phone asks for the next one it has been sitting in the cache for a
while. Making the *first* paragraph a short one is what keeps that single wait
down to a few seconds -- see ``opening_split`` below.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np

import archive
from paragraph_engine import ParagraphEngine

logger = logging.getLogger("breeze.remote")

# How many reads may be alive at once. The engine is one piece of hardware and
# there is one person holding the phone; more than a couple of live reads means
# something has leaked rather than that somebody is listening to three things.
MAX_LIVE_READS = 3

# A read nobody has touched for this long is over, whatever the phone thinks.
# Long enough to survive a walk between rooms, short enough that a forgotten
# read does not hold a hundred megabytes of audio until the server restarts.
IDLE_EXPIRY_SECONDS = 30 * 60

# What is sent per chunk while a paragraph is still being generated.
STREAM_BLOCK_SECONDS = 0.5

# How long a request may wait for its paragraph before giving up on it.
#
# Not a tuning knob so much as a deadlock guard: the paragraph being waited for
# is being generated as we wait, and generation runs faster than speech, so the
# wait is roughly how long the paragraph takes to say and always ends. This is
# the number that says a generator which has stopped producing without saying
# so is a fault, not a slow paragraph. It sits under the phone's own 60-second
# read timeout so that the server is the one that decides, and answers.
WAIT_CEILING = 45.0


def _wav_header(sample_rate: int, frames: int) -> bytes:
    """A 44-byte mono 16-bit PCM header for audio whose length is known."""
    data_bytes = frames * 2
    riff_bytes = data_bytes + 36
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_bytes, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_bytes,
    )


def _pcm16(block: np.ndarray) -> bytes:
    """One engine block as the 16-bit samples that go on the wire."""
    clipped = np.clip(np.asarray(block, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def opening_split(paragraph_ids: list[int]) -> list[int]:
    """Give the first sentence a paragraph to itself.

    Every paragraph after the first is generated while an earlier one is being
    spoken, so it is finished before anyone asks for it. The first has nobody
    ahead of it: the phone asks, and waits for as long as that paragraph takes
    to make, which is roughly as long as it takes to say. A paragraph of a
    hundred words is half a minute of waiting at a spinner.

    A chunk is a sentence, so making the first chunk its own paragraph turns
    that wait into a few seconds, and by the time that sentence has been read
    out the rest of its paragraph is ready. The cost is one extra stop for the
    skip button, at the end of the first sentence, and it buys the difference
    between a read that starts and one that looks broken.

    Ids only have to *change* where a paragraph changes -- ``ParagraphBook``
    groups runs of equal ids -- so this needs no renumbering, just one id that
    differs from its neighbour.

    Only the clipboard path is chunked in advance like this. A read written by
    the language model arrives paragraph by paragraph already, and its first
    one is short because the model has only just started writing.
    """
    if len(paragraph_ids) < 2 or paragraph_ids[0] != paragraph_ids[1]:
        return list(paragraph_ids)
    return [min(paragraph_ids) - 1, *paragraph_ids[1:]]


class RemoteRead:
    """One document being read by a phone.

    Holds a ``ParagraphEngine`` and nothing else of substance: the position, the
    cache and the generator all belong to the engine, and this is the part that
    knows about HTTP, expiry and the archive.
    """

    def __init__(
        self, prepared: dict[str, Any], text: str, owner: str = "phone"
    ) -> None:
        self.read_id = f"rd_{uuid.uuid4().hex[:16]}"
        self.owner = owner
        self.text = text
        self.created_at = time.time()
        self._touched = time.monotonic()
        # A copy, because the split below is the phone's business: the same
        # prepared job read aloud on this Mac keeps the paragraphs the document
        # actually has.
        self._job = dict(prepared["job"])
        self._job["paragraph_ids"] = opening_split(self._job.get("paragraph_ids") or [])
        prepared = {**prepared, "job": self._job}
        self._prepared = prepared
        self._recording: archive.Recording = prepared["recording"]
        self._tts = self._job["engine"]
        self._engine = ParagraphEngine(prepared, text)
        self._lock = threading.Lock()
        self._closed = False
        self._served: set[int] = set()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._engine.start()

    def touch(self) -> None:
        self._touched = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._touched

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, reason: str = "closed") -> None:
        """End the read, write its archive entry, and drop the audio."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._engine.stop()
        self._engine.join(30.0)

        rotator = self._engine.rotator
        if rotator is not None:
            self._recording.meta["reference_segments"] = rotator.segments
        try:
            self._job["cleanup"]()
        except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
            logger.exception("Job cleanup failed")

        error = self._engine.error
        entry = self._recording.write(
            input_text=self.text,
            sample_rate=self._tts.sample_rate,
            meta={
                "error": str(error) if error else None,
                "cancelled": reason != "finished",
                "output": "phone",
                "paragraphs": self._engine.count(),
                "paragraphs_spoken": sorted(self._served),
            },
        )
        if entry:
            archive.prune(self._prepared["archive_max"])
        self._engine.cache.clear()
        logger.info("Read %s %s after %d paragraph(s)",
                    self.read_id, reason, len(self._served))

    # -- what the routes ask for ------------------------------------------
    def manifest(self) -> dict[str, Any]:
        """Where the read is: how many paragraphs exist, which are ready."""
        self.touch()
        status = self._engine.status()
        error = self._engine.error
        return {
            "read_id": self.read_id,
            "paragraphs": status["paragraphs"],
            "final": status["paragraphs_final"],
            "ready": status["paragraphs_cached"],
            "position": status["paragraph"],
            "generating": status["paragraph_generating"],
            "error": str(error) if error else None,
            "created_at": self.created_at,
            "sample_rate": self._tts.sample_rate,
            "owner": self.owner,
        }

    def preview(self, index: int = 0, limit: int = 200) -> str:
        return self._engine.text(index)[:limit]

    def exists(self, index: int) -> bool:
        """Whether this paragraph will ever have text.

        Blocks only while the answer is genuinely unknown -- the language model
        is still writing and this index is one past the end.
        """
        if index < 0:
            return False
        if index < self._engine.count():
            return True
        if self._engine.final:
            return False
        return self._engine.wait_for(index, threading.Event())

    def paragraph(self, index: int) -> tuple[dict[str, str], Iterator[bytes]]:
        """One paragraph as a WAV response: headers, then the body.

        Asking for it is what moves the listener: the engine is told this is
        where the phone is, and only if nothing has made this paragraph is the
        engine sent to it.

        The response is always a finished paragraph with its true length, which
        is what lets the player show a duration, seek inside it, and -- the part
        that matters most -- know when it has ended and move to the next one.
        If it is not finished yet, the request waits for it. See this module's
        own notes for why the obvious alternative is not one.

        A caller may still put ``?wait=`` on the URL; earlier versions of the
        app do. It is accepted and ignored, because the answer no longer
        depends on it.
        """
        self.touch()
        self._engine.position = index
        with self._lock:
            self._served.add(index)

        if not self._engine.complete(index):
            self._await_paragraph(index, deadline=time.monotonic() + WAIT_CEILING)
        if not self._engine.complete(index):
            raise TimeoutError(
                f"Paragraph {index} was still being made after "
                f"{WAIT_CEILING:.0f} seconds"
            )

        frames = self._engine.cache.frames(index)
        headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(44 + frames * 2),
            "Accept-Ranges": "none",
            "X-Breeze-Paragraph": str(index),
            "X-Breeze-Seconds": f"{frames / self._tts.sample_rate:.2f}",
        }
        return headers, self._body(index, frames)

    def _await_paragraph(self, index: int, deadline: float) -> None:
        """Give the generator until ``deadline`` to finish this paragraph.

        The blocks are read and dropped: the cache is what holds them, and the
        point of reading is only to arrive at the end -- or at the deadline,
        whichever comes first. Giving up is not a failure here; it only means
        the audio goes out as a stream instead of as a file.
        """
        give_up = threading.Event()
        timer = threading.Timer(max(0.0, deadline - time.monotonic()), give_up.set)
        timer.daemon = True
        timer.start()
        try:
            for _block in self._engine.blocks(index, give_up):
                if give_up.is_set():
                    return
        finally:
            timer.cancel()

    def _body(self, index: int, frames: int) -> Iterator[bytes]:
        rate = self._tts.sample_rate
        stop = threading.Event()
        yield _wav_header(rate, frames)
        pending: list[bytes] = []
        pending_frames = 0
        flush_at = max(1, int(rate * STREAM_BLOCK_SECONDS))
        for block in self._engine.blocks(index, stop):
            pending.append(_pcm16(block))
            pending_frames += int(block.size)
            if pending_frames >= flush_at:
                yield b"".join(pending)
                pending, pending_frames = [], 0
        if pending:
            yield b"".join(pending)

    def playlist(self, base: str, token: str | None) -> str:
        """The read as an M3U, so any audio player can be the client.

        Not what the app uses -- it builds its own playlist so it can grow one
        while the model is still writing -- but it costs four lines and it means
        a read is never stuck behind one particular piece of software.
        """
        self.touch()
        query = f"?t={token}" if token else ""
        lines = ["#EXTM3U"]
        for index in range(max(1, self._engine.count())):
            lines.append(f"#EXTINF:-1,Paragraph {index + 1}")
            lines.append(f"{base}/v1/read/{self.read_id}/p{index}.wav{query}")
        return "\n".join(lines) + "\n"


class ReadRegistry:
    """The live reads, with the housekeeping that keeps them from piling up."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reads: dict[str, RemoteRead] = {}

    def create(
        self, prepared: dict[str, Any], text: str, owner: str = "phone"
    ) -> RemoteRead:
        """Start a read for one device, replacing only that device's own.

        A second phone, or this Mac's own speakers, is a different listener in
        a different room: it keeps whatever it was playing. What a new read
        replaces is the last one *the same device* asked for, which is what
        "read this instead" means when you press it twice.
        """
        read = RemoteRead(prepared, text, owner=owner)
        expired: list[RemoteRead] = []
        with self._lock:
            self._sweep(expired)
            for read_id, existing in list(self._reads.items()):
                if existing.owner == owner:
                    expired.append(self._reads.pop(read_id))
            while len(self._reads) >= MAX_LIVE_READS:
                oldest = max(self._reads.values(), key=lambda item: item.idle_seconds)
                expired.append(self._reads.pop(oldest.read_id))
            self._reads[read.read_id] = read
        for old in expired:
            old.close("replaced")
        read.start()
        return read

    def for_owner(self, owner: str) -> RemoteRead | None:
        with self._lock:
            for read in self._reads.values():
                if read.owner == owner:
                    return read
        return None

    def get(self, read_id: str) -> RemoteRead | None:
        expired: list[RemoteRead] = []
        with self._lock:
            self._sweep(expired)
            read = self._reads.get(read_id)
        for old in expired:
            old.close("expired")
        return read

    def drop(self, read_id: str, reason: str = "closed") -> bool:
        with self._lock:
            read = self._reads.pop(read_id, None)
        if read is None:
            return False
        read.close(reason)
        return True

    def clear(self, reason: str = "shutdown") -> None:
        with self._lock:
            reads = list(self._reads.values())
            self._reads.clear()
        for read in reads:
            read.close(reason)

    def live(self) -> list[dict[str, Any]]:
        with self._lock:
            reads = list(self._reads.values())
        return [
            {
                "read_id": read.read_id,
                "owner": read.owner,
                "created_at": read.created_at,
                "idle_seconds": round(read.idle_seconds, 1),
                "paragraphs": read.manifest()["paragraphs"],
            }
            for read in reads
        ]

    def _sweep(self, into: list[RemoteRead]) -> None:
        """Collect reads nobody has touched. Called with the lock held."""
        for read_id, read in list(self._reads.items()):
            if read.idle_seconds > IDLE_EXPIRY_SECONDS:
                into.append(self._reads.pop(read_id))


READS = ReadRegistry()
