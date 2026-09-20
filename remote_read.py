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

Two shapes of response, for one reason. A paragraph the engine has finished is
sent with its real length, so the phone can seek inside it and show a duration;
a paragraph still being made is sent as it arrives, so the first one starts
playing seconds before it is finished. Everything after the first is normally
finished before the phone asks.
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

# The length written into the header of a paragraph that is not finished yet.
# The convention for a WAV whose length is not known when the header goes out;
# players read the data chunk as running to the end of the stream.
_UNKNOWN_LENGTH = 0xFFFFFFFF


def _wav_header(sample_rate: int, frames: int | None) -> bytes:
    """A 44-byte mono 16-bit PCM header. ``frames`` of None means "still coming"."""
    data_bytes = _UNKNOWN_LENGTH if frames is None else frames * 2
    riff_bytes = _UNKNOWN_LENGTH if frames is None else data_bytes + 36
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


class RemoteRead:
    """One document being read by a phone.

    Holds a ``ParagraphEngine`` and nothing else of substance: the position, the
    cache and the generator all belong to the engine, and this is the part that
    knows about HTTP, expiry and the archive.
    """

    def __init__(self, prepared: dict[str, Any], text: str) -> None:
        self.read_id = f"rd_{uuid.uuid4().hex[:16]}"
        self.text = text
        self.created_at = time.time()
        self._touched = time.monotonic()
        self._prepared = prepared
        self._job = prepared["job"]
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

    def paragraph(
        self, index: int, wait: bool = False
    ) -> tuple[dict[str, str], Iterator[bytes]]:
        """One paragraph as a WAV response: headers, then the body.

        Asking for it is what moves the listener: the engine is told this is
        where the phone is, and only if nothing has made this paragraph is the
        engine sent to it.

        ``wait`` trades the first second or two of a paragraph nobody has made
        yet for a response that knows its own length. A player handed audio of
        unknown length has to guess at the duration, and what it guesses from a
        streamed WAV is nonsense -- so the app waits and gets a scrub bar, while
        a plain playlist in some other player streams and starts sooner.
        """
        self.touch()
        self._engine.position = index
        with self._lock:
            self._served.add(index)

        if wait and not self._engine.complete(index):
            self._await_paragraph(index)

        if self._engine.complete(index):
            frames = self._engine.cache.frames(index)
            headers = {
                "Content-Type": "audio/wav",
                "Content-Length": str(44 + frames * 2),
                "Accept-Ranges": "none",
                "X-Breeze-Paragraph": str(index),
                "X-Breeze-Cached": "1",
            }
            return headers, self._body(index, frames)

        headers = {
            "Content-Type": "audio/wav",
            "Cache-Control": "no-store",
            "X-Breeze-Paragraph": str(index),
            "X-Breeze-Cached": "0",
        }
        return headers, self._body(index, None)

    def _await_paragraph(self, index: int, attempts: int = 2) -> None:
        """Let the generator finish this paragraph, without keeping the audio.

        The blocks are read and dropped: the cache is what holds them, and the
        point of reading is only to arrive at the end. A paragraph the engine
        was pulled off part way through is asked for again, once.
        """
        stop = threading.Event()
        for _ in range(attempts):
            for _block in self._engine.blocks(index, stop):
                pass
            if self._engine.complete(index) or self._closed:
                return

    def _body(self, index: int, frames: int | None) -> Iterator[bytes]:
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

    def create(self, prepared: dict[str, Any], text: str) -> RemoteRead:
        read = RemoteRead(prepared, text)
        expired: list[RemoteRead] = []
        with self._lock:
            self._sweep(expired)
            while len(self._reads) >= MAX_LIVE_READS:
                oldest = max(self._reads.values(), key=lambda item: item.idle_seconds)
                expired.append(self._reads.pop(oldest.read_id))
            self._reads[read.read_id] = read
        for old in expired:
            old.close("replaced")
        read.start()
        return read

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
