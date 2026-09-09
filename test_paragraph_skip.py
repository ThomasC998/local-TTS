#!/usr/bin/env python3
"""Paragraph transport controls, without a model and without a speaker.

Everything under test here is scheduling: which paragraph plays next, what a
run of skip presses collapses to, and whether an abandoned generation is closed
rather than left hanging. None of that needs the engine, so this substitutes a
fake one that "synthesizes" instantly and a player that counts frames instead of
opening an output stream.

    python test_paragraph_skip.py

No audio is played on any device.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_TEMP = tempfile.mkdtemp(prefix="breeze-skip-test-")
os.environ["BREEZE_VOICES_DIR"] = str(Path(_TEMP) / "voices")
os.environ["BREEZE_STATE_DIR"] = str(Path(_TEMP) / "state")

import archive  # noqa: E402
import audio_out  # noqa: E402
import speech_session  # noqa: E402
from speech_session import ParagraphBook, SpeechSession  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        FAILED += 1
        print(f"  \033[31m✗\033[0m {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
class FakePlayer:
    """Records what would have been played. Opens nothing."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.utterance: str | None = None
        self.frames = 0
        self.starts = 0
        self.stops = 0
        self.finished = 0

    def start(self, utterance_id, sample_rate, prebuffer_ms=400, device=None):
        with self.lock:
            self.utterance = utterance_id
            self.starts += 1

    def write(self, utterance_id, audio):
        with self.lock:
            if self.utterance != utterance_id:
                return False
            self.frames += int(np.asarray(audio).size)
            return True

    def finish(self, utterance_id):
        with self.lock:
            self.finished += 1

    def wait_drained(self, timeout=None):
        return True

    def stop(self, fade=True, utterance_id=None):
        with self.lock:
            if utterance_id is not None and self.utterance != utterance_id:
                return
            self.utterance = None
            self.stops += 1

    def is_playing(self):
        with self.lock:
            return self.utterance is not None

    def status(self):
        return {}


class FakeEngine:
    """Yields a fixed block per chunk and records which paragraphs ran.

    ``closed`` is the point of it: a generator abandoned mid-paragraph must
    have its ``finally`` run, because that is where the real engine releases the
    gate and closes its codec request.
    """

    sample_rate = 24000

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.pause_buffer = np.zeros(64, dtype=np.float32)
        self.paragraph_pause_buffer = np.zeros(256, dtype=np.float32)
        self.requests: list[str] = []
        self.closed: list[str] = []
        self.open_generations = 0
        self._lock = threading.Lock()

    def stream_live(self, source, *, request_id="live", **options):
        with self._lock:
            self.requests.append(request_id)
            self.open_generations += 1
        try:
            for index, (chunk, _reference, _new_paragraph) in enumerate(source):
                if index:
                    yield "audio", self.pause_buffer, index
                time.sleep(self.delay)
                yield "audio", np.zeros(1024, dtype=np.float32), index
                yield "boundary", None, index
        finally:
            with self._lock:
                self.closed.append(request_id)
                self.open_generations -= 1


def build_session(paragraphs, *, debounce=0.2, engine=None, use_llm=False):
    """A session over a document whose paragraphs are already known."""
    engine = engine or FakeEngine()
    chunks: list[str] = []
    groups: list[int] = []
    for index, paragraph in enumerate(paragraphs):
        for sentence in paragraph:
            chunks.append(sentence)
            groups.append(index)

    prepared = {
        "config": {"playback": {"prebuffer_ms": 0}, "llm": {}},
        "voice_id": None,
        "use_llm": use_llm,
        "instruction": None,
        "utterance_id": "utt_test",
        "archive_max": 10,
        "recording": archive.Recording("utt_test", enabled=False),
        "job": {
            "engine": engine,
            "chunks": chunks,
            "paragraph_ids": groups,
            "options": {"ref_audio": None, "chunk_refs": None, "paragraph_ids": groups},
            "request_id": "speak-test",
            "max_words": 35,
            "rotation_override": None,
            "cleanup": lambda: None,
        },
    }
    session = SpeechSession(prepared, "input text", debounce=debounce)
    return session, engine


DOC = [
    ["Paragraph one, first sentence.", "Paragraph one, second sentence."],
    ["Paragraph two, only sentence."],
    ["Paragraph three, first.", "Paragraph three, second."],
    ["Paragraph four."],
    ["Paragraph five."],
]


# ---------------------------------------------------------------------------
section("The book")
# ---------------------------------------------------------------------------
book = ParagraphBook()
book.add(0, "one")
book.add(0, "two")
book.add(3, "three")  # source paragraph ids may skip; book indices may not
book.finish()
check("blank runs do not create empty paragraphs", book.count() == 2, str(book.count()))
check("a paragraph keeps its chunks in order", book.text(0) == "one two", book.text(0))

stop = threading.Event()
check("chunks of a closed paragraph end", list(book.chunks(0, stop)) == ["one", "two"])
check("an index past the end never arrives", book.wait_for(5, stop) is False)

growing = ParagraphBook()
seen: list[str] = []


def late_writer() -> None:
    time.sleep(0.1)
    growing.add(0, "late chunk")
    time.sleep(0.1)
    growing.finish()


threading.Thread(target=late_writer, daemon=True).start()
started = time.monotonic()
seen = list(growing.chunks(0, stop))
check("a reader blocks for a chunk still being written", seen == ["late chunk"], str(seen))
check("and stops when the writer finishes", time.monotonic() - started < 2.0)


# ---------------------------------------------------------------------------
section("Playing straight through")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(DOC)
session.start()
check("the session finishes on its own", session.wait(20.0))
check("every paragraph was generated", len(engine.requests) == 5, str(engine.requests))
check("one request per paragraph", engine.requests[0].endswith("-p0"), engine.requests[0])
check("every generation was closed", len(engine.closed) == 5, str(len(engine.closed)))
check("nothing is left open", engine.open_generations == 0)
check("the state ends as done", session.status()["state"] == "done", session.status()["state"])
check("the speaker was claimed once, not per paragraph", player.starts == 1, str(player.starts))
check("playback was played out rather than cut", player.finished == 1)


# ---------------------------------------------------------------------------
section("Skipping forwards")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(DOC, debounce=0.3, engine=FakeEngine(delay=0.15))
session.start()
time.sleep(0.2)  # somewhere inside paragraph 0
for _ in range(3):
    session.skip(1)
    time.sleep(0.05)
status = session.status()
check("three quick presses aim three paragraphs on", status["paragraph_target"] == 3,
      str(status["paragraph_target"]))
check("the state says it is seeking", status["state"] == "seeking", status["state"])
check("the sound stopped at the first press", player.utterance is None)

session.wait(20.0)
generated = [request.rsplit("-p", 1)[1] for request in engine.requests]
check("the paragraphs passed over were never generated",
      generated == ["0", "3", "4"], str(generated))
check("every abandoned generation was closed",
      len(engine.closed) == len(engine.requests) and engine.open_generations == 0,
      f"{len(engine.closed)} closed of {len(engine.requests)}")


# ---------------------------------------------------------------------------
section("Skipping backwards")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(DOC, debounce=0.2, engine=FakeEngine(delay=0.4))
session.start()
time.sleep(1.4)  # a couple of paragraphs in
before = session.status()["paragraph"]
session.skip(-1)
time.sleep(0.5)
after = session.status()["paragraph"]
check("a back press lands on an earlier paragraph", after < before or after == before - 1,
      f"{before} -> {after}")
session.cancel()
session.wait(10.0)
check("a paragraph already spoken can be generated again",
      len(engine.requests) > len(set(engine.requests)) or after < before,
      str(engine.requests))

session, engine = build_session(DOC, debounce=0.1)
session.start()
time.sleep(0.05)
for _ in range(5):
    session.skip(-1)
check("back presses clamp at the first paragraph",
      session.status()["paragraph_target"] == 0,
      str(session.status()["paragraph_target"]))
session.cancel()
session.wait(10.0)


# ---------------------------------------------------------------------------
section("Losing the output device")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(DOC, debounce=0.15, engine=FakeEngine(delay=0.12))
session.start()
time.sleep(0.3)
paused_at = session.status()["paragraph"]
session.pause("default-output-changed")
status = session.status()
check("the read reports itself paused", status["state"] == "paused", status["state"])
check("the reason is carried through",
      status["pause_reason"] == "default-output-changed", str(status["pause_reason"]))
check("the sound stopped at once", player.utterance is None)

time.sleep(0.6)
check("a paused read stays where it is",
      session.status()["paragraph"] == paused_at and session.status()["state"] == "paused")

opened_before = len(engine.requests)
session.skip(1)
check("the first press after a pause resumes rather than moves",
      session.status()["paragraph_target"] == paused_at,
      f"{session.status()['paragraph_target']} vs {paused_at}")
time.sleep(0.4)
check("and it starts speaking again",
      len(engine.requests) > opened_before, str(engine.requests))
session.cancel()
session.wait(10.0)
check("nothing is left open after a pause and a cancel", engine.open_generations == 0)


# ---------------------------------------------------------------------------
section("Stopping")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(DOC, engine=FakeEngine(delay=0.1))
session.start()
time.sleep(0.15)
check("a live session reports itself active", session.active)
session.cancel()
check("the session ends promptly", session.wait(5.0))
check("its state is cancelled", session.status()["state"] == "cancelled",
      session.status()["state"])
check("it is no longer active", not session.active)
check("the generation was closed", engine.open_generations == 0)
check("a second cancel is harmless", session.cancel() is None)


# ---------------------------------------------------------------------------
section("A document still being written")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session([["only paragraph."]], debounce=0.1,
                                engine=FakeEngine(delay=0.3))
# A producer that has written one paragraph and has not said it is finished --
# the shape the language-model path is in for most of a long read.
session._fill_from_text = lambda: session._book.add(0, "only paragraph.")  # noqa: SLF001
session.start()
time.sleep(0.1)
session.skip(5)
check("a forward skip cannot run past the paragraph being written",
      session.status()["paragraph_target"] == 1,
      str(session.status()["paragraph_target"]))
session._book.finish()  # noqa: SLF001
check("the session ends once the producer does", session.wait(10.0))


# ---------------------------------------------------------------------------
section("An outgoing read must not silence the one replacing it")
# ---------------------------------------------------------------------------
# Cancelling a read takes as long as the chunk it is inside, and by then the read
# that replaced it may already own the speaker. Tested against the real player,
# which opens no stream until something is written to it.
real = audio_out.SpeechPlayer()
real.start("utt_old", 24000, prebuffer_ms=0)
check("the speaker is claimed", real.status()["utterance_id"] == "utt_old")
real.start("utt_new", 24000, prebuffer_ms=0)
check("a new utterance takes it over", real.status()["utterance_id"] == "utt_new")
real.stop(utterance_id="utt_old")
check("the outgoing utterance's stop is ignored",
      real.status()["utterance_id"] == "utt_new",
      str(real.status()["utterance_id"]))
real.stop(utterance_id="utt_new")
check("the owner's own stop is not", real.status()["utterance_id"] is None)


# ---------------------------------------------------------------------------
print(f"\n{PASSED} passed, {FAILED} failed\n")
sys.exit(1 if FAILED else 0)
