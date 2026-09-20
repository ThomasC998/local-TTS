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
from speech_session import AudioCache, ParagraphBook, SpeechSession  # noqa: E402

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


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll rather than sleep a fixed time: three threads set their own pace."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
class FakePlayer:
    """Records what would have been played, and drains in real time. Opens nothing.

    The draining matters. The session keeps its queue short on purpose, because
    cached audio is handed over as fast as the player will take it and a player
    that swallows a paragraph whole would leave the session believing it was
    already at the end of something the listener has not started. A fake that
    always reports an empty queue would hide exactly that.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.utterance: str | None = None
        self.frames = 0
        self.starts = 0
        self.stops = 0
        self.finished = 0
        self.rate = 24000
        self._queued = 0.0
        self._at = time.monotonic()

    def _drain(self) -> None:
        now = time.monotonic()
        self._queued = max(0.0, self._queued - (now - self._at))
        self._at = now

    def start(self, utterance_id, sample_rate, prebuffer_ms=400, device=None):
        with self.lock:
            self.utterance = utterance_id
            self.starts += 1
            self.rate = int(sample_rate) or 24000
            self._queued = 0.0
            self._at = time.monotonic()

    def write(self, utterance_id, audio):
        with self.lock:
            if self.utterance != utterance_id:
                return False
            self._drain()
            size = int(np.asarray(audio).size)
            self.frames += size
            self._queued += size / float(self.rate)
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
            self._queued = 0.0
            self._at = time.monotonic()

    def is_playing(self):
        with self.lock:
            return self.utterance is not None

    def status(self):
        with self.lock:
            self._drain()
            return {"buffered_seconds": round(self._queued, 3)}


class FakeEngine:
    """Yields a fixed block per chunk and records which paragraphs ran.

    ``closed`` is the point of it: a generator abandoned mid-paragraph must
    have its ``finally`` run, because that is where the real engine releases the
    gate and closes its codec request.
    """

    sample_rate = 24000

    def __init__(self, delay: float = 0.02, frames: int = 1024) -> None:
        self.delay = delay
        self.frames = frames
        self.pause_buffer = np.zeros(64, dtype=np.float32)
        self.paragraph_pause_buffer = np.zeros(256, dtype=np.float32)
        self.requests: list[str] = []
        self.closed: list[str] = []
        self.open_generations = 0
        self._lock = threading.Lock()

    def paragraphs(self) -> list[str]:
        """The paragraph each generation was for, in the order they ran."""
        return [request.rsplit("-p", 1)[1] for request in self.requests]

    def stream_live(self, source, *, request_id="live", **options):
        with self._lock:
            self.requests.append(request_id)
            self.open_generations += 1
        try:
            for index, (chunk, _reference, _new_paragraph) in enumerate(source):
                if index:
                    yield "audio", self.pause_buffer, index
                time.sleep(self.delay)
                yield "audio", np.zeros(self.frames, dtype=np.float32), index
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

LONG_DOC = [[f"Paragraph {number}, its only sentence."] for number in range(12)]


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
section("The audio cache")
# ---------------------------------------------------------------------------
cache = AudioCache()
cache.begin(0)
cache.append(0, np.ones(100, dtype=np.float32))
check("a paragraph being made is live", cache.live(0) and not cache.complete(0))
cache.append(0, np.full(50, 0.5, dtype=np.float32))
cache.finish(0)
check("a finished paragraph is complete", cache.complete(0))
stop = threading.Event()
replay = [block.size for block in cache.blocks(0, stop)]
check("it replays every block it was given", replay == [100, 50], str(replay))
check("and again, as many times as asked",
      [block.size for block in cache.blocks(0, stop)] == [100, 50])
check("it knows what it holds", cache.cached() == [0], str(cache.cached()))

# The case a skip forwards lands in: a reader on a paragraph the engine is only
# part way through has to start now and keep up, not wait for the end.
cache.begin(1)
cache.append(1, np.zeros(10, dtype=np.float32))
seen: list[int] = []


def late_generator() -> None:
    time.sleep(0.15)
    cache.append(1, np.zeros(20, dtype=np.float32))
    time.sleep(0.15)
    cache.finish(1)


threading.Thread(target=late_generator, daemon=True).start()
started = time.monotonic()
seen = [block.size for block in cache.blocks(1, stop)]
check("a reader picks up blocks still being generated", seen == [10, 20], str(seen))
check("and ends when the generator does", time.monotonic() - started < 2.0)

cache.begin(2)
cache.append(2, np.zeros(30, dtype=np.float32))
cache.abandon(2)
check("an abandoned paragraph is not live", not cache.live(2) and not cache.complete(2))
cache.discard(2)
check("and its half a paragraph is dropped", cache.state(2) is None)

failing = AudioCache()
failing.begin(0)
failing.fail(0, RuntimeError("engine fell over"))
try:
    list(failing.blocks(0, stop))
    raised = False
except RuntimeError:
    raised = True
check("a generator's failure is raised at the reader", raised)

tight = AudioCache(budget_bytes=4 * 100 * 4)  # four blocks of a hundred floats
for index in range(6):
    tight.begin(index)
    tight.append(index, np.zeros(100, dtype=np.float32))
    tight.finish(index)
    tight.trim(5)
kept = tight.cached()
check("over budget, the paragraphs furthest from the ear go first",
      5 in kept and 0 not in kept, str(kept))
check("and it stays inside its budget", tight.size_bytes <= 4 * 100 * 4,
      str(tight.size_bytes))
tight.clear()
check("clearing empties it", tight.cached() == [] and tight.size_bytes == 0)


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
check("every paragraph was reached", set(engine.paragraphs()) == {"0", "1", "2", "3", "4"},
      str(engine.paragraphs()))
check("and none of them was generated twice",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))
check("every abandoned generation was closed",
      len(engine.closed) == len(engine.requests) and engine.open_generations == 0,
      f"{len(engine.closed)} closed of {len(engine.requests)}")

# A skip the engine cannot already have covered is the case that still has to
# generate: the paragraphs between here and there are never made at all.
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(LONG_DOC, debounce=0.3,
                                engine=FakeEngine(delay=0.25, frames=24000))
session.start()
time.sleep(0.1)
session.skip(9)
time.sleep(0.8)
reached = engine.paragraphs()
check("a skip past what is generated lands on a fresh generation",
      "9" in reached, str(reached))
check("and the paragraphs jumped over are never made",
      not ({"5", "6", "7", "8"} & set(reached)), str(reached))
session.cancel()
session.wait(10.0)
check("nothing is left open after a long skip", engine.open_generations == 0)


# ---------------------------------------------------------------------------
section("Skipping backwards")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

session, engine = build_session(LONG_DOC, debounce=0.2,
                                engine=FakeEngine(delay=0.05, frames=24000))
session.start()
check("the read gets a couple of paragraphs in",
      wait_until(lambda: session.status()["paragraph"] >= 2, 10.0),
      str(session.status()))

# The press's own reply is what says where it aimed: read the position
# separately and playback may have moved on between the two.
before = session.skip(-1)
target = before["paragraph_target"]
check("a back press aims one paragraph earlier",
      target == before["paragraph"] - 1, f"{before['paragraph']} -> {target}")
check("and lands there",
      wait_until(lambda: session.status()["paragraph"] <= target, 5.0),
      f"{target} vs {session.status()['paragraph']}")
after = session.status()
check("the paragraph gone back to is replayed, not made again",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))
check("and the engine carries on from where it was",
      after["paragraph_generating"] >= before["paragraph_generating"],
      f"{before['paragraph_generating']} -> {after['paragraph_generating']}")
session.cancel()
session.wait(10.0)
check("no paragraph was ever generated twice",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))
check("nothing is left open", engine.open_generations == 0)

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
section("Landing on audio that already exists")
# ---------------------------------------------------------------------------
player = FakePlayer()
audio_out.PLAYER = player
speech_session.audio_out.PLAYER = player

# An engine well ahead of the speaker, which is the ordinary case: a paragraph
# takes far less to make than to say.
session, engine = build_session(LONG_DOC, debounce=0.2,
                                engine=FakeEngine(delay=0.05, frames=36000))
session.start()
check("the engine runs in front of the ear",
      wait_until(lambda: session.status()["paragraph_generating"] >= 2, 8.0),
      str(session.status()))
check("but not indefinitely far in front",
      session.status()["paragraph_generating"]
      <= session.status()["paragraph"] + speech_session.GENERATE_AHEAD_PARAGRAPHS + 1,
      str(session.status()))
check("what it made is kept", session.status()["paragraphs_cached"] >= 2,
      str(session.status()))

before = session.status()
session.skip(1)
check("a skip onto generated audio starts playing it",
      wait_until(lambda: session.status()["paragraph"] == before["paragraph"] + 1, 5.0),
      f"{before['paragraph']} -> {session.status()['paragraph']}")
check("without generating the paragraph a second time",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))
check("and the engine was never sent back to where the ear is",
      session.status()["paragraph_generating"] >= before["paragraph_generating"],
      f"{before['paragraph_generating']} -> "
      f"{session.status()['paragraph_generating']}")

played = player.frames
check("the cached audio really is played",
      wait_until(lambda: player.frames > played, 5.0), str(player.frames))

session.cancel()
session.wait(10.0)
check("the cache goes with the session",
      session.status()["paragraphs_cached"] == 0
      and session._cache.size_bytes == 0,  # noqa: SLF001
      str(session.status()["paragraphs_cached"]))
check("nothing is left open", engine.open_generations == 0)


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

played_before = player.frames
session.skip(1)
check("the first press after a pause resumes rather than moves",
      session.status()["paragraph_target"] == paused_at,
      f"{session.status()['paragraph_target']} vs {paused_at}")
check("and it starts speaking again",
      wait_until(lambda: player.frames > played_before, 5.0),
      f"{played_before} -> {player.frames}")
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
session._paragraphs._fill_from_text = (  # noqa: SLF001
    lambda: session._book.add(0, "only paragraph.")  # noqa: SLF001
)
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
