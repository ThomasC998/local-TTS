#!/usr/bin/env python3
"""The read a phone drives, without a phone and without a model.

What matters here is the same thing that matters on the hotkey path: a
paragraph is generated once. The phone reaches a paragraph by asking for its
URL, so "asking twice" and "asking for one you already heard" have to be free,
and only "asking for one nothing has made" may cost a generation.

The engine is faked -- it "synthesizes" a fixed block per chunk instantly and
records which paragraph each generation was for, which is all the evidence any
of these claims needs.

    python test_remote_read.py

Nothing is played, nothing is downloaded, and no model is loaded.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_TEMP = tempfile.mkdtemp(prefix="breeze-read-test-")
os.environ["BREEZE_VOICES_DIR"] = str(Path(_TEMP) / "voices")
os.environ["BREEZE_STATE_DIR"] = str(Path(_TEMP) / "state")

import archive  # noqa: E402
import remote_read  # noqa: E402
from remote_read import ReadRegistry, RemoteRead  # noqa: E402

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
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class FakeEngine:
    """Generates instantly and remembers which paragraphs it was asked for."""

    sample_rate = 24000

    def __init__(self, delay: float = 0.02, frames: int = 2400) -> None:
        self.delay = delay
        self.frames = frames
        self.pause_buffer = np.zeros(64, dtype=np.float32)
        self.paragraph_pause_buffer = np.zeros(256, dtype=np.float32)
        self.requests: list[str] = []
        self.open_generations = 0
        self._lock = threading.Lock()

    def paragraphs(self) -> list[str]:
        return [request.rsplit("-p", 1)[1] for request in self.requests]

    def stream_live(self, source, *, request_id="live", **options):
        with self._lock:
            self.requests.append(request_id)
            self.open_generations += 1
        try:
            for index, (_chunk, _reference, _new) in enumerate(source):
                time.sleep(self.delay)
                yield "audio", np.full(self.frames, 0.25, dtype=np.float32), index
                yield "boundary", None, index
        finally:
            with self._lock:
                self.open_generations -= 1


def build_read(paragraphs, *, engine=None) -> tuple[RemoteRead, FakeEngine]:
    engine = engine or FakeEngine()
    chunks: list[str] = []
    groups: list[int] = []
    for index, paragraph in enumerate(paragraphs):
        for sentence in paragraph:
            chunks.append(sentence)
            groups.append(index)
    prepared = {
        "config": {"playback": {}, "llm": {}},
        "voice_id": None,
        "use_llm": False,
        "instruction": None,
        "utterance_id": "utt_read_test",
        "archive_max": 10,
        "recording": archive.Recording("utt_read_test", enabled=False),
        "job": {
            "engine": engine,
            "chunks": chunks,
            "paragraph_ids": groups,
            "options": {"ref_audio": None, "chunk_refs": None, "paragraph_ids": groups},
            "request_id": "read-test",
            "max_words": 35,
            "rotation_override": None,
            "cleanup": lambda: None,
        },
    }
    return RemoteRead(prepared, "input text"), engine


DOC = [[f"Paragraph {number}, its only sentence."] for number in range(8)]


def collect(read: RemoteRead, index: int) -> tuple[dict, bytes]:
    headers, body = read.paragraph(index)
    return headers, b"".join(body)


def wav_frames(blob: bytes) -> int:
    """Frames according to the header, which is what a player will believe."""
    (data_bytes,) = struct.unpack("<I", blob[40:44])
    return data_bytes // 2


# ---------------------------------------------------------------------------
section("A paragraph on the wire")
# ---------------------------------------------------------------------------
read, engine = build_read(DOC)
read.start()

headers, blob = collect(read, 0)
check("the body is a RIFF/WAVE file", blob[:4] == b"RIFF" and blob[8:12] == b"WAVE",
      str(blob[:12]))
# format, channels, sample rate, byte rate: PCM, mono, the engine's own rate.
check("mono 16-bit PCM at the engine's rate",
      struct.unpack("<HHII", blob[20:32]) == (1, 1, 24000, 48000),
      str(struct.unpack("<HHII", blob[20:32])))
check("the audio is as long as the engine made it",
      len(blob) - 44 == 2400 * 2, str(len(blob) - 44))

# The first paragraph is asked for while it is still being made, so it goes out
# as it arrives: no length in the header, because there is nothing to put there
# yet. That is what lets the phone start playing before the engine has finished.
check("a paragraph still being made streams",
      headers.get("Content-Length") is None
      and headers.get("X-Breeze-Cached") == "0", str(headers))
check("...and its header says the length is not known yet",
      wav_frames(blob) == 0xFFFFFFFF // 2, str(wav_frames(blob)))

# Asked for again, it is finished, and now it can be sent as a file with a
# length -- which is what gives the phone a duration and a working scrub bar.
generated_once = len(engine.requests)
headers, again = collect(read, 0)
check("a finished paragraph is sent with its length",
      headers.get("Content-Length") == str(len(again)),
      f"{headers.get('Content-Length')} vs {len(again)}")
check("...and says it came from the cache", headers.get("X-Breeze-Cached") == "1")
check("...with the real length in the header too",
      wav_frames(again) == 2400, str(wav_frames(again)))
check("the audio itself is identical either way", again[44:] == blob[44:])
check("and generating it again was not necessary",
      len(engine.requests) == generated_once, str(engine.paragraphs()))


# ---------------------------------------------------------------------------
section("Where the phone is")
# ---------------------------------------------------------------------------
check("the engine ran ahead of the paragraph asked for",
      wait_until(lambda: read.manifest()["generating"] >= 2, 5.0),
      str(read.manifest()))
import paragraph_engine  # noqa: E402

state = read.manifest()
check("but not indefinitely far ahead",
      state["generating"]
      <= state["position"] + paragraph_engine.GENERATE_AHEAD_PARAGRAPHS + 1,
      str(state))

before = read.manifest()
collect(read, 5)
after = read.manifest()
check("asking for a paragraph moves the position", after["position"] == 5,
      str(after["position"]))
check("a paragraph past the generated region is generated on demand",
      "5" in engine.paragraphs(), str(engine.paragraphs()))
check("and the ones jumped over are not made first",
      engine.paragraphs().count("5") == 1, str(engine.paragraphs()))

collect(read, 1)
check("going back replays rather than regenerates",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))
check("every paragraph it made, it made once",
      len(engine.requests) == len(set(engine.requests)), str(engine.paragraphs()))


# ---------------------------------------------------------------------------
section("The ends of the document")
# ---------------------------------------------------------------------------
check("a paragraph that exists is reported as existing", read.exists(7))
check("one past the end never will be", not read.exists(8))
check("nor does a negative index", not read.exists(-1))

manifest = read.manifest()
check("the manifest counts every paragraph", manifest["paragraphs"] == 8,
      str(manifest["paragraphs"]))
check("and knows the text is final", manifest["final"] is True)
check("it lists what is ready to play", len(manifest["ready"]) >= 3,
      str(manifest["ready"]))

playlist = read.playlist("https://mac.local:7860", "tok")
check("the playlist has one entry per paragraph",
      playlist.count("#EXTINF") == 8, str(playlist.count("#EXTINF")))
check("and carries the token, since a player sends no headers",
      "?t=tok" in playlist)


# ---------------------------------------------------------------------------
section("Ending a read")
# ---------------------------------------------------------------------------
read.close("finished")
check("the read reports itself closed", read.closed)
check("the cached audio is gone", read._engine.cache.size_bytes == 0,  # noqa: SLF001
      str(read._engine.cache.size_bytes))  # noqa: SLF001
check("nothing is left generating", engine.open_generations == 0)
check("closing twice is harmless", read.close("finished") is None)


# ---------------------------------------------------------------------------
section("The registry")
# ---------------------------------------------------------------------------
registry = ReadRegistry()
created = []
for _ in range(remote_read.MAX_LIVE_READS + 1):
    one, _engine = build_read(DOC[:2])
    registry._reads[one.read_id] = one  # noqa: SLF001 - stand in for create()
    one.start()
    created.append(one)
    time.sleep(0.01)

check("the registry holds what was put in it",
      len(registry.live()) == remote_read.MAX_LIVE_READS + 1)
fresh, _ = build_read(DOC[:2])
registry._reads[fresh.read_id] = fresh  # noqa: SLF001
fresh.start()
check("a read can be found by its id", registry.get(fresh.read_id) is fresh)
check("dropping it closes it", registry.drop(fresh.read_id) and fresh.closed)
check("dropping it twice says so", registry.drop(fresh.read_id) is False)

stale = created[0]
stale._touched -= remote_read.IDLE_EXPIRY_SECONDS + 1  # noqa: SLF001
registry.get("anything")
check("a read nobody touched is swept up",
      wait_until(lambda: stale.closed, 2.0) and registry.get(stale.read_id) is None)

registry.clear("shutdown")
check("clearing closes everything left",
      all(read.closed for read in created) and registry.live() == [])


# ---------------------------------------------------------------------------
print(f"\n{PASSED} passed, {FAILED} failed\n")
sys.exit(1 if FAILED else 0)
