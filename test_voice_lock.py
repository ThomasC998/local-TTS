"""Exercise stream_document's voice-lock logic without loading the model.

A stub stands in for the engine, so this runs in a second and needs no
checkpoint. It pins the contract the anchoring depends on: which chunks get a
reference, which template they switch to, how the anchor grows and where it
stops, and that the caller's own reference is never overwritten.

    python test_voice_lock.py
"""

import atexit
import shutil
import sys
import threading
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breeze_pipeline import BreezeEngine  # noqa: E402


class StubEngine:
    """Just enough of BreezeEngine for stream_document to run."""

    sample_rate = 24000
    default_seed = 42

    def __init__(self, seconds_per_chunk=3.0):
        self._gate = threading.Semaphore(1)
        self.pause_buffer = np.zeros(int(24000 * 0.22), dtype=np.float32)
        self.paragraph_pause_buffer = np.zeros(int(24000 * 0.7), dtype=np.float32)
        self.calls = []
        self.seconds = seconds_per_chunk
        self.swept = 0
        self.trimmed = 0
        self.trim_gate_free = None

    def release_stale_requests(self):
        self.swept += 1
        return 0

    def trim_if_needed(self):
        self.trimmed += 1
        # Releasing memory must happen with the gate already free, or it would
        # stall a request waiting to generate.
        self.trim_gate_free = self._gate.acquire(blocking=False)
        if self.trim_gate_free:
            self._gate.release()
        return None

    def stream_chunk(self, text, **kwargs):
        record = dict(kwargs)
        record["text"] = text
        if record.get("ref_audio"):
            # Prove the anchor file exists and is readable while it is in use.
            info = sf.info(record["ref_audio"])
            record["ref_seconds"] = round(info.frames / info.samplerate, 3)
        self.calls.append(record)
        yield np.full(int(self.sample_rate * self.seconds), 0.1, dtype=np.float32)

    _serialized = BreezeEngine._serialized
    stream_document = BreezeEngine.stream_document
    stream_live = BreezeEngine.stream_live
    _stream_items = BreezeEngine._stream_items


def run(chunks, *, voice_lock=True, seconds=3.0, **options):
    engine = StubEngine(seconds)
    events = list(engine.stream_document(chunks, voice_lock=voice_lock, **options))
    return engine, events


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


ok = True
chunks = ["One sentence here.", "Another sentence.", "A third one.", "And a fourth."]

print("voice lock ON, instruction given (guided -> edit):")
engine, events = run(chunks, voice_lock=True, instruction="A calm narrator.", seed=None)
ok &= check("chunk 0 has no reference", engine.calls[0].get("ref_audio") is None)
ok &= check("chunks 1+ are anchored", all(c.get("ref_audio") for c in engine.calls[1:]))
ok &= check("anchored chunks use edit mode",
            all(c["mode"] == "edit" for c in engine.calls[1:]))
ok &= check("anchor transcript starts with chunk 0",
            engine.calls[1]["ref_text"] == chunks[0])
ok &= check("seed defaults to 42 on every chunk",
            all(c["seed"] == 42 for c in engine.calls))
ok &= check("anchor grows to the 8s target (3s -> 6s -> 9s)",
            [c.get("ref_seconds") for c in engine.calls[1:]] == [3.0, 6.0, 9.0])
ok &= check("anchor stops growing once past target",
            engine.calls[3]["ref_text"] == " ".join(chunks[:3]))
ok &= check("one boundary per chunk",
            [e[2] for e in events if e[0] == "boundary"] == [0, 1, 2, 3])
ok &= check("pause inserted between chunks, not before the first",
            sum(1 for k, a, _ in events if k == "audio" and a is not None
                and a.size == engine.pause_buffer.size) == 3)

print("\nvoice lock ON, no instruction (plain -> clone):")
engine, _ = run(chunks, voice_lock=True, seed=7)
ok &= check("anchored chunks use clone mode",
            all(c["mode"] == "clone" for c in engine.calls[1:]))
ok &= check("explicit seed is preserved", all(c["seed"] == 7 for c in engine.calls))

print("\nvoice lock OFF:")
engine, _ = run(chunks, voice_lock=False, instruction="A calm narrator.")
ok &= check("no chunk is anchored", all(not c.get("ref_audio") for c in engine.calls))
ok &= check("seed still resolved for reproducibility",
            all(c["seed"] == 42 for c in engine.calls))

print("\nrequest already carries a reference:")
# Written here rather than shipped as a fixture. All this needs to be is a file
# soundfile can open and measure -- the stub engine reads its duration to prove
# the anchor is still readable at the moment it is used -- and a repository is
# not the place for a binary that three seconds of silence can stand in for.
_reference_dir = tempfile.mkdtemp(prefix="breeze-voice-lock-")
reference = Path(_reference_dir) / "reference.wav"
sf.write(reference, np.zeros(int(24000 * 3.0), dtype=np.float32), 24000)
atexit.register(shutil.rmtree, _reference_dir, True)
engine, _ = run(chunks, voice_lock=True, ref_audio=str(reference), ref_text="x",
                mode="clone")
ok &= check("every chunk uses the caller's reference, untouched",
            all(c["ref_audio"] == str(reference) for c in engine.calls))
ok &= check("caller's transcript is not overwritten",
            all(c["ref_text"] == "x" for c in engine.calls))

print("\nsingle chunk:")
engine, _ = run(["Only one."], voice_lock=True, instruction="A calm narrator.")
ok &= check("nothing to anchor to, so no reference", not engine.calls[0].get("ref_audio"))

print("\nanchor cap (long chunks):")
engine, _ = run(chunks, voice_lock=True, seconds=20.0)
ok &= check("anchor never exceeds the 15s cap",
            engine.calls[1]["ref_seconds"] <= 15.0)

print("\ntemp files cleaned up:")
# The anchor is written with tempfile.mkstemp, which puts it straight in TMPDIR
# -- so look there and nowhere else. Walking all of /var/folders instead used to
# work and then stopped: an AppTranslocation mount in it cannot be scanned, and
# the whole suite died on an OSError that had nothing to do with the anchor.
leftovers = list(Path(tempfile.gettempdir()).glob("breeze-anchor-*.wav"))
ok &= check("no breeze-anchor temp files left behind", not leftovers)

print("\nabandoned-stream cleanup:")
engine, _ = run(chunks, voice_lock=True, instruction="A calm narrator.")
ok &= check("stale codec requests swept before generating", engine.swept == 1)
ok &= check("memory trimmed once the document finished", engine.trimmed == 1)
ok &= check("trim runs with the gate released", engine.trim_gate_free is True)

# A client that aborts leaves the generator suspended mid-yield. Closing it must
# run the engine's cleanup rather than leaving a codec request open, which is
# what made the NEXT request fail with "already active request".
engine = StubEngine()
generator = engine.stream_document(chunks, voice_lock=True, instruction="A narrator.")
next(generator)                       # start it: chunk 0 is now in flight
anchor_files = list(Path(tempfile.gettempdir()).glob("breeze-anchor-*.wav"))
generator.close()                     # what an aborted request must trigger
ok &= check("closing an aborted generator releases the engine lock",
            engine._gate.acquire(blocking=False))
engine._gate.release()
ok &= check("closing an aborted generator removes its anchor file",
            not [path for path in anchor_files if path.exists()])


class LeakyCodec:
    """A codec runtime with a request an aborted generation left open."""

    def __init__(self):
        self.closed = []
        self.request_pool = type("Pool", (), {"active_req_ids": lambda self_: ["http-0"]})()

    def close_request(self, req_id):
        self.closed.append(req_id)


class LeakyEngine:
    class runtime:
        class audio_tokenizer:
            _stream_runtimes = {2: LeakyCodec()}

    _codec_runtimes = BreezeEngine._codec_runtimes
    release_stale_requests = BreezeEngine.release_stale_requests


print("\nstale codec request sweep:")
leaky = LeakyEngine()
released = leaky.release_stale_requests()
codec = LeakyEngine.runtime.audio_tokenizer._stream_runtimes[2]
ok &= check("the open request is closed", codec.closed == ["http-0"])
ok &= check("and counted", released == 1)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
