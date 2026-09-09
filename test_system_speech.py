"""Exercise multi-reference voices and the LLM-to-speech pipeline offline.

Nothing here loads the model or calls Vertex: a stub stands in for the engine
and a scripted iterator for the language model, so the whole file runs in about
a second. What it pins down is the behaviour that is hard to eyeball -- where a
tone change is allowed to land, what happens when the model stalls or fails
mid-sentence, and that a reference can never be stored without its transcript.

    python test_system_speech.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A scratch library, so a test run never touches the real voices directory.
_TEMP = tempfile.mkdtemp(prefix="breeze-tests-")
os.environ["BREEZE_VOICES_DIR"] = str(Path(_TEMP) / "voices")
os.environ["BREEZE_STATE_DIR"] = str(Path(_TEMP) / "state")

import voice_store  # noqa: E402
from breeze_pipeline import BreezeEngine, chunk_document  # noqa: E402
from llm_stream import SentinelFilter, sanitize  # noqa: E402
from speech_pipeline import ChunkPipe, TextAggregator, pump_llm  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}:")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
SPEECH = "This is the exact transcript of the reference recording, word for word."


def make_voice(name: str = "Test voice", *, seconds: float = 16.0) -> str:
    """A saved voice with one base reference, built without the model."""
    voice_store.ensure_dirs()
    audio = np.zeros(int(24000 * seconds), dtype=np.float32)
    preview = voice_store.save_preview(
        audio, 24000,
        text=SPEECH,
        generation={"instruction": "A calm narrator.", "cfg_scale": 4.0, "seed": 42},
        origin="designed",
    )
    voice = voice_store.create_voice_from_preview(
        preview["generated_voice_id"], name=name
    )
    return voice["voice_id"]


def add_variation(voice_id: str, label: str, tags: list[str],
                  *, seconds: float = 16.0) -> str:
    audio = np.full(int(24000 * seconds), 0.01, dtype=np.float32)
    preview = voice_store.save_preview(
        audio, 24000, text=SPEECH, generation={"instruction": label}, origin="cloned"
    )
    profile = voice_store.add_reference_from_preview(
        voice_id, preview["generated_voice_id"], label=label, tags=tags
    )
    return profile["references"][-1]["reference_id"]


class StubEngine:
    """Enough of BreezeEngine to run a document, with no model behind it."""

    sample_rate = 24000
    default_seed = 42

    def __init__(self) -> None:
        self._gate = threading.Semaphore(1)
        self.pause_buffer = np.zeros(240, dtype=np.float32)
        self.paragraph_pause_buffer = np.zeros(1680, dtype=np.float32)
        self.calls: list[dict] = []

    def release_stale_requests(self) -> int:
        return 0

    def trim_if_needed(self):
        return None

    def stream_chunk(self, text, *, request_id="x", _gate_held=False, **options):
        self.calls.append({"text": text, **options})
        yield np.full(2400, 0.1, dtype=np.float32)

    _serialized = BreezeEngine._serialized
    stream_document = BreezeEngine.stream_document
    stream_live = BreezeEngine.stream_live
    _stream_items = BreezeEngine._stream_items


# --------------------------------------------------------------------------
section("profile migration")
# --------------------------------------------------------------------------
voice_id = make_voice()
profile = voice_store.get_voice(voice_id)
check("a new voice starts with one reference", len(profile["references"]) == 1)
check("that reference is the default", profile["references"][0]["is_default"])
check("it carries the exact transcript", profile["references"][0]["text"] == SPEECH)

# Strip the v2 keys back off, the way a profile written by the old code looks.
raw_path = Path(os.environ["BREEZE_VOICES_DIR"]) / voice_id / "profile.json"
import json  # noqa: E402

legacy = json.loads(raw_path.read_text())
legacy.pop("references"), legacy.pop("rotation")
raw_path.write_text(json.dumps(legacy))
migrated = voice_store.get_voice(voice_id)
check("a v1 profile migrates on read", len(migrated["references"]) == 1)
check("migration keeps the transcript", migrated["references"][0]["text"] == SPEECH)
check("migration supplies rotation defaults",
      migrated["rotation"]["every_words"] == 1000)
check("reading a v1 profile does not rewrite it",
      "references" not in json.loads(raw_path.read_text()))

# --------------------------------------------------------------------------
section("references")
# --------------------------------------------------------------------------
bright = add_variation(voice_id, "bright", ["bright", "warm"])
profile = voice_store.get_voice(voice_id)
check("a variation is added", len(profile["references"]) == 2)
check("the base stays the default",
      profile["references"][0]["is_default"] and not profile["references"][1]["is_default"])
check("legacy 'reference' still points at the default",
      profile["reference"]["file"] == profile["references"][0]["file"])

voice_store.update_reference(voice_id, bright, {"is_default": True})
profile = voice_store.get_voice(voice_id)
check("promoting a reference moves the default",
      profile["references"][1]["is_default"])
check("...and the legacy key follows it",
      profile["reference"]["text"] == profile["references"][1]["text"])

try:
    voice_store.add_reference_from_audio(
        voice_id, voice_store.reference_audio_path(voice_id, bright), "  "
    )
    check("a reference without a transcript is refused", False)
except ValueError:
    check("a reference without a transcript is refused", True)

short = np.zeros(int(24000 * 3), dtype=np.float32)
short_preview = voice_store.save_preview(
    short, 24000, text=SPEECH, generation={}, origin="cloned"
)
try:
    voice_store.add_reference_from_preview(
        voice_id, short_preview["generated_voice_id"]
    )
    check("a too-short take is refused as a reference", False)
except ValueError:
    check("a too-short take is refused as a reference", True)

only = make_voice("Single")
try:
    voice_store.delete_reference(only, "ref_base")
    check("the last reference cannot be deleted", False)
except ValueError:
    check("the last reference cannot be deleted", True)
try:
    voice_store.update_reference(only, "ref_base", {"enabled": False})
    check("the last reference cannot be disabled", False)
except ValueError:
    check("the last reference cannot be disabled", True)

# --------------------------------------------------------------------------
section("saving the recording itself")
# --------------------------------------------------------------------------
# A cloned voice conditions on the recording every time it speaks, so the takes
# a clone produces are an audition rather than the voice. This is the path that
# skips them.
recording = Path(_TEMP) / "source.wav"
voice_store.write_wav(recording, np.zeros(int(24000 * 16), dtype=np.float32), 24000)

direct = voice_store.create_voice_from_recording(
    recording,
    name="Straight from the file",
    reference_text=SPEECH,
    generation={"instruction": "Warm and unhurried.", "cfg_scale": 4.0, "seed": 7},
    metadata={"description": "No take involved."},
)
check("a voice is created with no preview at all", bool(direct["voice_id"]))
check("the recording is the reference",
      direct["reference"]["source"] == "upload"
      and direct["reference"]["text"] == SPEECH)
check("...and is the audition sample too",
      direct["sample"]["file"] == direct["reference"]["file"])
check("the voice direction is kept", direct["generation"]["instruction"] == "Warm and unhurried.")
check("a direction makes it the edit branch", direct["generation"]["mode"] == "edit")
check("no direction makes it the clone branch",
      voice_store.create_voice_from_recording(
          recording, name="No direction", reference_text=SPEECH
      )["generation"]["mode"] == "clone")
check("it is marked as cloned", direct["origin"] == "cloned")

resolved = voice_store.voice_reference(direct["voice_id"])
check("it resolves to a reference the engine can use",
      resolved is not None and resolved[0].is_file() and resolved[1] == SPEECH)
check("variations can still be added to it",
      len(voice_store.add_reference_from_audio(
          direct["voice_id"], recording, SPEECH, label="second")["references"]) == 2)

try:
    voice_store.create_voice_from_recording(recording, name="x", reference_text="   ")
    check("a transcript is still required", False)
except ValueError:
    check("a transcript is still required", True)

tiny = Path(_TEMP) / "tiny.wav"
voice_store.write_wav(tiny, np.zeros(int(24000 * 4), dtype=np.float32), 24000)
try:
    voice_store.create_voice_from_recording(tiny, name="x", reference_text=SPEECH)
    check("a recording too short to clone from is refused", False)
except ValueError:
    check("a recording too short to clone from is refused", True)

# --------------------------------------------------------------------------
section("rotation")
# --------------------------------------------------------------------------
words = [10] * 30
paragraphs = [index // 5 for index in range(30)]  # a paragraph every 5 chunks

plan, segments = voice_store.build_reference_plan(only, words, paragraphs)
check("one reference means nothing to rotate", plan is None and segments == [])

plan, segments = voice_store.build_reference_plan(
    voice_id, words, paragraphs, rotation={"every_words": 40}, seed=1
)
check("two references rotate", plan is not None and len(segments) > 1)
check("a reference is assigned to every chunk", len(plan) == len(words))

starts = {segment["from_chunk"] for segment in segments}
check("every switch lands on a paragraph start",
      all(index % 5 == 0 for index in starts))
check("no switch repeats the voice already speaking",
      all(segments[i]["reference_id"] != segments[i + 1]["reference_id"]
          for i in range(len(segments) - 1)))
check("no segment is shorter than the threshold",
      all(segment["words"] >= 40 for segment in segments[:-1]))

plan_off, _ = voice_store.build_reference_plan(
    voice_id, words, paragraphs, rotation={"enabled": False}
)
check("rotation can be switched off", plan_off is None)

# Text with no blank lines has one paragraph and so no switch point at all;
# sentence starts have to stand in, or rotation would silently never happen.
flat, flat_segments = voice_store.build_reference_plan(
    voice_id, words, [0] * 30, rotation={"every_words": 40}, seed=1
)
check("a single-paragraph document still rotates", len(flat_segments) > 1)

tagged, tagged_segments = voice_store.build_reference_plan(
    voice_id, words, paragraphs, rotation={"every_words": 40, "tags": ["bright"]}
)
check("a tag filter narrowing to one reference stops rotation", tagged is None)
check("a tag filter matching nothing falls back to the whole pool",
      voice_store.build_reference_plan(
          voice_id, words, paragraphs,
          rotation={"every_words": 40, "tags": ["nonexistent"]})[0] is not None)

# --------------------------------------------------------------------------
section("chunking and paragraph gaps")
# --------------------------------------------------------------------------
chunks, groups = chunk_document("One. Two.\n\nThree. Four.")
check("chunks split by sentence", chunks == ["One.", "Two.", "Three.", "Four."])
check("paragraphs are tracked", groups == [0, 0, 1, 1])

engine = StubEngine()
events = list(engine.stream_document(chunks, paragraph_ids=groups, voice_lock=False))
gaps = [audio.size for kind, audio, _ in events
        if kind == "audio" and audio is not None and audio.size in (240, 1680)]
check("a blank line gets a longer pause than a sentence break",
      gaps == [240, 1680, 240])

engine = StubEngine()
refs = [(str(Path(_TEMP) / "a.wav"), "text a")] * 2 + [(str(Path(_TEMP) / "b.wav"), "text b")] * 2
list(engine.stream_document(chunks, chunk_refs=refs, paragraph_ids=groups))
check("per-chunk references reach the engine",
      [call["ref_audio"] for call in engine.calls] == [ref[0] for ref in refs])
check("per-chunk references switch template to clone",
      all(call["mode"] == "clone" for call in engine.calls))

# --------------------------------------------------------------------------
section("sentinel and sanitizer")
# --------------------------------------------------------------------------
sentinel = SentinelFilter()
check("a preamble before the marker is dropped",
      sentinel.feed("Sure, here you go:\n<<<SPEAK>>>\nHello.") == "Hello.")
check("text after the marker flows through", sentinel.feed(" More.") == " More.")

split = SentinelFilter()
split.feed("Of course! <<<SP")
check("a marker split across deltas still works",
      split.feed("EAK>>>Hi.") == "Hi.")

missing = SentinelFilter()
check("nothing is emitted while the marker may still arrive",
      missing.feed("Hello") == "")
check("...but the text is not lost when the stream ends",
      missing.flush() == "Hello")

check("markdown is stripped",
      sanitize("## Head\n- **bold** [link](http://x)") == "Head\nbold link")
check("invented tags are removed", sanitize("(pause) hi") == " hi")
check("supported vocal events survive", "(sigh)" in sanitize("(sigh) hello"))

# Brackets, dashes and numbers are all spoken correctly, so stripping them --
# which an earlier sanitizer did to every parenthetical -- lost the author's own
# asides to guard against a tag the model rarely invents.
check("a parenthetical the author wrote is kept",
      sanitize("The result (see below) held.") == "The result (see below) held.")
check("dashes and numbers are untouched",
      sanitize("A 3-4 Hz drift -- measurable at 44.1 kHz.")
      == "A 3-4 Hz drift -- measurable at 44.1 kHz.")
check("footnote markers are not",
      sanitize("As shown [12] earlier.") == "As shown earlier.")
check("an or-slash becomes a word",
      sanitize("read/write access") == "read or write access")
check("...but a path is left alone",
      sanitize("under /Users/thomas/Documents") == "under /Users/thomas/Documents")
check("...and a unit is left alone", sanitize("60 km/h") == "60 km/h")

# --------------------------------------------------------------------------
section("prompt migration")
# --------------------------------------------------------------------------
import speech_config  # noqa: E402

_OLD_DEFAULT = """You prepare text to be read aloud by a text-to-speech engine.

Your job is pacing, not rewriting."""

speech_config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
speech_config.CONFIG_PATH.write_text(
    json.dumps({"voice_id": "keep-me", "llm": {"prompt": _OLD_DEFAULT,
                                               "temperature": 0.7}}),
    encoding="utf-8",
)
check("an edited prompt is left alone",
      speech_config.load()["llm"]["prompt"] == _OLD_DEFAULT)

import hashlib  # noqa: E402

speech_config.SUPERSEDED_PROMPT_HASHES = frozenset(
    {hashlib.sha256(_OLD_DEFAULT.encode()).hexdigest()}
)
migrated = speech_config.load()
check("a stored copy of a superseded default is replaced",
      migrated["llm"]["prompt"] == speech_config.DEFAULT_LLM_PROMPT)
check("...without disturbing anything else the user set",
      migrated["voice_id"] == "keep-me" and migrated["llm"]["temperature"] == 0.7)
check("the new default asks for paragraphs of one idea",
      "one main concept" in speech_config.DEFAULT_LLM_PROMPT)
check("...and for abbreviations to be spoken",
      "by the way" in speech_config.DEFAULT_LLM_PROMPT
      and "hertz" in speech_config.DEFAULT_LLM_PROMPT)
check("...and says slashes are unsupported",
      "Slashes are NOT supported" in speech_config.DEFAULT_LLM_PROMPT)
speech_config.CONFIG_PATH.unlink(missing_ok=True)

# --------------------------------------------------------------------------
section("streaming aggregation")
# --------------------------------------------------------------------------
aggregator = TextAggregator()
emitted = []
for delta in ["The first ", "point. And a ", "second one. ", "\n\nNew idea. ", "Trailing"]:
    emitted += aggregator.feed(delta)
check("a sentence is released as soon as it is complete",
      emitted[0] == ("The first point.", 0))
check("nothing incomplete is released", ("Trailing", 1) not in emitted)
emitted += aggregator.flush()
check("the tail is flushed at the end", emitted[-1] == ("Trailing", 1))
check("paragraphs are numbered", [paragraph for _text, paragraph in emitted] == [0, 0, 1, 1])

decimal = TextAggregator()
check("a decimal point is not a sentence end",
      decimal.feed("It costs 3.") == [] and decimal.feed("5 dollars.") == [])

runon = TextAggregator(max_words=5)
produced = runon.feed("one two three, four five six, seven eight nine, ten eleven twelve, ")
check("a run-on with no terminal is released at clause boundaries", len(produced) > 0)

# --------------------------------------------------------------------------
section("the pipe under stress")
# --------------------------------------------------------------------------
def drive(deltas, *, delay=0.0):
    pipe = ChunkPipe(idle_timeout=2.0)
    aggregator = TextAggregator()

    def source():
        for delta in deltas:
            if delay:
                time.sleep(delay)
            yield delta

    thread = threading.Thread(target=pump_llm, args=(pipe, source()), kwargs={
        "sentinel_filter": SentinelFilter(),
        "sanitize": sanitize,
        "aggregator": aggregator,
    })
    thread.start()
    items = list(pipe.items())
    thread.join(5)
    return items

items = drive(["<<<SPEAK>>>", "One. ", "Two.\n\nThree."])
check("chunks arrive in order",
      [text for text, _ref, _new in items] == ["One.", "Two.", "Three."])
check("a paragraph change is flagged",
      [new for _text, _ref, new in items] == [False, False, True])

slow = drive(["<<<SPEAK>>>One. ", "Two."], delay=0.05)
check("a slow model still delivers every chunk", len(slow) == 2)


def failing():
    yield "<<<SPEAK>>>First sentence. "
    raise RuntimeError("vertex went away")


pipe = ChunkPipe(idle_timeout=2.0)
thread = threading.Thread(target=pump_llm, args=(pipe, failing()), kwargs={
    "sentinel_filter": SentinelFilter(), "sanitize": sanitize,
    "aggregator": TextAggregator(),
})
thread.start()
collected, raised = [], None
try:
    for item in pipe.items():
        collected.append(item)
except RuntimeError as exc:
    raised = exc
thread.join(5)
check("a mid-stream failure is raised at the consumer", raised is not None)
check("...after everything already produced was spoken", len(collected) == 1)

cancelled = ChunkPipe(idle_timeout=2.0)
cancelled.put("One.", 0)
cancelled.cancel()
check("cancelling stops the consumer immediately", list(cancelled.items()) == [])

idle = ChunkPipe(idle_timeout=0.3)
started = time.monotonic()
check("a producer that never finishes ends the utterance rather than hanging",
      list(idle.items()) == [] and time.monotonic() - started < 2.0)

# A stalled sentence is spoken rather than left buffered forever.
# The idle timeout is set well above STALL_FLUSH_SECONDS so this asserts the
# watchdog fired, not merely that one of the two timers went off first.
stall = ChunkPipe(idle_timeout=8.0)
aggregator = TextAggregator()


def stalling():
    yield "<<<SPEAK>>>A sentence that never gets its final full stop but is long enough"
    time.sleep(6.0)


thread = threading.Thread(target=pump_llm, args=(stall, stalling()), kwargs={
    "sentinel_filter": SentinelFilter(), "sanitize": sanitize, "aggregator": aggregator,
}, daemon=True)
thread.start()
first = next(iter(stall.items()), None)
stall.cancel()
check("a stalled sentence is eventually spoken", first is not None)

# --------------------------------------------------------------------------
section("live synthesis")
# --------------------------------------------------------------------------
engine = StubEngine()
pipe = ChunkPipe(idle_timeout=2.0)
rotator = voice_store.ReferenceRotator(
    voice_id, settings=voice_store.rotation_settings(voice_id, {"every_words": 20}),
    seed=3,
)
for index, text in enumerate([
    "First chunk of this document holds roughly a dozen plain words in it.",
    "Second chunk of this document holds roughly a dozen plain words in it.",
    "Third chunk of this document holds roughly a dozen plain words in it.",
]):
    pipe.put(text, index)
pipe.close()

events = list(engine.stream_live(
    pipe.items(rotator.take), ref_audio=None, ref_text=None, voice_lock=False,
    instruction=None, mode="clone",
))
check("live synthesis speaks every chunk", len(engine.calls) == 3)
check("live rotation switches between references",
      len({call["ref_audio"] for call in engine.calls}) > 1)
check("live synthesis emits boundaries",
      sum(1 for kind, _a, _i in events if kind == "boundary") == 3)


# --------------------------------------------------------------------------
section("playback continuity")
#
# The bug this guards against: 24 kHz audio handed to a 44.1 kHz output gets
# resampled per callback buffer, and a resampler that cannot see across a
# buffer edge leaves a discontinuity at every one of them -- heard as constant
# crackling, while the WAV written from the same samples is clean. So the
# streamed audio must match a one-shot conversion of the same signal.
# --------------------------------------------------------------------------
import audio_out  # noqa: E402

DEVICE_RATE = 44100
rng = np.random.default_rng(7)
seconds = 2.0
grid = np.arange(int(24000 * seconds)) / 24000
signal = (0.3 * np.sin(2 * np.pi * 220 * grid)
          + 0.05 * rng.standard_normal(grid.size)).astype(np.float32)

player = audio_out.SpeechPlayer()
player._ensure_stream = lambda utterance_id: None  # never open a real device
player.start("test", 24000, prebuffer_ms=0, device=None)
player._device_rate = DEVICE_RATE
player._fade_frames = max(1, int(DEVICE_RATE * (audio_out.FADE_MS / 1000.0)))
import soxr  # noqa: E402

player._resampler = soxr.ResampleStream(24000, DEVICE_RATE, 1,
                                        dtype="float32", quality="VHQ")

# Feed it the way the engine does: uneven blocks, including a tiny pause buffer.
position = 0
for size in [1920, 3840, 240, 7680, 1200, 4800] * 40:
    if position >= signal.size:
        break
    player.write("test", signal[position:position + size])
    position += size
player.finish("test")

# Drain through the callback exactly as PortAudio would, in fixed frames.
played = []
frame_count = 512
for _ in range(2000):
    out = np.zeros((frame_count, 1), dtype=np.float32)
    try:
        player._callback(out, frame_count, None, None)
    except Exception:  # CallbackStop, at the end of the utterance
        played.append(out[:, 0].copy())
        break
    played.append(out[:, 0].copy())
streamed = np.concatenate(played)

expected = soxr.resample(signal, 24000, DEVICE_RATE, quality="VHQ")
check("every sample the engine produced is played",
      abs(int(np.count_nonzero(streamed)) - expected.size) < DEVICE_RATE * 0.05)

# The opening and closing ramps are deliberate, so compare the steady middle:
# everything but the fade in at the start and the final partial callback, whose
# fade to silence is the whole point of the ramp.
head = player._fade_frames * 2
tail = frame_count + player._fade_frames * 2
usable = min(streamed.size, expected.size) - head - tail
error = np.abs(streamed[head:head + usable] - expected[head:head + usable]).max()
check(f"streamed audio matches a one-shot conversion (max error {error:.2e})",
      error < 1e-6)

# What the naive implementation would have produced, for contrast: this is the
# number that was audible as crackling.
naive, position = [], 0
for size in [1920, 3840, 240, 7680, 1200, 4800] * 40:
    if position >= signal.size:
        break
    naive.append(soxr.resample(signal[position:position + size], 24000, DEVICE_RATE))
    position += size
naive_error = np.abs(np.concatenate(naive)[:usable] - expected[:usable]).max()
check(f"...and per-block conversion would not have (max error {naive_error:.2e})",
      naive_error > 1e-3)

# No seam may exceed what the source itself contains: a click is a jump much
# larger than any real sample-to-sample step in speech.
biggest_source_step = float(np.abs(np.diff(expected)).max())
biggest_played_step = float(np.abs(np.diff(streamed[head:head + usable])).max())
check("no sample-to-sample jump larger than the source's own",
      biggest_played_step <= biggest_source_step * 1.05)

ending = streamed[streamed.size - frame_count:]
check("the utterance ends on a ramp to silence, not a cut",
      abs(float(ending[-1])) < 1e-6
      and float(np.abs(ending).max()) >= abs(float(ending[-1])))

player.stop(fade=False)

# --------------------------------------------------------------------------
import shutil  # noqa: E402

shutil.rmtree(_TEMP, ignore_errors=True)
print("\n" + ("ALL PASS" if not FAILURES else
              f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES)))
raise SystemExit(1 if FAILURES else 0)
