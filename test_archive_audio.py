#!/usr/bin/env python3
"""Sizing the archive, and handing its audio over as one zip.

The interesting part is not the arithmetic, it is the order: nothing may be
deleted until the zip has been written, reopened and checked to contain every
file it was given. This is the only bulk delete in the archive and it cannot be
undone, so the test asserts on what survives as much as on what goes.

    python test_archive_audio.py

No audio is played on any device.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_TEMP = tempfile.mkdtemp(prefix="breeze-archive-test-")
os.environ["BREEZE_STATE_DIR"] = str(Path(_TEMP) / "state")
os.environ["BREEZE_VOICES_DIR"] = str(Path(_TEMP) / "voices")

import archive  # noqa: E402

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


def record(index: int, seconds: float = 1.0, *, audio: bool = True) -> dict:
    """One archived utterance, with or without its recording."""
    recording = archive.Recording(f"utt_test{index:03d}", keep_audio=audio)
    recording.note_chunk(f"Chunk {index} of the spoken text.")
    recording.note_llm(f"<<<SPEAK>>> paragraph {index}")
    if audio:
        recording.note_audio(np.zeros(int(24000 * seconds), dtype=np.float32))
    recording.meta["voice_name"] = "Test voice"
    return recording.write(
        input_text=f"Input text for utterance {index}.", sample_rate=24000
    )


# ---------------------------------------------------------------------------
section("Measuring")
# ---------------------------------------------------------------------------
usage = archive.audio_usage()
check("an empty archive is zero", usage["bytes"] == 0 and usage["files"] == 0)
check("the threshold is reported so the UI need not hardcode it",
      usage["warn_bytes"] == archive.AUDIO_WARN_BYTES)
check("and it is 500 MB by default",
      archive.AUDIO_WARN_BYTES == 500 * 1024**2, str(archive.AUDIO_WARN_BYTES))

entries = [record(index, seconds=1.0) for index in range(1, 6)]
check("every utterance was archived", all(entries))

# 5 x 1s of 16-bit mono at 24 kHz, plus WAV headers.
expected = 5 * 24000 * 2
usage = archive.audio_usage()
check("each write is added to the running total, with no walk",
      expected <= usage["bytes"] <= expected + 5 * 200,
      f"{usage['bytes']} vs ~{expected}")
check("...and the file count with it", usage["files"] == 5, str(usage["files"]))

measured = archive.measure_audio()
check("a full sweep agrees with the running total",
      measured["bytes"] == usage["bytes"], f"{measured['bytes']} vs {usage['bytes']}")

record(6, audio=False)
check("an utterance archived without audio adds nothing",
      archive.audio_usage()["files"] == 5)

check("under the threshold, nothing is flagged", archive.audio_usage()["over"] is False)
archive.AUDIO_WARN_BYTES = 1000
check("over the threshold, it is", archive.audio_usage()["over"] is True)
archive.AUDIO_WARN_BYTES = 500 * 1024**2


# ---------------------------------------------------------------------------
section("Exporting")
# ---------------------------------------------------------------------------
before = archive.recent(50)
result = archive.export_audio()
check("every recording went into the zip", result["entries"] == 5, str(result))
check("the freed figure matches what was held",
      result["freed_bytes"] == usage["bytes"],
      f"{result['freed_bytes']} vs {usage['bytes']}")

bundle_path = Path(result["path"])
check("the zip is where it says it is", bundle_path.is_file())
with zipfile.ZipFile(bundle_path) as bundle:
    names = bundle.namelist()
    manifest = json.loads(bundle.read("manifest.json"))
check("it holds one entry per recording, plus a manifest",
      len(names) == 6 and "manifest.json" in names, str(names))
check("entries keep their archive path, so they can be put back",
      all(name.count("/") == 2 and name.endswith("/audio.wav")
          for name in names if name != "manifest.json"))
check("the manifest describes what is inside",
      manifest["count"] == 5
      and all(item.get("utterance_id") for item in manifest["entries"]),
      str(manifest["entries"][:1]))

check("the audio is gone from the archive",
      not list(archive.ARCHIVE_DIR.glob("*/*/audio.wav")))
check("...and the total reflects that", archive.audio_usage()["bytes"] == 0)


# ---------------------------------------------------------------------------
section("What must survive")
# ---------------------------------------------------------------------------
after = archive.recent(50)
check("no utterance was lost", len(after) == len(before), f"{len(after)} vs {len(before)}")
check("the input text is still there",
      all((archive.ARCHIVE_DIR / entry["directory"] / "input.txt").is_file()
          for entry in after))
check("so is what was actually spoken",
      all((archive.ARCHIVE_DIR / entry["directory"] / "spoken.txt").is_file()
          for entry in after))
check("so is the model's raw output",
      all((archive.ARCHIVE_DIR / entry["directory"] / "llm_output.txt").is_file()
          for entry in after))
check("the previews still read the same",
      [entry["input_preview"] for entry in after]
      == [entry["input_preview"] for entry in before])

check("the index no longer claims the audio is playable",
      not any(entry.get("has_audio") for entry in after))
check("...and says where it went",
      all(entry.get("audio_exported_to") == result["file"]
          for entry in after if entry.get("audio_seconds")))

# The newest entry is the one archived without audio, so pick one that had some.
spoke = next(entry for entry in after if entry.get("audio_seconds"))
meta = json.loads(
    (archive.ARCHIVE_DIR / spoke["directory"] / "meta.json").read_text()
)
check("each utterance's own metadata was updated too",
      meta["has_audio"] is False and meta["audio_exported_to"] == result["file"])

silent = next(entry for entry in after if not entry.get("audio_seconds"))
silent_meta = json.loads(
    (archive.ARCHIVE_DIR / silent["directory"] / "meta.json").read_text()
)
check("an utterance that never had audio is not marked as exported",
      "audio_exported_to" not in silent_meta)


# ---------------------------------------------------------------------------
section("Edges")
# ---------------------------------------------------------------------------
try:
    archive.export_audio()
    check("exporting nothing is refused rather than writing an empty zip", False)
except ValueError:
    check("exporting nothing is refused rather than writing an empty zip", True)

listed = archive.exports()
check("the export is listed for collection",
      len(listed) == 1 and listed[0]["file"] == result["file"], str(listed))
check("its path resolves", archive.export_path(result["file"]).is_file())
try:
    archive.export_path("../../escape.zip")
    check("a path outside the exports directory is refused", False)
except ValueError:
    check("a path outside the exports directory is refused", True)

record(7, seconds=1.0)
check("a later utterance archives its audio normally",
      archive.audio_usage()["files"] == 1)
second = archive.export_audio()
check("a second export gets its own file", second["file"] != result["file"])
check("both are listed", len(archive.exports()) == 2)

archive.clear()
check("clearing the history zeroes the total",
      archive.audio_usage()["bytes"] == 0 and archive.audio_usage()["files"] == 0)


# ---------------------------------------------------------------------------
print(f"\n{PASSED} passed, {FAILED} failed\n")
sys.exit(1 if FAILED else 0)
