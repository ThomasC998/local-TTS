"""Keep what was said, what it was made from, and how it was made.

One directory per utterance, grouped by day::

    state/archive/
        index.jsonl                     -- one line per utterance, append-only
        2026-09-01/
            utt_<id>/
                meta.json               -- everything about this generation
                input.txt               -- the text as it arrived (clipboard)
                llm_output.txt          -- the model's raw reply, sentinel included
                spoken.txt              -- what the engine was actually given
                audio.wav               -- the rendered speech

Per-utterance metadata plus an append-only index, rather than one large JSON
that every generation rewrites: appending cannot corrupt what is already there,
and a half-written entry costs one utterance instead of the whole history.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger("breeze.archive")

STATE_DIR = Path(
    os.getenv("BREEZE_STATE_DIR", Path(__file__).resolve().parent / "state")
)
ARCHIVE_DIR = STATE_DIR / "archive"
INDEX_PATH = ARCHIVE_DIR / "index.jsonl"
EXPORTS_DIR = STATE_DIR / "exports"

# Speech is the only thing here with any real weight -- the text of a thousand
# utterances is a few megabytes, the audio of a hundred is a gigabyte. Past this
# the UI says so and offers to hand the audio over as one zip.
AUDIO_WARN_BYTES = int(float(os.getenv("BREEZE_ARCHIVE_AUDIO_WARN_MB", "500")) * 1024**2)

# How often the size is recomputed while the server runs. Walking the archive is
# cheap, but it is not free, and nothing here changes fast enough to need more.
USAGE_INTERVAL_SECONDS = 3600.0

_LOCK = threading.RLock()

# Recomputed on startup and every hour; nudged by each utterance in between, so
# the figure the UI shows stays live without walking the tree on every write.
_USAGE: dict[str, Any] = {
    "bytes": 0,
    "files": 0,
    "checked_at": None,
    "measured": False,
}


class Recording:
    """Collects one utterance's text and audio, then writes it out.

    Nothing here is allowed to break synthesis: every failure is logged and
    swallowed, because losing the archive copy is not worth losing the speech.
    """

    def __init__(self, utterance_id: str, *, enabled: bool = True,
                 keep_audio: bool = True) -> None:
        self.utterance_id = utterance_id
        self.enabled = enabled
        self.keep_audio = keep_audio
        self.started_at = time.time()
        self.directory = (
            ARCHIVE_DIR
            / datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d")
            / utterance_id
        )
        self._llm_raw: list[str] = []
        self._spoken: list[str] = []
        self._audio: list[np.ndarray] = []
        self.meta: dict[str, Any] = {}

    def note_llm(self, delta: str) -> None:
        if self.enabled:
            self._llm_raw.append(delta)

    def note_chunk(self, text: str) -> None:
        if self.enabled:
            self._spoken.append(text)

    def note_audio(self, audio: np.ndarray) -> None:
        if self.enabled and self.keep_audio and audio is not None and audio.size:
            self._audio.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    @property
    def llm_output(self) -> str:
        return "".join(self._llm_raw)

    @property
    def spoken_text(self) -> str:
        return "\n".join(self._spoken)

    def write(self, *, input_text: str, sample_rate: int,
              meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        entry: dict[str, Any] = {}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "input.txt").write_text(input_text, encoding="utf-8")
            if self._spoken:
                (self.directory / "spoken.txt").write_text(
                    self.spoken_text, encoding="utf-8"
                )
            if self._llm_raw:
                (self.directory / "llm_output.txt").write_text(
                    self.llm_output, encoding="utf-8"
                )

            duration = 0.0
            if self._audio:
                audio = np.concatenate(self._audio)
                duration = float(audio.size / sample_rate)
                path = self.directory / "audio.wav"
                sf.write(path, audio, sample_rate, subtype="PCM_16")
                _note_audio_written(path)

            entry = {
                "utterance_id": self.utterance_id,
                "created_at_unix": int(self.started_at),
                "created_at": datetime.fromtimestamp(self.started_at).isoformat(
                    timespec="seconds"
                ),
                "directory": str(self.directory.relative_to(ARCHIVE_DIR)),
                "input_chars": len(input_text),
                "input_preview": input_text.strip()[:300],
                "spoken_chars": len(self.spoken_text),
                "audio_seconds": round(duration, 3),
                "has_audio": bool(self._audio),
                "sample_rate": sample_rate,
                "elapsed_seconds": round(time.time() - self.started_at, 3),
                **(meta or {}),
                **self.meta,
            }
            (self.directory / "meta.json").write_text(
                json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _append_index(entry)
        except Exception:  # noqa: BLE001 - the archive is never load-bearing
            logger.exception("Could not archive utterance %s", self.utterance_id)
            return None
        return entry


def _append_index(entry: dict[str, Any]) -> None:
    with _LOCK:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        with INDEX_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def recent(limit: int = 50) -> list[dict[str, Any]]:
    """The newest entries first. A malformed line is skipped, not fatal."""
    if not INDEX_PATH.is_file():
        return []
    try:
        lines = INDEX_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(entries) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def entry_path(relative: str) -> Path:
    """Resolve an archived file, refusing anything outside the archive."""
    target = (ARCHIVE_DIR / relative).resolve()
    if not str(target).startswith(str(ARCHIVE_DIR.resolve())):
        raise ValueError("path escapes the archive directory")
    return target


# --------------------------------------------------------------------------
# How much disk the audio is using
#
# The text of an utterance is kilobytes and worth keeping forever. The audio is
# megabytes each and worth keeping only until you have listened to it -- so the
# two are counted, warned about and disposed of separately.
# --------------------------------------------------------------------------
def _audio_files() -> list[Path]:
    """Every archived recording, in no particular order."""
    if not ARCHIVE_DIR.is_dir():
        return []
    return [path for path in ARCHIVE_DIR.glob("*/*/audio.wav") if path.is_file()]


def measure_audio() -> dict[str, Any]:
    """Walk the archive and total the audio. Refreshes the cached figure."""
    total = 0
    count = 0
    for path in _audio_files():
        try:
            total += path.stat().st_size
        except OSError:  # noqa: PERF203 - a file removed mid-walk is not an error
            continue
        count += 1
    with _LOCK:
        _USAGE.update(bytes=total, files=count, checked_at=time.time(), measured=True)
        return dict(_USAGE)


def _note_audio_written(path: Path) -> None:
    """Add one new recording to the running total, without a full walk."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    with _LOCK:
        _USAGE["bytes"] += size
        _USAGE["files"] += 1


def audio_usage() -> dict[str, Any]:
    """What the UI shows: the cached total, the threshold, and whether it is over.

    Deliberately does not measure. This is polled by every open tab, and the
    whole point of the hourly sweep is that a poll costs nothing.
    """
    with _LOCK:
        snapshot = dict(_USAGE)
    if not snapshot["measured"]:
        snapshot = measure_audio()
    return {
        "bytes": snapshot["bytes"],
        "files": snapshot["files"],
        "checked_at": snapshot["checked_at"],
        "warn_bytes": AUDIO_WARN_BYTES,
        "over": snapshot["bytes"] > AUDIO_WARN_BYTES,
    }


def export_audio() -> dict[str, Any]:
    """Zip every archived recording, then delete the recordings.

    The text stays. What is removed is only ``audio.wav``; ``input.txt``,
    ``spoken.txt``, ``llm_output.txt`` and ``meta.json`` are what make the
    history worth having, and they weigh nothing.

    Nothing is deleted until the zip has been written, closed, reopened and
    checked to contain every file it was given. Deleting is irreversible and
    this is the one place that does it in bulk, so the order matters more than
    the few seconds it costs.
    """
    files = _audio_files()
    if not files:
        raise ValueError("There is no archived audio to export")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    destination = EXPORTS_DIR / f"breeze-audio-{stamp}.zip"
    # Two exports inside the same second would otherwise land on the same name,
    # and the second would overwrite a zip whose audio has already been deleted
    # from the archive -- the one way this feature could lose a recording.
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = EXPORTS_DIR / f"breeze-audio-{stamp}-{suffix}.zip"

    with _LOCK:
        entries = {entry.get("directory"): entry for entry in recent(100_000)}
        manifest: list[dict[str, Any]] = []
        written: list[tuple[Path, str]] = []

        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for path in files:
                name = f"{path.parent.parent.name}/{path.parent.name}/audio.wav"
                bundle.write(path, arcname=name)
                written.append((path, name))
                entry = entries.get(f"{path.parent.parent.name}/{path.parent.name}")
                manifest.append(
                    {
                        "file": name,
                        "bytes": path.stat().st_size,
                        "utterance_id": (entry or {}).get("utterance_id"),
                        "created_at": (entry or {}).get("created_at"),
                        "audio_seconds": (entry or {}).get("audio_seconds"),
                        "voice_name": (entry or {}).get("voice_name"),
                        "input_preview": (entry or {}).get("input_preview"),
                    }
                )
            # A zip that cannot say what is in it is a zip you will not open.
            bundle.writestr(
                "manifest.json",
                json.dumps(
                    {"exported_at": datetime.now().isoformat(timespec="seconds"),
                     "count": len(manifest), "entries": manifest},
                    indent=2, ensure_ascii=False,
                ),
            )

        # Reopen and verify before anything is removed.
        with zipfile.ZipFile(destination) as bundle:
            if bundle.testzip() is not None:
                raise OSError(f"{destination.name} failed its own integrity check")
            stored = set(bundle.namelist())
        missing = [name for _path, name in written if name not in stored]
        if missing:
            raise OSError(
                f"{len(missing)} recording(s) did not reach {destination.name}; "
                "nothing was deleted"
            )

        freed = 0
        removed = 0
        for path, _name in written:
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:
                logger.exception("Could not remove %s after exporting it", path)
                continue
            removed += 1
            _mark_audio_exported(path.parent, destination.name)

        _rewrite_index(destination.name)
        _USAGE.update(bytes=0, files=0, checked_at=time.time(), measured=True)

    logger.info("Exported %d recording(s) to %s and freed %.1f MiB",
                removed, destination, freed / 1024**2)
    return {
        "file": destination.name,
        "path": str(destination),
        "entries": removed,
        "freed_bytes": freed,
        "zip_bytes": destination.stat().st_size,
    }


def _mark_audio_exported(directory: Path, archive_name: str) -> None:
    """Record on the utterance that its audio moved into a zip."""
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["has_audio"] = False
        meta["audio_exported_to"] = archive_name
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not update %s", meta_path)


def _rewrite_index(archive_name: str) -> None:
    """Clear ``has_audio`` across the index, atomically.

    The index is append-only in normal use, which is what makes a half-written
    utterance cost one line instead of the whole history. This is the one
    operation that has to go back and change what is already there, so it writes
    a new file and renames it over the old one.
    """
    if not INDEX_PATH.is_file():
        return
    try:
        lines = INDEX_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.exception("Could not read the archive index")
        return

    updated: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            updated.append(line)  # keep what we cannot parse rather than drop it
            continue
        if entry.get("has_audio"):
            entry["has_audio"] = False
            entry["audio_exported_to"] = archive_name
        updated.append(json.dumps(entry, ensure_ascii=False))

    try:
        tmp = INDEX_PATH.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(updated) + "\n", encoding="utf-8")
        os.replace(tmp, INDEX_PATH)
    except OSError:
        logger.exception("Could not rewrite the archive index")


def export_path(name: str) -> Path:
    """Resolve an exported zip, refusing anything outside the exports directory."""
    target = (EXPORTS_DIR / name).resolve()
    if not str(target).startswith(str(EXPORTS_DIR.resolve())):
        raise ValueError("path escapes the exports directory")
    return target


def exports() -> list[dict[str, Any]]:
    """Zips waiting to be moved somewhere else, newest first."""
    if not EXPORTS_DIR.is_dir():
        return []
    found = []
    for path in EXPORTS_DIR.glob("breeze-audio-*.zip"):
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append({"file": path.name, "path": str(path),
                      "bytes": stat.st_size, "created_at_unix": int(stat.st_mtime)})
    return sorted(found, key=lambda item: item["created_at_unix"], reverse=True)


def start_usage_monitor(interval: float = USAGE_INTERVAL_SECONDS) -> threading.Thread:
    """Measure now, then once an hour, logging whenever the archive is over."""

    def run() -> None:
        while True:
            usage = measure_audio()
            if usage["bytes"] > AUDIO_WARN_BYTES:
                logger.warning(
                    "Archived audio is %.2f GiB across %d utterance(s), over the "
                    "%.0f MiB mark. Export it from the web UI to free the space.",
                    usage["bytes"] / 1024**3, usage["files"],
                    AUDIO_WARN_BYTES / 1024**2,
                )
            else:
                logger.info("Archived audio: %.1f MiB across %d utterance(s)",
                            usage["bytes"] / 1024**2, usage["files"])
            time.sleep(interval)

    thread = threading.Thread(target=run, name="archive-usage", daemon=True)
    thread.start()
    return thread


def prune(max_entries: int = 500) -> int:
    """Drop the oldest utterance directories once the archive grows past a cap."""
    if not ARCHIVE_DIR.is_dir():
        return 0
    with _LOCK:
        directories: list[tuple[float, Path]] = []
        for day in ARCHIVE_DIR.iterdir():
            if not day.is_dir():
                continue
            for utterance in day.iterdir():
                if utterance.is_dir():
                    directories.append((utterance.stat().st_mtime, utterance))

        directories.sort(reverse=True)
        removed = 0
        for _stamp, path in directories[max_entries:]:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1

        # Empty day directories left behind read as gaps in the history.
        for day in ARCHIVE_DIR.iterdir():
            if day.is_dir() and not any(day.iterdir()):
                day.rmdir()
    # Only when something actually went: prune runs after every utterance and
    # almost always removes nothing, so the walk is not worth doing each time.
    if removed:
        measure_audio()
    return removed


def clear() -> None:
    with _LOCK:
        shutil.rmtree(ARCHIVE_DIR, ignore_errors=True)
        _USAGE.update(bytes=0, files=0, checked_at=time.time(), measured=True)
