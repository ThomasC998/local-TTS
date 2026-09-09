"""On-disk voice profiles and design/clone previews.

A *voice profile* is everything needed to reproduce a speaker: the reference
audio fragment, the exact transcript of that fragment, and every generation
setting that produced it. Profiles live one directory per voice::

    voices/
        voice_<id>/
            profile.json     -- metadata, settings, generation parameters
            sample.wav       -- the audition clip (what the voice sounds like)
            reference.wav    -- what later requests condition on (cloned voices)
        .previews/
            gvi_<id>/        -- unsaved candidates, pruned automatically

Designed voices point ``reference`` at ``sample.wav``: the take you picked *is*
the reference. Cloned voices keep the uploaded recording as the reference and
use the generated clip only for auditioning, which mirrors the upstream API --
a saved clone synthesizes from the stored sample, not from its preview.

Nothing here imports the model, so the server can serve the voice library while
a generation holds the engine lock.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
VOICES_DIR = Path(
    os.getenv("BREEZE_VOICES_DIR", Path(__file__).resolve().parent / "voices")
)
PREVIEWS_DIR = VOICES_DIR / ".previews"

PROFILE_NAME = "profile.json"
SAMPLE_NAME = "sample.wav"
REFERENCE_NAME = "reference.wav"

# Previews are cheap to regenerate and pile up fast, so they expire.
PREVIEW_MAX_AGE_SECONDS = 24 * 3600
PREVIEW_MAX_COUNT = 400

# One lock for every mutation. Endpoints run in FastAPI's threadpool, so two
# requests really can write the same directory at once.
_LOCK = threading.RLock()

# --------------------------------------------------------------------------
# Metadata taxonomy
#
# Mirrors the upstream voice-metadata options, minus the multilingual half:
# language_code is stored as free-form text and never validated or enforced.
# --------------------------------------------------------------------------
GENDER_CODES = ("male", "female", "neutral")
AGE_CODES = ("child", "young", "middle_aged", "old")
TONE_CODES = (
    "warm", "calm", "bright", "gentle", "energetic", "authoritative", "sincere",
    "weary", "precise", "refined", "urgent", "friendly", "articulate", "steady",
    "playful", "compassionate", "reflective", "measured", "rhythmic", "passionate",
)
TONE_MAX_ITEMS = 3
ACCENT_CODES = (
    "american", "british", "scottish", "irish", "australian", "canadian",
    "us_southern", "us_new_york", "indian", "south_african", "russian",
    "japanese", "korean", "chinese", "mandarin_guangdong",
    "mandarin_northeastern", "mandarin_shaanxi", "mandarin_shanghai",
    "mandarin_sichuan", "mandarin_yunnan", "mandarin_henan", "cantonese",
)
TAG_MAX_ITEMS = 20
TAG_MAX_LENGTH = 128

ORIGINS = ("designed", "cloned")

# Where a reference recording came from. ``variation`` is a take generated from
# an existing reference with a new instruction -- the same speaker, differently
# coloured -- which is what rotation cycles through.
REFERENCE_SOURCES = ("upload", "designed_preview", "variation", "recording")

# Rotation keeps a long read from flattening into one tone. It swaps the
# reference every ``every_words`` words, but only at a boundary, so the switch
# never lands inside a sentence.
ROTATION_MODES = ("random", "sequence")
ROTATION_BOUNDARIES = ("paragraph", "sentence")
DEFAULT_ROTATION: dict[str, Any] = {
    "enabled": True,
    "mode": "random",
    "every_words": 1000,
    "boundary": "paragraph",
    "tags": [],
}

# Every generation knob a profile remembers. Stored verbatim so a saved voice
# replays exactly, including the fields the request left at their defaults.
GENERATION_FIELDS = (
    "instruction",
    "mode",
    "cfg_scale",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "greedy",
    "repetition_penalty",
    "max_new_tokens",
    "max_seq_len",
    "codec_chunk_frames",
    "max_words",
)

# Reference-audio guidance, surfaced in the UI and in /v1/voice-metadata/options.
REFERENCE_MIN_SECONDS = 10.0
REFERENCE_GOOD_SECONDS = 15.0


class VoiceNotFound(KeyError):
    """No profile with that id."""


class PreviewNotFound(KeyError):
    """No preview with that id, or it has already expired."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write: a crash mid-save must not leave a half-parsed profile."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    """Write float32 mono audio; return its duration in seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return float(audio.size / sample_rate)


def audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:  # noqa: BLE001 - duration is informational only
        return 0.0


def sample_rate_of(path: Path) -> int | None:
    """The file's own rate. None when it cannot be read; nothing depends on it."""
    try:
        return int(sf.info(str(path)).samplerate)
    except Exception:  # noqa: BLE001 - reported, never load-bearing
        return None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def normalize_tags(tags: Any) -> list[str]:
    """Canonical lowercase tags: NFC-normalized, trimmed, deduplicated."""
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [piece for piece in tags.split(",")]
    if not isinstance(tags, (list, tuple)):
        raise ValueError("tags must be an array of strings")

    seen: list[str] = []
    for tag in tags:
        text = unicodedata.normalize("NFC", str(tag)).strip().lower()
        if not text:
            continue
        if len(text) > TAG_MAX_LENGTH:
            raise ValueError(f"tag exceeds {TAG_MAX_LENGTH} characters: {text[:40]}...")
        if text not in seen:
            seen.append(text)
    if len(seen) > TAG_MAX_ITEMS:
        raise ValueError(f"at most {TAG_MAX_ITEMS} tags are allowed")
    return seen


def _validate_choice(value: Any, codes: tuple[str, ...], field: str) -> str | None:
    """An empty value clears the field; anything else must be a known code."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text not in codes:
        raise ValueError(f"{field} must be one of: {', '.join(codes)}")
    return text


def validate_metadata(patch: dict[str, Any]) -> dict[str, Any]:
    """Check the descriptive fields of a create/update payload.

    Only keys present in ``patch`` are touched, so a PATCH that omits a field
    preserves it while an explicitly empty one clears it.
    """
    clean: dict[str, Any] = {}

    if "name" in patch:
        name = str(patch["name"] or "").strip()
        if not name:
            raise ValueError("name must not be empty")
        if len(name) > 200:
            raise ValueError("name must be 200 characters or fewer")
        clean["name"] = name

    if "description" in patch:
        description = patch["description"]
        clean["description"] = str(description).strip() if description else None

    if "notes" in patch:
        notes = patch["notes"]
        clean["notes"] = str(notes).strip() if notes else None

    if "language_code" in patch:
        # Kept as free-form metadata: multilingual behaviour is deliberately
        # not implemented, so this is a label rather than a switch.
        language = patch["language_code"]
        clean["language_code"] = str(language).strip().lower() if language else None

    if "gender" in patch:
        clean["gender"] = _validate_choice(patch["gender"], GENDER_CODES, "gender")
    if "age" in patch:
        clean["age"] = _validate_choice(patch["age"], AGE_CODES, "age")
    if "accent" in patch:
        clean["accent"] = _validate_choice(patch["accent"], ACCENT_CODES, "accent")

    if "tone" in patch:
        tone_value = patch["tone"]
        if tone_value is None or tone_value == "":
            tones: list[str] = []
        else:
            if isinstance(tone_value, str):
                tone_value = [piece for piece in tone_value.split(",")]
            tones = []
            for item in tone_value:
                code = _validate_choice(item, TONE_CODES, "tone")
                if code and code not in tones:
                    tones.append(code)
        if len(tones) > TONE_MAX_ITEMS:
            raise ValueError(f"tone accepts at most {TONE_MAX_ITEMS} codes")
        clean["tone"] = tones

    if "tags" in patch:
        clean["tags"] = normalize_tags(patch["tags"])

    if "is_favorited" in patch:
        clean["is_favorited"] = bool(patch["is_favorited"])

    return clean


def metadata_options() -> dict[str, Any]:
    """Every valid code, for a UI form or an automated edit to validate against."""
    return {
        "gender_codes": list(GENDER_CODES),
        "age_codes": list(AGE_CODES),
        "tone_codes": list(TONE_CODES),
        "tone_max_items": TONE_MAX_ITEMS,
        "accent_codes": list(ACCENT_CODES),
        "tag_max_items": TAG_MAX_ITEMS,
        "tag_max_length": TAG_MAX_LENGTH,
        "origins": list(ORIGINS),
        "generation_fields": list(GENERATION_FIELDS),
        "reference_audio": {
            "min_seconds": REFERENCE_MIN_SECONDS,
            "recommended_seconds": REFERENCE_GOOD_SECONDS,
            "guidance": (
                "Cloning works from about 10 seconds of speech; 15 seconds or more "
                "is noticeably better. Use one speaker, no music or background "
                "noise, and pick a passage that is as varied as possible -- mixed "
                "sentence lengths, questions and statements, a range of pitch and "
                "pace. A flat, monotone sample clones as a flat, monotone voice."
            ),
        },
        "language_code": (
            "Stored as a free-form label. Multilingual synthesis is deliberately "
            "not implemented, so this field never changes generation behaviour."
        ),
    }


# --------------------------------------------------------------------------
# Previews
# --------------------------------------------------------------------------
def _preview_dir(generated_voice_id: str) -> Path:
    return PREVIEWS_DIR / generated_voice_id


def save_preview(
    audio: np.ndarray,
    sample_rate: int,
    *,
    text: str,
    generation: dict[str, Any],
    origin: str,
    name: str | None = None,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one candidate take and return its descriptor.

    ``reference_audio`` is the recording a *cloned* candidate was conditioned on.
    It is copied in now, while the upload still exists, so saving the preview as
    a voice later needs nothing from the original request.
    """
    generated_voice_id = _new_id("gvi")
    directory = _preview_dir(generated_voice_id)
    directory.mkdir(parents=True, exist_ok=True)

    duration = write_wav(directory / "audio.wav", audio, sample_rate)

    stored_reference: str | None = None
    if reference_audio is not None and Path(reference_audio).is_file():
        shutil.copyfile(reference_audio, directory / REFERENCE_NAME)
        stored_reference = REFERENCE_NAME

    payload = {
        "generated_voice_id": generated_voice_id,
        "origin": origin,
        "name": name,
        "text": text,
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "created_at_unix": _now(),
        "generation": {key: generation.get(key) for key in GENERATION_FIELDS},
        "reference_file": stored_reference,
        "reference_text": reference_text,
        **(extra or {}),
    }
    _write_json(directory / "meta.json", payload)
    return payload


def load_preview(generated_voice_id: str) -> dict[str, Any]:
    path = _preview_dir(generated_voice_id) / "meta.json"
    if not path.is_file():
        raise PreviewNotFound(generated_voice_id)
    return _read_json(path)


def preview_audio_path(generated_voice_id: str) -> Path:
    path = _preview_dir(generated_voice_id) / "audio.wav"
    if not path.is_file():
        raise PreviewNotFound(generated_voice_id)
    return path


def prune_previews(
    max_age_seconds: int = PREVIEW_MAX_AGE_SECONDS, max_count: int = PREVIEW_MAX_COUNT
) -> int:
    """Drop expired candidates. Returns how many directories were removed."""
    if not PREVIEWS_DIR.is_dir():
        return 0

    with _LOCK:
        entries: list[tuple[float, Path]] = []
        for directory in PREVIEWS_DIR.iterdir():
            if directory.is_dir():
                entries.append((directory.stat().st_mtime, directory))

        cutoff = time.time() - max_age_seconds
        doomed = [path for stamp, path in entries if stamp < cutoff]
        survivors = sorted(
            ((stamp, path) for stamp, path in entries if stamp >= cutoff),
            reverse=True,
        )
        doomed.extend(path for _stamp, path in survivors[max_count:])

        for path in doomed:
            shutil.rmtree(path, ignore_errors=True)
        return len(doomed)


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------
def _voice_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id


def _profile_path(voice_id: str) -> Path:
    return _voice_dir(voice_id) / PROFILE_NAME


def _migrate_references(profile: dict[str, Any]) -> bool:
    """Give a v1 profile the v2 shape in place. True if anything changed.

    v1 stored exactly one ``reference`` object. v2 keeps that key -- every
    existing caller reads it, and it always mirrors the default entry -- and
    adds ``references``, the list rotation draws from.
    """
    changed = False

    if not isinstance(profile.get("references"), list) or not profile["references"]:
        legacy = profile.get("reference") or {}
        profile["references"] = [
            {
                "reference_id": "ref_base",
                "file": legacy.get("file") or SAMPLE_NAME,
                "text": legacy.get("text") or "",
                "label": "base",
                "tags": ["base"],
                "duration_seconds": legacy.get("duration_seconds") or 0.0,
                "source": legacy.get("source") or "designed_preview",
                "enabled": True,
                "is_default": True,
                "created_at_unix": profile.get("created_at_unix") or _now(),
                "instruction": (profile.get("generation") or {}).get("instruction"),
                "generation": dict(profile.get("generation") or {}),
                "generated_voice_id": profile.get("generated_voice_id"),
            }
        ]
        changed = True

    if not isinstance(profile.get("rotation"), dict):
        profile["rotation"] = dict(DEFAULT_ROTATION)
        changed = True
    else:
        for key, value in DEFAULT_ROTATION.items():
            if key not in profile["rotation"]:
                profile["rotation"][key] = value
                changed = True

    return changed


def _sync_default_reference(profile: dict[str, Any]) -> None:
    """Keep the legacy ``reference`` key pointing at the default entry.

    Anything written before multi-reference existed reads ``reference``; this is
    what keeps those paths correct after a variation is added or removed.
    """
    entries = profile.get("references") or []
    if not entries:
        return
    default = next(
        (entry for entry in entries if entry.get("is_default") and entry.get("enabled")),
        None,
    )
    if default is None:
        default = next((entry for entry in entries if entry.get("enabled")), entries[0])
        for entry in entries:
            entry["is_default"] = entry is default
    profile["reference"] = {
        "file": default["file"],
        "text": default.get("text") or "",
        "duration_seconds": default.get("duration_seconds") or 0.0,
        "source": default.get("source") or "upload",
    }


def _public(profile: dict[str, Any]) -> dict[str, Any]:
    """Add the derived fields an API response carries but disk does not.

    Migration happens here rather than on disk: a v1 profile that is only ever
    read keeps its original bytes, and the first mutation writes the v2 shape.
    """
    voice_id = profile["voice_id"]
    view = dict(profile)
    _migrate_references(view)
    view["preview_url"] = f"/v1/voices/{voice_id}/sample"
    view["reference_url"] = f"/v1/voices/{voice_id}/reference"
    view["references"] = [
        {
            **entry,
            "audio_url": f"/v1/voices/{voice_id}/references/{entry['reference_id']}/audio",
            "exists": (_voice_dir(voice_id) / entry["file"]).is_file(),
        }
        for entry in view.get("references", [])
    ]
    return view


def create_voice_from_preview(
    generated_voice_id: str,
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a preview as a reusable voice.

    The candidate's audio becomes the audition sample. For a designed voice that
    same clip is also the reference every later request conditions on; for a
    cloned voice the original recording keeps that job.
    """
    preview = load_preview(generated_voice_id)
    source = _preview_dir(generated_voice_id)

    clean = validate_metadata({"name": name, **(metadata or {})})
    voice_id = _new_id("voice")
    directory = _voice_dir(voice_id)

    with _LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "audio.wav", directory / SAMPLE_NAME)

        sample_text = preview["text"]
        sample_duration = preview.get("duration_seconds") or audio_duration(
            directory / SAMPLE_NAME
        )

        # A cloned voice references the recording it was cloned from; a designed
        # voice has no earlier recording, so the take itself becomes the anchor.
        if preview.get("reference_file") and (source / REFERENCE_NAME).is_file():
            shutil.copyfile(source / REFERENCE_NAME, directory / REFERENCE_NAME)
            reference = {
                "file": REFERENCE_NAME,
                "text": preview.get("reference_text") or "",
                "duration_seconds": round(audio_duration(directory / REFERENCE_NAME), 3),
                "source": "upload",
            }
        else:
            reference = {
                "file": SAMPLE_NAME,
                "text": sample_text,
                "duration_seconds": round(sample_duration, 3),
                "source": "designed_preview",
            }

        generation = {key: preview["generation"].get(key) for key in GENERATION_FIELDS}
        profile = _new_profile(
            voice_id,
            clean,
            origin=preview.get("origin", "designed"),
            generation=generation,
            sample={
                "file": SAMPLE_NAME,
                "text": sample_text,
                "duration_seconds": round(sample_duration, 3),
            },
            reference=reference,
            sample_rate=preview.get("sample_rate"),
            generated_voice_id=generated_voice_id,
        )
        _write_json(_profile_path(voice_id), profile)

    return _public(profile)


def _new_profile(
    voice_id: str,
    clean: dict[str, Any],
    *,
    origin: str,
    generation: dict[str, Any],
    sample: dict[str, Any],
    reference: dict[str, Any],
    sample_rate: int | None,
    generated_voice_id: str | None,
) -> dict[str, Any]:
    """The on-disk shape of a voice, however it was created.

    Shared by both routes into the library -- keeping a generated take, and
    keeping the recording itself -- because a profile written two ways is a
    profile that drifts apart on the next field anyone adds.
    """
    return {
        "voice_id": voice_id,
        "name": clean["name"],
        "description": clean.get("description"),
        "notes": clean.get("notes"),
        "origin": origin,
        "voice_type": "personal",
        "created_at_unix": _now(),
        "updated_at_unix": _now(),
        "is_favorited": bool(clean.get("is_favorited", False)),
        "tags": clean.get("tags", []),
        "language_code": clean.get("language_code"),
        "gender": clean.get("gender"),
        "age": clean.get("age"),
        "tone": clean.get("tone", []),
        "accent": clean.get("accent"),
        "settings": {"guidance_scale": generation.get("cfg_scale") or 1.0},
        "generation": generation,
        "sample": sample,
        "reference": reference,
        "references": [
            {
                "reference_id": "ref_base",
                "file": reference["file"],
                "text": reference["text"],
                "label": "base",
                "tags": ["base"],
                "duration_seconds": reference["duration_seconds"],
                "source": reference["source"],
                "enabled": True,
                "is_default": True,
                "created_at_unix": _now(),
                "instruction": generation.get("instruction"),
                "generation": dict(generation),
                "generated_voice_id": generated_voice_id,
            }
        ],
        "rotation": dict(DEFAULT_ROTATION),
        "sample_rate": sample_rate,
        "generated_voice_id": generated_voice_id,
    }


def create_voice_from_recording(
    audio_path: Path,
    *,
    name: str,
    reference_text: str,
    generation: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save the recording itself as a voice, with no generated take involved.

    ``create_voice_from_preview`` keeps a candidate the model produced, and
    makes the recording behind it the reference. This keeps the recording and
    stops there -- which is what you want when the clone is already right and
    the takes were only ever a way of checking it.

    Nothing is lost by skipping them. A cloned voice conditions every later
    request on the *recording*, never on the take, so the take was only ever an
    audition. Here the recording is the audition too: the library plays back the
    thing the voice will actually sound like.

    ``generation`` carries the voice direction and the sampling settings, so a
    voice saved this way answers exactly as one saved from a take would.
    """
    reference_text = _validate_reference_text(reference_text)
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise ValueError(f"reference audio not found: {audio_path}")

    duration = audio_duration(audio_path)
    if duration and duration < REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"Recording is {duration:.1f}s. Cloning works from about "
            f"{REFERENCE_MIN_SECONDS:.0f}s; "
            f"{REFERENCE_GOOD_SECONDS:.0f}s or more is better."
        )

    clean = validate_metadata({"name": name, **(metadata or {})})
    supplied = dict(generation or {})
    resolved = {key: supplied.get(key) for key in GENERATION_FIELDS}
    # The voice is anchored to a recording, so the mode is the reference branch
    # of whichever template pair the direction implies -- the same rule
    # _build_job applies when the voice is spoken with.
    resolved["mode"] = resolved.get("mode") or (
        "edit" if resolved.get("instruction") else "clone"
    )

    voice_id = _new_id("voice")
    directory = _voice_dir(voice_id)

    with _LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio_path, directory / REFERENCE_NAME)
        stored_duration = round(audio_duration(directory / REFERENCE_NAME), 3)

        descriptor = {
            "file": REFERENCE_NAME,
            "text": reference_text,
            "duration_seconds": stored_duration,
            "source": "upload",
        }
        profile = _new_profile(
            voice_id,
            clean,
            origin="cloned",
            generation=resolved,
            # One file, two jobs. The recording is both what the model
            # conditions on and what the library plays when you audition the
            # voice, and REFERENCE_NAME is never swept by reference deletion.
            sample={
                "file": REFERENCE_NAME,
                "text": reference_text,
                "duration_seconds": stored_duration,
            },
            reference=descriptor,
            sample_rate=sample_rate_of(directory / REFERENCE_NAME),
            generated_voice_id=None,
        )
        _write_json(_profile_path(voice_id), profile)

    return _public(profile)


def get_voice(voice_id: str) -> dict[str, Any]:
    path = _profile_path(voice_id)
    if not path.is_file():
        raise VoiceNotFound(voice_id)
    return _public(_read_json(path))


def voice_reference(
    voice_id: str, reference_id: str | None = None
) -> tuple[Path, str] | None:
    """The audio path and exact transcript a saved voice conditions on.

    Without ``reference_id`` this is the default entry, which is what every
    caller predating multi-reference support gets.
    """
    profile = get_voice(voice_id)
    if reference_id:
        entry = next(
            (
                item
                for item in profile.get("references", [])
                if item["reference_id"] == reference_id
            ),
            None,
        )
        if entry is None:
            raise VoiceNotFound(f"{voice_id} has no reference {reference_id}")
        reference = entry
    else:
        reference = profile.get("reference") or {}

    file_name = reference.get("file")
    text = (reference.get("text") or "").strip()
    if not file_name or not text:
        return None
    path = _voice_dir(voice_id) / file_name
    if not path.is_file():
        return None
    return path, text


def sample_path(voice_id: str) -> Path:
    profile = get_voice(voice_id)
    path = _voice_dir(voice_id) / profile["sample"]["file"]
    if not path.is_file():
        raise VoiceNotFound(f"{voice_id} has no sample audio")
    return path


def reference_path(voice_id: str) -> Path:
    reference = voice_reference(voice_id)
    if reference is None:
        raise VoiceNotFound(f"{voice_id} has no reference audio")
    return reference[0]


def _matches_search(profile: dict[str, Any], needle: str) -> bool:
    haystack = " ".join(
        str(value)
        for value in (
            profile.get("name"),
            profile.get("description"),
            profile.get("notes"),
            profile.get("voice_id"),
            (profile.get("generation") or {}).get("instruction"),
            " ".join(profile.get("tags") or []),
            " ".join(profile.get("tone") or []),
            profile.get("accent"),
            profile.get("gender"),
            profile.get("age"),
        )
        if value
    ).lower()
    return needle.lower() in haystack


def list_voices(
    *,
    search: str | None = None,
    origin: str | None = None,
    tags: list[str] | None = None,
    favorites_only: bool = False,
    sort: str = "created_at_unix",
    sort_direction: str = "desc",
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """Filter, sort and paginate the library.

    Text matching is plain substring: there is no embedding index locally, and a
    library of personal voices is small enough that it does not need one.
    """
    if not VOICES_DIR.is_dir():
        profiles: list[dict[str, Any]] = []
    else:
        profiles = []
        for directory in sorted(VOICES_DIR.iterdir()):
            if directory.name.startswith(".") or not directory.is_dir():
                continue
            path = directory / PROFILE_NAME
            if path.is_file():
                try:
                    profiles.append(_read_json(path))
                except (OSError, json.JSONDecodeError):
                    continue  # a corrupt profile must not take down the list

    if search:
        profiles = [item for item in profiles if _matches_search(item, search)]
    if origin:
        profiles = [item for item in profiles if item.get("origin") == origin]
    if favorites_only:
        profiles = [item for item in profiles if item.get("is_favorited")]
    if tags:
        wanted = normalize_tags(tags)
        profiles = [
            item for item in profiles if set(wanted) <= set(item.get("tags") or [])
        ]

    if sort not in {"created_at_unix", "updated_at_unix", "name"}:
        raise ValueError("sort must be created_at_unix, updated_at_unix or name")
    reverse = str(sort_direction).lower() != "asc"
    profiles.sort(
        key=lambda item: (str(item.get("name", "")).lower() if sort == "name"
                          else item.get(sort) or 0),
        reverse=reverse,
    )

    total = len(profiles)
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    start = (page - 1) * page_size
    window = profiles[start : start + page_size]

    return {
        "voices": [_public(item) for item in window],
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": start + len(window) < total,
    }


def update_voice(voice_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a descriptive-metadata patch. Omitted keys keep their value."""
    clean = validate_metadata(patch)
    with _LOCK:
        path = _profile_path(voice_id)
        if not path.is_file():
            raise VoiceNotFound(voice_id)
        profile = _read_json(path)
        profile.update(clean)
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)
    return _public(profile)


def update_settings(voice_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Change the stored generation defaults, e.g. the guidance scale or seed.

    ``guidance_scale`` and ``cfg_scale`` are the same number under two names --
    the upstream field and the one this runtime uses -- so they stay in step.
    """
    with _LOCK:
        path = _profile_path(voice_id)
        if not path.is_file():
            raise VoiceNotFound(voice_id)
        profile = _read_json(path)
        generation = dict(profile.get("generation") or {})

        if "guidance_scale" in patch and patch["guidance_scale"] is not None:
            generation["cfg_scale"] = float(patch["guidance_scale"])
        for field in GENERATION_FIELDS:
            if field in patch:
                generation[field] = patch[field]

        profile["generation"] = generation
        profile["settings"] = {"guidance_scale": generation.get("cfg_scale") or 1.0}
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)
    return _public(profile)


def set_tags(voice_id: str, tags: Any) -> dict[str, Any]:
    return update_voice(voice_id, {"tags": tags})


def set_favorite(voice_id: str, favorited: bool) -> dict[str, Any]:
    return update_voice(voice_id, {"is_favorited": favorited})


def delete_voice(voice_id: str) -> None:
    with _LOCK:
        directory = _voice_dir(voice_id)
        if not (directory / PROFILE_NAME).is_file():
            raise VoiceNotFound(voice_id)
        shutil.rmtree(directory, ignore_errors=True)


# --------------------------------------------------------------------------
# References
#
# A voice is one speaker, but a two-page article read in a single unvarying
# tone gets tiring. So a voice may hold several reference recordings of that
# same speaker -- a base take plus variations generated from it ("warmer",
# "more urgent") -- and a long read rotates between them.
#
# Every entry carries its own exact transcript. The model requires the words
# spoken in the reference alongside the audio, so a reference without a
# transcript is not a reference; it is rejected at write time rather than
# discovered as a bad clone later.
# --------------------------------------------------------------------------
def _load_for_write(voice_id: str) -> tuple[Path, dict[str, Any]]:
    """Read a profile in v2 shape, ready to mutate. Caller holds ``_LOCK``."""
    path = _profile_path(voice_id)
    if not path.is_file():
        raise VoiceNotFound(voice_id)
    profile = _read_json(path)
    _migrate_references(profile)
    return path, profile


def _validate_reference_text(text: Any) -> str:
    clean = str(text or "").strip()
    if not clean:
        raise ValueError(
            "Every reference needs the exact words spoken in it: the model "
            "conditions on audio and transcript together"
        )
    return clean


def list_references(voice_id: str) -> list[dict[str, Any]]:
    return get_voice(voice_id)["references"]


def reference_entry(voice_id: str, reference_id: str) -> dict[str, Any]:
    for entry in get_voice(voice_id)["references"]:
        if entry["reference_id"] == reference_id:
            return entry
    raise VoiceNotFound(f"{voice_id} has no reference {reference_id}")


def reference_audio_path(voice_id: str, reference_id: str) -> Path:
    entry = reference_entry(voice_id, reference_id)
    path = _voice_dir(voice_id) / entry["file"]
    if not path.is_file():
        raise VoiceNotFound(f"{voice_id}/{reference_id} has no audio on disk")
    return path


def _add_reference(
    voice_id: str,
    *,
    source_audio: Path,
    text: str,
    label: str | None,
    tags: Any,
    source: str,
    instruction: str | None = None,
    generation: dict[str, Any] | None = None,
    generated_voice_id: str | None = None,
    make_default: bool = False,
) -> dict[str, Any]:
    text = _validate_reference_text(text)
    clean_tags = normalize_tags(tags)
    if source not in REFERENCE_SOURCES:
        raise ValueError(f"source must be one of: {', '.join(REFERENCE_SOURCES)}")
    source_audio = Path(source_audio)
    if not source_audio.is_file():
        raise ValueError(f"reference audio not found: {source_audio}")

    with _LOCK:
        path, profile = _load_for_write(voice_id)
        reference_id = _new_id("ref")
        file_name = f"reference_{reference_id}.wav"
        shutil.copyfile(source_audio, _voice_dir(voice_id) / file_name)

        entry = {
            "reference_id": reference_id,
            "file": file_name,
            "text": text,
            "label": (str(label).strip() if label else None) or (clean_tags[0] if clean_tags else "variation"),
            "tags": clean_tags,
            "duration_seconds": round(audio_duration(_voice_dir(voice_id) / file_name), 3),
            "source": source,
            "enabled": True,
            "is_default": False,
            "created_at_unix": _now(),
            # Kept so a variation can be regenerated later: the descriptor that
            # produced it plus the exact sampling settings.
            "instruction": instruction,
            "generation": {key: (generation or {}).get(key) for key in GENERATION_FIELDS},
            "generated_voice_id": generated_voice_id,
        }
        profile["references"].append(entry)
        if make_default:
            for item in profile["references"]:
                item["is_default"] = item is entry
        _sync_default_reference(profile)
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)

    return _public(profile)


def add_reference_from_preview(
    voice_id: str,
    generated_voice_id: str,
    *,
    label: str | None = None,
    tags: Any = None,
    make_default: bool = False,
) -> dict[str, Any]:
    """Attach a generated take to a voice as another reference.

    The transcript is the script the take was generated from, so it is exact by
    construction -- nothing is transcribed and nothing can drift.
    """
    preview = load_preview(generated_voice_id)
    audio = preview_audio_path(generated_voice_id)
    duration = preview.get("duration_seconds") or audio_duration(audio)
    if duration < REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"That take is {duration:.1f}s. A reference needs about "
            f"{REFERENCE_MIN_SECONDS:.0f}s to clone from reliably; generate the "
            f"variation on a longer script."
        )
    return _add_reference(
        voice_id,
        source_audio=audio,
        text=preview["text"],
        label=label,
        tags=tags,
        source="variation",
        instruction=(preview.get("generation") or {}).get("instruction"),
        generation=preview.get("generation") or {},
        generated_voice_id=generated_voice_id,
        make_default=make_default,
    )


def add_reference_from_audio(
    voice_id: str,
    audio_path: Path,
    text: str,
    *,
    label: str | None = None,
    tags: Any = None,
    make_default: bool = False,
) -> dict[str, Any]:
    """Attach an uploaded recording of the same speaker as another reference."""
    duration = audio_duration(Path(audio_path))
    if duration and duration < REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"Recording is {duration:.1f}s; a reference needs about "
            f"{REFERENCE_MIN_SECONDS:.0f}s."
        )
    return _add_reference(
        voice_id,
        source_audio=audio_path,
        text=text,
        label=label,
        tags=tags,
        source="recording",
        make_default=make_default,
    )


def update_reference(
    voice_id: str, reference_id: str, patch: dict[str, Any]
) -> dict[str, Any]:
    """Retag, rename, enable/disable, or promote a reference to default."""
    with _LOCK:
        path, profile = _load_for_write(voice_id)
        entry = next(
            (item for item in profile["references"] if item["reference_id"] == reference_id),
            None,
        )
        if entry is None:
            raise VoiceNotFound(f"{voice_id} has no reference {reference_id}")

        if "label" in patch:
            entry["label"] = str(patch["label"]).strip() or entry["label"]
        if "tags" in patch:
            entry["tags"] = normalize_tags(patch["tags"])
        if "text" in patch:
            entry["text"] = _validate_reference_text(patch["text"])
        if "enabled" in patch:
            enabled = bool(patch["enabled"])
            if not enabled and sum(
                1 for item in profile["references"] if item.get("enabled")
            ) <= 1:
                raise ValueError("A voice must keep at least one enabled reference")
            entry["enabled"] = enabled
        if patch.get("is_default"):
            entry["enabled"] = True
            for item in profile["references"]:
                item["is_default"] = item is entry

        _sync_default_reference(profile)
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)
    return _public(profile)


def delete_reference(voice_id: str, reference_id: str) -> dict[str, Any]:
    with _LOCK:
        path, profile = _load_for_write(voice_id)
        entries = profile["references"]
        entry = next(
            (item for item in entries if item["reference_id"] == reference_id), None
        )
        if entry is None:
            raise VoiceNotFound(f"{voice_id} has no reference {reference_id}")
        if len(entries) <= 1:
            raise ValueError("A voice must keep at least one reference")

        entries.remove(entry)
        # The base reference doubles as the audition sample on designed voices,
        # so only files this feature created are removed from disk.
        if entry["file"] not in {SAMPLE_NAME, REFERENCE_NAME}:
            (_voice_dir(voice_id) / entry["file"]).unlink(missing_ok=True)

        _sync_default_reference(profile)
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)
    return _public(profile)


def validate_rotation(patch: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    if "enabled" in patch:
        clean["enabled"] = bool(patch["enabled"])
    if "mode" in patch:
        mode = str(patch["mode"] or "").strip().lower()
        if mode not in ROTATION_MODES:
            raise ValueError(f"mode must be one of: {', '.join(ROTATION_MODES)}")
        clean["mode"] = mode
    if "boundary" in patch:
        boundary = str(patch["boundary"] or "").strip().lower()
        if boundary not in ROTATION_BOUNDARIES:
            raise ValueError(
                f"boundary must be one of: {', '.join(ROTATION_BOUNDARIES)}"
            )
        clean["boundary"] = boundary
    if "every_words" in patch:
        words = int(patch["every_words"])
        if not 20 <= words <= 100_000:
            raise ValueError("every_words must be between 20 and 100000")
        clean["every_words"] = words
    if "tags" in patch:
        clean["tags"] = normalize_tags(patch["tags"])
    return clean


def update_rotation(voice_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    clean = validate_rotation(patch)
    with _LOCK:
        path, profile = _load_for_write(voice_id)
        profile["rotation"] = {**profile["rotation"], **clean}
        profile["updated_at_unix"] = _now()
        _write_json(path, profile)
    return _public(profile)


def reference_pool(
    voice_id: str, tags: list[str] | None = None
) -> list[dict[str, Any]]:
    """Enabled references that exist on disk, optionally filtered by tag.

    A tag filter that matches nothing falls back to the whole pool: a stale tag
    in the settings should not silently mute the voice.
    """
    entries = [
        entry
        for entry in get_voice(voice_id)["references"]
        if entry.get("enabled") and entry.get("exists") and (entry.get("text") or "").strip()
    ]
    wanted = normalize_tags(tags) if tags else []
    if wanted:
        filtered = [
            entry for entry in entries if set(wanted) & set(entry.get("tags") or [])
        ]
        if filtered:
            return filtered
    return entries


def rotation_settings(
    voice_id: str, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The voice's rotation settings, with a request-level override applied."""
    settings = {**DEFAULT_ROTATION, **(get_voice(voice_id).get("rotation") or {})}
    if override:
        settings.update(validate_rotation(override))
    return settings


class ReferenceRotator:
    """Picks which reference each chunk of a long read is conditioned on.

    Switching is deliberately conservative. It happens only when two things are
    true at once: enough words have gone by since the last switch, and the next
    chunk starts a new group -- a paragraph, or a sentence where the text has no
    paragraphs to use. Both are boundaries *between* chunks, so a tone change is
    never audible inside a sentence.

    The same object serves a document whose chunks are all known up front and
    one still arriving from a language model, so the two cannot drift apart.
    """

    def __init__(
        self,
        voice_id: str,
        *,
        settings: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.voice_id = voice_id
        self.settings = settings or rotation_settings(voice_id)
        self.pool = reference_pool(voice_id, self.settings.get("tags"))
        self.directory = _voice_dir(voice_id)
        self._rng = random.Random(seed)
        self._order = list(range(len(self.pool)))
        if self.settings.get("mode") == "random":
            self._rng.shuffle(self._order)
        self._cursor = 0
        self._words = 0
        self._index = 0
        self.segments: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = self.pool[self._order[0]] if self.pool else None
        # Segments describe an actual rotation. With nothing to rotate between
        # there is no story to tell, and an empty list says exactly that.
        if self.active:
            self._open_segment()

    @property
    def active(self) -> bool:
        """Whether there is anything to rotate between."""
        return bool(self.settings.get("enabled")) and len(self.pool) > 1

    def _open_segment(self) -> None:
        assert self._current is not None
        self.segments.append(
            {
                "reference_id": self._current["reference_id"],
                "label": self._current.get("label"),
                "tags": self._current.get("tags") or [],
                "from_chunk": self._index,
                "words": 0,
            }
        )

    def _switch(self) -> None:
        self._cursor += 1
        if self.settings.get("mode") == "random" and self._cursor % len(self.pool) == 0:
            # Reshuffle each full cycle, but never repeat across the seam: a
            # "switch" that lands on the voice already speaking is not a switch.
            previous = self._order[-1]
            self._rng.shuffle(self._order)
            if len(self._order) > 1 and self._order[0] == previous:
                self._order[0], self._order[-1] = self._order[-1], self._order[0]
        self._current = self.pool[self._order[self._cursor % len(self.pool)]]
        self._words = 0
        self._open_segment()

    def take(self, chunk_words: int, starts_group: bool) -> tuple[str, str] | None:
        """The (path, transcript) for the next chunk, advancing the rotation."""
        if not self.active or self._current is None:
            return None
        if starts_group and self._words >= int(self.settings["every_words"]):
            self._switch()
        reference = (
            str(self.directory / self._current["file"]),
            self._current["text"],
        )
        self._words += chunk_words
        self.segments[-1]["words"] += chunk_words
        self._index += 1
        return reference


def build_reference_plan(
    voice_id: str,
    chunk_words: list[int],
    group_ids: list[int],
    *,
    rotation: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[list[tuple[str, str]] | None, list[dict[str, Any]]]:
    """Assign a reference to every chunk of a document that is fully known.

    Returns ``(per_chunk, segments)``. ``None`` means there is nothing to rotate
    -- one reference, or rotation switched off -- which lets the caller take its
    ordinary single-reference path unchanged.
    """
    settings = rotation_settings(voice_id, rotation)
    rotator = ReferenceRotator(voice_id, settings=settings, seed=seed)
    if not rotator.active or not chunk_words:
        return None, []

    # With no blank lines in the source there is exactly one paragraph and so no
    # switch point at all, which would silently disable rotation on the most
    # common input there is: a passage copied out of an article. Sentence starts
    # are the fallback, and are still never mid-sentence.
    boundaries = list(group_ids)
    if settings.get("boundary") == "sentence" or len(set(boundaries)) < 2:
        boundaries = list(range(len(chunk_words)))

    per_chunk: list[tuple[str, str]] = []
    for index, words in enumerate(chunk_words):
        starts_group = index > 0 and boundaries[index] != boundaries[index - 1]
        reference = rotator.take(words, starts_group)
        if reference is None:
            return None, []
        per_chunk.append(reference)
    return per_chunk, rotator.segments


def ensure_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
