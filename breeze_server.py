"""Persistent local HTTP server for Breeze TTS 2.

The model is loaded once at startup and reused for every request, so any
codebase on this machine can call it over plain HTTP. Which runtime loads it --
MLX on Apple Silicon, PyTorch on an NVIDIA GPU -- is decided in ``tts_backends``
and is invisible from here.

    python breeze_server.py --host 127.0.0.1 --port 7860

The checkpoint directory defaults to the one the active backend expects, and
can be given as a positional argument or in BREEZE_MODEL.

Endpoints
    GET  /                 -- browser UI
    GET  /health           -- readiness, sample rate, model path
    GET  /v1/capabilities  -- machine-readable parameter reference
    POST /v1/text/prepare  -- Gemini Flash Lite text preparation only
    POST /v1/audio/speech  -- synthesis (JSON or multipart/form-data)

Every /v1/audio/speech parameter is documented in PARAMETERS below, with an
example value for each. That table is also served from /v1/capabilities.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

import archive
import audio_out
import hotkeys
import llm_providers
import llm_stream
import platform_support
import speech_config
import speech_session
import tts_backends
import voice_store
from breeze_pipeline import (
    CHINESE_VOCAL_EVENTS,
    ENGLISH_VOCAL_EVENTS,
    MAX_WORDS_PER_CHUNK,
    BreezeEngine,
    chunk_document,
    validate_and_chunk_text,
)
from speech_pipeline import (
    STALL_FLUSH_SECONDS,
    ChunkPipe,
    TextAggregator,
    pump_llm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
# Read before any module-level default is computed from the environment: the
# provider registry, the backend choice and the model path are all resolved
# from it, and a .env read after that would be read too late to matter.
platform_support.load_env()

logger = logging.getLogger("breeze.server")

STATE: dict[str, Any] = {"engine": None, "model_path": None}
WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

# ---------------------------------------------------------------------------
# Parameter reference for POST /v1/audio/speech.
#
# Send these as a JSON object, or as multipart/form-data fields when you need to
# upload reference audio. In multipart every value is a string ("4.0", "true");
# the server coerces them. Unknown fields are ignored.
# ---------------------------------------------------------------------------
PARAMETERS: list[dict[str, Any]] = [
    # --- What to say -------------------------------------------------------
    {
        "name": "text",
        "type": "string",
        "default": None,
        "required": True,
        "example": "(sigh) Welcome aboard. Your journey begins now.",
        "doc": "The words to speak. Accepts the alias 'input'. Inline vocal "
               "events are allowed: "
               f"{' '.join(ENGLISH_VOCAL_EVENTS)} in English, "
               f"{' '.join(CHINESE_VOCAL_EVENTS)} in Chinese. Any other "
               "bracketed tag is read out loud literally. Long text is split "
               "automatically -- see max_words.",
    },
    # --- How it should sound ----------------------------------------------
    {
        "name": "instruction",
        "type": "string",
        "default": None,
        "example": "A deep, booming movie-trailer narrator, dramatic and intense.",
        "doc": "Plain-language description of the voice and delivery. On its "
               "own this is voice design (invent a voice from nothing). "
               "Combined with reference audio it becomes voice direction "
               "(keep the speaker, change the delivery). Supplying this is "
               "what unlocks cfg_scale.",
    },
    # --- Whose voice -------------------------------------------------------
    {
        "name": "ref_audio",
        "type": "file upload (multipart only)",
        "default": None,
        "example": "@reference_zh.wav",
        "doc": "Reference recording to clone the speaker from. A few seconds "
               "of clean speech is enough, and it may be in a different "
               "language than 'text' -- cross-lingual cloning works. Requires "
               "ref_text. Use multipart/form-data to send this.",
    },
    {
        "name": "ref_audio_path",
        "type": "string",
        "default": None,
        "example": "/Users/thomas/Documents/BreezeTTS2/reference_zh.wav",
        "doc": "Alternative to uploading: an absolute path the SERVER can "
               "read. Handy for JSON requests and for large files you do not "
               "want to send over the wire. Requires ref_text.",
    },
    {
        "name": "ref_text",
        "type": "string",
        "default": None,
        "example": "这是中文参考音频的准确文字稿。",
        "doc": "The exact transcript of the reference audio, word for word. "
               "Accuracy matters: a wrong transcript degrades the clone badly. "
               "Must be supplied together with ref_audio/ref_audio_path.",
    },
    # --- Guidance ----------------------------------------------------------
    {
        "name": "cfg_scale",
        "type": "float > 0",
        "default": 1.0,
        "example": 4.0,
        "doc": "Classifier-free guidance: how hard the model is pushed toward "
               "the instruction. ~4.0 is a good default when using an "
               "instruction; higher is more dramatic but can distort. Only "
               "meaningful in modes 'guided' and 'edit'. In 'plain' and "
               "'clone' there is no negative branch, so this is ignored "
               "(clamped to 1.0) rather than rejected.",
    },
    # --- Mode --------------------------------------------------------------
    {
        "name": "mode",
        "type": "auto | plain | guided | clone | edit",
        "default": "auto",
        "example": "edit",
        "doc": "Which prompt template to use. 'auto' picks from what you sent: "
               "nothing extra -> plain; instruction only -> guided (voice "
               "design); reference only -> clone; both -> edit (voice "
               "direction). Set it explicitly to force a template and get a "
               "clear error if the required fields are missing.",
    },
    # --- Text preparation --------------------------------------------------
    {
        "name": "prepare",
        "type": "boolean",
        "default": False,
        "example": True,
        "doc": "Run the text through Gemini Flash Lite on Vertex AI first. It "
               "fixes punctuation, breaks run-on sentences into lines, and "
               "inserts supported vocal events -- turning raw LLM or chat "
               "output into something that reads well aloud. It never "
               "translates or rewords. If Vertex is unreachable the request "
               "still succeeds using the raw text, and the reason is returned "
               "in the X-Breeze-Prep-Error header.",
    },
    {
        "name": "prep_instruction",
        "type": "string",
        "default": None,
        "example": "Keep it terse and clipped; this is a status update.",
        "doc": "Extra direction for the preparation model only. Does not "
               "affect the voice. Ignored unless prepare is true.",
    },
    # --- Output format -----------------------------------------------------
    {
        "name": "stream",
        "type": "boolean",
        "default": False,
        "example": True,
        "doc": "Stream audio as it is generated instead of waiting for the "
               "whole file. Emits raw PCM (see 'format'), first bytes in about "
               "0.2s. Use this for conversational latency; use the default "
               "for a file you are going to save.",
    },
    {
        "name": "format",
        "type": "wav | pcm | sse",
        "default": "wav",
        "example": "sse",
        "doc": "'wav' returns a complete RIFF file with a header. 'pcm' "
               "returns bare samples: signed 16-bit little-endian, 24000 Hz, "
               "mono, NO header -- feed it straight to an audio device. "
               "'sse' returns Server-Sent Events carrying base64 PCM plus "
               "sentence boundaries and a live realtime factor, which is what "
               "an adaptive player needs to widen its buffer without "
               "glitching. Setting stream=true implies pcm. Accepts the alias "
               "'response_format'.",
    },
    # --- Chunking ----------------------------------------------------------
    {
        "name": "max_words",
        "type": "integer",
        "default": MAX_WORDS_PER_CHUNK,
        "example": 35,
        "doc": "Word ceiling per synthesized chunk. Text is split on sentence "
               "endings first, then on clause boundaries for any sentence "
               "longer than this, and the pieces are rejoined with a 220 ms "
               "breath pause. Keep it near the default: on a 96-word sentence, "
               "chunked output ran 39s while unchunked ran 79s because the "
               "model drifted and padded.",
    },
    # --- Sampling ----------------------------------------------------------
    {
        "name": "seed",
        "type": "integer",
        "default": 42,
        "example": 1234,
        "doc": "Random seed. Reuse the same seed with identical settings to "
               "reproduce a take exactly; change it to draw a different "
               "delivery from the same description.",
    },
    {
        "name": "temperature",
        "type": "float > 0",
        "default": 0.9,
        "example": 0.7,
        "doc": "Sampling randomness. Lower is steadier and more predictable; "
               "higher is more expressive but more likely to wander. Applied "
               "to both the backbone and the depth decoder.",
    },
    {
        "name": "top_k",
        "type": "integer >= 0",
        "default": 50,
        "example": 40,
        "doc": "Keep only the k most likely tokens at each step. 0 disables "
               "the cutoff.",
    },
    {
        "name": "top_p",
        "type": "float in (0, 1]",
        "default": 1.0,
        "example": 0.95,
        "doc": "Nucleus sampling: keep the smallest set of tokens whose "
               "probabilities sum to p. 1.0 disables it.",
    },
    {
        "name": "greedy",
        "type": "boolean",
        "default": False,
        "example": True,
        "doc": "Always take the most likely token (argmax) instead of "
               "sampling. Fully deterministic, but tends to sound flatter. "
               "Overrides temperature/top_k/top_p in practice.",
    },
    {
        "name": "repetition_penalty",
        "type": "float > 0",
        "default": 1.1,
        "example": 1.2,
        "doc": "Discourages reusing tokens already generated. Raise it if the "
               "voice stutters, loops a syllable, or trails into babble; 1.0 "
               "disables it.",
    },
    # --- Limits ------------------------------------------------------------
    {
        "name": "max_new_tokens",
        "type": "integer > 0",
        "default": 1500,
        "example": 2000,
        "doc": "Hard ceiling on generated codec frames per chunk, i.e. the "
               "maximum audio length. Raise it only if long chunks are being "
               "cut off mid-word.",
    },
    {
        "name": "max_seq_len",
        "type": "integer > 0",
        "default": 2048,
        "example": 3072,
        "doc": "Maximum prompt length in tokens. A very long instruction plus "
               "a long reference transcript can hit this; the request fails "
               "rather than silently truncating.",
    },
    {
        "name": "codec_chunk_frames",
        "type": "integer > 0",
        "default": 2,
        "example": 1,
        "doc": "How many codec frames are decoded per step while streaming. "
               "Lower means smaller, more frequent audio packets (slightly "
               "lower latency, slightly more overhead); higher is more "
               "efficient but chunkier. Only affects streaming feel, not the "
               "final audio.",
    },
    # --- Saved voices ------------------------------------------------------
    {
        "name": "voice_id",
        "type": "string",
        "default": None,
        "required": False,
        "example": "voice_9f2c1a7b4e6d5c3a2b1f0e9d",
        "doc": "Speak with a saved voice profile. Every generation setting "
               "stored on that profile -- instruction, seed, temperature, "
               "top_k, top_p, cfg_scale, repetition_penalty and the rest -- is "
               "restored, and any field sent alongside overrides it for this "
               "request only. POST /v1/text-to-speech/{voice_id} is the same "
               "thing with the id in the path.",
    },
    {
        "name": "voice_mode",
        "type": "'anchor' | 'params'",
        "default": "anchor",
        "required": False,
        "example": "anchor",
        "doc": "How a saved voice is reproduced. 'anchor' conditions "
               "generation on the profile's stored reference recording and its "
               "exact transcript, which is what actually holds the speaker "
               "identical across text it has never seen. 'params' replays the "
               "saved settings only -- reproducible for identical text, but "
               "the voice drifts once the words change, because without a "
               "reference the model re-invents a speaker from the instruction.",
    },
    {
        "name": "rotation",
        "type": "boolean | object",
        "default": "the voice's own rotation settings",
        "required": False,
        "example": '{"enabled": true, "every_words": 1000}',
        "doc": "Tone rotation for a saved voice that holds more than one "
               "reference recording. A long read swaps between them so it does "
               "not flatten into a single unvarying tone -- same speaker, "
               "different colour. Switches land only on a paragraph start (or a "
               "sentence start, when the text has no blank lines to use), so a "
               "change is never audible mid-sentence. Pass false to turn it off "
               "for one request, or an object with any of 'enabled', 'mode' "
               "(random | sequence), 'every_words', 'boundary' (paragraph | "
               "sentence) and 'tags' to override the voice's stored settings. "
               "Ignored when the request brings its own reference audio.",
    },
    {
        "name": "voice_lock",
        "type": "boolean",
        "default": True,
        "required": False,
        "example": True,
        "doc": "Keep one voice across a chunked paragraph. The first chunk is "
               "generated normally and then becomes the reference the rest are "
               "conditioned on, so sentence two sounds like sentence one. Only "
               "engages when there are several chunks and no reference audio "
               "was supplied; a request that already has a reference is "
               "anchored throughout anyway. Set false to hear the raw "
               "per-sentence behaviour.",
    },
]

_PARAM_INDEX = {entry["name"]: entry for entry in PARAMETERS}
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _as_bool(value: Any, default: bool | None = False) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise HTTPException(400, f"Expected a boolean, got {value!r}")


def _as_float(value: Any, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Expected a number, got {value!r}") from exc


def _as_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Expected an integer, got {value!r}") from exc


def _engine() -> BreezeEngine:
    engine = STATE["engine"]
    if engine is None:
        raise HTTPException(503, "Model is still loading")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    voice_store.ensure_dirs()
    removed = voice_store.prune_previews()
    if removed:
        logger.info("Pruned %d expired voice preview(s)", removed)

    model_path = STATE["model_path"] or tts_backends.default_model_path()
    STATE["engine"] = BreezeEngine(
        model_path, audio_device=os.getenv("BREEZE_AUDIO_DEVICE", "auto")
    )
    STATE["model_path"] = str(Path(model_path).resolve())

    # Headphones disconnecting should stop the read, not redirect it to the
    # laptop speakers. Nothing here touches PortAudio, so it is safe to run for
    # the whole life of the process whether anything is speaking or not.
    audio_out.WATCHER.on_change(_on_output_device_change)
    audio_out.WATCHER.start()

    # Sized once now and once an hour after. Audio is the only part of the
    # archive with any weight, and it accumulates silently -- a warning that
    # only appears after you go looking is not a warning.
    archive.start_usage_monitor()
    yield
    audio_out.WATCHER.stop()
    _speak_cancel("cancelled")
    STATE["engine"] = None


app = FastAPI(title="Breeze TTS 2", version="1.2.0", lifespan=lifespan)


async def _read_params(request: Request) -> tuple[dict[str, Any], bytes | None]:
    """Accept JSON or multipart/form-data; return (fields, ref_audio_bytes)."""
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "JSON body must be an object")
        return payload, None

    form = await request.form()
    fields: dict[str, Any] = {}
    ref_bytes: bytes | None = None
    for key, value in form.multi_items():
        if hasattr(value, "read"):  # UploadFile
            if key == "ref_audio":
                ref_bytes = await value.read()
            continue
        fields[key] = value
    return fields, ref_bytes


def _pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _synthesize_events(
    engine: BreezeEngine, chunks: list[str], options: dict[str, Any]
) -> Iterator[tuple[str, np.ndarray | None, int]]:
    """Yield ("audio", samples, chunk_index) and ("boundary", None, chunk_index).

    A boundary marks the end of one sentence chunk. Clients cannot detect these
    in a raw PCM byte stream, but they are exactly where a player can pause to
    rebuild its buffer without the listener hearing a glitch.

    ``voice_lock`` travels inside ``options``; the engine applies it and the
    resolved seed, so every caller here gets the same reproducibility.

    The request id is unique per call. Reusing "http-0" for every request meant
    an aborted generation whose cleanup had not run yet collided with the next
    one by name, and a late cleanup could close a request that a *live*
    generation was using.
    """
    request_id = f"http-{uuid.uuid4().hex[:8]}"
    yield from engine.stream_document(chunks, request_id=request_id, **options)


def _synthesize_stream(
    engine: BreezeEngine, chunks: list[str], options: dict[str, Any]
) -> Iterator[np.ndarray]:
    """Audio only, for the WAV and raw-PCM responses."""
    for kind, audio, _index in _synthesize_events(engine, chunks, options):
        if kind == "audio" and audio is not None:
            yield audio


_EXHAUSTED = object()


async def _drain(
    generator: Iterator[bytes], cleanup: Callable[[], None]
) -> AsyncIterator[bytes]:
    """Pump a synthesis generator from the event loop and always close it.

    Starlette would otherwise iterate the generator in a worker thread and, when
    the client disconnects, simply stop asking for items -- leaving the
    generator suspended mid-yield with its ``finally`` blocks unrun until the
    garbage collector notices. That is what left a codec request open and made
    the *next* request fail with "already active request".

    Owning it here fixes the timing: the disconnect cancels the await below,
    which runs this ``finally``, which closes the generator, which runs the
    engine's own cleanup -- releasing the codec request, the engine lock and the
    voice-lock anchor file before the next request starts.
    """
    try:
        while True:
            item = await run_in_threadpool(next, generator, _EXHAUSTED)
            if item is _EXHAUSTED:
                break
            yield item
    finally:
        # Closed inline rather than awaited: this runs while the request is
        # already being cancelled, and an await here would be cancelled too,
        # skipping the cleanup entirely. Closing the generator only unwinds its
        # finally blocks -- release the gate, close the codec request, unlink
        # the anchor file -- so it costs microseconds on the event loop.
        try:
            generator.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            logger.exception("Failed to close an aborted synthesis stream")
        cleanup()


def _collect_audio(
    engine: BreezeEngine, chunks: list[str], options: dict[str, Any]
) -> np.ndarray:
    """Render a whole document to one float32 array."""
    parts = [audio for audio in _synthesize_stream(engine, chunks, options)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    """One Server-Sent Event. Base64 keeps audio newline-free for the wire."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _sse_stream(
    engine: BreezeEngine,
    chunks: list[str],
    options: dict[str, Any],
    meta: dict[str, Any],
    collect: list[np.ndarray] | None = None,
    events: Iterator[tuple[str, np.ndarray | None, int]] | None = None,
) -> Iterator[bytes]:
    """Audio plus live timing, so a client can schedule playback adaptively.

    Every event carries the realtime factor measured so far (audio produced /
    wall time). Above 1.0 the model is outrunning playback; near or below 1.0 a
    player should widen its buffer, ideally at a boundary.

    ``collect`` receives every audio block, so a caller streaming a voice
    preview can keep the finished take without generating it twice.

    ``events`` substitutes an already-built event iterator, which is how the
    system-speech path -- whose chunks are still arriving from a language model
    -- reuses this wire format unchanged.
    """
    sample_rate = engine.sample_rate
    started = time.perf_counter()
    total_samples = 0
    seq = 0

    source = events if events is not None else _synthesize_events(engine, chunks, options)
    total_chunks = len(chunks) if chunks else 0

    yield _sse("start", {"sample_rate": sample_rate, **meta})
    try:
        for kind, audio, index in source:
            elapsed = time.perf_counter() - started
            audio_seconds = total_samples / sample_rate
            rtf = (audio_seconds / elapsed) if elapsed > 0 else 0.0

            if kind == "audio" and audio is not None:
                if collect is not None:
                    collect.append(audio)
                total_samples += audio.size
                yield _sse(
                    "audio",
                    {
                        "seq": seq,
                        "chunk": index,
                        "samples": int(audio.size),
                        "pcm": base64.b64encode(_pcm16(audio)).decode("ascii"),
                        "audio_seconds": round(total_samples / sample_rate, 3),
                        "elapsed": round(elapsed, 3),
                        "rtf": round(rtf, 3),
                    },
                )
                seq += 1
            elif kind == "boundary":
                yield _sse(
                    "boundary",
                    {
                        "chunk": index,
                        "chunks_total": total_chunks,
                        "audio_seconds": round(audio_seconds, 3),
                        "elapsed": round(elapsed, 3),
                        "rtf": round(rtf, 3),
                    },
                )
    except Exception as exc:  # noqa: BLE001 - the stream is already committed
        logger.exception("Streaming synthesis failed")
        yield _sse("error", {"message": str(exc)})
        return

    elapsed = time.perf_counter() - started
    audio_seconds = total_samples / sample_rate
    yield _sse(
        "end",
        {
            "audio_seconds": round(audio_seconds, 3),
            "elapsed": round(elapsed, 3),
            "rtf": round(audio_seconds / elapsed, 3) if elapsed > 0 else 0.0,
        },
    )


@app.get("/")
def index() -> FileResponse:
    """Browser UI. The API is fully usable without it."""
    if not WEB_INDEX.is_file():
        raise HTTPException(404, "web/index.html is missing")
    return FileResponse(WEB_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict[str, Any]:
    engine = STATE["engine"]
    payload = {
        "status": "ok" if engine is not None else "loading",
        "model_path": STATE["model_path"],
        "sample_rate": engine.sample_rate if engine is not None else None,
        "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
        "platform": platform_support.describe(),
        "backend": engine.backend.NAME if engine is not None else None,
    }
    if engine is not None:
        payload["memory"] = engine.memory_stats()
        payload["engine"] = engine.backend.describe()
    return payload


@app.post("/v1/memory/trim")
def trim_memory_endpoint() -> dict[str, Any]:
    """Release cached-but-free allocator blocks. The model stays loaded.

    Generation trims on its own; this is for checking the effect by hand.
    """
    engine = _engine()
    before = engine.memory_stats()
    released = engine.trim_memory()
    return {"released": released, "before": before, "after": engine.memory_stats()}


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    """Machine-readable parameter reference -- the same table as PARAMETERS."""
    return {
        "endpoint": "/v1/audio/speech",
        "content_types": ["application/json", "multipart/form-data"],
        "sample_rate": STATE["engine"].sample_rate if STATE["engine"] else None,
        "pcm_encoding": "signed 16-bit little-endian, mono, no header",
        "modes": {
            "plain": "no instruction, no reference; cfg ignored",
            "guided": "instruction only -- voice design; cfg applies",
            "clone": "reference only -- voice clone; cfg ignored",
            "edit": "reference + instruction -- voice direction; cfg applies",
        },
        "vocal_events": {
            "english": list(ENGLISH_VOCAL_EVENTS),
            "chinese": list(CHINESE_VOCAL_EVENTS),
        },
        "parameters": PARAMETERS,
        "engine": {
            "backend": STATE["engine"].backend.NAME if STATE["engine"] else None,
            "backends": tts_backends.survey(),
            "platform": platform_support.describe(),
        },
        "llm": {
            "provider": llm_stream.status().get("provider"),
            "providers": llm_providers.describe_all(),
            "note": (
                "Set BREEZE_LLM_PROVIDER in .env. Unset, the first provider "
                "with a credential present is used, API keys before Vertex."
            ),
        },
        "system_speech": {
            "speak": "POST /v1/speak",
            "toggle": "POST /v1/speak/toggle",
            "stop": "POST /v1/speak/stop",
            "skip": "POST /v1/speak/skip  {delta: +1 | -1}",
            "resume": "POST /v1/speak/resume",
            "status": "GET /v1/speak/status",
            "hotkeys": "GET|PUT /v1/hotkeys",
            "hotkey_host": hotkeys.host(),
            "note": (
                "One hotkey drives toggle: it silences a read in progress, and "
                "the next press starts a new one from the clipboard. Skips move "
                "whole paragraphs and are debounced, so holding the key scrolls "
                "the document rather than synthesizing every paragraph passed. "
                "Losing the output device pauses the read where it stands; a "
                "skip press picks it up again. The shortcuts that drive all of "
                "this are configured at /v1/hotkeys and read from there by "
                "whichever hotkey host this platform uses."
            ),
        },
        "voices": {
            "design": "POST /v1/voice-previews/design",
            "clone": "POST /v1/voice-previews/clone",
            "audition": "GET /v1/voice-previews/{generated_voice_id}/stream",
            "save": "POST /v1/voice-previews/{generated_voice_id}/save",
            "save_recording": "POST /v1/voices/from-recording",
            "list": "GET /v1/voices",
            "read": "GET /v1/voices/{voice_id}",
            "sample": "GET /v1/voices/{voice_id}/sample",
            "edit": "PATCH /v1/voices/{voice_id}",
            "settings": "PATCH /v1/voices/{voice_id}/settings",
            "tags": "PUT /v1/voices/{voice_id}/tags",
            "favorite": "PUT|DELETE /v1/voices/{voice_id}/favorite",
            "delete": "DELETE /v1/voices/{voice_id}",
            "speak": "POST /v1/text-to-speech/{voice_id}",
            "speak_stream": "POST /v1/text-to-speech/{voice_id}/stream",
            "metadata_options": "GET /v1/voice-metadata/options",
            "preview_count_max": PREVIEW_COUNT_MAX,
            "default_preview_script": DEFAULT_PREVIEW_SCRIPT,
        },
        "reproducibility": {
            "default_seed": STATE["engine"].default_seed if STATE["engine"] else None,
            "note": (
                "A seed makes a take repeatable for identical text. It does not "
                "carry a voice across different text: with no reference "
                "recording the model invents a speaker from the instruction and "
                "the words in front of it. Save the take you like as a voice "
                "and call it with voice_mode=anchor (the default) to keep the "
                "speaker on text it has never seen."
            ),
        },
    }


@app.post("/v1/text/prepare")
async def prepare_endpoint(request: Request) -> JSONResponse:
    """Run text preparation on its own, without synthesizing anything."""
    fields, _ = await _read_params(request)
    text = (fields.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Field 'text' is required")

    from text_prep import prepare_text

    # Network call: keep it off the event loop.
    result = await run_in_threadpool(prepare_text, text, fields.get("prep_instruction"))
    if not result["success"]:
        raise HTTPException(502, f"Text preparation failed: {result['error']}")
    return JSONResponse({"text": result["text"], "original": text})


# ---------------------------------------------------------------------------
# Synthesis core, shared by /v1/audio/speech and /v1/text-to-speech/{voice_id}
# ---------------------------------------------------------------------------

# Everything that reaches BreezeEngine.stream_chunk as a sampling knob.
# instruction, mode and max_words are handled separately: the first two decide
# the template, the third decides chunking rather than generation.
_SAMPLING_KEYS = (
    "cfg_scale", "seed", "temperature", "top_k", "top_p", "greedy",
    "repetition_penalty", "max_new_tokens", "max_seq_len", "codec_chunk_frames",
)


def _generation_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Parse the sampling knobs. None means 'this request did not specify it'.

    Kept distinct from 'specified as the default value' so a saved voice can
    fill the gaps without overriding anything the caller actually asked for.
    """
    return {
        "cfg_scale": _as_float(
            fields.get("cfg_scale", fields.get("guidance_scale")), None
        ),
        "seed": _as_int(fields.get("seed"), None),
        "temperature": _as_float(fields.get("temperature"), None),
        "top_k": _as_int(fields.get("top_k"), None),
        "top_p": _as_float(fields.get("top_p"), None),
        "greedy": _as_bool(fields.get("greedy"), None),
        "repetition_penalty": _as_float(fields.get("repetition_penalty"), None),
        "max_new_tokens": _as_int(fields.get("max_new_tokens"), None),
        "max_seq_len": _as_int(fields.get("max_seq_len"), None),
        "codec_chunk_frames": _as_int(fields.get("codec_chunk_frames"), None),
    }


def _materialize_reference(
    fields: dict[str, Any], ref_bytes: bytes | None
) -> tuple[Path | None, bool]:
    """Return (path, is_temporary). Uploads must hit disk for the tokenizer."""
    if ref_bytes:
        tmp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.write(ref_bytes)
        tmp_file.flush()
        tmp_file.close()
        return Path(tmp_file.name), True
    if fields.get("ref_audio_path"):
        path = Path(str(fields["ref_audio_path"])).expanduser()
        if not path.is_file():
            raise HTTPException(400, f"ref_audio_path not found: {path}")
        return path, False
    return None, False


def _rotation_override(fields: dict[str, Any]) -> dict[str, Any] | None:
    """Per-request rotation settings, on top of whatever the voice stores.

    ``rotation`` accepts an object, or a plain boolean for the common case of
    turning it off for one request.
    """
    value = fields.get("rotation")
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    parsed = _as_bool(value, None)
    if parsed is None:
        raise HTTPException(400, "rotation must be a boolean or an object")
    return {"enabled": parsed}


def _load_voice(voice_id: str) -> dict[str, Any]:
    try:
        return voice_store.get_voice(voice_id)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc


def _build_job(
    fields: dict[str, Any],
    ref_bytes: bytes | None,
    *,
    text: str,
    voice_id: str | None = None,
) -> dict[str, Any]:
    """Resolve one synthesis request into chunks, options and headers.

    A saved voice contributes two things: its stored settings, which fill in
    every knob the request left unset, and -- in the default ``anchor`` mode --
    its reference recording, which is what actually keeps the speaker stable on
    text the voice has never spoken.
    """
    engine = _engine()
    generation = _generation_fields(fields)

    instruction_given = "instruction" in fields
    instruction = (fields.get("instruction") or None) if instruction_given else None
    mode = (fields.get("mode") or "").strip() or None
    max_words = _as_int(fields.get("max_words"), None)
    voice_lock = _as_bool(fields.get("voice_lock"), True)

    ref_path, uploaded = _materialize_reference(fields, ref_bytes)
    ref_text = fields.get("ref_text") or None

    voice_id = voice_id or (fields.get("voice_id") or None)
    voice_meta: dict[str, Any] = {}

    if voice_id:
        profile = _load_voice(voice_id)
        saved = profile.get("generation") or {}
        voice_mode = (fields.get("voice_mode") or "anchor").strip().lower()
        if voice_mode not in {"anchor", "params"}:
            raise HTTPException(
                400, f"voice_mode must be 'anchor' or 'params', got {voice_mode!r}"
            )

        for key in _SAMPLING_KEYS:
            if generation[key] is None and saved.get(key) is not None:
                generation[key] = saved[key]
        if not instruction_given:
            instruction = saved.get("instruction")
        if max_words is None and saved.get("max_words") is not None:
            max_words = saved["max_words"]

        if voice_mode == "anchor":
            if ref_path is None:
                reference = voice_store.voice_reference(voice_id)
                if reference is None:
                    raise HTTPException(
                        400,
                        f"Voice {voice_id} has no reference recording; "
                        "call it with voice_mode=params instead",
                    )
                ref_path, ref_text = reference[0], reference[1]
            # The reference branch of the same template pair the voice was
            # designed with: plain -> clone, guided -> edit.
            mode = mode or ("edit" if instruction else "clone")
        else:
            mode = mode or saved.get("mode") or "auto"
            if mode in {"clone", "edit"} and ref_path is None:
                # A cloned voice *is* its recording; there is no parameter-only
                # version of it to replay.
                raise HTTPException(
                    400,
                    f"Voice {voice_id} was cloned from a recording, so "
                    "voice_mode=params has nothing to reproduce it from. Use "
                    "voice_mode=anchor.",
                )

        voice_meta = {
            "voice_id": voice_id,
            "voice_name": profile.get("name"),
            "voice_mode": voice_mode,
            "voice_origin": profile.get("origin"),
        }

    if generation["cfg_scale"] is None:
        generation["cfg_scale"] = 1.0
    mode = mode or "auto"
    if max_words is None:
        max_words = MAX_WORDS_PER_CHUNK

    options = {
        "instruction": instruction,
        "ref_audio": str(ref_path) if ref_path else None,
        "ref_text": ref_text,
        "mode": mode,
        "voice_lock": voice_lock,
        **generation,
    }

    chunks, paragraph_ids = chunk_document(text, max_words=max_words)
    if not chunks:
        raise HTTPException(400, "No synthesizable text after chunking")

    # Reference rotation: a long read swaps between recordings of the same
    # speaker so it does not flatten into one tone. Only meaningful when a saved
    # voice supplies the references, and only when the request did not bring a
    # reference of its own.
    rotation_segments: list[dict[str, Any]] = []
    chunk_refs: list[tuple[str, str]] | None = None
    if voice_id and voice_meta.get("voice_mode") == "anchor" and not uploaded:
        override = _rotation_override(fields)
        try:
            chunk_refs, rotation_segments = voice_store.build_reference_plan(
                voice_id,
                [len(chunk.split()) for chunk in chunks],
                paragraph_ids,
                rotation=override,
                seed=generation["seed"],
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def cleanup() -> None:
        if uploaded and ref_path is not None:
            ref_path.unlink(missing_ok=True)

    # Validate before committing to a streaming response, so bad requests get a
    # JSON 400 instead of a truncated audio stream. This touches no model state,
    # so it is safe to run while another request is generating.
    try:
        resolved_mode, _, effective_cfg = engine.validate(
            instruction=instruction,
            ref_audio=options["ref_audio"],
            ref_text=ref_text,
            cfg_scale=generation["cfg_scale"],
            mode=mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        cleanup()
        raise HTTPException(400, str(exc)) from exc

    headers = {
        "X-Breeze-Chunks": str(len(chunks)),
        "X-Breeze-Sample-Rate": str(engine.sample_rate),
        "X-Breeze-Mode": resolved_mode,
        "X-Breeze-Cfg-Scale": f"{effective_cfg:g}",
        "X-Breeze-Seed": str(options["seed"] if options["seed"] is not None
                             else engine.default_seed),
    }
    if voice_meta:
        headers["X-Breeze-Voice-Id"] = voice_meta["voice_id"]
        headers["X-Breeze-Voice-Mode"] = voice_meta["voice_mode"]
    if rotation_segments:
        headers["X-Breeze-Reference-Rotation"] = str(len(rotation_segments))

    # These are named parameters of stream_document rather than per-chunk
    # generation options, so they ride along in the same dict without ever
    # reaching stream_chunk.
    options["chunk_refs"] = chunk_refs
    options["paragraph_ids"] = paragraph_ids

    return {
        "engine": engine,
        "chunks": chunks,
        "paragraph_ids": paragraph_ids,
        "options": options,
        "headers": headers,
        "cleanup": cleanup,
        "resolved_mode": resolved_mode,
        "effective_cfg": effective_cfg,
        "voice": voice_meta,
        "rotation": rotation_segments,
    }


async def _prepare_text_field(fields: dict[str, Any], text: str) -> tuple[str, str | None]:
    """Optional Gemini pass. A failure is reported, never fatal."""
    if not _as_bool(fields.get("prepare"), False):
        return text, None

    from text_prep import prepare_text

    result = await run_in_threadpool(prepare_text, text, fields.get("prep_instruction"))
    if result["success"]:
        return result["text"], None
    logger.warning("Continuing with raw text; prep failed: %s", result["error"])
    return text, result["error"]


async def _respond(
    job: dict[str, Any],
    fields: dict[str, Any],
    prep_error: str | None,
    *,
    default_format: str = "wav",
) -> StreamingResponse:
    """Render a prepared job as WAV, raw PCM, or an SSE event stream."""
    engine = job["engine"]
    chunks = job["chunks"]
    options = job["options"]
    cleanup = job["cleanup"]
    headers = dict(job["headers"])
    if prep_error:
        headers["X-Breeze-Prep-Error"] = prep_error[:200].replace("\n", " ")

    audio_format = (
        fields.get("format") or fields.get("response_format") or default_format
    ).strip().lower()
    if audio_format not in {"wav", "pcm", "sse"}:
        cleanup()
        raise HTTPException(
            400, f"format must be 'wav', 'pcm' or 'sse', got {audio_format!r}"
        )
    want_stream = _as_bool(fields.get("stream"), False)

    if audio_format == "sse":
        # Audio plus boundary/timing events, for an adaptive client-side queue.
        events = _sse_stream(
            engine,
            chunks,
            options,
            {
                "chunks": len(chunks),
                "mode": job["resolved_mode"],
                "cfg_scale": job["effective_cfg"],
                "prep_error": prep_error,
                **job["voice"],
            },
        )
        return StreamingResponse(
            _drain(events, cleanup),
            media_type="text/event-stream",
            headers={
                **headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # never buffer an event stream
            },
        )

    if want_stream or audio_format == "pcm":
        def pcm_iter() -> Iterator[bytes]:
            for audio in _synthesize_stream(engine, chunks, options):
                yield _pcm16(audio)

        headers["X-Breeze-Format"] = f"pcm_s16le_{engine.sample_rate}hz_mono"
        return StreamingResponse(
            _drain(pcm_iter(), cleanup), media_type="audio/L16", headers=headers
        )

    # Generation is blocking and GPU-bound; running it in the threadpool keeps
    # the event loop free to serve /health and flush other responses.
    try:
        audio = await run_in_threadpool(_collect_audio, engine, chunks, options)
    finally:
        cleanup()
    if not audio.size:
        raise HTTPException(500, "Model produced no audio")

    payload = _wav_bytes(audio, engine.sample_rate)
    headers["X-Breeze-Duration"] = f"{audio.size / engine.sample_rate:.2f}"
    return StreamingResponse(io.BytesIO(payload), media_type="audio/wav", headers=headers)


@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request) -> StreamingResponse:
    """Synthesize speech. See PARAMETERS above for every accepted field."""
    fields, ref_bytes = await _read_params(request)

    text = (fields.get("text") or fields.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "Field 'text' is required")

    text, prep_error = await _prepare_text_field(fields, text)
    job = _build_job(fields, ref_bytes, text=text)
    return await _respond(job, fields, prep_error)


@app.post("/v1/text-to-speech/{voice_id}")
async def tts_voice_endpoint(voice_id: str, request: Request) -> StreamingResponse:
    """Speak with a saved voice. Identical body to /v1/audio/speech, minus the id."""
    fields, ref_bytes = await _read_params(request)

    text = (fields.get("text") or fields.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "Field 'text' is required")

    text, prep_error = await _prepare_text_field(fields, text)
    job = _build_job(fields, ref_bytes, text=text, voice_id=voice_id)
    return await _respond(job, fields, prep_error)


@app.post("/v1/text-to-speech/{voice_id}/stream")
async def tts_voice_stream_endpoint(voice_id: str, request: Request) -> StreamingResponse:
    """Streaming twin of the above; defaults to raw PCM for a lower time to first audio."""
    fields, ref_bytes = await _read_params(request)

    text = (fields.get("text") or fields.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "Field 'text' is required")

    text, prep_error = await _prepare_text_field(fields, text)
    job = _build_job(fields, ref_bytes, text=text, voice_id=voice_id)
    return await _respond(job, fields, prep_error, default_format="pcm")


# ---------------------------------------------------------------------------
# Voice previews: design and clone candidates, auditioned before they are kept
# ---------------------------------------------------------------------------

# Long enough to clone from (~18 s), and deliberately varied: statement,
# question and exclamation, mixed sentence lengths, a wide spread of vowels and
# consonants. A flat sample clones as a flat voice.
DEFAULT_PREVIEW_SCRIPT = (
    "Every clear morning I walk the same quiet road down to the harbour, just "
    "to watch the boats come in. Did you know the water turns almost silver an "
    "hour before sunrise? It is strange, joyful work: counting the waves, "
    "naming the gulls, and forgetting the time completely."
)

PREVIEW_COUNT_MAX = 20

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # never buffer an event stream
}


def _wants_sse(fields: dict[str, Any]) -> bool:
    return (
        fields.get("format") or fields.get("response_format") or ""
    ).strip().lower() == "sse"


def _require_single_take(count: int) -> None:
    """Streaming returns one take: the events of several would interleave."""
    if count != 1:
        raise HTTPException(
            400,
            "format=sse streams a single take; request preview_count=1 per call "
            "and vary the seed yourself",
        )


def _preview_generation(job: dict[str, Any], seed: int) -> dict[str, Any]:
    """The exact settings that produced a candidate, for the profile to keep."""
    options = job["options"]
    return {
        "instruction": options.get("instruction"),
        "mode": job["resolved_mode"],
        "cfg_scale": job["effective_cfg"],
        "seed": seed,
        "temperature": options.get("temperature"),
        "top_k": options.get("top_k"),
        "top_p": options.get("top_p"),
        "greedy": options.get("greedy"),
        "repetition_penalty": options.get("repetition_penalty"),
        "max_new_tokens": options.get("max_new_tokens"),
        "max_seq_len": options.get("max_seq_len"),
        "codec_chunk_frames": options.get("codec_chunk_frames"),
        # Set by the caller, which knows what the request asked to chunk at.
        "max_words": None,
    }


async def _render_candidates(
    fields: dict[str, Any],
    *,
    text: str,
    count: int,
    base_seed: int,
    origin: str,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Generate ``count`` takes of one script, each with its own seed.

    Every candidate is rendered with voice lock on, so a preview long enough to
    clone from speaks in one voice from beginning to end rather than drifting
    between its own sentences.
    """
    engine = _engine()
    previews: list[dict[str, Any]] = []

    for index in range(count):
        seed = base_seed + index
        candidate_fields = {**fields, "seed": seed, "voice_lock": True}
        candidate_fields.pop("voice_id", None)
        if reference_audio is not None:
            candidate_fields["ref_audio_path"] = str(reference_audio)
            candidate_fields["ref_text"] = reference_text

        job = _build_job(candidate_fields, None, text=text)
        try:
            audio = await run_in_threadpool(
                _collect_audio, engine, job["chunks"], job["options"]
            )
        finally:
            job["cleanup"]()
        if not audio.size:
            raise HTTPException(500, "Model produced no audio")

        generation = _preview_generation(job, seed)
        generation["max_words"] = _as_int(fields.get("max_words"), MAX_WORDS_PER_CHUNK)
        preview = voice_store.save_preview(
            audio,
            engine.sample_rate,
            text=text,
            generation=generation,
            origin=origin,
            name=name,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
        preview["audio_base_64"] = base64.b64encode(
            _wav_bytes(audio, engine.sample_rate)
        ).decode("ascii")
        preview["stream_url"] = (
            f"/v1/voice-previews/{preview['generated_voice_id']}/stream"
        )
        previews.append(preview)

    return previews


def _candidate_job(
    fields: dict[str, Any],
    *,
    text: str,
    seed: int,
    reference_audio: Path | None,
    reference_text: str | None,
) -> dict[str, Any]:
    """Prepare one candidate take. Voice lock on, so the preview is one voice."""
    candidate_fields = {**fields, "seed": seed, "voice_lock": True}
    candidate_fields.pop("voice_id", None)
    if reference_audio is not None:
        candidate_fields["ref_audio_path"] = str(reference_audio)
        candidate_fields["ref_text"] = reference_text
    return _build_job(candidate_fields, None, text=text)


def _preview_sse(
    fields: dict[str, Any],
    *,
    text: str,
    seed: int,
    origin: str,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    name: str | None = None,
) -> Iterator[bytes]:
    """Stream one candidate as it is generated, then save and announce it.

    Same event shape as /v1/audio/speech, plus a ``preview`` event carrying the
    ``generated_voice_id`` just before ``end`` -- so the browser can play the
    take live and still have something it can save afterwards, from a single
    generation.
    """
    engine = _engine()
    job = _candidate_job(
        fields,
        text=text,
        seed=seed,
        reference_audio=reference_audio,
        reference_text=reference_text,
    )
    collected: list[np.ndarray] = []
    try:
        frames = _sse_stream(
            engine,
            job["chunks"],
            job["options"],
            {
                "chunks": len(job["chunks"]),
                "mode": job["resolved_mode"],
                "cfg_scale": job["effective_cfg"],
                "prep_error": None,
                "seed": seed,
                "origin": origin,
            },
            collect=collected,
        )
        for frame in frames:
            # Save on the way past the final event, so the id reaches the
            # client while the connection is still open.
            if frame.startswith(b"event: end") and collected:
                generation = _preview_generation(job, seed)
                generation["max_words"] = _as_int(
                    fields.get("max_words"), MAX_WORDS_PER_CHUNK
                )
                preview = voice_store.save_preview(
                    np.concatenate(collected),
                    engine.sample_rate,
                    text=text,
                    generation=generation,
                    origin=origin,
                    name=name,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                )
                preview["stream_url"] = (
                    f"/v1/voice-previews/{preview['generated_voice_id']}/stream"
                )
                yield _sse("preview", preview)
            yield frame
    finally:
        job["cleanup"]()


@app.post("/v1/voice-previews/design")
async def design_previews(request: Request) -> JSONResponse:
    """Invent voices from a description. Returns one candidate per requested take.

    Each candidate uses ``seed + n``, so every take is a different voice and
    every take is reproducible on its own seed. Nothing is persisted as a voice
    until one is saved.
    """
    fields, _ = await _read_params(request)

    description = (
        fields.get("voice_description") or fields.get("instruction") or ""
    ).strip()
    if not description:
        raise HTTPException(400, "Field 'voice_description' is required")

    text = (fields.get("text") or DEFAULT_PREVIEW_SCRIPT).strip()
    count = _as_int(fields.get("preview_count"), 1) or 1
    if not 1 <= count <= PREVIEW_COUNT_MAX:
        raise HTTPException(
            400, f"preview_count must be between 1 and {PREVIEW_COUNT_MAX}"
        )
    base_seed = _as_int(fields.get("seed"), None)
    if base_seed is None:
        base_seed = _engine().default_seed

    if _wants_sse(fields):
        _require_single_take(count)
        return StreamingResponse(
            _drain(
                _preview_sse(
                    {**fields, "instruction": description},
                    text=text,
                    seed=base_seed,
                    origin="designed",
                    name=fields.get("name") or None,
                ),
                voice_store.prune_previews,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    previews = await _render_candidates(
        {**fields, "instruction": description},
        text=text,
        count=count,
        base_seed=base_seed,
        origin="designed",
        name=fields.get("name") or None,
    )
    voice_store.prune_previews()
    return JSONResponse(
        {
            "previews": previews,
            "text": text,
            "voice_description": description,
            "sample_rate": _engine().sample_rate,
        }
    )


@app.post("/v1/voice-previews/clone")
async def clone_previews(request: Request) -> JSONResponse:
    """Clone a voice from a recording, and audition it on a script.

    Unlike the hosted API this runtime has no speech recognition, so the exact
    transcript of the recording must be supplied as ``ref_text``. A wrong
    transcript degrades the clone noticeably.
    """
    fields, ref_bytes = await _read_params(request)

    reference_text = (fields.get("ref_text") or "").strip()
    if not reference_text:
        raise HTTPException(
            400,
            "Field 'ref_text' is required: this runtime cannot transcribe the "
            "recording, so it needs the exact words spoken in it",
        )

    ref_path, uploaded = _materialize_reference(fields, ref_bytes)
    if ref_path is None:
        raise HTTPException(400, "Upload 'ref_audio' or pass 'ref_audio_path'")

    text = (fields.get("text") or DEFAULT_PREVIEW_SCRIPT).strip()
    count = _as_int(fields.get("preview_count"), 1) or 1
    if not 1 <= count <= PREVIEW_COUNT_MAX:
        raise HTTPException(
            400, f"preview_count must be between 1 and {PREVIEW_COUNT_MAX}"
        )
    base_seed = _as_int(fields.get("seed"), None)
    if base_seed is None:
        base_seed = _engine().default_seed

    duration = voice_store.audio_duration(ref_path)
    warnings: list[str] = []
    if duration and duration < voice_store.REFERENCE_MIN_SECONDS:
        warnings.append(
            f"Reference is {duration:.1f}s. Cloning works from about "
            f"{voice_store.REFERENCE_MIN_SECONDS:.0f}s; "
            f"{voice_store.REFERENCE_GOOD_SECONDS:.0f}s or more is better."
        )

    if _wants_sse(fields):
        _require_single_take(count)

        def finished() -> None:
            if uploaded:
                ref_path.unlink(missing_ok=True)
            voice_store.prune_previews()

        return StreamingResponse(
            _drain(
                _preview_sse(
                    fields,
                    text=text,
                    seed=base_seed,
                    origin="cloned",
                    reference_audio=ref_path,
                    reference_text=reference_text,
                    name=fields.get("name") or None,
                ),
                finished,
            ),
            media_type="text/event-stream",
            headers={**_SSE_HEADERS, "X-Breeze-Reference-Seconds": f"{duration:.2f}"},
        )

    try:
        previews = await _render_candidates(
            fields,
            text=text,
            count=count,
            base_seed=base_seed,
            origin="cloned",
            reference_audio=ref_path,
            reference_text=reference_text,
            name=fields.get("name") or None,
        )
    finally:
        if uploaded:
            ref_path.unlink(missing_ok=True)

    voice_store.prune_previews()
    return JSONResponse(
        {
            "previews": previews,
            "text": text,
            "reference_seconds": round(duration, 2),
            "warnings": warnings,
            "sample_rate": _engine().sample_rate,
        }
    )


@app.get("/v1/voice-previews/{generated_voice_id}/stream")
def stream_preview(generated_voice_id: str) -> FileResponse:
    """Audition a candidate that has not been saved yet."""
    try:
        path = voice_store.preview_audio_path(generated_voice_id)
    except voice_store.PreviewNotFound as exc:
        raise HTTPException(404, f"No such preview: {generated_voice_id}") from exc
    return FileResponse(path, media_type="audio/wav")


@app.post("/v1/voice-previews/{generated_voice_id}/save")
async def save_preview_endpoint(
    generated_voice_id: str, request: Request
) -> JSONResponse:
    """Persist a candidate as a reusable voice profile."""
    fields, _ = await _read_params(request)

    name = (fields.get("voice_name") or fields.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Field 'voice_name' is required")

    metadata = _voice_metadata(fields)

    try:
        voice = voice_store.create_voice_from_preview(
            generated_voice_id, name=name, metadata=metadata
        )
    except voice_store.PreviewNotFound as exc:
        raise HTTPException(404, f"No such preview: {generated_voice_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(voice, status_code=201)


def _voice_metadata(fields: dict[str, Any]) -> dict[str, Any]:
    """The library fields of a save request, whatever it is saving."""
    metadata: dict[str, Any] = {}
    if "voice_description" in fields or "description" in fields:
        metadata["description"] = fields.get("voice_description") or fields.get(
            "description"
        )
    for key in ("notes", "language_code", "gender", "age", "tone", "accent", "tags"):
        if key in fields:
            metadata[key] = fields[key]
    return metadata


@app.post("/v1/voices/from-recording")
async def create_voice_from_recording_endpoint(request: Request) -> JSONResponse:
    """Save a recording as a voice directly, without generating a take first.

    The takes a clone produces are an audition, not the voice: every later
    request is conditioned on the *recording*, so the take was only ever there
    to check the clone was right. When it already is, this skips them -- upload,
    transcript, optional direction, saved.

    Same body as ``/v1/voice-previews/clone`` minus the preview script: an
    ``ref_audio`` upload or ``ref_audio_path``, the exact ``ref_text``, an
    optional ``instruction``, and whatever sampling settings the voice should
    answer with. Nothing is synthesized here, so it returns immediately and
    works while the model is still loading.
    """
    fields, ref_bytes = await _read_params(request)

    name = (fields.get("voice_name") or fields.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Field 'voice_name' is required")

    reference_text = (fields.get("ref_text") or "").strip()
    if not reference_text:
        raise HTTPException(
            400,
            "Field 'ref_text' is required: this runtime cannot transcribe the "
            "recording, so it needs the exact words spoken in it",
        )

    ref_path, uploaded = _materialize_reference(fields, ref_bytes)
    if ref_path is None:
        raise HTTPException(400, "Upload 'ref_audio' or pass 'ref_audio_path'")

    generation = _generation_fields(fields)
    generation["instruction"] = (fields.get("instruction") or "").strip() or None
    generation["mode"] = (fields.get("mode") or "").strip() or None
    if generation["cfg_scale"] is None:
        generation["cfg_scale"] = 1.0

    try:
        voice = voice_store.create_voice_from_recording(
            ref_path,
            name=name,
            reference_text=reference_text,
            generation=generation,
            metadata=_voice_metadata(fields),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if uploaded:
            ref_path.unlink(missing_ok=True)
    return JSONResponse(voice, status_code=201)


# ---------------------------------------------------------------------------
# Saved voices
# ---------------------------------------------------------------------------
@app.get("/v1/voice-metadata/options")
def voice_metadata_options() -> dict[str, Any]:
    """Every valid metadata code, for a form or an automated edit to check against."""
    return voice_store.metadata_options()


@app.get("/v1/voices")
def list_voices_endpoint(request: Request) -> dict[str, Any]:
    """List, search, filter and page through saved voices."""
    query = request.query_params
    try:
        return voice_store.list_voices(
            search=query.get("search") or None,
            origin=query.get("origin") or None,
            tags=query.getlist("tags") or None,
            favorites_only=bool(_as_bool(query.get("favorites_only"), False)),
            sort=query.get("sort") or "created_at_unix",
            sort_direction=query.get("sort_direction") or "desc",
            page=_as_int(query.get("page"), 1) or 1,
            page_size=_as_int(query.get("page_size"), 50) or 50,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/v1/voices/{voice_id}")
def get_voice_endpoint(voice_id: str) -> dict[str, Any]:
    return _load_voice(voice_id)


@app.get("/v1/voices/{voice_id}/sample")
def voice_sample(voice_id: str) -> FileResponse:
    """The audition clip: what this voice sounds like."""
    try:
        return FileResponse(voice_store.sample_path(voice_id), media_type="audio/wav")
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/v1/voices/{voice_id}/reference")
def voice_reference_audio(voice_id: str) -> FileResponse:
    """The recording every anchored request conditions on."""
    try:
        return FileResponse(
            voice_store.reference_path(voice_id), media_type="audio/wav"
        )
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@app.patch("/v1/voices/{voice_id}")
async def update_voice_endpoint(voice_id: str, request: Request) -> dict[str, Any]:
    """Edit descriptive metadata. Omitted fields keep their value."""
    fields, _ = await _read_params(request)
    try:
        return voice_store.update_voice(voice_id, fields)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/v1/voices/{voice_id}/settings")
async def update_voice_settings(voice_id: str, request: Request) -> dict[str, Any]:
    """Change the generation defaults this voice replays."""
    fields, _ = await _read_params(request)
    try:
        return voice_store.update_settings(voice_id, fields)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/v1/voices/{voice_id}/tags")
async def replace_voice_tags(voice_id: str, request: Request) -> dict[str, Any]:
    fields, _ = await _read_params(request)
    try:
        return voice_store.set_tags(voice_id, fields.get("tags", []))
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/v1/voices/{voice_id}/favorite")
def favorite_voice(voice_id: str) -> dict[str, Any]:
    try:
        return voice_store.set_favorite(voice_id, True)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc


@app.delete("/v1/voices/{voice_id}/favorite")
def unfavorite_voice(voice_id: str) -> dict[str, Any]:
    try:
        return voice_store.set_favorite(voice_id, False)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc


@app.delete("/v1/voices/{voice_id}")
def delete_voice_endpoint(voice_id: str) -> JSONResponse:
    try:
        voice_store.delete_voice(voice_id)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, f"No such voice: {voice_id}") from exc
    return JSONResponse({"deleted": True, "voice_id": voice_id})


# ---------------------------------------------------------------------------
# Voice references: several recordings of one speaker
#
# A voice holds a base reference plus any number of variations of it -- the same
# person, differently coloured -- and a long read rotates between them so two
# pages do not arrive in one unchanging tone.
# ---------------------------------------------------------------------------

# A variation has to be long enough to clone from, so it is auditioned on the
# same deliberately varied script a new voice is.
DEFAULT_VARIATION_SCRIPT = DEFAULT_PREVIEW_SCRIPT


@app.get("/v1/voices/{voice_id}/references")
def list_references_endpoint(voice_id: str) -> dict[str, Any]:
    try:
        return {
            "voice_id": voice_id,
            "references": voice_store.list_references(voice_id),
            "rotation": _load_voice(voice_id).get("rotation"),
        }
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/v1/voices/{voice_id}/references/{reference_id}/audio")
def reference_audio_endpoint(voice_id: str, reference_id: str) -> FileResponse:
    try:
        path = voice_store.reference_audio_path(voice_id, reference_id)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="audio/wav")


@app.post("/v1/voices/{voice_id}/references")
async def add_reference_endpoint(voice_id: str, request: Request) -> dict[str, Any]:
    """Attach another reference recording of the same speaker.

    Either a generated take (``generated_voice_id``, from
    /v1/voices/{id}/variations) or a recording uploaded as ``ref_audio``. A
    recording must come with ``ref_text``: this runtime cannot transcribe, and a
    reference without its exact transcript clones badly.
    """
    fields, ref_bytes = await _read_params(request)
    _load_voice(voice_id)

    label = fields.get("label")
    tags = fields.get("tags")
    make_default = _as_bool(fields.get("make_default"), False)

    try:
        generated_voice_id = (fields.get("generated_voice_id") or "").strip()
        if generated_voice_id:
            return voice_store.add_reference_from_preview(
                voice_id, generated_voice_id,
                label=label, tags=tags, make_default=make_default,
            )

        ref_path, uploaded = _materialize_reference(fields, ref_bytes)
        if ref_path is None:
            raise HTTPException(
                400, "Provide 'generated_voice_id', or upload 'ref_audio'"
            )
        try:
            return voice_store.add_reference_from_audio(
                voice_id, ref_path, fields.get("ref_text") or "",
                label=label, tags=tags, make_default=make_default,
            )
        finally:
            if uploaded:
                ref_path.unlink(missing_ok=True)
    except voice_store.PreviewNotFound as exc:
        raise HTTPException(404, f"No such preview: {exc}") from exc
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/v1/voices/{voice_id}/references/{reference_id}")
async def update_reference_endpoint(
    voice_id: str, reference_id: str, request: Request
) -> dict[str, Any]:
    """Retag, rename, enable/disable, or promote a reference to the default."""
    fields, _ = await _read_params(request)
    patch = {
        key: fields[key]
        for key in ("label", "tags", "text", "enabled", "is_default")
        if key in fields
    }
    for key in ("enabled", "is_default"):
        if key in patch:
            patch[key] = _as_bool(patch[key], False)
    try:
        return voice_store.update_reference(voice_id, reference_id, patch)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/v1/voices/{voice_id}/references/{reference_id}")
def delete_reference_endpoint(voice_id: str, reference_id: str) -> dict[str, Any]:
    try:
        return voice_store.delete_reference(voice_id, reference_id)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/v1/voices/{voice_id}/rotation")
async def update_rotation_endpoint(voice_id: str, request: Request) -> dict[str, Any]:
    """Change how this voice rotates between its references."""
    fields, _ = await _read_params(request)
    patch = {
        key: fields[key]
        for key in ("enabled", "mode", "every_words", "boundary", "tags")
        if key in fields
    }
    if "enabled" in patch:
        patch["enabled"] = _as_bool(patch["enabled"], True)
    if "every_words" in patch:
        patch["every_words"] = _as_int(patch["every_words"], 1000)
    try:
        return voice_store.update_rotation(voice_id, patch)
    except voice_store.VoiceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/voices/{voice_id}/variations")
async def voice_variations(voice_id: str, request: Request) -> JSONResponse:
    """Audition the same speaker under a new direction.

    This is the design tab's first step aimed at a voice that already exists:
    the voice's own reference recording plus a fresh descriptor, which is
    ``edit`` mode -- the identity is held by the recording while the instruction
    moves the delivery. Nothing is attached to the voice until a take is saved
    with POST /v1/voices/{id}/references.
    """
    fields, _ = await _read_params(request)
    profile = _load_voice(voice_id)

    instruction = (fields.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(
            400,
            "Field 'instruction' is required: it is what makes this take differ "
            "from the voice you already have (for example 'brighter and quicker')",
        )

    reference_id = (fields.get("reference_id") or "").strip() or None
    reference = voice_store.voice_reference(voice_id, reference_id)
    if reference is None:
        raise HTTPException(
            400, f"Voice {voice_id} has no usable reference recording to vary"
        )

    text = (fields.get("text") or DEFAULT_VARIATION_SCRIPT).strip()
    count = _as_int(fields.get("preview_count"), 1) or 1
    if not 1 <= count <= PREVIEW_COUNT_MAX:
        raise HTTPException(400, f"preview_count must be between 1 and {PREVIEW_COUNT_MAX}")

    base_seed = _as_int(fields.get("seed"), None)
    if base_seed is None:
        base_seed = _engine().default_seed

    warnings: list[str] = []
    words = len(text.split())
    if words < 45:
        warnings.append(
            f"That script is {words} words. A reference needs about "
            f"{voice_store.REFERENCE_MIN_SECONDS:.0f}s of speech, so takes from a "
            "short script cannot be attached as a reference."
        )

    # A voice's own settings are the baseline; the descriptor is the only thing
    # the caller is expected to supply.
    saved = profile.get("generation") or {}
    candidate_fields = {
        **{key: saved[key] for key in ("cfg_scale", "temperature", "top_k", "top_p",
                                       "repetition_penalty", "greedy")
           if saved.get(key) is not None},
        **fields,
        "instruction": instruction,
    }
    candidate_fields.pop("voice_id", None)

    if _wants_sse(candidate_fields):
        _require_single_take(count)
        return StreamingResponse(
            _drain(
                _preview_sse(
                    candidate_fields,
                    text=text,
                    seed=base_seed,
                    origin="cloned",
                    reference_audio=reference[0],
                    reference_text=reference[1],
                    name=f"{profile.get('name')} · {instruction[:40]}",
                ),
                voice_store.prune_previews,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    previews = await _render_candidates(
        candidate_fields,
        text=text,
        count=count,
        base_seed=base_seed,
        origin="cloned",
        reference_audio=reference[0],
        reference_text=reference[1],
        name=f"{profile.get('name')} · {instruction[:40]}",
    )
    voice_store.prune_previews()
    return JSONResponse(
        {
            "voice_id": voice_id,
            "instruction": instruction,
            "text": text,
            "previews": previews,
            "warnings": warnings,
        }
    )


# ---------------------------------------------------------------------------
# System speech: the global hotkey path
#
# Hammerspoon reads the clipboard and posts it here; everything else -- voice
# selection, the LLM rewrite, rotation, playback -- happens in this process.
# That keeps the trigger side to one HTTP call with no audio code in it.
# ---------------------------------------------------------------------------
# Two shapes of "currently speaking" live here, because they are two different
# things. ``session`` is the hotkey path: audio on this machine's speakers, with
# transport controls, owned by the server. ``stream`` is a caller streaming the
# same utterance over HTTP, where cancellation is the client hanging up.
SPEAK: dict[str, Any] = {
    "session": None,
    "stream": None,
    "meta": {},
}
_SPEAK_LOCK = threading.Lock()


def _clipboard_text() -> str:
    """Whatever is on the clipboard, read from the user's own session."""
    try:
        return platform_support.clipboard_text()
    except platform_support.ClipboardUnavailable as exc:
        raise HTTPException(500, f"Could not read the clipboard: {exc}") from exc


def _resolve_speak_voice(fields: dict[str, Any], config: dict[str, Any]) -> str:
    """The voice this utterance uses: request, then setting, then whatever exists."""
    voice_id = (fields.get("voice_id") or "").strip() or config.get("voice_id")
    if voice_id:
        _load_voice(voice_id)  # 404s here rather than halfway through speaking
        return voice_id
    library = voice_store.list_voices(page_size=1)
    if not library["voices"]:
        raise HTTPException(
            400, "No saved voices yet. Design or clone one in the web UI first."
        )
    return library["voices"][0]["voice_id"]


def _active_session() -> speech_session.SpeechSession | None:
    """The session currently speaking on this machine, if there is one."""
    with _SPEAK_LOCK:
        session = SPEAK.get("session")
    return session if session is not None and session.active else None


def _speak_cancel(reason: str = "cancelled") -> None:
    """Stop whatever is speaking. Safe to call when nothing is."""
    with _SPEAK_LOCK:
        session = SPEAK.get("session")
        stream = SPEAK.get("stream")
        if stream is not None and stream.get("state") in {"speaking", "starting"}:
            stream["state"] = reason
    if session is not None:
        session.cancel(reason)
    if stream is not None:
        cancel, pipe = stream.get("cancel"), stream.get("pipe")
        if cancel is not None:
            cancel.set()
        if pipe is not None:
            pipe.cancel()
    audio_out.PLAYER.stop()


def _on_output_device_change(reason: str) -> None:
    """The output moved, or went away. Stop, and remember the place.

    Called from the device watcher's own thread. It does the least it can: the
    session pauses itself, and PortAudio is only told to look at the world again
    the next time a stream is opened, which is the one moment that is safe.
    """
    playback = (speech_config.load().get("playback") or {})
    audio_out.request_refresh()
    if not playback.get("pause_on_device_change", True):
        return
    session = _active_session()
    if session is None:
        audio_out.PLAYER.stop()
        return
    logger.info("Output device event (%s); pausing the current utterance", reason)
    session.pause(reason)


def _speak_synthesis_fields(
    fields: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Merge the stored synthesis defaults under whatever the request specified."""
    synthesis = config.get("synthesis") or {}
    merged: dict[str, Any] = {}
    for key in ("cfg_scale", "max_words", "seed"):
        if synthesis.get(key) is not None:
            merged[key] = synthesis[key]
    if synthesis.get("voice_lock") is not None:
        merged["voice_lock"] = synthesis["voice_lock"]
    if config.get("rotation", {}).get("override"):
        merged["rotation"] = {
            key: value
            for key, value in config["rotation"].items()
            if key != "override"
        }
    merged.update({key: value for key, value in fields.items() if value is not None})
    return merged


def _live_events(
    engine: BreezeEngine,
    job: dict[str, Any],
    *,
    text: str,
    voice_id: str,
    config: dict[str, Any],
    instruction: str | None,
    recording: archive.Recording,
    cancel: threading.Event,
) -> Iterator[tuple[str, np.ndarray | None, int]]:
    """Speak the model's output as it is written, rather than after it is done.

    The producer thread reads Vertex flat out into an unbounded text queue --
    text is small, so the model is never made to wait on the speaker -- while
    this generator pulls whole sentences off the other end. If the model pauses,
    the engine waits with its gate held so the utterance stays contiguous.
    """
    llm_settings = config.get("llm") or {}
    rotation_override = job.get("rotation_override")
    rotator = voice_store.ReferenceRotator(
        voice_id,
        settings=voice_store.rotation_settings(voice_id, rotation_override),
        seed=job["options"].get("seed"),
    )

    pipe = ChunkPipe()
    with _SPEAK_LOCK:
        if SPEAK.get("stream") is not None:
            SPEAK["stream"]["pipe"] = pipe

    aggregator = TextAggregator(
        max_words=_as_int(job.get("max_words"), MAX_WORDS_PER_CHUNK)
    )
    sentinel = llm_stream.SentinelFilter()

    def produce() -> None:
        deltas = llm_stream.stream_text(
            text,
            prompt=llm_settings.get("prompt") or speech_config.DEFAULT_LLM_PROMPT,
            instruction=instruction,
            model=llm_settings.get("model"),
            temperature=float(llm_settings.get("temperature", 0.3)),
            max_output_tokens=int(llm_settings.get("max_output_tokens", 8192)),
        )
        pump_llm(
            pipe,
            deltas,
            sentinel_filter=sentinel,
            sanitize=llm_stream.sanitize,
            aggregator=aggregator,
            on_raw=recording.note_llm,
        )

    producer = threading.Thread(target=produce, name="llm-producer", daemon=True)
    producer.start()

    def reference_for(words: int, starts_paragraph: bool) -> tuple[str, str] | None:
        return rotator.take(words, starts_paragraph)

    def source() -> Iterator[tuple[str, tuple[str, str] | None, bool]]:
        for item in pipe.items(reference_for):
            if cancel.is_set():
                return
            recording.note_chunk(item[0])
            yield item

    options = dict(job["options"])
    options.pop("chunk_refs", None)
    options.pop("paragraph_ids", None)
    try:
        yield from engine.stream_live(source(), request_id=job["request_id"], **options)
    finally:
        pipe.cancel()
        recording.meta["reference_segments"] = rotator.segments


def _speak_events(
    engine: BreezeEngine,
    job: dict[str, Any],
    *,
    text: str,
    voice_id: str,
    config: dict[str, Any],
    use_llm: bool,
    instruction: str | None,
    recording: archive.Recording,
    cancel: threading.Event,
) -> Iterator[tuple[str, np.ndarray | None, int]]:
    """One event stream for both paths, so playback and SSE never diverge."""
    if use_llm:
        events = _live_events(
            engine, job, text=text, voice_id=voice_id, config=config,
            instruction=instruction, recording=recording, cancel=cancel,
        )
    else:
        for chunk in job["chunks"]:
            recording.note_chunk(chunk)
        recording.meta["reference_segments"] = job.get("rotation") or []
        events = _synthesize_events(engine, job["chunks"], job["options"])

    for event in events:
        if cancel.is_set():
            break
        if event[0] == "audio" and event[1] is not None:
            recording.note_audio(event[1])
        yield event


def _prepare_utterance(fields: dict[str, Any], text: str) -> dict[str, Any]:
    """Resolve settings, voice and engine options for one utterance."""
    config = speech_config.load()
    voice_id = _resolve_speak_voice(fields, config)
    use_llm = _as_bool(fields.get("llm"), False)
    instruction = (
        fields.get("llm_instruction")
        if "llm_instruction" in fields
        else (config.get("llm") or {}).get("instruction")
    ) or None

    job_fields = _speak_synthesis_fields(
        {key: fields.get(key) for key in
         ("cfg_scale", "max_words", "seed", "voice_lock", "rotation", "instruction")},
        config,
    )
    # The live path chunks its own text as it arrives, so the job is built for
    # its options and reference only; the placeholder never reaches the model.
    job = _build_job(
        job_fields, None,
        text="Placeholder." if use_llm else text,
        voice_id=voice_id,
    )
    job["request_id"] = f"speak-{uuid.uuid4().hex[:8]}"
    job["max_words"] = job_fields.get("max_words")
    job["rotation_override"] = job_fields.get("rotation")

    archive_settings = config.get("archive") or {}
    enabled = _as_bool(fields.get("archive"), archive_settings.get("enabled", True))
    utterance_id = f"utt_{uuid.uuid4().hex[:16]}"
    recording = archive.Recording(
        utterance_id,
        enabled=bool(enabled),
        keep_audio=bool(archive_settings.get("keep_audio", True)),
    )
    recording.meta.update(
        {
            "voice_id": voice_id,
            "voice_name": (job.get("voice") or {}).get("voice_name"),
            "llm": use_llm,
            "llm_model": (config.get("llm") or {}).get("model") or llm_stream.status().get("model")
            if use_llm else None,
            "llm_instruction": instruction if use_llm else None,
            "mode": job["resolved_mode"],
            "cfg_scale": job["effective_cfg"],
            "seed": job["options"].get("seed"),
        }
    )

    return {
        "config": config,
        "voice_id": voice_id,
        "use_llm": use_llm,
        "instruction": instruction,
        "job": job,
        "recording": recording,
        "utterance_id": utterance_id,
        "archive_max": int(archive_settings.get("max_entries", 500)),
    }


def _speak_debounce_seconds(config: dict[str, Any]) -> float:
    controls = config.get("controls") or {}
    return max(0.0, int(controls.get("skip_debounce_ms") or 0) / 1000.0)


@app.post("/v1/speak")
async def speak_endpoint(request: Request) -> Any:
    """Speak text on this machine's speakers. The hotkey path's single call.

    With no ``text`` the clipboard is used, so a caller that cannot read it --
    or would rather not -- can post an empty body. ``llm`` routes the text
    through Gemini first, streaming, so speech starts before the rewrite ends.

    ``output`` selects where the audio goes: ``device`` (default) plays it here
    and returns immediately, ``sse`` streams it to the caller in the same event
    format the web UI already plays, and ``wav`` returns a finished file.
    """
    fields, _ = await _read_params(request)
    return await _start_speaking(fields)


async def _start_speaking(fields: dict[str, Any]) -> Any:
    """Build one utterance and set it going. Shared by /speak and /speak/toggle."""
    text = (fields.get("text") or "").strip()
    if not text:
        text = _clipboard_text().strip()
    if not text:
        raise HTTPException(400, "Nothing to speak: no 'text' and the clipboard is empty")

    output = (fields.get("output") or "device").strip().lower()
    if output not in {"device", "sse", "wav"}:
        raise HTTPException(400, "output must be 'device', 'sse' or 'wav'")

    # A second press replaces the first, which is what a person pressing a
    # hotkey twice means. Do this before building the job so the engine lock is
    # released by the outgoing utterance.
    _speak_cancel("replaced")
    prepared = await run_in_threadpool(_prepare_utterance, fields, text)

    meta = {
        "voice_id": prepared["voice_id"],
        "llm": prepared["use_llm"],
        "chars": len(text),
        "output": output,
        "preview": text[:200],
    }

    if output == "device":
        session = speech_session.SpeechSession(
            prepared,
            text,
            debounce=_speak_debounce_seconds(prepared["config"]),
        )
        with _SPEAK_LOCK:
            SPEAK["session"] = session
            SPEAK["stream"] = None
            SPEAK["meta"] = meta
        session.start()
        return JSONResponse(
            {
                "action": "started",
                "voice_id": prepared["voice_id"],
                "llm": prepared["use_llm"],
                "chars": len(text),
                "output": output,
                **session.status(),
            }
        )

    cancel = threading.Event()
    with _SPEAK_LOCK:
        SPEAK["session"] = None
        SPEAK["meta"] = meta
        SPEAK["stream"] = {
            "utterance_id": prepared["utterance_id"],
            "state": "starting",
            "error": None,
            "cancel": cancel,
            "pipe": None,
            "started_at": time.time(),
        }

    engine = prepared["job"]["engine"]
    events = _speak_events(
        engine,
        prepared["job"],
        text=text,
        voice_id=prepared["voice_id"],
        config=prepared["config"],
        use_llm=prepared["use_llm"],
        instruction=prepared["instruction"],
        recording=prepared["recording"],
        cancel=cancel,
    )

    def finished() -> None:
        prepared["job"]["cleanup"]()
        with _SPEAK_LOCK:
            stream = SPEAK.get("stream")
            if stream is not None and stream["utterance_id"] == prepared["utterance_id"]:
                stream["state"] = "cancelled" if cancel.is_set() else "done"
                stream["pipe"] = None
        entry = prepared["recording"].write(
            input_text=text, sample_rate=engine.sample_rate,
            meta={"cancelled": cancel.is_set(), "output": output},
        )
        if entry:
            archive.prune(prepared["archive_max"])

    if output == "sse":
        with _SPEAK_LOCK:
            SPEAK["stream"]["state"] = "speaking"
        frames = _sse_stream(
            engine,
            prepared["job"]["chunks"] if not prepared["use_llm"] else [],
            prepared["job"]["options"],
            {
                "utterance_id": prepared["utterance_id"],
                "chunks": 0 if prepared["use_llm"] else len(prepared["job"]["chunks"]),
                "mode": prepared["job"]["resolved_mode"],
                "cfg_scale": prepared["job"]["effective_cfg"],
                "prep_error": None,
                "llm": prepared["use_llm"],
                **prepared["job"]["voice"],
            },
            events=events,
        )
        return StreamingResponse(
            _drain(frames, finished),
            media_type="text/event-stream",
            headers={**prepared["job"]["headers"], **_SSE_HEADERS},
        )

    def collect() -> np.ndarray:
        parts = [audio for kind, audio, _ in events if kind == "audio" and audio is not None]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    try:
        audio = await run_in_threadpool(collect)
    finally:
        finished()
    if not audio.size:
        raise HTTPException(500, "Model produced no audio")
    return StreamingResponse(
        io.BytesIO(_wav_bytes(audio, engine.sample_rate)),
        media_type="audio/wav",
        headers={
            **prepared["job"]["headers"],
            "X-Breeze-Utterance-Id": prepared["utterance_id"],
        },
    )


@app.post("/v1/speak/stop")
def speak_stop() -> dict[str, Any]:
    """Silence the current utterance immediately."""
    _speak_cancel("cancelled")
    return {"stopped": True, **audio_out.PLAYER.status()}


@app.post("/v1/speak/toggle")
async def speak_toggle(request: Request) -> Any:
    """One key for both directions: silence what is playing, or start reading.

    The hotkey is the same physical press either way, which is the point. It
    also means a fresh copy is never one press away from being spoken over the
    top of the last one: the first press silences, the second speaks whatever is
    on the clipboard by then.
    """
    fields, _ = await _read_params(request)
    session = _active_session()
    if session is not None:
        _speak_cancel("cancelled")
        return JSONResponse(
            {"action": "stopped", **session.status(),
             "playback": audio_out.PLAYER.status()}
        )
    return await _start_speaking(fields)


@app.post("/v1/speak/skip")
async def speak_skip(request: Request) -> dict[str, Any]:
    """Move the read forwards or backwards by whole paragraphs.

    ``delta`` is in paragraphs and may be negative. Every press cancels what is
    generating and pushes a deadline out; the engine only starts on the
    paragraph you landed on once the presses stop, so holding the key scrolls
    rather than synthesizing everything on the way past.

    While the read is paused -- which is what losing the output device does to
    it -- the first press resumes instead of moving, because the paragraph it
    stopped in has to be spoken again from the start regardless.
    """
    fields, _ = await _read_params(request)
    delta = _as_int(fields.get("delta"), 1)
    if delta is None:
        delta = 1
    session = _active_session()
    if session is None:
        return {"action": "idle", "state": "idle"}
    return {"action": "skipped", "delta": delta, **session.skip(delta)}


@app.post("/v1/speak/resume")
def speak_resume() -> dict[str, Any]:
    """Pick a paused read back up where it stopped, without the skip delay."""
    session = _active_session()
    if session is None:
        return {"action": "idle", "state": "idle"}
    return {"action": "resumed", **session.resume()}


@app.get("/v1/speak/status")
def speak_status() -> dict[str, Any]:
    with _SPEAK_LOCK:
        session = SPEAK.get("session")
        stream = SPEAK.get("stream")
        meta = SPEAK.get("meta") or {}
    if session is not None:
        state = session.status()
    elif stream is not None:
        state = {key: value for key, value in stream.items()
                 if key not in {"cancel", "pipe"}}
    else:
        state = {"utterance_id": None, "state": "idle", "error": None,
                 "started_at": None}
    return {**state, "meta": meta, "playback": audio_out.PLAYER.status()}


@app.get("/v1/speak/clipboard")
def speak_clipboard() -> dict[str, Any]:
    """What the hotkey would speak right now. Used by the settings page."""
    text = _clipboard_text()
    return {"chars": len(text), "preview": text[:2000], "words": len(text.split())}


# ---------------------------------------------------------------------------
# System-speech settings and history
# ---------------------------------------------------------------------------
@app.get("/v1/system-speech/config")
def system_speech_config() -> dict[str, Any]:
    return {
        "config": speech_config.load(),
        "defaults": speech_config.DEFAULTS,
        "default_prompt": speech_config.DEFAULT_LLM_PROMPT,
        "llm": llm_stream.status(),
        "llm_providers": llm_providers.describe_all(),
        "model_chain": list(llm_stream.MODEL_CHAIN),
        "hotkeys": get_hotkeys(),
        "devices": audio_out.devices(),
        "platform": platform_support.describe(),
        "rotation_modes": list(voice_store.ROTATION_MODES),
        "rotation_boundaries": list(voice_store.ROTATION_BOUNDARIES),
    }


@app.patch("/v1/system-speech/config")
async def update_system_speech_config(request: Request) -> dict[str, Any]:
    fields, _ = await _read_params(request)
    try:
        return {"config": speech_config.save(fields)}
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------------------
# Hotkeys
#
# The shortcuts live in the same config as everything else on this path, and
# both hotkey hosts read them from here: Hammerspoon on startup and on reload,
# the Python daemon on startup. Neither hard-codes a combination, so changing
# one in the browser changes it on whichever machine is asking.
#
# Applying a change still needs the host to pick it up -- Hammerspoon reloads
# its config, the daemon is restarted -- which is what ``restart_required``
# says. Rebinding a live global hook from an HTTP handler is a good way to end
# up with a shortcut bound twice and nothing able to unbind it.
# ---------------------------------------------------------------------------
@app.get("/v1/hotkeys")
def get_hotkeys() -> dict[str, Any]:
    """The configured shortcuts, plus what a UI needs to render an editor."""
    config = speech_config.load()
    return {
        "bindings": config.get("hotkeys") or dict(hotkeys.DEFAULTS),
        "defaults": dict(hotkeys.DEFAULTS),
        "actions": hotkeys.describe(),
        "host": hotkeys.host(),
        "named_keys": sorted(hotkeys.bindings.NAMED_KEYS),
        "modifiers": ["ctrl", "alt", "shift", "cmd"],
    }


@app.put("/v1/hotkeys")
async def set_hotkeys(request: Request) -> dict[str, Any]:
    """Replace the shortcuts. Rejects anything a hotkey host could not bind."""
    fields, _ = await _read_params(request)
    bindings = fields.get("bindings") if isinstance(fields.get("bindings"), dict) else fields
    try:
        validated = hotkeys.validate(
            {key: value for key, value in bindings.items() if key in hotkeys.DEFAULTS}
        )
    except hotkeys.InvalidBinding as exc:
        raise HTTPException(400, str(exc)) from exc
    saved = speech_config.save({"hotkeys": validated})
    return {
        "bindings": saved["hotkeys"],
        "host": hotkeys.host(),
        "restart_required": (
            "Reload the Hammerspoon config to apply."
            if hotkeys.host() == "hammerspoon"
            else "Restart the hotkey daemon to apply."
        ),
    }


@app.post("/v1/system-speech/prompt/reset")
def reset_system_speech_prompt() -> dict[str, Any]:
    return {"config": speech_config.reset_prompt()}


@app.get("/v1/system-speech/history")
def system_speech_history(limit: int = 50) -> dict[str, Any]:
    return {"entries": archive.recent(max(1, min(500, limit)))}


@app.get("/v1/system-speech/history/{directory:path}/audio")
def system_speech_history_audio(directory: str) -> FileResponse:
    try:
        path = archive.entry_path(f"{directory}/audio.wav")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "That utterance has no archived audio")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/v1/system-speech/history")
def clear_system_speech_history() -> dict[str, Any]:
    archive.clear()
    return {"cleared": True, "audio_usage": archive.audio_usage()}


@app.get("/v1/system-speech/audio-usage")
def system_speech_audio_usage() -> dict[str, Any]:
    """How much disk the archived recordings take, and whether that is too much.

    Answered from a figure measured at startup and refreshed hourly, nudged by
    each new utterance in between. Every open tab polls this, so it must not
    walk the archive.
    """
    return {**archive.audio_usage(), "exports": archive.exports()}


@app.post("/v1/system-speech/audio-export")
async def export_system_speech_audio() -> dict[str, Any]:
    """Zip every archived recording and delete the recordings, keeping the text.

    The zip is written to ``state/exports/`` and left there to be moved
    somewhere else; it carries a ``manifest.json`` so it can still be read
    months later. Nothing is deleted until the zip has been reopened and
    verified to hold every file, because this is the only bulk delete in the
    archive and it cannot be undone.
    """
    try:
        return await run_in_threadpool(archive.export_audio)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/v1/system-speech/audio-export/{name}")
def download_system_speech_audio(name: str) -> FileResponse:
    """Hand an exported zip to the browser, so moving it needs no terminal."""
    try:
        path = archive.export_path(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, f"No such export: {name}")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.delete("/v1/system-speech/audio-export/{name}")
def delete_system_speech_audio_export(name: str) -> dict[str, Any]:
    """Remove an export once it has been moved somewhere safe."""
    try:
        path = archive.export_path(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, f"No such export: {name}")
    path.unlink()
    return {"deleted": name, "exports": archive.exports()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Breeze TTS 2 HTTP server")
    parser.add_argument(
        "model",
        nargs="?",
        default=tts_backends.default_model_path(),
        help="checkpoint directory; defaults to the active backend's own",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--audio-device",
        # "auto" means the same thing to both backends -- put the codec wherever
        # the main model went. The rest are for pinning it by hand: mps on a
        # Mac, cuda on Windows, cpu anywhere to rule the accelerator out.
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="where the audio codec runs (default: with the model)",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "torch"),
        help="force a speech backend instead of detecting one",
    )
    args = parser.parse_args()

    STATE["model_path"] = args.model
    os.environ["BREEZE_AUDIO_DEVICE"] = args.audio_device
    if args.backend:
        os.environ["BREEZE_BACKEND"] = args.backend
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
