"""Persistent in-memory Breeze TTS 2 engine.

Loads the model once and reuses it across many requests, and adds sentence
chunking so long paragraphs never build one enormous attention context.

The model itself is built by whichever backend this machine can run -- MLX on
Apple Silicon, PyTorch on CUDA -- and everything below that line is shared. See
``tts_backends`` for what the two have to agree on; it is a short list, and it
is short deliberately: chunking, the voice lock, paragraph gaps, request
serialization and cancellation are all decisions about how a document should
sound, not about which chip is doing the arithmetic, so they are written once.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import tts_backends

# Prompt collation and sampling are the same code on both platforms; they are
# stored in the MLX checkout only because that is the vendored port this
# project started from, and they import no MLX. See breeze_tts_torch/_shared.py.
_RUNTIME_DIR = Path(__file__).resolve().parent / "breeze-tts-mlx"
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

from breeze_tts_mlx.sampling import NumpySampler, SamplingConfig  # noqa: E402
from breeze_tts_mlx.templates import get_template, prepare_inputs  # noqa: E402

# Configuration & guardrails
MAX_WORDS_PER_CHUNK = 35
SILENCE_GAP_MS = 220  # Natural breath pause between sentences
# A blank line is a conceptual break, not a breath. Held long enough that a
# listener registers the point just made before the next one starts.
PARAGRAPH_GAP_MS = 700

# The sampler is a single RNG stream living on the runtime. Left alone it keeps
# advancing across chunks and across requests, so two identical requests return
# different audio and every sentence of a paragraph draws a different voice.
# Reseeding per chunk from a known default makes generation reproducible.
DEFAULT_SEED = 42

# Voice lock: how much audio to accumulate as the anchor a paragraph's later
# chunks are conditioned on. Below ~8 s a reference clones weakly; past ~15 s
# the extra context buys nothing and costs encode time on every chunk.
VOICE_LOCK_TARGET_SECONDS = 8.0
VOICE_LOCK_MAX_SECONDS = 15.0

# Only these four modes exist upstream. plain/clone are unconditional branches,
# so classifier-free guidance is undefined for them and the runtime rejects it.
_TEMPLATE_FOR_MODE = {
    "plain": "tts_plain",
    "guided": "tts_instruction",
    "clone": "ref_clone_tata",
    "edit": "ref_edit_tata",
}
_CFG_CAPABLE_MODES = {"guided", "edit"}
# Dual CFG (separate uncond/ref/ins branches) exists in the shared template code
# but this MLX runtime rejects it in _build_branches: "The MLX CLI currently
# supports no CFG or single CFG". So it is not exposed.

_SENTENCE_TERMINALS = r"(?<=[.!?。！？])\s+"
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_CLAUSE_TERMINALS = r"(?<=[,;:，；：])\s*"

# Vocal events the model actually understands. Anything else is spoken aloud.
ENGLISH_VOCAL_EVENTS = ("(laugh)", "(cough)", "(clears throat)", "(sigh)")
CHINESE_VOCAL_EVENTS = ("[笑]", "[咳嗽]", "[清嗓子]", "[叹气]")


def resolve_mode(*, has_ref: bool, has_instruction: bool) -> str:
    """Mirror infer.py's ``--mode auto`` selection."""
    if has_ref and has_instruction:
        return "edit"
    if has_ref:
        return "clone"
    if has_instruction:
        return "guided"
    return "plain"


def chunk_document(
    text: str, max_words: int = MAX_WORDS_PER_CHUNK
) -> tuple[list[str], list[int]]:
    """Chunk text and report which paragraph each chunk came from.

    Returns ``(chunks, paragraph_ids)`` of equal length. The paragraph ids are
    what reference rotation switches on: a blank line is the only place a
    listener expects the delivery to shift, so it is the only place it may.
    """
    chunks: list[str] = []
    paragraphs: list[int] = []
    for index, paragraph in enumerate(_PARAGRAPH_BREAK.split(text.strip())):
        produced = validate_and_chunk_text(paragraph, max_words=max_words)
        chunks.extend(produced)
        paragraphs.extend([index] * len(produced))
    return chunks, paragraphs


def validate_and_chunk_text(
    text: str, max_words: int = MAX_WORDS_PER_CHUNK
) -> list[str]:
    """Split text into sentences, sub-splitting any that exceed ``max_words``.

    Vocal-event tags -- ``(sigh)`` in English, ``[叹气]`` in Chinese -- carry no
    sentence terminals, so they survive the split intact.
    """
    final_chunks: list[str] = []

    for sentence in re.split(_SENTENCE_TERMINALS, text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= max_words:
            final_chunks.append(sentence)
            continue

        # Too long: fall back to clause boundaries to bound the attention span.
        buffer: list[str] = []
        for clause in re.split(_CLAUSE_TERMINALS, sentence):
            clause = clause.strip()
            if not clause:
                continue
            if buffer and len(" ".join(buffer + [clause]).split()) > max_words:
                final_chunks.append(" ".join(buffer))
                buffer = [clause]
            else:
                buffer.append(clause)
        if buffer:
            final_chunks.append(" ".join(buffer))

    # A single clause longer than max_words still has to go somewhere: emit it
    # rather than silently dropping text.
    return [chunk for chunk in final_chunks if chunk]


class BreezeEngine:
    """Holds one quantized model in unified memory for the process lifetime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        audio_device: str = "auto",
        seed: int = 42,
        max_new_tokens: int = 1500,
        max_seq_len: int = 2048,
        repetition_penalty: float = 1.1,
        codec_chunk_frames: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        backend: Any = None,
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")

        self.backend = backend or tts_backends.load_backend()
        print(
            f"Loading model from {model_path} with the {self.backend.NAME} backend..."
        )
        sampling = SamplingConfig(
            temperature=temperature, top_k=top_k, top_p=top_p, do_sample=True
        )
        self.model_path = model_path
        self.default_seed = seed
        self.runtime = self.backend.build(
            model_path,
            audio_device=audio_device,
            seed=seed,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            repetition_penalty=repetition_penalty,
            codec_chunk_frames=codec_chunk_frames,
        )
        # Kept so per-request overrides always start from a known baseline.
        self.base_config = self.runtime.runtime_config

        # Every allocator involved keeps freed blocks for reuse rather than
        # returning them, so a burst of long generations leaves the process
        # holding several spare gigabytes. Each backend bounds what it can at
        # build time and names the one pool it cannot, which is what gets
        # watched here. Trimming never touches the model: weights are live
        # allocations and stay resident.
        self._growth_key = self.backend.GROWTH_KEY
        self.cache_budget = int(
            float(os.getenv(self.backend.GROWTH_BUDGET_ENV, "1.0")) * 1024**3
        )
        self._growth_baseline = self.memory_stats().get(self._growth_key, 0)
        # Serializes generation: one MLX runtime cannot render two clips at
        # once. A semaphore rather than an RLock because generation runs inside
        # generators that the web server drives from a thread pool -- successive
        # steps, and the cleanup when a client aborts mid-stream, can land on
        # different threads. A thread-owned lock cannot be released by a thread
        # that did not take it, which stranded it and hung every later request.
        self._gate = threading.Semaphore(1)
        self.pause_buffer = np.zeros(
            int(self.sample_rate * (SILENCE_GAP_MS / 1000.0)), dtype=np.float32
        )
        self.paragraph_pause_buffer = np.zeros(
            int(self.sample_rate * (PARAGRAPH_GAP_MS / 1000.0)), dtype=np.float32
        )
        print(f"Model ready. Sample rate: {self.sample_rate} Hz")

    @property
    def sample_rate(self) -> int:
        return self.runtime.sample_rate

    @contextmanager
    def _serialized(self, *, held: bool) -> Iterator[None]:
        """Hold the generation gate, unless an enclosing call already holds it.

        Replaces re-entrancy: ``stream_document`` takes the gate for a whole
        document and tells each chunk it is already held, so nesting never
        depends on which thread is running.
        """
        if held:
            yield
            return
        self._gate.acquire()
        try:
            yield
        finally:
            self._gate.release()

    def memory_stats(self) -> dict[str, int]:
        """Accelerator memory accounting, in bytes.

        The keys are the backend's -- MLX reports its unified-memory pools and
        the MPS ones behind the audio tokenizer, CUDA reports allocated,
        reserved and what the driver has left. Both add the process RSS, which
        is the only figure that means the same thing on either machine.
        """
        stats = self.backend.memory_stats(self.runtime)
        try:
            import psutil

            stats["process_rss"] = int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - reporting must never break a request
            pass
        return stats

    def trim_memory(self) -> dict[str, int]:
        """Hand cached-but-free blocks back to the system.

        This does **not** unload or reload anything: model weights are live
        allocations, untouched here, so the checkpoint stays resident and is
        reused exactly as before. Only the allocators' pools of already-freed
        blocks are released -- the memory that would otherwise stay claimed by
        this process after a burst of work.

        Returns the bytes released by each allocator.
        """
        return self.backend.trim_memory(self.runtime)

    def trim_if_needed(self) -> dict[str, int] | None:
        """Trim after a document, but only once the spare pool has piled up.

        Each backend names one allocator it cannot bound at build time -- the
        MPS pool behind the audio tokenizer on the Mac, the CUDA caching
        allocator on Windows -- and both grow the same way: roughly 0.6 GiB per
        long take here, because anchored chunks re-encode a reference recording
        that plain synthesis never touched. Trimming on a budget keeps the
        process flat without paying the reallocation cost on every request.
        """
        stats = self.memory_stats()
        grown = stats.get(self._growth_key, 0) - self._growth_baseline
        if grown <= self.cache_budget:
            return None
        return self.trim_memory()

    def _codec_runtimes(self) -> list[Any]:
        """Every streaming codec runtime the tokenizer has built so far.

        One is cached per ``codec_chunk_frames`` value, and that is a
        per-request override, so a leak can live in any of them.
        """
        tokenizer = self.runtime.audio_tokenizer
        cached = getattr(tokenizer, "_stream_runtimes", None)
        if isinstance(cached, dict) and cached:
            return list(cached.values())
        return [tokenizer.stream_runtime(self.runtime.runtime_config.codec_chunk_frames)]

    def release_stale_requests(self) -> int:
        """Close codec requests an abandoned generation left open.

        When a client aborts a stream, the generator feeding it is dropped
        mid-yield and its cleanup runs whenever the interpreter gets around to
        it -- which can be *after* the next request has already started. The
        codec runtime still lists the old request as active and refuses to
        reopen it: "open_request received is_first_decode=True for an already
        active request".

        Generations are serialized by the gate, so anything still active
        when a new document starts is by definition abandoned and safe to close.
        """
        released = 0
        for codec in self._codec_runtimes():
            try:
                active = list(codec.request_pool.active_req_ids())
            except AttributeError:  # a runtime that does not expose the pool
                continue
            for request_id in active:
                try:
                    codec.close_request(request_id)
                except Exception:  # noqa: BLE001 - never block a new request
                    print(f"Warning: could not close stale codec request {request_id}")
                else:
                    released += 1
        return released

    @contextmanager
    def _overrides(
        self,
        *,
        seed: int | None,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        greedy: bool | None,
        repetition_penalty: float | None,
        max_new_tokens: int | None,
        max_seq_len: int | None,
        codec_chunk_frames: int | None,
    ) -> Iterator[None]:
        """Temporarily swap generation settings for one request.

        ``runtime_config`` and ``sampler`` are plain attributes on the runtime,
        so per-request settings are applied by swapping them under the lock and
        restoring them afterwards. Anything left as None keeps the engine default.
        """
        base = self.base_config
        sampling_changes = {
            key: value
            for key, value in (
                ("temperature", temperature),
                ("top_k", top_k),
                ("top_p", top_p),
                ("do_sample", None if greedy is None else not greedy),
            )
            if value is not None
        }
        sampling = (
            replace(base.backbone_sampling, **sampling_changes)
            if sampling_changes
            else base.backbone_sampling
        )
        config_changes: dict[str, Any] = {
            key: value
            for key, value in (
                ("repetition_penalty", repetition_penalty),
                ("max_new_tokens", max_new_tokens),
                ("max_seq_len", max_seq_len),
                ("codec_chunk_frames", codec_chunk_frames),
            )
            if value is not None
        }
        config = replace(
            base, backbone_sampling=sampling, depth_sampling=sampling, **config_changes
        )
        config.validate()

        saved_config = self.runtime.runtime_config
        saved_sampler = self.runtime.sampler
        self.runtime.runtime_config = config
        if seed is not None:
            self.runtime.sampler = NumpySampler(seed)
        try:
            yield
        finally:
            self.runtime.runtime_config = saved_config
            self.runtime.sampler = saved_sampler

    def validate(
        self,
        *,
        instruction: str | None,
        ref_audio: str | Path | None,
        ref_text: str | None,
        cfg_scale: float,
        mode: str,
    ) -> tuple[str, str | None, float]:
        """Check arguments without touching the model.

        Cheap and thread-safe, so a server can return 400s for bad requests
        while another request is mid-generation. Returns the resolved mode, the
        normalized instruction, and the effective CFG scale.
        """
        has_ref = ref_audio is not None
        has_ref_text = bool(ref_text and ref_text.strip())
        if has_ref != has_ref_text:
            raise ValueError("ref_audio and ref_text must be provided together")
        if has_ref and not Path(ref_audio).is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise ValueError("cfg_scale must be finite and greater than zero")

        instruction = instruction.strip() if instruction and instruction.strip() else None
        if mode == "auto":
            mode = resolve_mode(has_ref=has_ref, has_instruction=instruction is not None)
        if mode not in _TEMPLATE_FOR_MODE:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: auto, "
                f"{', '.join(_TEMPLATE_FOR_MODE)}"
            )
        if mode in {"clone", "edit"} and not has_ref:
            raise ValueError(f"mode {mode} requires ref_audio and ref_text")
        if mode in {"guided", "edit"} and instruction is None:
            raise ValueError(f"mode {mode} requires an instruction")

        # plain/clone have no negative branch, so CFG is silently clamped rather
        # than raising -- callers copying the doc's cfg_scale=4.0 still work.
        effective_cfg = cfg_scale if mode in _CFG_CAPABLE_MODES else 1.0
        return mode, instruction, effective_cfg

    def _build_inputs(
        self,
        text: str,
        *,
        instruction: str | None,
        ref_audio: str | Path | None,
        ref_text: str | None,
        cfg_scale: float,
        mode: str,
        request_id: str,
    ) -> dict[str, Any]:
        mode, instruction, effective_cfg = self.validate(
            instruction=instruction,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_scale=cfg_scale,
            mode=mode,
        )

        request: dict[str, Any] = {"id": request_id, "text": text, "speaker": "S0"}
        if instruction is not None:
            request["instruction"] = instruction
        if ref_audio is not None:
            request["ref_audio_path"] = str(ref_audio)
            request["ref_text"] = ref_text.strip()

        return prepare_inputs(
            self.runtime.tokenizer,
            self.runtime.audio_tokenizer,
            self.runtime,
            [request],
            get_template(_TEMPLATE_FOR_MODE[mode]),
            guidance_scale=effective_cfg,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

    def stream_chunk(
        self,
        text: str,
        *,
        instruction: str | None = None,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        cfg_scale: float = 1.0,
        mode: str = "auto",
        seed: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        max_seq_len: int | None = None,
        codec_chunk_frames: int | None = None,
        request_id: str = "breeze-request",
        _gate_held: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield float32 mono audio segments as they are decoded."""
        with self._serialized(held=_gate_held), self._overrides(
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            codec_chunk_frames=codec_chunk_frames,
        ):
            inputs = self._build_inputs(
                text,
                instruction=instruction,
                ref_audio=ref_audio,
                ref_text=ref_text,
                cfg_scale=cfg_scale,
                mode=mode,
                request_id=request_id,
            )
            for chunk in self.runtime.iter_audio_chunks(inputs, request_id=request_id):
                audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio

    def generate_chunk(self, text: str, **kwargs: Any) -> np.ndarray:
        """Run one chunk to completion and return float32 mono audio."""
        parts = list(self.stream_chunk(text, **kwargs))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def stream_document(
        self,
        chunks: list[str],
        *,
        voice_lock: bool = True,
        request_id: str = "doc",
        chunk_refs: list[tuple[str, str]] | None = None,
        paragraph_ids: list[int] | None = None,
        **options: Any,
    ) -> Iterator[tuple[str, np.ndarray | None, int]]:
        """Synthesize a chunked document, yielding audio and boundary events.

        Yields ``("audio", samples, chunk_index)`` and ``("boundary", None,
        chunk_index)``. A boundary marks the end of one sentence chunk -- the
        only place a player can pause without the listener hearing a glitch.

        Two guarantees the per-chunk path cannot give on its own:

        *Determinism.* The seed is resolved once and applied to every chunk, so
        an unseeded request no longer inherits whatever RNG state the previous
        request happened to leave behind.

        *One voice.* Without a reference recording the model invents a speaker
        from the instruction and the text in front of it, so each sentence of a
        paragraph arrives in a different voice. Voice lock closes that: the
        opening chunk is generated normally, then becomes the reference every
        later chunk is conditioned on. The anchor keeps growing with those
        already-anchored chunks until it is long enough to clone from reliably.
        A request that already carries reference audio needs none of this --
        every chunk is anchored to the same recording from the start.

        ``chunk_refs`` overrides the reference per chunk, one ``(path, text)``
        pair per entry, which is how a long read rotates between recordings of
        the same speaker. Supplying it means every chunk is already anchored, so
        voice lock has nothing left to do and stays off.

        ``paragraph_ids`` marks where blank lines were, so the gap at a
        conceptual break is longer than the breath between two sentences.
        """
        # Lock only where the drift exists: several chunks, no reference yet.
        lock_active = (
            voice_lock
            and not options.get("ref_audio")
            and not chunk_refs
            and len(chunks) > 1
        )

        groups = paragraph_ids or [0] * len(chunks)

        def items() -> Iterator[tuple[str, tuple[str, str] | None, bool]]:
            for index, chunk in enumerate(chunks):
                reference = (
                    chunk_refs[index]
                    if chunk_refs is not None and index < len(chunk_refs)
                    else None
                )
                new_paragraph = index > 0 and groups[index] != groups[index - 1]
                yield chunk, reference, new_paragraph

        yield from self._stream_items(
            items(), lock_active=lock_active, request_id=request_id, options=options
        )

    def stream_live(
        self,
        source: Iterator[tuple[str, tuple[str, str] | None, bool]],
        *,
        voice_lock: bool = True,
        request_id: str = "live",
        **options: Any,
    ) -> Iterator[tuple[str, np.ndarray | None, int]]:
        """Synthesize a document whose chunks are still being written.

        The same event stream as ``stream_document``, but fed by an iterator
        that may block between items -- the shape a language model's output
        arrives in. ``source`` yields ``(text, reference_or_None)``; blocking in
        its ``__next__`` is expected, and the engine gate stays held across the
        wait so the document's chunks remain contiguous. Items are
        ``(text, reference_or_None, starts_new_paragraph)``.

        Voice lock is deliberately available here too: it needs no length, only
        a first chunk to anchor on.
        """
        lock_active = voice_lock and not options.get("ref_audio")
        yield from self._stream_items(
            source, lock_active=lock_active, request_id=request_id, options=options
        )

    def _stream_items(
        self,
        source: Iterator[tuple[str, tuple[str, str] | None, bool]],
        *,
        lock_active: bool,
        request_id: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, np.ndarray | None, int]]:
        """The shared body of ``stream_document`` and ``stream_live``."""
        options = dict(options)
        if options.get("seed") is None:
            options["seed"] = self.default_seed

        instruction = options.get("instruction")
        anchor_path: Path | None = None
        anchor_audio: list[np.ndarray] = []
        anchor_text: list[str] = []
        anchor_seconds = 0.0

        def write_anchor() -> None:
            """Persist the accumulated anchor; the tokenizer reads from disk."""
            nonlocal anchor_path
            if anchor_path is None:
                handle, name = tempfile.mkstemp(prefix="breeze-anchor-", suffix=".wav")
                os.close(handle)
                anchor_path = Path(name)
            sf.write(anchor_path, np.concatenate(anchor_audio), self.sample_rate)

        try:
            # Held for the whole document, so its chunks stay contiguous.
            with self._serialized(held=False):
                # An earlier request that the client aborted may still be
                # holding a codec request open. Nothing else can be generating
                # here, so clear it before opening ours.
                stale = self.release_stale_requests()
                if stale:
                    print(f"Released {stale} codec request(s) from an aborted generation")

                for index, (chunk, reference, new_paragraph) in enumerate(source):
                    if index:
                        yield "audio", (
                            self.paragraph_pause_buffer
                            if new_paragraph
                            else self.pause_buffer
                        ), index

                    chunk_options = dict(options)
                    if reference is not None:
                        chunk_options["ref_audio"] = reference[0]
                        chunk_options["ref_text"] = reference[1]
                        chunk_options["mode"] = "edit" if instruction else "clone"
                    elif lock_active and anchor_path is not None:
                        chunk_options["ref_audio"] = str(anchor_path)
                        chunk_options["ref_text"] = " ".join(anchor_text)
                        # plain -> clone, guided -> edit: the reference branch of
                        # the same template pair, so CFG behaves as documented.
                        chunk_options["mode"] = "edit" if instruction else "clone"

                    produced: list[np.ndarray] = []
                    for audio in self.stream_chunk(
                        chunk,
                        request_id=f"{request_id}-{index}",
                        _gate_held=True,
                        **chunk_options,
                    ):
                        produced.append(audio)
                        yield "audio", audio, index
                    yield "boundary", None, index

                    if (
                        lock_active
                        and produced
                        and anchor_seconds < VOICE_LOCK_TARGET_SECONDS
                    ):
                        # Chunk 0 sets the identity; later chunks were themselves
                        # anchored to it, so adding them strengthens the
                        # reference rather than blending in a second speaker.
                        segment = np.concatenate(produced)
                        room = VOICE_LOCK_MAX_SECONDS - anchor_seconds
                        keep = min(segment.size, int(room * self.sample_rate))
                        if keep > 0:
                            anchor_audio.append(segment[:keep])
                            anchor_text.append(chunk)
                            anchor_seconds += keep / self.sample_rate
                            write_anchor()
        finally:
            if anchor_path is not None:
                anchor_path.unlink(missing_ok=True)
            # Outside the gate: nothing is generating, so releasing spare
            # blocks cannot stall a request that is mid-flight.
            released = self.trim_if_needed()
            if released:
                freed = sum(released.values()) / (1024**3)
                print(f"Trimmed {freed:.2f} GiB of spare allocator blocks")

    def synthesize_long_text(
        self,
        text: str,
        output_file: str | Path,
        *,
        max_words: int = MAX_WORDS_PER_CHUNK,
        voice_lock: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Chunk a paragraph, synthesize each piece, and join with breath pauses."""
        chunks = validate_and_chunk_text(text, max_words=max_words)
        if not chunks:
            raise ValueError("No synthesizable text after chunking")
        if verbose:
            print(f"Processing {len(chunks)} guarded sentence chunks:")

        audio_stream: list[np.ndarray] = []
        reported = -1
        for kind, audio, index in self.stream_document(
            chunks, voice_lock=voice_lock, request_id="chunk", **kwargs
        ):
            if verbose and index != reported:
                reported = index
                print(
                    f'  [{index + 1}/{len(chunks)}] '
                    f'({len(chunks[index].split())} words): "{chunks[index]}"'
                )
            if kind == "audio" and audio is not None and audio.size:
                audio_stream.append(audio)

        if not audio_stream:
            raise RuntimeError("Model produced no audio")

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_audio = np.concatenate(audio_stream)
        sf.write(output_file, final_audio, self.sample_rate)
        if verbose:
            duration = final_audio.size / self.sample_rate
            print(f"Saved {output_file} ({duration:.2f}s)")
        return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Long-form Breeze TTS 2 synthesis")
    parser.add_argument(
        "model", nargs="?", default=tts_backends.default_model_path()
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-words", type=int, default=MAX_WORDS_PER_CHUNK)
    parser.add_argument(
        "--no-voice-lock",
        action="store_true",
        help="Let every sentence invent its own voice instead of anchoring to the first",
    )
    parser.add_argument("--output", default="./outputs/long_form.wav")
    args = parser.parse_args()

    engine = BreezeEngine(args.model)
    engine.synthesize_long_text(
        args.text,
        args.output,
        max_words=args.max_words,
        voice_lock=not args.no_voice_lock,
        instruction=args.instruction,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy or None,
    )
