"""One spoken document, seekable a paragraph at a time.

The hotkey path used to be a straight line: text in, one long generation, audio
out, and the only control was "stop". That is fine until you want to skip the
paragraph you are half way through -- at which point you need three things the
straight line cannot give you.

*A paragraph you can go back to.* Audio is thrown away as it is played, so
going backwards means synthesizing the paragraph again, which means still
having its text. The ``ParagraphBook`` keeps every paragraph the document has
produced so far, whether it came from the clipboard whole or is still being
written by the language model one token at a time.

*A generation you can abandon without leaking.* Each paragraph is its own
``stream_live`` call, so cancelling one closes its codec request, releases the
engine gate and trims the allocator pools on the way out -- the same lifecycle
a web-UI request gets, and the reason skipping through twenty paragraphs costs
no more memory than speaking one.

*A press that means "and the next one too".* Skips are debounced. Each press
cancels what is playing and moves a target index; only once the presses stop
for a moment does the engine start on where you landed. Holding the key scrolls
through the document instead of synthesizing every paragraph on the way past.

Pausing is the same machinery under another name. When the output device
disappears -- headphones out of battery, the usual case -- playback stops where
it is and the session sits paused, exactly like a video player, until a skip
press picks it back up.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import archive
import audio_out
import llm_stream
import speech_config
import voice_store
from breeze_pipeline import MAX_WORDS_PER_CHUNK
from speech_pipeline import TextAggregator, pump_llm

logger = logging.getLogger("breeze.session")

# How long the session waits after the last skip press before it starts
# generating. Long enough to press again without hearing anything spin up,
# short enough that a single press does not feel like a stall.
SKIP_DEBOUNCE_SECONDS = 2.0

# States a caller can see. Only the first four are live.
LIVE_STATES = frozenset({"starting", "speaking", "seeking", "paused"})


class _AnyEvent:
    """Set when any of the events it wraps is set."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class ParagraphBook:
    """The document's paragraphs, as sentence chunks, as they become known.

    Written by one producer -- the clipboard splitter, or the language model's
    stream -- and read by the synthesis worker, which may be anywhere in the
    document including behind the producer. Indices are dense over paragraphs
    that actually have text: a run of blank lines is a formatting artefact, not
    a paragraph a listener would count.
    """

    def __init__(self) -> None:
        self._paragraphs: list[list[str]] = []
        self._source_id: int | None = None
        self._complete = False
        self._error: BaseException | None = None
        self._condition = threading.Condition()

    # -- producer side ----------------------------------------------------
    def add(self, source_paragraph: int, chunk: str) -> None:
        with self._condition:
            if not self._paragraphs or source_paragraph != self._source_id:
                self._paragraphs.append([])
                self._source_id = source_paragraph
            self._paragraphs[-1].append(chunk)
            self._condition.notify_all()

    def finish(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._error = error
            self._complete = True
            self._condition.notify_all()

    # -- reader side ------------------------------------------------------
    @property
    def complete(self) -> bool:
        with self._condition:
            return self._complete

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    def count(self) -> int:
        with self._condition:
            return len(self._paragraphs)

    def text(self, index: int) -> str:
        with self._condition:
            if 0 <= index < len(self._paragraphs):
                return " ".join(self._paragraphs[index])
        return ""

    def _closed(self, index: int) -> bool:
        """Whether paragraph ``index`` can still grow.

        A later paragraph having started is what closes an earlier one: the
        model moved on, so nothing more is coming for this index.
        """
        return self._complete or len(self._paragraphs) > index + 1

    def chunks(self, index: int, stop: Any) -> Iterator[str]:
        """Paragraph ``index``'s chunks, blocking until each one exists.

        Ends when the paragraph is closed and exhausted, or when ``stop`` is
        set. A producer-side failure is raised here, on the consumer's thread,
        so it surfaces as a failed utterance rather than a silent short one.
        """
        cursor = 0
        while True:
            with self._condition:
                while True:
                    if stop.is_set():
                        return
                    if self._error is not None:
                        raise self._error
                    ready = (
                        index < len(self._paragraphs)
                        and cursor < len(self._paragraphs[index])
                    )
                    if ready:
                        break
                    if self._closed(index):
                        return
                    self._condition.wait(0.2)
                chunk = self._paragraphs[index][cursor]
                cursor += 1
            yield chunk

    def wait_for(self, index: int, stop: Any) -> bool:
        """Block until paragraph ``index`` exists. False means it never will."""
        with self._condition:
            while True:
                if stop.is_set():
                    return False
                if index < len(self._paragraphs):
                    return True
                if self._complete:
                    return False
                self._condition.wait(0.2)


class _BookSink:
    """The pipe interface ``pump_llm`` writes into, backed by a book.

    ``pump_llm`` already does the hard parts -- holding back a partial sentence,
    flushing one the model stalled on, tagging every chunk with its paragraph --
    and none of that is worth reimplementing. It only ever needs four things
    from the thing it writes to, so the book supplies exactly those.
    """

    def __init__(self, book: ParagraphBook, stop: threading.Event) -> None:
        self._book = book
        self._stop = stop

    def put(self, chunk: str, paragraph: int) -> None:
        self._book.add(paragraph, chunk)

    def close(self) -> None:
        self._book.finish()

    def fail(self, error: BaseException) -> None:
        self._book.finish(error)

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()


class SpeechSession:
    """One utterance on the machine's speakers, with transport controls.

    Owns two threads: a producer filling the book, and a worker taking one
    paragraph at a time out of it and into the engine. Everything a hotkey does
    -- skip, pause, stop -- only sets flags; the worker is what closes the
    generator, because a generator may only be closed from the thread iterating
    it.
    """

    def __init__(
        self,
        prepared: dict[str, Any],
        text: str,
        *,
        debounce: float = SKIP_DEBOUNCE_SECONDS,
    ) -> None:
        self._prepared = prepared
        self._text = text
        self._job = prepared["job"]
        self._engine = self._job["engine"]
        self._recording: archive.Recording = prepared["recording"]
        self._config = prepared["config"]
        self._debounce = max(0.0, debounce)

        self.utterance_id: str = prepared["utterance_id"]
        self.started_at = time.time()

        self._book = ParagraphBook()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._done = threading.Event()

        self._state = "starting"
        self._error: str | None = None
        self._index = 0
        self._seek_target: int | None = None
        self._seek_deadline = 0.0
        self._paused = False
        self._pause_reason: str | None = None
        self._player_open = False
        self._spoken: list[int] = []

        self._rotator: voice_store.ReferenceRotator | None = None
        self._worker: threading.Thread | None = None
        self._producer: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin producing text and speaking it. Returns immediately."""
        self._build_rotator(self._prepared["voice_id"])

        if self._prepared["use_llm"]:
            self._producer = threading.Thread(
                target=self._produce_llm, name="speak-llm", daemon=True
            )
            self._producer.start()
        else:
            self._fill_from_text()

        with self._lock:
            self._state = "speaking"
        self._worker = threading.Thread(
            target=self._run, name="speak-worker", daemon=True
        )
        self._worker.start()

    def _build_rotator(self, voice_id: str | None) -> None:
        """The reference rotation, carried across paragraphs so it keeps its place.

        One rotator for the whole session rather than one per paragraph: the
        rotation is measured in words since the last switch, and restarting that
        count at every paragraph would switch on every one of them.
        """
        if not voice_id:
            self._rotator = None
            return
        try:
            self._rotator = voice_store.ReferenceRotator(
                voice_id,
                settings=voice_store.rotation_settings(
                    voice_id, self._job.get("rotation_override")
                ),
                seed=self._job["options"].get("seed"),
            )
        except Exception:  # noqa: BLE001 - a voice with no pool simply does not rotate
            logger.exception("Could not build the reference rotation")
            self._rotator = None

    def _fill_from_text(self) -> None:
        """The clipboard path: every paragraph is known before a word is said."""
        chunks = self._job["chunks"]
        groups = self._job["paragraph_ids"] or [0] * len(chunks)
        for chunk, group in zip(chunks, groups):
            self._book.add(group, chunk)
        self._book.finish()

    def _produce_llm(self) -> None:
        """The model path: paragraphs arrive while earlier ones are being said.

        Deliberately not throttled by playback. Text is kilobytes and the model
        is far faster than synthesis, so letting it run to the end costs nothing
        and buys the one thing playback cannot recover on its own: the text of a
        paragraph the listener has already heard and wants to hear again.
        """
        llm_settings = self._config.get("llm") or {}
        sink = _BookSink(self._book, self._stop)
        aggregator = TextAggregator(
            max_words=int(self._job.get("max_words") or MAX_WORDS_PER_CHUNK)
        )
        try:
            deltas = llm_stream.stream_text(
                self._text,
                prompt=llm_settings.get("prompt") or speech_config.DEFAULT_LLM_PROMPT,
                instruction=self._prepared["instruction"],
                model=llm_settings.get("model"),
                temperature=float(llm_settings.get("temperature", 0.3)),
                max_output_tokens=int(llm_settings.get("max_output_tokens", 8192)),
            )
            pump_llm(
                sink,
                deltas,
                sentinel_filter=llm_stream.SentinelFilter(),
                sanitize=llm_stream.sanitize,
                aggregator=aggregator,
                on_raw=self._recording.note_llm,
            )
        except BaseException as exc:  # noqa: BLE001 - handed to the worker
            logger.exception("The language model stream failed")
            self._book.finish(exc)

    # -- the worker -------------------------------------------------------
    def _run(self) -> None:
        error: str | None = None
        try:
            while not self._stop.is_set():
                if not self._settle():
                    break
                index = self._index
                if not self._book.wait_for(index, self._stop):
                    break  # nothing at this index, and nothing more is coming
                self._interrupt.clear()
                interrupted = self._speak_paragraph(index)
                if self._stop.is_set():
                    break
                if interrupted:
                    continue  # a skip or a pause decides where we go next
                with self._lock:
                    if self._seek_target is None and not self._paused:
                        self._index = index + 1
        except Exception as exc:  # noqa: BLE001 - reported through status()
            error = str(exc)
            logger.exception("Speaking failed")
        finally:
            self._finish(error)

    def _settle(self) -> bool:
        """Wait out a pause or a run of skip presses. False means give up.

        The deadline is re-read every time round: another press while we are
        waiting pushes it out, which is what makes holding the key cheap.
        """
        while not self._stop.is_set():
            with self._lock:
                if self._paused and self._seek_target is None:
                    self._wake.wait(0.25)
                    continue
                if self._seek_target is not None:
                    remaining = self._seek_deadline - time.monotonic()
                    if remaining > 0:
                        self._wake.wait(min(remaining, 0.25))
                        continue
                    self._index = self._clamp(self._seek_target)
                    self._seek_target = None
                    self._paused = False
                    self._pause_reason = None
                    self._interrupt.clear()
                    self._state = "speaking"
                return True
        return False

    def _speak_paragraph(self, index: int) -> bool:
        """Synthesize and play one paragraph. True if it was cut short.

        One ``stream_live`` per paragraph is what makes this affordable: the
        engine gate, the codec request and the allocator trim all live and die
        inside this call, so an abandoned paragraph leaves nothing behind for
        the next one to trip over.
        """
        engine = self._engine
        player = audio_out.PLAYER
        playback = self._config.get("playback") or {}

        options = {
            key: value
            for key, value in self._job["options"].items()
            if key not in {"chunk_refs", "paragraph_ids"}
        }
        stop_source = _AnyEvent(self._stop, self._interrupt)
        sentence_groups = self._book.complete and self._book.count() <= 1

        def source() -> Iterator[tuple[str, tuple[str, str] | None, bool]]:
            for position, chunk in enumerate(self._book.chunks(index, stop_source)):
                self._recording.note_chunk(chunk)
                starts_group = position == 0 or sentence_groups
                # A per-chunk reference wins over the job's own inside the
                # engine, so the two never need reconciling here.
                reference = (
                    self._rotator.take(len(chunk.split()), starts_group)
                    if self._rotator is not None
                    else None
                )
                yield chunk, reference, position == 0

        with self._lock:
            open_now = not self._player_open
            self._player_open = True
        if open_now:
            player.start(
                self.utterance_id,
                engine.sample_rate,
                prebuffer_ms=int(playback.get("prebuffer_ms") or 400),
                device=playback.get("device"),
            )
        else:
            # The conceptual gap between paragraphs, written before the next
            # generation starts so it also covers the engine spinning up.
            player.write(self.utterance_id, engine.paragraph_pause_buffer)

        interrupted = False
        events = engine.stream_live(
            source(), request_id=f"{self._job['request_id']}-p{index}", **options
        )
        try:
            for kind, audio, _position in events:
                if self._interrupt.is_set() or self._stop.is_set():
                    interrupted = True
                    break
                if kind == "audio" and audio is not None and audio.size:
                    self._recording.note_audio(audio)
                    if not player.write(self.utterance_id, audio):
                        interrupted = True
                        break
        finally:
            # Closed here, on the thread that iterated it, so the engine's own
            # finally runs now rather than whenever the collector notices.
            events.close()

        if not interrupted:
            with self._lock:
                if not self._spoken or self._spoken[-1] != index:
                    self._spoken.append(index)
        return interrupted or self._interrupt.is_set()

    def _finish(self, error: str | None) -> None:
        player = audio_out.PLAYER
        cancelled = self._stop.is_set()
        with self._lock:
            open_player = self._player_open
            self._player_open = False
        if error or cancelled:
            player.stop(utterance_id=self.utterance_id)
        elif open_player:
            # Play out what is queued rather than cutting the last words off.
            player.finish(self.utterance_id)
            player.wait_drained(30.0)

        book_error = self._book.error
        if error is None and book_error is not None:
            error = str(book_error)

        with self._lock:
            self._error = error
            self._state = (
                "error" if error else "cancelled" if cancelled else "done"
            )
            spoken = list(self._spoken)
            if self._rotator is not None:
                self._recording.meta["reference_segments"] = self._rotator.segments

        try:
            self._job["cleanup"]()
        except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
            logger.exception("Job cleanup failed")

        entry = self._recording.write(
            input_text=self._text,
            sample_rate=self._engine.sample_rate,
            meta={
                "error": error,
                "cancelled": cancelled,
                "output": "device",
                "paragraphs": self._book.count(),
                "paragraphs_spoken": spoken,
            },
        )
        if entry:
            archive.prune(self._prepared["archive_max"])
        self._done.set()

    # -- controls ---------------------------------------------------------
    def _clamp(self, target: int) -> int:
        """Keep a seek inside the document, allowing for one still being written.

        One past the last known paragraph is legal while the model is still
        producing: it means "the next one, when it exists", and the worker
        blocks there until it does.
        """
        count = self._book.count()
        ceiling = max(0, count - 1) if self._book.complete else count
        return max(0, min(target, ceiling))

    def _silence(self) -> None:
        """Stop the sound, but only if this session still owns the speaker."""
        audio_out.PLAYER.stop(utterance_id=self.utterance_id)
        with self._lock:
            self._player_open = False

    def skip(self, delta: int) -> dict[str, Any]:
        """Move ``delta`` paragraphs and arm the debounce.

        A press while paused resumes rather than moves. The paragraph was cut
        off part way through, so it has to be spoken again from its start
        whatever happens; landing on it is what the listener meant. Press twice
        and the second press does move, because by then the first one has
        already resumed.
        """
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            resuming = self._paused and self._seek_target is None
            base = self._seek_target if self._seek_target is not None else self._index
            self._seek_target = self._clamp(base if resuming else base + delta)
            self._seek_deadline = time.monotonic() + self._debounce
            self._paused = False
            self._pause_reason = None
            self._state = "seeking"
            self._wake.notify_all()
        self._interrupt.set()
        self._silence()
        return self.status()

    def pause(self, reason: str = "paused") -> dict[str, Any]:
        """Stop the sound but keep the place. Resumed with a skip press."""
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            self._paused = True
            self._pause_reason = reason
            self._seek_target = None
            self._state = "paused"
            self._wake.notify_all()
        self._interrupt.set()
        self._silence()
        return self.status()

    def resume(self) -> dict[str, Any]:
        """Pick up at the paragraph that was interrupted, without waiting."""
        with self._lock:
            if self._state not in LIVE_STATES:
                return self.status()
            self._seek_target = self._clamp(self._index)
            self._seek_deadline = time.monotonic()
            self._paused = False
            self._pause_reason = None
            self._state = "seeking"
            self._wake.notify_all()
        self._interrupt.set()
        return self.status()

    def cancel(self, reason: str = "cancelled") -> None:
        """End the session. Safe to call more than once, and from any thread."""
        with self._lock:
            if self._state in LIVE_STATES:
                self._state = reason
            self._wake.notify_all()
        self._stop.set()
        self._interrupt.set()
        self._silence()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    # -- reporting --------------------------------------------------------
    @property
    def active(self) -> bool:
        with self._lock:
            return self._state in LIVE_STATES

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "utterance_id": self.utterance_id,
                "state": self._state,
                "error": self._error,
                "started_at": self.started_at,
                "paragraph": self._index,
                "paragraph_target": self._seek_target,
                "paragraphs": self._book.count(),
                "paragraphs_final": self._book.complete,
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "paragraph_preview": self._book.text(self._index)[:200],
            }
