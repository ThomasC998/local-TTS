"""A document turned into paragraph audio, a few paragraphs ahead of whoever is listening.

This is the half of a read that does not care where the sound comes out. It
holds the text as it becomes known, generates each paragraph once, keeps what it
generated, and lets a listener take any paragraph back out -- the one being made
right now, or one heard ten minutes ago.

There are two listeners in this project and they want the same thing for
different reasons. ``speech_session.SpeechSession`` plays on this machine's
speakers and moves through the document with a hotkey. ``remote_read.RemoteRead``
serves the same paragraphs over HTTP to a phone, which moves through the
document with the buttons on its lock screen. Neither wants to wait for a
paragraph that was already made, and neither wants the engine restarted because
somebody pressed a button.

So the contract is a position and a cache:

*The listener sets ``position``* to the paragraph it is on. The generator stays
``GENERATE_AHEAD_PARAGRAPHS`` in front of it and no further -- far enough that a
skip or two lands on audio that already exists, near enough that scrolling
through a document does not pay for a dozen paragraphs nobody waits to hear.

*The listener takes ``blocks(index)``* and gets audio, whether that means
replaying what is cached or following along behind the engine as it is made.
Only a paragraph nothing has made -- a skip past the generated region -- costs a
generation, and then the listener says so with ``aim``.

Everything is threads and no async: the engine is a blocking, GIL-releasing
compute job, and the HTTP side runs it in a worker anyway.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

import archive
import llm_stream
import speech_config
import voice_store
from breeze_pipeline import MAX_WORDS_PER_CHUNK
from speech_pipeline import TextAggregator, pump_llm

logger = logging.getLogger("breeze.paragraphs")

# How far in front of the listener the engine is allowed to get. Far enough that
# a skip or two lands on audio that already exists; near enough that holding the
# key does not pay for a dozen paragraphs nobody waits to hear.
GENERATE_AHEAD_PARAGRAPHS = 3

# What the cache may hold before the paragraphs furthest from the listener are
# dropped. Engine audio is float32 at 24 kHz, so this is roughly ninety minutes
# of speech -- longer than anything a clipboard read produces, and a hard stop
# for the pathological case rather than a limit anyone is meant to reach.
CACHE_BUDGET_BYTES = 512 * 1024 * 1024


class AnyEvent:
    """Set when any of the events it wraps is set."""

    def __init__(self, *events: Any) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class ParagraphBook:
    """The document's paragraphs, as sentence chunks, as they become known.

    Written by one producer -- the clipboard splitter, or the language model's
    stream -- and read by the generator, which may be anywhere in the document
    including behind the producer. Indices are dense over paragraphs that
    actually have text: a run of blank lines is a formatting artefact, not a
    paragraph a listener would count.
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


class _CachedParagraph:
    """One paragraph's audio, and whether more of it is coming."""

    __slots__ = ("blocks", "state", "epoch", "size", "error")

    def __init__(self, epoch: int) -> None:
        self.blocks: list[np.ndarray] = []
        self.state = AudioCache.GENERATING
        self.epoch = epoch
        self.size = 0
        self.error: BaseException | None = None


class AudioCache:
    """Paragraph audio the engine has already produced, for this read only.

    Blocks are kept exactly as the engine yielded them -- float32 at the engine's
    own rate, before anything resamples or encodes them -- so a paragraph sounds
    the same whether it is heard as it is made or replayed ten minutes later.

    A reader may be taking blocks out of a paragraph while the generator is
    still putting them in, which is the case that matters: arriving at a
    paragraph the engine started a second ago should start playing now and keep
    up as the rest arrives, not wait for the end and not start again. So each
    paragraph carries a state, and ``blocks`` blocks while that state says more
    is coming.

    This is not the archive. Nothing here is written to disk, and it is dropped
    whole when the read ends.
    """

    GENERATING, COMPLETE, ABANDONED = "generating", "complete", "abandoned"

    def __init__(self, budget_bytes: int = CACHE_BUDGET_BYTES) -> None:
        self._budget = max(0, int(budget_bytes))
        self._entries: dict[int, _CachedParagraph] = {}
        self._epoch = 0
        self._bytes = 0
        self._condition = threading.Condition()

    # -- generator side ---------------------------------------------------
    def begin(self, index: int) -> None:
        """Claim ``index`` for a fresh generation, dropping any partial audio."""
        with self._condition:
            existing = self._entries.get(index)
            if existing is not None:
                self._bytes -= existing.size
            self._epoch += 1
            self._entries[index] = _CachedParagraph(self._epoch)
            self._condition.notify_all()

    def append(self, index: int, block: np.ndarray) -> None:
        audio = np.asarray(block, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        with self._condition:
            entry = self._entries.get(index)
            if entry is None or entry.state is not self.GENERATING:
                return
            entry.blocks.append(audio)
            entry.size += audio.nbytes
            self._bytes += audio.nbytes
            self._condition.notify_all()

    def finish(self, index: int) -> None:
        """The paragraph is whole and can be replayed from now on."""
        self._settle(index, self.COMPLETE)

    def abandon(self, index: int) -> None:
        """Generation stopped part way. What is here is not a paragraph."""
        self._settle(index, self.ABANDONED)

    def fail(self, index: int, error: BaseException) -> None:
        with self._condition:
            entry = self._entries.get(index)
            if entry is not None:
                entry.state = self.ABANDONED
                entry.error = error
            self._condition.notify_all()

    def _settle(self, index: int, state: str) -> None:
        with self._condition:
            entry = self._entries.get(index)
            if entry is not None and entry.state is self.GENERATING:
                entry.state = state
            self._condition.notify_all()

    # -- reader side ------------------------------------------------------
    def state(self, index: int) -> str | None:
        with self._condition:
            entry = self._entries.get(index)
            return entry.state if entry is not None else None

    def live(self, index: int) -> bool:
        """Whether this paragraph's audio exists or is on its way.

        The one question a listener asks: yes means listen, no means somebody
        has to generate it first.
        """
        return self.state(index) in {self.GENERATING, self.COMPLETE}

    def complete(self, index: int) -> bool:
        return self.state(index) == self.COMPLETE

    def cached(self) -> list[int]:
        with self._condition:
            return sorted(
                index
                for index, entry in self._entries.items()
                if entry.state is self.COMPLETE
            )

    def frames(self, index: int) -> int:
        """How many samples a finished paragraph holds. 0 if it is not finished."""
        with self._condition:
            entry = self._entries.get(index)
            if entry is None or entry.state is not self.COMPLETE:
                return 0
            return sum(int(block.size) for block in entry.blocks)

    def blocks(self, index: int, stop: Any) -> Iterator[np.ndarray]:
        """Paragraph ``index``'s audio, waiting for blocks still being made.

        Ends when the paragraph is finished and exhausted, when it is abandoned
        or regenerated underneath the reader, or when ``stop`` is set. A
        generator-side failure is raised here, on the reader's thread, so a
        failed paragraph fails the read rather than ending it quietly.
        """
        cursor = 0
        epoch: int | None = None
        while True:
            with self._condition:
                while True:
                    if stop.is_set():
                        return
                    entry = self._entries.get(index)
                    if entry is not None:
                        if epoch is None:
                            epoch = entry.epoch
                        elif entry.epoch != epoch:
                            return  # regenerated under us; that reader owns it
                        if entry.error is not None:
                            raise entry.error
                        if cursor < len(entry.blocks):
                            break
                        if entry.state is not self.GENERATING:
                            return
                    elif epoch is not None:
                        return  # dropped under us
                    self._condition.wait(0.2)
                block = entry.blocks[cursor]
                cursor += 1
            yield block

    # -- housekeeping -----------------------------------------------------
    def trim(self, keep_near: int) -> None:
        """Drop whole paragraphs, furthest from the listener first, to fit the budget.

        Only settled paragraphs are candidates, and never the one being listened
        to. A dropped paragraph is not lost -- it is regenerated if the listener
        ever goes back that far -- so this trades the rarest case for a bound on
        how much a very long read can hold.
        """
        with self._condition:
            if self._bytes <= self._budget:
                return
            candidates = sorted(
                (
                    index
                    for index, entry in self._entries.items()
                    if entry.state is not self.GENERATING and index != keep_near
                ),
                key=lambda index: (-abs(index - keep_near), index),
            )
            for index in candidates:
                if self._bytes <= self._budget:
                    break
                entry = self._entries.pop(index)
                self._bytes -= entry.size
                logger.debug("Dropped cached audio for paragraph %d", index)

    def discard(self, index: int) -> None:
        """Forget a paragraph that is neither whole nor being made.

        What a cut-short generation left behind is half a paragraph: replaying
        it would speak half the words and call it done. Dropping it is what
        makes a reader wait for the generation that is about to replace it
        instead of reading the remains of the one before.
        """
        with self._condition:
            entry = self._entries.get(index)
            if entry is not None and entry.state is self.ABANDONED:
                del self._entries[index]
                self._bytes -= entry.size
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._entries.clear()
            self._bytes = 0
            self._condition.notify_all()

    @property
    def size_bytes(self) -> int:
        with self._condition:
            return self._bytes


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


class ParagraphEngine:
    """The text and the audio of one read, generated a little ahead of the ear.

    Owns two threads: a producer filling the book -- the clipboard splitter
    returns immediately, the language model streams for as long as it takes --
    and a generator turning paragraphs into audio. Listeners own neither; they
    set ``position``, take ``blocks``, and ``aim`` the generator on the rare
    occasion they land somewhere nothing has been made.
    """

    def __init__(
        self,
        prepared: dict[str, Any],
        text: str,
        *,
        budget_bytes: int = CACHE_BUDGET_BYTES,
    ) -> None:
        self._prepared = prepared
        self._text = text
        self._job = prepared["job"]
        self._config = prepared["config"]
        self._recording: archive.Recording = prepared["recording"]

        self.engine = self._job["engine"]
        self.book = ParagraphBook()
        self.cache = AudioCache(budget_bytes)

        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._interrupt = threading.Event()

        self._position = 0
        self._generating = 0
        self._aimed = False

        self.started_at = time.time()
        self._rotator: voice_store.ReferenceRotator | None = None
        self._producer: threading.Thread | None = None
        self._generator: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin producing text and audio. Returns immediately."""
        self._build_rotator(self._prepared["voice_id"])
        if self._prepared["use_llm"]:
            self._producer = threading.Thread(
                target=self._produce_llm, name="read-llm", daemon=True
            )
            self._producer.start()
        else:
            self._fill_from_text()
        self._generator = threading.Thread(
            target=self._generate, name="read-generator", daemon=True
        )
        self._generator.start()

    def stop(self) -> None:
        """Wind the engine up. Safe from any thread, and more than once."""
        with self._lock:
            self._stop.set()
            self._interrupt.set()
            self._wake.notify_all()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the generator to leave the engine. Call after ``stop``."""
        generator = self._generator
        if generator is not None and generator is not threading.current_thread():
            generator.join(timeout)

    def close(self) -> None:
        """Stop, wait, and drop the audio. The end of a read."""
        self.stop()
        self.join(30.0)
        self.cache.clear()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _build_rotator(self, voice_id: str | None) -> None:
        """The reference rotation, carried across paragraphs so it keeps its place.

        One rotator for the whole read rather than one per paragraph: the
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

    @property
    def rotator(self) -> voice_store.ReferenceRotator | None:
        return self._rotator

    def _fill_from_text(self) -> None:
        """The clipboard path: every paragraph is known before a word is said."""
        chunks = self._job["chunks"]
        groups = self._job["paragraph_ids"] or [0] * len(chunks)
        for chunk, group in zip(chunks, groups):
            self.book.add(group, chunk)
        self.book.finish()

    def _produce_llm(self) -> None:
        """The model path: paragraphs arrive while earlier ones are being said.

        Deliberately not throttled by the listener. Text is kilobytes and the
        model is far faster than synthesis, so letting it run to the end costs
        nothing and buys the one thing a listener cannot recover on its own: the
        text of a paragraph already heard that somebody wants to hear again.
        """
        llm_settings = self._config.get("llm") or {}
        sink = _BookSink(self.book, self._stop)
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
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            logger.exception("The language model stream failed")
            self.book.finish(exc)

    # -- what a listener uses ---------------------------------------------
    @property
    def position(self) -> int:
        with self._lock:
            return self._position

    @position.setter
    def position(self, index: int) -> None:
        """Where the listener is. The generator stays just in front of it."""
        with self._lock:
            if index == self._position:
                return
            self._position = index
            self._wake.notify_all()

    @property
    def generating(self) -> int:
        with self._lock:
            return self._generating

    def wait_for(self, index: int, stop: Any) -> bool:
        """Block until paragraph ``index`` has text. False means it never will."""
        return self.book.wait_for(index, AnyEvent(self._stop, stop))

    def blocks(self, index: int, stop: Any) -> Iterator[np.ndarray]:
        """Paragraph ``index``'s audio, generating it first if nothing has.

        The listener does not have to know which case it is in -- replaying what
        was heard before, following the engine through a paragraph it started a
        moment ago, or waiting on one nothing has touched. Only the last of
        those moves the engine.
        """
        if not self.cache.live(index):
            self.aim(index)
        yield from self.cache.blocks(index, AnyEvent(self._stop, stop))

    def complete(self, index: int) -> bool:
        return self.cache.complete(index)

    def count(self) -> int:
        return self.book.count()

    @property
    def final(self) -> bool:
        return self.book.complete

    @property
    def error(self) -> BaseException | None:
        return self.book.error

    def text(self, index: int) -> str:
        return self.book.text(index)

    def clamp(self, target: int) -> int:
        """Keep a seek inside the document, allowing for one still being written.

        One past the last known paragraph is legal while the model is still
        producing: it means "the next one, when it exists", and a listener
        blocks there until it does.
        """
        count = self.book.count()
        ceiling = max(0, count - 1) if self.book.complete else count
        return max(0, min(target, ceiling))

    def aim(self, index: int) -> None:
        """Point the engine at ``index`` and cut short whatever it was making.

        Only worth calling for a paragraph that has no audio and none coming,
        which is what a skip past the generated region means.
        """
        with self._lock:
            if self._generating == index and self.cache.live(index):
                return
            self.cache.discard(index)
            self._generating = index
            self._aimed = True
            self._interrupt.set()
            self._wake.notify_all()

    # -- the generator ----------------------------------------------------
    def _generate(self) -> None:
        """Keep the cache a few paragraphs in front of the listener.

        Walks forwards on its own and is only ever moved by a listener asking
        for a paragraph nothing has made yet. Landing on cached audio does not
        touch this thread at all: it carries on with the paragraph it was in,
        which is the whole point of generating ahead.
        """
        while not self._stop.is_set():
            with self._lock:
                # Cleared under the lock that ``aim`` takes to set it, or an aim
                # landing between the two would be cleared away and the engine
                # would generate the paragraph it had just been moved off.
                self._interrupt.clear()
                index = self._generating
                self._aimed = False

            if not self._await_room(index):
                continue
            if self._stop.is_set():
                break

            arrival = AnyEvent(self._stop, self._interrupt)
            if not self.book.wait_for(index, arrival):
                if self._stop.is_set():
                    break
                # The end of the document, or an aim while waiting for text.
                self._park()
                continue

            if not self.cache.complete(index):
                self._generate_paragraph(index)
                self.cache.trim(self.position)

            with self._lock:
                if not self._aimed and not self._interrupt.is_set():
                    self._generating = index + 1
                    self._wake.notify_all()

    def _await_room(self, index: int) -> bool:
        """Hold the engine within ``GENERATE_AHEAD_PARAGRAPHS`` of the listener.

        False means stop, or that an aim moved the engine while it waited -- in
        both cases the caller starts its loop again rather than generating the
        paragraph it was about to.
        """
        with self._lock:
            while not self._stop.is_set() and not self._aimed:
                if index - self._position <= GENERATE_AHEAD_PARAGRAPHS:
                    return True
                self._wake.wait(0.25)
        return False

    def _park(self) -> None:
        """Nothing left to generate. Sleep until an aim needs something made."""
        with self._lock:
            while not self._stop.is_set() and not self._aimed:
                self._wake.wait(0.25)

    def _generate_paragraph(self, index: int) -> None:
        """Synthesize one paragraph into the cache.

        One ``stream_live`` per paragraph is what makes this affordable: the
        engine gate, the codec request and the allocator trim all live and die
        inside this call, so an abandoned paragraph leaves nothing behind for
        the next one to trip over.
        """
        options = {
            key: value
            for key, value in self._job["options"].items()
            if key not in {"chunk_refs", "paragraph_ids"}
        }
        stop_source = AnyEvent(self._stop, self._interrupt)
        sentence_groups = self.book.complete and self.book.count() <= 1

        def source() -> Iterator[tuple[str, tuple[str, str] | None, bool]]:
            for position, chunk in enumerate(self.book.chunks(index, stop_source)):
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

        self.cache.begin(index)
        whole = False
        events = self.engine.stream_live(
            source(), request_id=f"{self._job['request_id']}-p{index}", **options
        )
        try:
            for kind, audio, _position in events:
                if self._interrupt.is_set() or self._stop.is_set():
                    break
                if kind == "audio" and audio is not None and audio.size:
                    self._recording.note_audio(audio)
                    self.cache.append(index, audio)
            else:
                whole = not (self._interrupt.is_set() or self._stop.is_set())
        except BaseException as exc:  # noqa: BLE001 - handed to the listener
            logger.exception("Generating paragraph %d failed", index)
            self.cache.fail(index, exc)
            return
        finally:
            # Closed here, on the thread that iterated it, so the engine's own
            # finally runs now rather than whenever the collector notices.
            events.close()

        if whole:
            self.cache.finish(index)
        else:
            self.cache.abandon(index)

    # -- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paragraph": self._position,
                "paragraph_generating": self._generating,
                "paragraphs": self.book.count(),
                "paragraphs_final": self.book.complete,
                "paragraphs_cached": self.cache.cached(),
                "cache_bytes": self.cache.size_bytes,
            }
