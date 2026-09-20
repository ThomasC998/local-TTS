"""One spoken document, seekable a paragraph at a time.

The hotkey path used to be a straight line: text in, one long generation, audio
out, and the only control was "stop". That is fine until you want to skip the
paragraph you are half way through -- at which point you need four things the
straight line cannot give you.

*A paragraph you can go back to.* The ``ParagraphBook`` keeps every paragraph
the document has produced so far, whether it came from the clipboard whole or
is still being written by the language model one token at a time.

*The audio of a paragraph you can go back to.* Text alone only buys the right
to synthesize it again, which is seconds of silence for something that was
computed a minute ago. The ``AudioCache`` keeps what the engine produced, so a
paragraph heard once is replayed rather than made again. It lives and dies with
the session: a stop, or a new read, starts from an empty one.

*A generation you can abandon without leaking.* Each paragraph is its own
``stream_live`` call, so cancelling one closes its codec request, releases the
engine gate and trims the allocator pools on the way out -- the same lifecycle
a web-UI request gets, and the reason skipping through twenty paragraphs costs
no more memory than speaking one.

*A press that means "and the next one too".* Skips are debounced. Each press
silences what is playing and moves a target index; only once the presses stop
for a moment does the session land. Holding the key scrolls through the
document instead of speaking every paragraph on the way past.

Generation and playback are separate threads, because the engine is faster than
the speaker and there is no reason to make it wait. The generator runs a few
paragraphs in front of the ear, filling the cache; playback takes paragraphs
out of it. That gap is what makes a skip cheap: the paragraph you land on has
usually been generated already, so the press costs the debounce and nothing
else, and the generator carries on from where it was rather than starting over.
Only a skip past the generator's position moves it -- there the paragraph
really does have to be made, exactly as it always was.

Everything the session reports -- the paragraph number, where a skip counts
from -- is measured at the speaker, not at the engine. The two are seconds
apart, and the listener only knows about one of them.

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

import numpy as np

import archive
import audio_out
import llm_stream
import speech_config
import voice_store
from breeze_pipeline import MAX_WORDS_PER_CHUNK
from speech_pipeline import TextAggregator, pump_llm

logger = logging.getLogger("breeze.session")

# How long the session waits after the last skip press before it lands. Long
# enough to press again without hearing anything start up, short enough that a
# single press does not feel like a stall.
SKIP_DEBOUNCE_SECONDS = 2.0

# How far in front of the speaker the engine is allowed to get. Far enough that
# a skip or two lands on audio that already exists; near enough that holding the
# key does not pay for a dozen paragraphs nobody waits to hear.
GENERATE_AHEAD_PARAGRAPHS = 3

# How much audio may sit in the player's queue. The queue is unbounded and
# cached audio would fill it instantly, which would put the sound seconds behind
# the paragraph the session thinks it is on -- and a skip press is aimed at what
# is being heard.
PLAYER_LEAD_SECONDS = 1.0

# ...and how little of it may be left before the session calls a paragraph done
# and moves to the next. A paragraph can be shorter than the lead, so handing
# the last block over is not evidence anybody has heard it. What is left covers
# the gap while the next paragraph's first block is found.
PARAGRAPH_HANDOVER_SECONDS = 0.25

# The largest piece handed to the player at once. The queue is how the session
# measures its distance from the ear, and a piece is only counted once it is
# whole, so anything longer than this would overshoot the lead by the difference.
WRITE_SLICE_SECONDS = 0.25

# What the cache may hold before the paragraphs furthest from the ear are
# dropped. Engine audio is float32 at 24 kHz, so this is roughly ninety minutes
# of speech -- longer than anything a clipboard read produces, and a hard stop
# for the pathological case rather than a limit anyone is meant to reach.
CACHE_BUDGET_BYTES = 512 * 1024 * 1024

# States a caller can see. Only the first four are live.
LIVE_STATES = frozenset({"starting", "speaking", "seeking", "paused"})


def _paced(blocks: Iterator[np.ndarray], sample_rate: int) -> Iterator[np.ndarray]:
    """Engine blocks, cut down to something the queue can be measured in.

    Slices are views, not copies, and a block already short enough passes
    straight through -- which is every block the engine produces live. It is
    cached audio replayed at once that this exists for.
    """
    limit = max(1, int(sample_rate * WRITE_SLICE_SECONDS))
    for block in blocks:
        if block.size <= limit:
            yield block
            continue
        for start in range(0, block.size, limit):
            yield block[start : start + limit]


class _AnyEvent:
    """Set when any of the events it wraps is set."""

    def __init__(self, *events: threading.Event) -> None:
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
    """Paragraph audio the engine has already produced, for this session only.

    Blocks are kept exactly as the engine yielded them -- float32 at the engine's
    own rate, before the player resamples anything -- so a paragraph sounds the
    same whether it is heard as it is made or replayed ten minutes later.

    A reader may be taking blocks out of a paragraph while the generator is
    still putting them in, which is the case that matters: skipping onto a
    paragraph the engine started a second ago should start speaking now and keep
    up as the rest arrives, not wait for the end and not start again. So each
    paragraph carries a state, and ``blocks`` blocks while that state says more
    is coming.

    This is not the archive. Nothing here is written to disk, and it is dropped
    whole when the session ends.
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

        The one question playback asks: yes means listen, no means somebody has
        to generate it first.
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

    def blocks(self, index: int, stop: Any) -> Iterator[np.ndarray]:
        """Paragraph ``index``'s audio, waiting for blocks still being made.

        Ends when the paragraph is finished and exhausted, when it is abandoned
        or regenerated underneath the reader, or when ``stop`` is set. A
        generator-side failure is raised here, on the reader's thread, so a
        failed paragraph fails the utterance rather than ending it quietly.
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
        """Drop whole paragraphs, furthest from the ear first, to fit the budget.

        Only finished paragraphs are candidates, and never the one being
        listened to. A dropped paragraph is not lost -- it is regenerated if the
        listener ever goes back that far -- so this trades the rarest case for a
        bound on how much a very long read can hold.
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


class SpeechSession:
    """One utterance on the machine's speakers, with transport controls.

    Owns three threads: a producer filling the book with text, a generator
    turning paragraphs into audio a little ahead of the ear, and a playback
    thread taking that audio out of the cache and onto the speaker. Everything a
    hotkey does -- skip, pause, stop -- only sets flags; the threads are what
    close their own generators, because a generator may only be closed from the
    thread iterating it.
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
        self._cache = AudioCache()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._generation_interrupt = threading.Event()
        self._done = threading.Event()

        self._state = "starting"
        self._error: str | None = None
        self._index = 0              # what the listener is hearing
        self._generating = 0         # what the engine is working on
        self._aimed = False          # a seek moved the engine; do not advance it
        self._seek_target: int | None = None
        self._seek_deadline = 0.0
        self._paused = False
        self._pause_reason: str | None = None
        self._player_open = False
        self._spoken: list[int] = []

        self._rotator: voice_store.ReferenceRotator | None = None
        self._player: threading.Thread | None = None
        self._generator: threading.Thread | None = None
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
        self._generator = threading.Thread(
            target=self._generate, name="speak-generator", daemon=True
        )
        self._generator.start()
        self._player = threading.Thread(
            target=self._run, name="speak-player", daemon=True
        )
        self._player.start()

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
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            logger.exception("The language model stream failed")
            self._book.finish(exc)

    # -- the generator ----------------------------------------------------
    def _generate(self) -> None:
        """Keep the cache a few paragraphs in front of the speaker.

        Walks forwards on its own and is only ever moved by playback asking for
        a paragraph nothing has made yet. A skip that lands on cached audio does
        not touch this thread at all: it carries on with the paragraph it was
        in, which is the whole point of generating ahead.
        """
        while not self._stop.is_set():
            with self._lock:
                # Cleared under the lock that a seek takes to set it, or a seek
                # landing between the two would be cleared away and the engine
                # would generate the paragraph it had just been moved off.
                self._generation_interrupt.clear()
                index = self._generating
                self._aimed = False

            if not self._await_room(index):
                continue
            if self._stop.is_set():
                break

            arrival = _AnyEvent(self._stop, self._generation_interrupt)
            if not self._book.wait_for(index, arrival):
                if self._stop.is_set():
                    break
                # The end of the document, or a redirect while waiting for text.
                self._park()
                continue

            if not self._cache.complete(index):
                self._generate_paragraph(index)
                self._cache.trim(self._index)

            with self._lock:
                if not self._aimed and not self._generation_interrupt.is_set():
                    self._generating = index + 1
                    self._wake.notify_all()

    def _await_room(self, index: int) -> bool:
        """Hold the engine within ``GENERATE_AHEAD_PARAGRAPHS`` of the ear.

        False means stop, or that a seek moved the engine while it waited -- in
        both cases the caller starts its loop again rather than generating the
        paragraph it was about to.
        """
        with self._lock:
            while not self._stop.is_set() and not self._aimed:
                if index - self._index <= GENERATE_AHEAD_PARAGRAPHS:
                    return True
                self._wake.wait(0.25)
        return False

    def _park(self) -> None:
        """Nothing left to generate. Sleep until a seek needs something made."""
        with self._lock:
            while not self._stop.is_set() and not self._aimed:
                self._wake.wait(0.25)

    def _aim_generator(self, index: int) -> None:
        """Point the engine at ``index`` and cut short whatever it was making.

        Only called for a paragraph that has no audio and none coming, which is
        what a skip past the generated region means.
        """
        with self._lock:
            if self._generating == index and self._cache.live(index):
                return
            self._cache.discard(index)
            self._generating = index
            self._aimed = True
            self._generation_interrupt.set()
            self._wake.notify_all()

    def _generate_paragraph(self, index: int) -> None:
        """Synthesize one paragraph into the cache.

        One ``stream_live`` per paragraph is what makes this affordable: the
        engine gate, the codec request and the allocator trim all live and die
        inside this call, so an abandoned paragraph leaves nothing behind for
        the next one to trip over.
        """
        engine = self._engine
        options = {
            key: value
            for key, value in self._job["options"].items()
            if key not in {"chunk_refs", "paragraph_ids"}
        }
        stop_source = _AnyEvent(self._stop, self._generation_interrupt)
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

        self._cache.begin(index)
        whole = False
        events = engine.stream_live(
            source(), request_id=f"{self._job['request_id']}-p{index}", **options
        )
        try:
            for kind, audio, _position in events:
                if self._generation_interrupt.is_set() or self._stop.is_set():
                    break
                if kind == "audio" and audio is not None and audio.size:
                    self._recording.note_audio(audio)
                    self._cache.append(index, audio)
            else:
                whole = not (
                    self._generation_interrupt.is_set() or self._stop.is_set()
                )
        except BaseException as exc:  # noqa: BLE001 - handed to the listener
            logger.exception("Generating paragraph %d failed", index)
            self._cache.fail(index, exc)
            return
        finally:
            # Closed here, on the thread that iterated it, so the engine's own
            # finally runs now rather than whenever the collector notices.
            events.close()

        if whole:
            self._cache.finish(index)
        else:
            self._cache.abandon(index)

    # -- playback ---------------------------------------------------------
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
                interrupted = self._play_paragraph(index)
                if self._stop.is_set():
                    break
                if interrupted:
                    continue  # a skip, a pause or a lost paragraph decides next
                with self._lock:
                    if self._seek_target is None and not self._paused:
                        self._index = index + 1
                        self._wake.notify_all()
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
                    self._wake.notify_all()
                return True
        return False

    def _play_paragraph(self, index: int) -> bool:
        """Put paragraph ``index`` on the speaker. True if it was cut short.

        Nothing is synthesized here. Either the audio is in the cache -- made a
        moment ago by the generator running ahead, or heard once already -- or
        the generator is pointed at this paragraph and its blocks are taken as
        they land. Both look the same from here, which is why a skip backwards
        and a skip onto a paragraph the engine is half way through both start
        speaking the instant the presses stop.
        """
        player = audio_out.PLAYER
        playback = self._config.get("playback") or {}

        if not self._cache.live(index):
            self._aim_generator(index)

        with self._lock:
            open_now = not self._player_open
            self._player_open = True
        if open_now:
            player.start(
                self.utterance_id,
                self._engine.sample_rate,
                prebuffer_ms=int(playback.get("prebuffer_ms") or 400),
                device=playback.get("device"),
            )
        else:
            # The conceptual gap between paragraphs, written before the next
            # one starts so it also covers the engine spinning up.
            player.write(self.utterance_id, self._engine.paragraph_pause_buffer)

        # Never hold less than the player is waiting for before it opens the
        # stream, or nothing would ever start playing and nothing would drain.
        lead = max(
            PLAYER_LEAD_SECONDS,
            int(playback.get("prebuffer_ms") or 400) / 1000.0 + WRITE_SLICE_SECONDS,
        )

        interrupted = False
        cached = self._cache.blocks(index, _AnyEvent(self._stop, self._interrupt))
        for block in _paced(cached, self._engine.sample_rate):
            if self._interrupt.is_set() or self._stop.is_set():
                interrupted = True
                break
            if not player.write(self.utterance_id, block):
                interrupted = True
                break
            if not self._wait_until(player, lead):
                interrupted = True
                break

        if self._interrupt.is_set() or self._stop.is_set():
            interrupted = True
        if interrupted:
            return True
        if not self._cache.complete(index):
            # The cache lost this paragraph underneath us -- generation was
            # aimed elsewhere, or it was trimmed. Stay put and have it made.
            return True

        with self._lock:
            if not self._spoken or self._spoken[-1] != index:
                self._spoken.append(index)
        # Heard, not merely handed over: a paragraph can be shorter than the
        # queue, and moving on now would put the position a paragraph in front
        # of the listener with no way for a skip press to mean what it says.
        return not self._wait_until(player, PARAGRAPH_HANDOVER_SECONDS)

    def _wait_until(self, player: Any, buffered_seconds: float) -> bool:
        """Wait for the queue to fall to ``buffered_seconds``. False: interrupted.

        A player that cannot say how much it holds is not waited on, and neither
        is one that is not playing yet -- before the prebuffer is reached
        nothing is draining, so waiting would be waiting for good.
        """
        while not (self._stop.is_set() or self._interrupt.is_set()):
            try:
                if not player.is_playing():
                    return True
                buffered = float((player.status() or {}).get("buffered_seconds") or 0.0)
            except Exception:  # noqa: BLE001 - a player that cannot say never waits
                return True
            if buffered <= buffered_seconds:
                return True
            time.sleep(0.05)
        return False

    def _finish(self, error: str | None) -> None:
        player = audio_out.PLAYER
        cancelled = self._stop.is_set()

        # Wind the engine up first: it writes into the recording, and the
        # archive is about to be written from it.
        with self._lock:
            self._stop.set()
            self._generation_interrupt.set()
            self._wake.notify_all()
        generator = self._generator
        if generator is not None and generator is not threading.current_thread():
            generator.join(30.0)

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
        # The cache is the session's, not the document's: the archive keeps
        # what was said, and nothing here outlives the read that made it.
        self._cache.clear()
        self._done.set()

    # -- controls ---------------------------------------------------------
    def _clamp(self, target: int) -> int:
        """Keep a seek inside the document, allowing for one still being written.

        One past the last known paragraph is legal while the model is still
        producing: it means "the next one, when it exists", and playback blocks
        there until it does.
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
        """Move ``delta`` paragraphs from what is being heard and arm the debounce.

        Counted from the speaker, not the engine, which is usually a paragraph
        or two further on. The engine is left alone: if the paragraph landed on
        has been made already -- forwards into what was generated ahead, or
        backwards into what was heard before -- it is replayed from the cache
        and generation carries on undisturbed. Only a landing past everything
        made so far moves the engine, and then it starts there.

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
            self._stop.set()
            self._interrupt.set()
            self._generation_interrupt.set()
            self._wake.notify_all()
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
                "paragraph_generating": self._generating,
                "paragraphs": self._book.count(),
                "paragraphs_final": self._book.complete,
                "paragraphs_cached": len(self._cache.cached()),
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "paragraph_preview": self._book.text(self._index)[:200],
            }
