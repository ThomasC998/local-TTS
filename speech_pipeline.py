"""Turn a language model's token stream into chunks the TTS engine can speak.

The shape of the problem: Gemini emits fragments of words, the engine wants
whole sentences, and the listener wants audio to start now. So text is buffered
only until a complete sentence exists, then handed straight over.

Two properties keep this robust when the model is slow or pauses mid-sentence:

*No backpressure on the model.* Text is tiny -- a whole article is a few
kilobytes -- so the queue between the two is effectively unbounded and the
producer never blocks. That removes the failure this pipeline would otherwise
be prone to: stalling the HTTP stream because the consumer stopped reading.

*The consumer waits, it does not die.* The engine holds its gate across a wait
so the document's chunks stay contiguous, and gives up only after a long idle
timeout -- which ends the utterance cleanly rather than hanging forever.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Iterator
from typing import Any

from breeze_pipeline import MAX_WORDS_PER_CHUNK, validate_and_chunk_text

logger = logging.getLogger("breeze.speech")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
# A terminal only counts once whitespace follows it: until then "3." could be
# the start of "3.5" and the sentence is not over.
_COMPLETE_SENTENCE = re.compile(r"[.!?。！？]['\"”’)\]]*(?=\s)")

# How long the engine waits for the next chunk before ending the utterance.
IDLE_TIMEOUT_SECONDS = 45.0
# How long a partial sentence may sit unfinished before it is spoken anyway.
STALL_FLUSH_SECONDS = 3.0
# ...and how many words it needs for that to be worth doing. Below this, a
# fragment spoken on its own just sounds clipped.
STALL_FLUSH_MIN_WORDS = 12


class TextAggregator:
    """Accumulates deltas and emits ``(text, paragraph_index)`` chunks.

    A chunk is released as soon as it is a complete sentence, so the first audio
    starts after one sentence rather than after the whole rewrite. Paragraph
    indices ride along because a blank line is where both the longer pause and
    the reference rotation happen.
    """

    def __init__(self, max_words: int = MAX_WORDS_PER_CHUNK) -> None:
        self.max_words = max_words
        self._buffer = ""
        self._paragraph = 0

    def feed(self, delta: str) -> list[tuple[str, int]]:
        self._buffer += delta
        return self._drain()

    def _emit(self, text: str, out: list[tuple[str, int]]) -> None:
        for chunk in validate_and_chunk_text(text, max_words=self.max_words):
            out.append((chunk, self._paragraph))

    def _drain(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        while True:
            # A blank line closes the paragraph, whatever else is buffered.
            match = _PARAGRAPH_BREAK.search(self._buffer)
            if match:
                self._emit(self._buffer[: match.start()], out)
                self._buffer = self._buffer[match.end():]
                self._paragraph += 1
                continue

            # Everything up to the last completed sentence can go now.
            ends = list(_COMPLETE_SENTENCE.finditer(self._buffer))
            if ends:
                cut = ends[-1].end()
                self._emit(self._buffer[:cut], out)
                self._buffer = self._buffer[cut:].lstrip()
                continue

            # A run-on with no terminal in sight would otherwise buffer forever.
            # Release all but the tail, which may still be growing.
            if len(self._buffer.split()) > self.max_words * 2:
                pieces = validate_and_chunk_text(self._buffer, max_words=self.max_words)
                if len(pieces) > 1:
                    for piece in pieces[:-1]:
                        out.append((piece, self._paragraph))
                    self._buffer = pieces[-1]
            break
        return out

    def flush_stalled(self) -> list[tuple[str, int]]:
        """Speak an unfinished sentence that the model has stopped extending.

        The whole buffer goes, not all-but-the-tail: a sentence with no commas
        in it splits into exactly one piece, and holding that piece back as
        "still growing" is how a stalled fragment would sit unsaid forever.
        Once the model has been silent this long, waiting is the worse option.

        Only worth doing at all once there is enough text to stand on its own; a
        three-word fragment read with a falling intonation sounds worse than the
        pause it was meant to avoid.
        """
        if len(self._buffer.split()) < STALL_FLUSH_MIN_WORDS:
            return []
        pieces = validate_and_chunk_text(self._buffer, max_words=self.max_words)
        self._buffer = ""
        return [(piece, self._paragraph) for piece in pieces]

    def flush(self) -> list[tuple[str, int]]:
        """Everything left when the model's stream ends."""
        out: list[tuple[str, int]] = []
        remaining, self._buffer = self._buffer.strip(), ""
        if remaining:
            self._emit(remaining, out)
        return out


class ChunkPipe:
    """Thread-safe hand-off of ready chunks from producer to engine.

    Unbounded on purpose: see the module docstring. ``fail`` propagates a
    producer-side error to the consumer instead of ending the utterance early
    and silently, which would be indistinguishable from the model finishing.
    """

    _CLOSED = object()

    def __init__(self, idle_timeout: float = IDLE_TIMEOUT_SECONDS) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._idle_timeout = idle_timeout
        self._error: BaseException | None = None
        self._cancelled = threading.Event()

    def put(self, chunk: str, paragraph: int) -> None:
        self._queue.put((chunk, paragraph))

    def close(self) -> None:
        self._queue.put(self._CLOSED)

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._queue.put(self._CLOSED)

    def cancel(self) -> None:
        """Stop the consumer at the next chunk boundary."""
        self._cancelled.set()
        self._queue.put(self._CLOSED)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def items(
        self, reference_for: Any = None
    ) -> Iterator[tuple[str, tuple[str, str] | None, bool]]:
        """Chunks in engine form, blocking until each one is ready.

        ``reference_for(words, starts_paragraph)`` supplies the reference
        rotation, and is called here rather than in the producer so the switch
        is decided against what is actually about to be spoken.
        """
        previous_paragraph: int | None = None
        while not self._cancelled.is_set():
            try:
                item = self._queue.get(timeout=self._idle_timeout)
            except queue.Empty:
                logger.warning(
                    "No chunk for %.0fs; ending the utterance", self._idle_timeout
                )
                return
            if item is self._CLOSED:
                if self._error is not None:
                    raise self._error
                return

            chunk, paragraph = item
            starts_paragraph = (
                previous_paragraph is not None and paragraph != previous_paragraph
            )
            previous_paragraph = paragraph
            words = len(chunk.split())
            reference = (
                reference_for(words, starts_paragraph) if reference_for else None
            )
            yield chunk, reference, starts_paragraph


def pump_llm(
    pipe: ChunkPipe,
    deltas: Iterator[str],
    *,
    sentinel_filter: Any,
    sanitize: Any,
    aggregator: TextAggregator,
    on_raw: Any = None,
) -> None:
    """Read the model's stream to the end, feeding ready chunks into ``pipe``.

    Runs on its own thread. It reads as fast as the model produces, which is the
    point: the model is never made to wait on the speaker, so a slow synthesis
    can never stall the HTTP connection to Vertex.

    A watchdog covers the opposite stall. If the model goes quiet partway
    through a sentence, the engine would otherwise sit idle holding a fragment
    it was never told to speak; after ``STALL_FLUSH_SECONDS`` that fragment is
    released at its last clause boundary instead.
    """
    guard = threading.Lock()
    last_delta = [time.monotonic()]
    done = threading.Event()

    def emit(pieces: list[tuple[str, int]]) -> None:
        for chunk, paragraph in pieces:
            clean = sanitize(chunk).strip()
            if clean:
                pipe.put(clean, paragraph)

    def watchdog() -> None:
        while not done.wait(0.5):
            if pipe.cancelled:
                return
            if time.monotonic() - last_delta[0] < STALL_FLUSH_SECONDS:
                continue
            with guard:
                stalled = aggregator.flush_stalled()
            if stalled:
                logger.info("Model stalled mid-sentence; speaking %d buffered chunk(s)",
                            len(stalled))
                emit(stalled)
                last_delta[0] = time.monotonic()

    watcher = threading.Thread(target=watchdog, name="llm-stall-watchdog", daemon=True)
    watcher.start()

    try:
        for delta in deltas:
            if pipe.cancelled:
                break
            last_delta[0] = time.monotonic()
            if on_raw is not None:
                on_raw(delta)
            visible = sentinel_filter.feed(delta)
            if not visible:
                continue
            with guard:
                ready = aggregator.feed(visible)
            emit(ready)

        with guard:
            tail = sentinel_filter.flush()
            ready = aggregator.feed(tail) if tail else []
            ready += aggregator.flush()
        emit(ready)
    except BaseException as exc:  # noqa: BLE001 - handed to the consumer
        logger.exception("LLM stream failed")
        done.set()
        pipe.fail(exc)
        return
    done.set()
    pipe.close()
