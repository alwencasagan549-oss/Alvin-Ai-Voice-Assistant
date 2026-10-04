"""Low-latency streaming Text-to-Speech for Alvin.

Pipeline::

    text -> clean_text_for_tts -> chunk_sentences
         -> producer tasks: edge-tts MP3 fetch, up to TTS_PREFETCH_SENTENCES ahead
         -> one bounded queue of MP3 chunks per sentence
         -> consumer: progressive MP3 decode (miniaudio, off the event loop)
         -> EdgeSilenceTrimmer (drops leading/trailing dead air)
         -> TTS_SENTENCE_GAP_MS of digital silence between sentences
         -> 16 kHz / 16-bit / mono PCM frames

``synthesize_stream`` yields ``bytes`` audio frames and reports failures as
``{"type": "error", ...}`` events, so a caller can retry, fall back to a local
voice, or play a filler sound. ``synthesize_pcm`` is the bytes-only wrapper that
raises :class:`TTSSynthesisError` instead.

Latency notes (measured with ``scripts/bench_tts.py``):

* edge-tts delivers a sentence's MP3 as a burst, so progressive decode saves
  roughly the burst window (~0.1-0.3 s) rather than the whole synthesis.
* Prefetching the next sentences concurrently hides most of the network wait
  for sentences 2..N, which is where the multi-second win comes from.
* The trimmer holds back ``TTS_TRIM_EDGE_MS`` of audio to be able to drop
  end-of-sentence silence; set it to 0 for minimum time-to-first-audio at the
  cost of keeping the trailing silence.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import TypedDict

import edge_tts
import miniaudio
import numpy as np

from .config import settings

log = logging.getLogger("alvin.tts")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC = re.compile(r"\*(.+?)\*")
_MARKDOWN_HEADING = re.compile(r"^#+\s*", re.MULTILINE)
_MARKDOWN_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_MARKDOWN_INLINE_CODE = re.compile(r"`(.+?)`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff\U00002700-\U000027bf\U0001f900-\U0001f9ff]"
)

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLE_WIDTH = 2
TARGET_CHANNELS = 1

# Shortest phrase worth synthesizing: one word keeps the warm-up cheap while
# still exercising the edge-tts connection and the progressive MP3 decode.
WARMUP_TEXT = "Ready."

_PCM_DTYPE = np.dtype("<i2")

# Marks the end of a sentence's MP3 stream inside its queue.
_SENTENCE_DONE = object()


class TTSErrorEvent(TypedDict):
    """Error event yielded in place of audio when a sentence fails."""

    type: str
    stage: str
    message: str
    sentence_index: int
    sentence: str
    fatal: bool


class TTSSynthesisError(RuntimeError):
    """Raised by :func:`synthesize_pcm` when the stream reports a failure."""

    def __init__(self, event: TTSErrorEvent) -> None:
        super().__init__(event["message"])
        self.event = event


def clean_text_for_tts(text: str) -> str:
    """Strip Markdown artifacts, URLs, and emojis for clean TTS input."""
    if not text:
        return ""

    text = _MARKDOWN_CODE_BLOCK.sub("", text)
    text = _MARKDOWN_INLINE_CODE.sub(r"\1", text)
    text = _MARKDOWN_BOLD.sub(r"\1", text)
    text = _MARKDOWN_ITALIC.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = _URL.sub("", text)
    text = _EMOJI.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks for streaming TTS."""
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_END.split(text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks


def dbfs_to_amplitude(dbfs: float) -> int:
    """Convert a dBFS threshold to an int16 amplitude."""
    return max(1, min(32767, int(32768 * (10 ** (dbfs / 20)))))


def silence_gap_bytes(gap_ms: float) -> bytes:
    """Digital silence used to join two trimmed sentences."""
    frames = int(gap_ms * TARGET_SAMPLE_RATE / 1000)
    return np.zeros(frames, dtype=_PCM_DTYPE).tobytes()


def _decode_mp3(mp3: bytes) -> np.ndarray:
    """Decode MP3 to 16 kHz / signed 16-bit / mono samples."""
    decoded = miniaudio.decode(
        mp3,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=TARGET_CHANNELS,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    return np.frombuffer(decoded.samples.tobytes(), dtype=_PCM_DTYPE)


class EdgeSilenceTrimmer:
    """Drops the silence at both ends of a sentence while it streams.

    Frames quieter than ``threshold`` are considered silence. Leading silence is
    dropped immediately; a rolling ``hold`` window of the tail is kept back so
    end-of-sentence silence can still be discarded by :meth:`flush`.
    """

    def __init__(self, threshold: int, hold: int) -> None:
        self._threshold = threshold
        self._hold = max(0, hold)
        self._pending = np.empty(0, dtype=_PCM_DTYPE)
        self._speech_started = False

    @property
    def speech_started(self) -> bool:
        return self._speech_started

    def push(self, pcm: bytes) -> bytes:
        """Feed decoded PCM, returning the frames that are ready to send."""
        frames = np.frombuffer(pcm, dtype=_PCM_DTYPE)
        if frames.size == 0:
            return b""
        if self._pending.size:
            frames = np.concatenate((self._pending, frames))

        if not self._speech_started:
            loud = np.flatnonzero(np.abs(frames) >= self._threshold)
            if loud.size == 0:
                # Keep only the most recent hold frames so the onset survives.
                self._pending = (
                    frames[-self._hold :]
                    if self._hold
                    else np.empty(0, dtype=_PCM_DTYPE)
                )
                return b""
            self._speech_started = True
            self._pending = np.empty(0, dtype=_PCM_DTYPE)
            return frames[int(loud[0]) :].tobytes()

        if self._hold == 0:
            return frames.tobytes()
        if frames.size <= self._hold:
            self._pending = frames
            return b""
        self._pending = frames[-self._hold :]
        return frames[: -self._hold].tobytes()

    def flush(self) -> bytes:
        """End of sentence: emit the held tail unless it is pure silence."""
        pending, self._pending = self._pending, np.empty(0, dtype=_PCM_DTYPE)
        if pending.size == 0:
            return b""
        if not self._speech_started:
            return b""  # whole sentence was below the threshold
        if not (np.abs(pending) >= self._threshold).any():
            return b""
        return pending.tobytes()


class _PcmAssembler:
    """Incrementally decodes one sentence's MP3 stream into trimmed PCM.

    miniaudio decodes every complete frame in the buffer it is given and the
    result is a prefix-stable extension of the previous decode, so re-decoding
    the buffer and emitting only the new tail is artifact-free.
    """

    def __init__(self, threshold: int, hold: int, interval_bytes: int) -> None:
        self._buffer = bytearray()
        self._decoded_bytes = 0
        self._emitted_samples = 0
        self._interval = max(1, interval_bytes)
        self._trimmer = EdgeSilenceTrimmer(threshold, hold)

    async def push(self, mp3_chunk: bytes) -> bytes:
        """Add MP3 bytes, returning any newly available trimmed PCM."""
        self._buffer.extend(mp3_chunk)
        if len(self._buffer) - self._decoded_bytes < self._interval:
            return b""
        return await self._drain()

    async def finish(self) -> bytes:
        """Decode whatever is left and emit the held tail."""
        out = await self._drain()
        return out + self._trimmer.flush()

    async def _drain(self) -> bytes:
        if not self._buffer:
            return b""
        data = bytes(self._buffer)
        self._decoded_bytes = len(data)
        frames = await asyncio.to_thread(_decode_mp3, data)
        fresh = frames[self._emitted_samples :]
        if fresh.size == 0:
            return b""
        self._emitted_samples += fresh.size
        return self._trimmer.push(fresh.tobytes())


async def _fetch_sentence_mp3(
    sentence: str,
    voice: str,
    queue: asyncio.Queue,
    retries: int | None = None,
    backoff_ms: int | None = None,
) -> None:
    """Producer: stream one sentence's MP3 chunks into its queue.

    edge-tts occasionally fails a single fetch (429 / network blip / a dropped
    audio chunk). A failed sentence was previously dropped straight to an error
    event, which the caller skips -- audible to the user as "a word that never
    gets spoken". Retry the whole fetch a few times before giving up so
    transient failures stop losing words. ``CancelledError`` (barge-in) is
    propagated immediately, never retried.
    """
    retries = settings.tts_retries if retries is None else retries
    backoff_ms = settings.tts_retry_backoff_ms if backoff_ms is None else backoff_ms
    try:
        last_exc: BaseException | None = None
        got_audio = False
        for attempt in range(retries + 1):
            try:
                communicate = edge_tts.Communicate(sentence, voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        queue.put_nowait(chunk["data"])
                        got_audio = True
                return  # fully streamed; nothing left to do
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry, then forward
                last_exc = exc
                log.warning(
                    "edge-tts attempt %d/%d failed for %r: %s",
                    attempt + 1,
                    retries + 1,
                    sentence[:60],
                    exc,
                )
                # Only restart when nothing was emitted yet, so a retry can't
                # double-speak a sentence that already partially streamed.
                if got_audio or attempt >= retries:
                    break
                await asyncio.sleep(backoff_ms / 1000.0)
        if last_exc is not None:
            queue.put_nowait(last_exc)
    finally:
        queue.put_nowait(_SENTENCE_DONE)


def _error_event(exc: BaseException, index: int, sentence: str) -> TTSErrorEvent:
    return TTSErrorEvent(
        type="error",
        stage="tts",
        message=f"{type(exc).__name__}: {exc}",
        sentence_index=index,
        sentence=sentence[:120],
        fatal=False,
    )


async def synthesize_sentence_stream(
    sentence: str,
    voice: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream a single sentence as trimmed 16 kHz 16-bit mono PCM bytes.

    Provided for callers that already have one sentence in hand; prefer
    :func:`synthesize_stream`, which prefetches subsequent sentences.
    """
    if not sentence or not sentence.strip():
        return

    assembler = _PcmAssembler(
        threshold=dbfs_to_amplitude(settings.tts_silence_threshold_dbfs),
        hold=int(settings.tts_trim_edge_ms * TARGET_SAMPLE_RATE / 1000),
        interval_bytes=settings.tts_decode_interval_bytes,
    )
    queue: asyncio.Queue = asyncio.Queue()
    producer = asyncio.create_task(
        _fetch_sentence_mp3(sentence, voice or settings.tts_voice, queue)
    )
    try:
        while True:
            item = await queue.get()
            if item is _SENTENCE_DONE:
                break
            if isinstance(item, Exception):
                raise item
            out = await assembler.push(item)
            if out:
                yield out
        out = await assembler.finish()
        if out:
            yield out
    finally:
        producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - cleanup must never raise
            log.warning("error cleaning up TTS producer task")


async def synthesize_stream(
    text: str,
    voice: str | None = None,
    prefetch: int | None = None,
) -> AsyncGenerator[bytes | TTSErrorEvent, None]:
    """Stream text as sentence-chunked PCM audio, prefetching upcoming sentences.

    Yields 16 kHz / 16-bit / mono ``bytes`` frames. A sentence that fails yields a
    ``{"type": "error", ...}`` event instead and the stream continues, so the
    caller always receives the sentences that did work.
    """
    cleaned = clean_text_for_tts(text)
    sentences = chunk_sentences(cleaned)
    if not sentences:
        return

    voice = voice or settings.tts_voice
    window = max(1, prefetch if prefetch is not None else settings.tts_prefetch)
    threshold = dbfs_to_amplitude(settings.tts_silence_threshold_dbfs)
    hold = int(settings.tts_trim_edge_ms * TARGET_SAMPLE_RATE / 1000)
    gap = silence_gap_bytes(settings.tts_sentence_gap_ms)

    queues: list[asyncio.Queue] = [asyncio.Queue() for _ in sentences]
    producers: dict[int, asyncio.Task] = {}

    def spawn(index: int) -> None:
        if index < len(sentences) and index not in producers:
            producers[index] = asyncio.create_task(
                _fetch_sentence_mp3(sentences[index], voice, queues[index])
            )

    spoken_any = False
    try:
        for index in range(min(window, len(sentences))):
            spawn(index)

        for index, sentence in enumerate(sentences):
            spawn(index + window)  # keep the prefetch window full
            queue = queues[index]
            assembler = _PcmAssembler(
                threshold, hold, settings.tts_decode_interval_bytes
            )
            # The joining gap belongs between sentences, not between the chunks
            # a single sentence is decoded into.
            sentence_started = False

            async def emit(frames: bytes):
                """Yield PCM, prefixing the first frame of each sentence."""
                nonlocal spoken_any, sentence_started
                if spoken_any and not sentence_started and gap:
                    yield gap
                if frames:
                    spoken_any = True
                    sentence_started = True
                    yield frames

            failed: BaseException | None = None
            while True:
                item = await queue.get()
                if item is _SENTENCE_DONE:
                    break
                if isinstance(item, Exception):
                    failed = item
                    break
                out = await assembler.push(item)
                if out:
                    async for frame in emit(out):
                        yield frame

            if failed is None:
                out = await assembler.finish()
                if out:
                    async for frame in emit(out):
                        yield frame
            else:
                log.warning("TTS failed for sentence %d: %s", index, failed)
                yield _error_event(failed, index, sentence)
    finally:
        for task in producers.values():
            if not task.done():
                task.cancel()
        if producers:
            await asyncio.gather(*producers.values(), return_exceptions=True)
        # Clear the producers dict to release task references
        producers.clear()


async def synthesize_pcm(
    text: str,
    voice: str | None = None,
    prefetch: int | None = None,
) -> AsyncGenerator[bytes, None]:
    """Like :func:`synthesize_stream` but raises instead of yielding errors."""
    async for item in synthesize_stream(text, voice, prefetch):
        if isinstance(item, dict):
            raise TTSSynthesisError(item)
        yield item


async def warm_up(voice: str | None = None) -> bool:
    """Synthesize and discard one tiny phrase to prime the TTS path.

    The first edge-tts request in a process pays a one-off cost -- endpoint
    setup, TLS, and the first progressive MP3 decode -- measured at roughly
    +0.3-0.6 s of time-to-first-audio. Running that during service startup
    keeps it off the first user turn.

    Note that each sentence opens its own edge-tts connection, so this does not
    pool connections: it removes the cold-start cost from the request path, not
    per-sentence overhead. Returns ``True`` when audio actually came back and
    never raises; a failed warm-up only means the first turn is slower.
    """
    start = time.perf_counter()
    audio_bytes = 0
    errors = 0
    try:
        async for item in synthesize_stream(WARMUP_TEXT, voice):
            if isinstance(item, dict):
                errors += 1
                continue
            audio_bytes += len(item)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - warm-up is best effort
        log.warning(
            "TTS warm-up failed (%s: %s); the first turn pays the cold start",
            type(exc).__name__,
            exc,
        )
        return False

    if not audio_bytes:
        log.warning(
            "TTS warm-up produced no audio (%d error event(s)); the first turn "
            "pays the cold start",
            errors,
        )
        return False

    log.info(
        "TTS warm in %.0f ms (%s, %d bytes discarded)",
        (time.perf_counter() - start) * 1000,
        voice or settings.tts_voice,
        audio_bytes,
    )
    return True
