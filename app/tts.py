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
    text = _URL.sub(" ", text)  # Replace URL with space to separate words
    text = _EMOJI.sub(" ", text)  # Replace emoji with space
    # Replace newlines with spaces for clean TTS
    text = text.replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    if not text:
        return []
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def dbfs_to_amplitude(dbfs: float) -> int:
    """Convert dBFS threshold to 16-bit PCM amplitude."""
    return int(32767 * 10 ** (dbfs / 20.0))


def silence_gap_bytes(ms: int) -> bytes:
    """Return ``ms`` milliseconds of digital silence at 16 kHz mono 16-bit."""
    if ms <= 0:
        return b""
    frames = TARGET_SAMPLE_RATE * ms // 1000
    return b"\x00" * (frames * 2)


class EdgeSilenceTrimmer:
    """Incrementally drops leading/trailing silence below a threshold.

    Leading silence is suppressed until the first frame exceeds the threshold.
    Trailing silence is held back by ``hold`` frames and only emitted if it
    contains speech; otherwise it is dropped.
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
    timeout_sec = getattr(settings, 'tts_timeout_sec', 10.0)
    try:
        last_exc: BaseException | None = None
        got_audio = False
        for attempt in range(retries + 1):
            try:
                communicate = edge_tts.Communicate(sentence, voice)
                stream = communicate.stream()

                async def _consume():
                    nonlocal got_audio
                    async for chunk in stream:
                        if chunk["type"] == "audio":
                            queue.put_nowait(chunk["data"])
                            got_audio = True

                consume_task = asyncio.create_task(_consume())
                try:
                    await asyncio.wait_for(consume_task, timeout=timeout_sec)
                except asyncio.TimeoutError:
                    consume_task.cancel()
                    raise asyncio.TimeoutError(f"TTS timeout after {timeout_sec}s")
                return  # fully streamed; nothing left to do
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                last_exc = exc
                log.warning(
                    "edge-tts attempt %d/%d timed out for %r",
                    attempt + 1,
                    retries + 1,
                    sentence[:60],
                )
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

    # Start the first `window` producers.
    for i in range(min(window, len(sentences))):
        spawn(i)

    try:
        for index in range(len(sentences)):
            # Wait for the next producer to finish (or fail).
            if index in producers:
                await producers[index]
            # Drain this sentence's queue.
            assembler = _PcmAssembler(
                threshold=threshold,
                hold=hold,
                interval_bytes=settings.tts_decode_interval_bytes,
            )
            q = queues[index]
            while True:
                item = await q.get()
                if item is _SENTENCE_DONE:
                    break
                if isinstance(item, Exception):
                    yield _error_event(item, index, sentences[index])
                    # Skip to next sentence on failure.
                    break
                out = await assembler.push(item)
                if out:
                    yield out
            # Flush any held audio for this sentence.
            tail = await assembler.finish()
            if tail:
                yield tail
            # Emit gap silence between sentences (except after the last).
            if index + 1 < len(sentences) and gap:
                yield gap
            # Start the next producer if we're still within the window.
            if index + window < len(sentences):
                spawn(index + window)
    finally:
        # Cancel any outstanding producers.
        for task in producers.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - cleanup must never raise
                log.warning("error cleaning up TTS producer task")


def _decode_mp3(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 to 16 kHz mono 16-bit PCM using miniaudio."""
    if not mp3_bytes:
        return np.empty(0, dtype=_PCM_DTYPE)
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=TARGET_CHANNELS,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    # decoded.samples is an array.array, convert to numpy
    return np.frombuffer(decoded.samples, dtype=_PCM_DTYPE)


def synthesize_pcm(text: str, voice: str | None = None):
    """Synthesize full text to PCM frames, raising on error events.

    This is a compatibility wrapper that yields frames and raises
    :class:`TTSSynthesisError` when an error event is encountered.
    """
    async def _gen():
        async for chunk in synthesize_stream(text, voice):
            if isinstance(chunk, bytes):
                yield chunk
            elif isinstance(chunk, dict) and chunk.get("type") == "error":
                raise TTSSynthesisError(chunk)

    return _gen()


async def warm_up(voice: str | None = None) -> bool:
    """Pre-warm the TTS pipeline at startup so the first real turn is fast."""
    try:
        async for chunk in synthesize_stream(WARMUP_TEXT, voice or settings.tts_voice):
            # Check for error events from synthesize_stream
            if isinstance(chunk, dict) and chunk.get("type") == "error":
                log.warning("TTS warm-up error event: %s", chunk.get("message"))
                return False
        log.info("TTS warm-up completed")
        return True
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        log.warning("TTS warm-up failed: %s", exc)
        return False