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
* The trimmer holds back ``TTS_TRIM_TRAILING_MS`` of audio to be able to drop
  end-of-sentence silence; set it to 0 for minimum time-to-first-audio at
  the cost of keeping the trailing silence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncGenerator
from math import gcd
from pathlib import Path
from typing import TypedDict

import edge_tts
import edge_tts.communicate as _edge_communicate
import edge_tts.constants as _edge_constants
import miniaudio
import numpy as np
from piper import PiperVoice

from .audio_pool import run_in_media_pool
from .config import settings

log = logging.getLogger("alvin.tts")

# ── Patch edge-tts for the new October 2026 endpoint ──────────────────────
# Microsoft retired the old speech.platform.bing.com host (returns 403).
# The new service lives at api.msedgeservices.com and authenticates with
# Ocp-Apim-Subscription-Key instead of TrustedClientToken.  We monkey-patch
# the constants that edge-tts read at import time so no library code changes
# are needed.
_LEGACY_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
_edge_constants.BASE_URL = "api.msedgeservices.com/tts/cognitiveservices"
_edge_constants.WSS_URL = (
    f"wss://{_edge_constants.BASE_URL}/websocket/v1"
    f"?Ocp-Apim-Subscription-Key={_LEGACY_TOKEN}"
)
_edge_constants.VOICE_LIST = (
    f"https://{_edge_constants.BASE_URL}/voices/list"
    f"?Ocp-Apim-Subscription-Key={_LEGACY_TOKEN}"
)
_edge_constants.TRUSTED_CLIENT_TOKEN = _LEGACY_TOKEN
_edge_communicate.WSS_URL = _edge_constants.WSS_URL
# drm.py also imported TRUSTED_CLIENT_TOKEN at module load
import edge_tts.drm as _edge_drm

_edge_drm.TRUSTED_CLIENT_TOKEN = _LEGACY_TOKEN
# Update VOICE_HEADERS Authority to match new endpoint
_edge_constants.VOICE_HEADERS["Authority"] = _edge_constants.BASE_URL.split("/")[0]
# voices.py imports VOICE_LIST from constants at module load, patch it too
import edge_tts.voices as _edge_voices

_edge_voices.VOICE_LIST = _edge_constants.VOICE_LIST
log.debug("Patched edge-tts for new endpoint: %s", _edge_constants.WSS_URL[:60])

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

# Warm-up phrase: longer than a single word so it exercises the sentence-splitting
# code path, gap-silence injection, and the multi-sentence prefetch machinery —
# not just the TLS handshake.
WARMUP_TEXT = "Ready to help. Ask me anything."

_PCM_DTYPE = np.dtype("<i2")

# Marks the end of a sentence's MP3 stream inside its queue.
_SENTENCE_DONE = object()

# Engine usage counters (edge-tts vs Piper fallback) for diagnostics / /health.
engine_sentences: dict[str, int] = {"edge_tts": 0, "piper": 0}


class TTSErrorEvent(TypedDict):
    """Error event yielded in place of audio when a sentence fails."""

    type: str
    stage: str
    error: str
    sentence_index: int
    sentence: str
    fatal: bool


class TTSSynthesisError(RuntimeError):
    """Raised by :func:`synthesize_pcm` when the stream reports a failure."""

    def __init__(self, event: TTSErrorEvent) -> None:
        super().__init__(
            event.get("error", event.get("message", "TTS synthesis failed"))
        )
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
    Trailing silence is held back by ``trailing_hold`` frames and only emitted if it
    contains speech; otherwise it is dropped. Leading hold is separate and used
    only to preserve speech onset during leading silence detection.
    """

    def __init__(self, threshold: int, leading_hold: int, trailing_hold: int) -> None:
        self._threshold = threshold
        self._leading_hold = max(0, leading_hold)
        self._trailing_hold = max(0, trailing_hold)
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
                # Keep only the most recent leading_hold frames so the onset survives.
                self._pending = (
                    frames[-self._leading_hold :]
                    if self._leading_hold
                    else np.empty(0, dtype=_PCM_DTYPE)
                )
                return b""
            self._speech_started = True
            self._pending = np.empty(0, dtype=_PCM_DTYPE)
            return frames[int(loud[0]) :].tobytes()

        # Speech has started - use trailing_hold for end-of-sentence buffering
        if self._trailing_hold == 0:
            return frames.tobytes()
        if frames.size <= self._trailing_hold:
            self._pending = frames
            return b""
        self._pending = frames[-self._trailing_hold :]
        return frames[: -self._trailing_hold].tobytes()

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


class PiperTTS:
    """Local Piper TTS fallback using ONNX voice models.

    Loads a Piper voice model on first use and synthesizes 16 kHz mono PCM.
    Provides both streaming (sentence-by-sentence) and full-text synthesis.
    """

    _instance: PiperTTS | None = None
    _voice: PiperVoice | None = None
    _model_path: Path | None = None
    _config_path: Path | None = None

    def __new__(cls) -> PiperTTS:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._loaded = False

    def _load(self) -> bool:
        """Load the Piper voice model. Returns True on success."""
        if self._loaded:
            return self._voice is not None

        model_dir = Path(settings.piper_tts_model_dir)
        voice_name = settings.piper_tts_voice
        self._model_path = model_dir / f"{voice_name}.onnx"
        self._config_path = model_dir / f"{voice_name}.onnx.json"

        if not self._model_path.exists():
            log.warning(
                "Piper model not found at %s; run: python -m piper.download_voices %s --download-dir %s",
                self._model_path,
                voice_name,
                model_dir,
            )
            self._loaded = True
            return False

        try:
            self._voice = PiperVoice.load(
                str(self._model_path),
                config_path=str(self._config_path)
                if self._config_path.exists()
                else None,
                use_cuda=False,
            )
            self._loaded = True
            log.info("Piper TTS loaded voice: %s", voice_name)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load Piper voice %s: %s", voice_name, exc)
            self._voice = None
            self._loaded = True
            return False

    def synthesize_pcm(self, text: str) -> bytes | None:
        """Synthesize text to 16 kHz mono 16-bit PCM bytes."""
        if not self._load() or self._voice is None:
            return None

        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=settings.piper_tts_length_scale,
                noise_scale=settings.piper_tts_noise_scale,
                noise_w_scale=settings.piper_tts_noise_w,
            )
            audio = self._voice.synthesize(text, syn_config=syn_config)
            # Buffer the whole sentence before resampling so we resample once
            # with a proper anti-aliasing filter instead of per-chunk.
            pcm_all = b"".join(
                chunk.audio_int16_bytes
                for chunk in audio
            )
            native_rate = int(getattr(self._voice.config, "sample_rate", 22050))
            if native_rate != TARGET_SAMPLE_RATE and pcm_all:
                pcm_all = self._resample_pcm(pcm_all, native_rate, TARGET_SAMPLE_RATE)
            return pcm_all
        except Exception as exc:  # noqa: BLE001
            log.warning("Piper TTS synthesis failed for %r: %s", text[:60], exc)
            return None

    @staticmethod
    def _resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Resample int16 mono PCM from src_rate to dst_rate with anti-aliasing.

        ``soxr`` is preferred when available; ``scipy.signal.resample_poly`` is
        used as a fallback. Neither dependency is required at import time, so
        environments without either degrade gracefully (synthesis returns None)
        instead of crashing ``import app.tts``.
        """
        if src_rate == dst_rate or not pcm:
            return pcm
        try:
            import soxr  # type: ignore[import-not-found]

            arr = np.frombuffer(pcm, dtype=_PCM_DTYPE).astype(np.float32) / 32768.0
            out = soxr.resample(arr, src_rate, dst_rate, quality="HQ")
            return (np.clip(out, -1.0, 1.0) * 32767.0).astype(_PCM_DTYPE).tobytes()
        except ImportError:
            pass

        try:
            from scipy.signal import resample_poly  # type: ignore[import-not-found]
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Piper resampling needs either 'soxr' or 'scipy'; "
                "install one of them (e.g. pip install soxr) or set "
                "PIPER_TTS_ENABLED=false to disable the fallback."
            ) from exc

        g = gcd(dst_rate, src_rate)
        up = dst_rate // g
        down = src_rate // g
        arr = np.frombuffer(pcm, dtype=_PCM_DTYPE).astype(np.float32)
        out = resample_poly(arr, up, down)
        return np.clip(out, -32768.0, 32767.0).astype(_PCM_DTYPE).tobytes()

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize text sentence-by-sentence, yielding PCM bytes.

        An async generator so the module-level :func:`synthesize_stream` fallback
        path (``async for pcm in piper_tts.synthesize_stream(...)``) works. Each
        sentence is synthesized on the bounded audio pool so the CPU-bound Piper
        inference and resampling do not block the event loop.
        """
        if not self._load() or self._voice is None:
            return

        sentences = chunk_sentences(text)
        gap = silence_gap_bytes(settings.tts_sentence_gap_ms)
        for index, sentence in enumerate(sentences):
            pcm = await run_in_media_pool(self.synthesize_pcm, sentence)
            if pcm:
                yield pcm
            if index + 1 < len(sentences) and gap:
                yield gap


class _PcmAssembler:
    """Incrementally decodes one sentence's MP3 stream into trimmed PCM.

    miniaudio decodes every complete frame in the buffer it is given and the
    result is a prefix-stable extension of the previous decode, so re-decoding
    the buffer and emitting only the new tail is artifact-free.
    """

    def __init__(
        self,
        threshold: int,
        leading_hold: int,
        trailing_hold: int,
        interval_bytes: int,
    ) -> None:
        self._buffer = bytearray()
        self._decoded_bytes = 0
        self._emitted_samples = 0
        self._interval = max(1, interval_bytes)
        # Leading hold is cheap (just delays first audio); trailing hold needs
        # to be larger to detect real end-of-sentence silence.
        self._trimmer = EdgeSilenceTrimmer(threshold, leading_hold, trailing_hold)

    async def push(self, mp3_chunk: bytes) -> bytes:
        """Add MP3 bytes, returning any newly available trimmed PCM."""
        self._buffer.extend(mp3_chunk)
        new_bytes = len(self._buffer) - self._decoded_bytes
        # Decode schedule: first decode at 512 bytes, subsequent decodes at the
        # configured interval (not 2x). The 2x was an optimization that made
        # behavior harder to reason about and could starve the first decode
        # if it emitted no samples. Using the interval consistently is clearer.
        if self._emitted_samples == 0:
            interval = 512
        else:
            interval = self._interval
        if new_bytes < interval:
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
        frames = await run_in_media_pool(_decode_mp3, data)
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
    """Producer: stream one sentence's MP3 chunks into its queue."""
    retries = settings.tts_retries if retries is None else retries
    backoff_ms = settings.tts_retry_backoff_ms if backoff_ms is None else backoff_ms
    timeout_sec = getattr(settings, "tts_timeout_sec", 10.0)

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
                except TimeoutError:
                    consume_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await consume_task
                    raise TimeoutError(f"TTS timeout after {timeout_sec}s")
                return
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                last_exc = exc
                log.warning(
                    "edge-tts attempt %d/%d timed out for %r",
                    attempt + 1,
                    retries + 1,
                    sentence[:60],
                )
                if got_audio or attempt >= retries:
                    break
                await asyncio.sleep(backoff_ms / 1000.0)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning(
                    "edge-tts attempt %d/%d failed for %r: %s",
                    attempt + 1,
                    retries + 1,
                    sentence[:60],
                    exc,
                )
                if got_audio or attempt >= retries:
                    break
                if "429" in str(exc):
                    await asyncio.sleep((backoff_ms * (attempt + 3)) / 1000.0)
                else:
                    await asyncio.sleep(backoff_ms / 1000.0)
        if last_exc is not None:
            queue.put_nowait(last_exc)
    finally:
        queue.put_nowait(_SENTENCE_DONE)


def _error_event(exc: BaseException, index: int, sentence: str) -> TTSErrorEvent:
    return TTSErrorEvent(
        type="error",
        stage="tts",
        error=f"{type(exc).__name__}: {exc}",
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
        leading_hold=int(settings.tts_trim_leading_ms * TARGET_SAMPLE_RATE / 1000),
        trailing_hold=int(settings.tts_trim_trailing_ms * TARGET_SAMPLE_RATE / 1000),
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
    leading_hold = int(settings.tts_trim_leading_ms * TARGET_SAMPLE_RATE / 1000)
    trailing_hold = int(settings.tts_trim_trailing_ms * TARGET_SAMPLE_RATE / 1000)
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
            # Ensure the producer for this sentence has been started.
            spawn(index)
            # Drain this sentence's queue progressively as MP3 chunks arrive
            # from the producer — do NOT await the producer task first, because
            # that would block until the entire sentence is fetched, defeating
            # the progressive-decode pipeline.
            assembler = _PcmAssembler(
                threshold=threshold,
                leading_hold=leading_hold,
                trailing_hold=trailing_hold,
                interval_bytes=settings.tts_decode_interval_bytes,
            )
            q = queues[index]
            edge_failed = False
            while True:
                item = await q.get()
                if item is _SENTENCE_DONE:
                    break
                if isinstance(item, Exception):
                    edge_failed = True
                    engine_sentences["piper"] += 1
                    log.warning(
                        "sentence %d served by PIPER (edge-tts failed): %r",
                        index,
                        sentences[index][:60],
                    )
                    break
                out = await assembler.push(item)
                if out:
                    yield out
            if edge_failed:
                if settings.piper_tts_enabled:
                    piper_tts = PiperTTS()
                    async for pcm in piper_tts.synthesize_stream(sentences[index]):
                        if pcm:
                            yield pcm
                else:
                    yield _error_event(
                        RuntimeError("edge-tts failed and Piper TTS disabled"),
                        index,
                        sentences[index],
                    )
                # Emit gap after fallback sentence too (except after last).
                if index + 1 < len(sentences) and gap:
                    yield gap
                # Start the next producer if we're still within the window.
                if index + window < len(sentences):
                    spawn(index + window)
                continue
            # Flush any held audio for this sentence.
            tail = await assembler.finish()
            if tail:
                yield tail
            engine_sentences["edge_tts"] += 1
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
    before = dict(engine_sentences)
    edge_ok = False
    try:
        async for chunk in synthesize_stream(WARMUP_TEXT, voice or settings.tts_voice):
            # Check for error events from synthesize_stream
            if isinstance(chunk, dict) and chunk.get("type") == "error":
                log.warning("TTS warm-up error event: %s", chunk.get("error"))
                break
        else:
            edge_ok = True
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        log.warning("TTS warm-up failed: %s", exc)

    used_edge = engine_sentences["edge_tts"] - before["edge_tts"]
    used_piper = engine_sentences["piper"] - before["piper"]
    if used_piper and not used_edge:
        log.warning(
            "TTS warm-up: ALL %d sentence(s) served by PIPER — edge-tts is down. "
            "Check the edge-tts endpoint patch / network.",
            used_piper,
        )
        edge_ok = False
    elif used_piper:
        log.warning(
            "TTS warm-up: %d edge-tts, %d piper sentence(s) — edge-tts partially failing",
            used_edge,
            used_piper,
        )
    elif used_edge:
        log.info("TTS warm-up completed (%d edge-tts sentence(s))", used_edge)

    # Also pre-load Piper TTS voice if enabled as fallback (side effect only).
    if settings.piper_tts_enabled:
        piper_tts = PiperTTS()
        if piper_tts._load():
            log.info("Piper TTS warm-up completed")
        else:
            log.warning("Piper TTS warm-up skipped (voice model not found)")

    return edge_ok
