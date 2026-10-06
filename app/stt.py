"""STT service with Groq primary (in-memory Opus compression) and local small.en fallback with a duration-scaled circuit breaker.

Three guards keep a turn cheap and honest:

* a **silence gate** (:func:`audio_dbfs`) short-circuits near-silent buffers in
  microseconds, because Whisper answers near-silence with hallucinations
  ("you", "thank you", "Subtitles by...") instead of empty text;
* an **in-memory Opus encoder** (:mod:`soundfile`, libsndfile bundled in the
  wheel) so no system ``ffmpeg`` binary is needed -- the previous pydub path
  raised ``FileNotFoundError`` on machines without ffmpeg, which silently pushed
  every turn onto the local model;
* a **duration-scaled circuit breaker**, since encode plus upload time grows
  with the utterance and a fixed 1.8 s cut long segments off mid-flight.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import re
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from groq import AsyncGroq

from .config import settings

log = logging.getLogger("alvin.stt")


def audio_dbfs(audio: np.ndarray) -> float:
    """Full-scale RMS level of a float32 signal in dBFS.

    ``-inf`` means digital silence. Works on the float32 samples in [-1, 1]
    that the connection layer builds, so no extra conversion is needed.
    """
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def normalize_spoken_text(text: str) -> str:
    """Inverse Text Normalization (ITN) helper for spoken email formatting.

    Converts "test at domain dot org" -> "test@domain.org".
    Applied to both Groq and local fallback transcription results.
    """
    email_pattern = r'\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+\s+dot\s+[a-zA-Z]{2,})\b'

    def replace_email(match):
        local_part = match.group(1)
        domain_part = match.group(2).replace(" dot ", ".")
        return f"{local_part}@{domain_part}"

    return re.sub(email_pattern, replace_email, text, flags=re.IGNORECASE)


def breaker_timeout(audio_seconds: float) -> float:
    """Cloud attempt budget for an utterance of ``audio_seconds``.

    The floor is ``settings.groq_timeout_sec``; longer audio gets
    ``base + slope * duration`` so a 12 s utterance is not cut off while its
    Opus payload is still uploading.
    """
    scaled = (
        settings.groq_timeout_base_sec
        + max(0.0, audio_seconds) * settings.groq_timeout_slope_sec
    )
    return max(settings.groq_timeout_sec, scaled)


@dataclass
class TranscriptionResult:
    """One transcription outcome, whatever backend produced it."""

    text: str
    confidence: float
    latency_ms: float
    model: str


class GroqBackend:
    """Groq Whisper API backend with Opus compression."""

    def __init__(self) -> None:
        self._client = None
        self._lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is not None:
            return self._client
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    def _compress_audio(self, audio: np.ndarray) -> bytes:
        """Encode mono audio as Ogg/Opus entirely in memory.

        ``soundfile`` ships libsndfile inside the wheel and links through
        cffi, so this needs no ``ffmpeg`` binary on PATH. Called from a worker
        thread (via :meth:`transcribe_async`) because it is CPU-bound.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=tuple(range(audio.ndim - 1)))
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        buffer = io.BytesIO()
        sf.write(buffer, pcm16, settings.sample_rate, format="OGG", subtype="OPUS")
        buffer.seek(0)
        return buffer.read()

    async def warm_up(self) -> bool:
        """Open the TLS connection and prime the HTTP pool before the first turn.

        A cheap authenticated ``GET`` against the same host keeps the handshake
        (and the httpx connection pool) warm; the first real turn then starts
        with an already-established socket instead of paying ~0.5-2.7 s for
        TCP + TLS. It returns ``True`` when the cloud answered.
        """
        start = time.perf_counter()
        try:
            client = await self._get_client()
            await client.models.list()
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            log.warning(
                "Groq warm-up failed (%s: %s); the first turn pays the handshake",
                type(exc).__name__,
                exc,
            )
            return False
        log.info(
            "Groq connection warm in %.0f ms", (time.perf_counter() - start) * 1000
        )
        return True

    async def close(self) -> None:
        """Close the httpx client to release resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must never raise
                log.warning("error closing Groq client: %s", exc)
            self._client = None

    async def transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> TranscriptionResult:
        """Transcribe via Groq; Opus compression plus the call run under the lock.

        The lock serialises uploads so the API rate limit trips gracefully
        instead of failing every concurrent turn.
        """
        start = time.perf_counter()
        prompt = settings.groq_initial_prompt.strip() or None
        try:
            client = await self._get_client()
            ogg_bytes = await asyncio.to_thread(self._compress_audio, audio)
            async with self._lock:
                transcription = await client.audio.transcriptions.create(
                    file=("audio.ogg", ogg_bytes, "audio/ogg"),
                    model=settings.groq_model,
                    language=language or settings.groq_language,
                    response_format="verbose_json",
                    temperature=0.0,
                    **({"prompt": prompt} if prompt else {}),
                )
            latency_ms = (time.perf_counter() - start) * 1000
            text = normalize_spoken_text(transcription.text.strip())
            confidence = (getattr(transcription, "x_groq", None) or {}).get("confidence", 0.9)
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                latency_ms=latency_ms,
                model=settings.groq_model,
            )
        except Exception:
            log.exception("Groq transcription failed")
            raise


class LocalWhisperBackend:
    """Local faster-whisper backend with pre-warming and greedy decoding."""

    def __init__(self) -> None:
        self.model_name = settings.fallback_model
        self.model = None
        self._lock = None
        self._loading = None

    async def ensure_loaded(self) -> None:
        """Load the model once, coalescing concurrent callers on one future.

        A future is used instead of a lock so no task can block the event loop
        while the ~500 MB model is being read from disk.
        """
        if self.model is not None:
            return
        if self._loading is not None:
            await self._loading
            return

        loop = asyncio.get_running_loop()
        self._loading = loop.create_future()
        try:
            self.model = await asyncio.to_thread(self._load)
            self._lock = asyncio.Lock()
            self._loading.set_result(None)
        except BaseException as exc:
            self._loading.set_exception(exc)
            self._loading = None
            raise

    def _load(self) -> WhisperModel:
        log.info(
            "Loading local fallback model: %s (int8, %d threads)",
            self.model_name,
            settings.cpu_threads,
        )
        model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=settings.cpu_threads,
        )
        # Warm up ctranslate2 kernels so the first real turn is not the slow one.
        dummy_pcm = np.zeros(int(settings.sample_rate * 0.5), dtype=np.float32)
        list(
            model.transcribe(
                dummy_pcm,
                language="en",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
        )
        log.info("Local fallback model warm and ready in RAM")
        return model

    async def transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> TranscriptionResult:
        """Greedy-decode on the local model, serialised per instance."""
        if self.model is None:
            await self.ensure_loaded()
        assert self.model is not None and self._lock is not None
        start = time.perf_counter()
        async with self._lock:
            text, confidence = await asyncio.to_thread(
                self._transcribe, audio, language
            )
        latency_ms = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, float]:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size == 0 or audio.ndim not in (1, 2):
            return "", 0.0
        if audio.ndim == 2:
            audio = audio.mean(axis=tuple(range(audio.ndim - 1)))
        # Shape/emptiness is the only gate here: no RMS silence gate, because the
        # VAD in app.vad.py already decides what counts as a segment.
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = normalize_spoken_text(" ".join(seg.text for seg in segments).strip())
        confidence = getattr(info, "transcription_probability", 0.8)
        return text, confidence


class STTService:
    """STT service with Groq primary (Opus compression) and local fallback with circuit breaker."""

    def __init__(self) -> None:
        self._groq_backend = None
        self._local_backend = None

    def _get_groq_backend(self) -> GroqBackend:
        if self._groq_backend is None:
            self._groq_backend = GroqBackend()
        return self._groq_backend

    def _get_local_backend(self) -> LocalWhisperBackend:
        if self._local_backend is None:
            self._local_backend = LocalWhisperBackend()
        return self._local_backend

    @property
    def model(self) -> object | None:
        """The loaded local model, or ``None`` before the fallback is created."""
        backend = getattr(self, "_local_backend", None)
        if backend is not None and hasattr(backend, "model"):
            return getattr(backend, "model", None)
        return None

    async def ensure_loaded(self) -> None:
        """Pre-warm local fallback at startup if enabled."""
        if settings.preload_fallback:
            log.info("Pre-warming local fallback model at startup...")
            await self._get_local_backend().ensure_loaded()

    async def warm_up_cloud(self) -> bool:
        """Prime the cloud connection pool. Never raises."""
        return await self._get_groq_backend().warm_up()

    @property
    def is_ready(self) -> bool:
        # If preload is disabled, we consider the service ready even if the model
        # isn't loaded yet (it will be loaded on first fallback)
        if not settings.preload_fallback:
            return True
        model = getattr(self._local_backend, "model", None)
        return model is not None

    async def shutdown(self) -> None:
        """Release any resident model memory and drop both backends."""
        for backend in (self._local_backend, self._groq_backend):
            if backend is None:
                continue
            # Close Groq httpx client if present
            closer = getattr(backend, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception as exc:  # noqa: BLE001 - releasing must never raise
                    log.warning("error closing backend: %s", exc)
            # Close/unload local model if present
            model = getattr(backend, "model", None)
            model_closer = getattr(model, "close", None) or getattr(model, "unload", None)
            if model_closer is not None:
                try:
                    model_closer()
                except Exception as exc:  # noqa: BLE001 - releasing must never raise
                    log.warning("error releasing STT model: %s", exc)
        self._local_backend = None
        self._groq_backend = None

    async def transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> tuple[str, float]:
        """Transcribe with Groq primary (Opus compressed) and a fallback breaker.

        Returns ``(text, confidence)``. A Groq timeout or any other Groq error
        falls back to the local model, so the caller always gets a transcript.
        Buffers quieter than ``settings.silence_threshold_dbfs`` return
        immediately with empty text: no upload, no local decode, and no
        Whisper hallucination of "you" / "thank you".
        """
        level_dbfs = audio_dbfs(audio)
        if level_dbfs < settings.silence_threshold_dbfs:
            log.debug(
                "silence gate: %.1f dBFS < %.1f dBFS, skipping backends",
                level_dbfs,
                settings.silence_threshold_dbfs,
            )
            return "", 0.0

        groq_backend = self._get_groq_backend()
        audio_seconds = audio.size / settings.sample_rate
        timeout = breaker_timeout(audio_seconds)

        if settings.groq_fallback:
            try:
                result = await asyncio.wait_for(
                    groq_backend.transcribe_async(audio, language),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "Groq STT timeout (%.1fs for %.1fs of audio), falling back to local %s",
                    timeout,
                    audio_seconds,
                    settings.fallback_model,
                )
                local_backend = self._get_local_backend()
                if local_backend.model is None:
                    await local_backend.ensure_loaded()
                result = await local_backend.transcribe_async(audio, language)
                log.info("Fallback transcription succeeded")
            except Exception:  # noqa: BLE001 - any Groq failure must fall back
                log.warning(
                    "Groq STT error, falling back to local %s", settings.fallback_model
                )
                local_backend = self._get_local_backend()
                if local_backend.model is None:
                    await local_backend.ensure_loaded()
                result = await local_backend.transcribe_async(audio, language)
                log.info("Fallback transcription succeeded")
        else:
            # Groq explicitly disabled - use local backend directly
            log.debug("Groq disabled, using local backend")
            local_backend = self._get_local_backend()
            if local_backend.model is None:
                await local_backend.ensure_loaded()
            result = await local_backend.transcribe_async(audio, language)
            log.info("Local transcription succeeded")

        log.debug(
            "STT [%s] latency=%.1fms confidence=%.3f text=%r",
            result.model,
            result.latency_ms,
            result.confidence,
            result.text[:50],
        )
        return result.text, result.confidence


stt_service = STTService()
