"""STT service with Cloudflare Workers AI primary and Groq fallback.

Two guards keep a turn cheap and honest:

* a **silence gate** (:func:`audio_dbfs`) short-circuits near-silent buffers in
  microseconds, because Whisper answers near-silence with hallucinations
  ("you", "thank you", "Subtitles by...") instead of empty text;
* an **in-memory Opus encoder** (:mod:`soundfile`, libsndfile bundled in the
  wheel) so no system ``ffmpeg`` binary is needed -- the previous pydub path
  raised ``FileNotFoundError`` on machines without ffmpeg, which silently pushed
  every turn onto the local model;
* a **duration-scaled circuit breaker**, since encode plus upload time grows
  with the utterance and a fixed timeout cuts long segments off mid-flight.
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
from groq import AsyncGroq

from .config import settings

log = logging.getLogger("alvin.stt")

# Phrases Whisper reliably emits when fed near-silence. Matching them **only
# below the silence threshold + a small margin** keeps real speech ("yes",
# "thank you" said to the assistant) from being dropped.
_COMMON_HALLUCINATIONS: frozenset[str] = frozenset({
    "you", "yeah", "bye", "okay", "ok", "yes", "no",
    "thank you", "thanks", "thank you for watching",
    "thanks for watching", "please subscribe",
    "subscribe", "like and subscribe",
    "subtitles by", "subtitles", "amara.org",
    "transcription by", "captions by",
    "the end", "...",
})

# Extend into matched sub-phrases so "thank you very much" also matches.
_HALLUCINATION_PREFIXES = (
    "thank you", "thanks", "please subscribe", "subtitles by",
    "amara.org", "transcription by", "captions by", "transcribed by",
)

# How far above SILENCE_THRESHOLD_DBFS a segment must be before its transcript
# is trusted unconditionally. Below this margin (but above the silence gate),
# only non-hallucination transcripts are accepted.
_HALLUCINATION_GUARD_MARGIN_DB = 6.0


def _looks_like_hallucination(text: str) -> bool:
    """True when a transcript matches a known Whisper silence artifact."""
    cleaned = text.strip().strip(".!?,;:").lower()
    if not cleaned:
        return True  # empty / punctuation-only
    if cleaned in _COMMON_HALLUCINATIONS:
        return True
    for prefix in _HALLUCINATION_PREFIXES:
        if cleaned.startswith(prefix):
            return True
    return False


def _filter_hallucination(
    text: str, confidence: float, level_dbfs: float
) -> tuple[str, float, bool]:
    """Return ``(text, confidence, filtered)`` for a transcribed segment.

    If the audio is quiet (close to the silence gate) and the transcript
    matches a known Whisper silence artifact, return empty text. A loud
    segment saying "thank you" is *not* filtered — that's real speech.
    """
    if not text.strip():
        return "", confidence, True
    if level_dbfs >= settings.silence_threshold_dbfs + _HALLUCINATION_GUARD_MARGIN_DB:
        # Clearly not near-silence — trust the transcript.
        return text, confidence, False
    if _looks_like_hallucination(text):
        log.debug(
            "hallucination filter: dropped %r (%.1f dBFS, conf=%.2f)",
            text[:60],
            level_dbfs,
            confidence,
        )
        return "", 0.0, True
    return text, confidence, False


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


class CloudflareSTTBackend:
    """Cloudflare Workers AI STT backend using @cf/openai/whisper-large-v3-turbo.

    Free on Workers AI (no separate billing). Uses the same CF_API_KEY and
    CF_ACCOUNT_ID as the LLM provider. Audio is sent as multipart/form-data.
    """

    def __init__(self) -> None:
        self._client = None
        self._lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is not None:
            return self._client
        if not settings.cf_api_key or not settings.cf_account_id:
            raise RuntimeError("CF_API_KEY and CF_ACCOUNT_ID required for Cloudflare STT")
        import httpx
        self._client = httpx.AsyncClient(
            base_url=f"{settings.cf_base_url}/{settings.cf_account_id}/ai/run",
            headers={"Authorization": f"Bearer {settings.cf_api_key}"},
            timeout=settings.cf_stt_timeout_sec,
        )
        return self._client

    def _compress_audio(self, audio: np.ndarray) -> bytes:
        """Encode mono audio as Ogg/Opus entirely in memory (same as Groq)."""
        if audio.ndim > 1:
            audio = audio.mean(axis=tuple(range(audio.ndim - 1)))
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        buffer = io.BytesIO()
        sf.write(buffer, pcm16, settings.sample_rate, format="OGG", subtype="OPUS")
        buffer.seek(0)
        return buffer.read()

    async def warm_up(self) -> bool:
        """Warm up the Cloudflare connection pool."""
        start = time.perf_counter()
        try:
            client = await self._get_client()
            # Light GET to warm the connection
            await client.get("/models", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            log.warning(
                "Cloudflare STT warm-up failed (%s: %s); the first turn pays the handshake",
                type(exc).__name__,
                exc,
            )
            return False
        log.info(
            "Cloudflare STT connection warm in %.0f ms", (time.perf_counter() - start) * 1000
        )
        return True

    async def close(self) -> None:
        """Close the httpx client to release resources."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001 - cleanup must never raise
                log.warning("error closing Cloudflare STT client: %s", exc)
            self._client = None

    async def transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> TranscriptionResult:
        """Transcribe via Cloudflare Workers AI; Opus compression under the lock."""
        start = time.perf_counter()
        try:
            client = await self._get_client()
            ogg_bytes = await asyncio.to_thread(self._compress_audio, audio)
            async with self._lock:
                files = {"audio": ("audio.ogg", ogg_bytes, "audio/ogg")}
                data = {
                    "model": settings.cf_stt_model,
                }
                if language:
                    data["language"] = language
                response = await client.post(
                    f"/{settings.cf_stt_model}",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()
            latency_ms = (time.perf_counter() - start) * 1000
            # Cloudflare returns {"result": {"text": "..."}} or {"error": ...}
            if "result" in result and "text" in result["result"]:
                text = normalize_spoken_text(result["result"]["text"].strip())
            else:
                log.warning("Cloudflare STT unexpected response: %s", result)
                text = ""
            confidence = 0.9  # Cloudflare doesn't return confidence; use reasonable default
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                latency_ms=latency_ms,
                model=settings.cf_stt_model,
            )
        except Exception:
            log.exception("Cloudflare STT transcription failed")
            raise


class STTService:
    """STT service with Cloudflare Workers AI primary and Groq fallback.

    Fallback chain:
      Cloudflare STT (@cf/openai/whisper-large-v3-turbo)
        → Groq (whisper-large-v3-turbo)

    When CF_STT disabled:
      Groq only
    """

    def __init__(self) -> None:
        self._groq_backend = None
        self._cf_stt_backend = None

    def _get_groq_backend(self) -> GroqBackend:
        if self._groq_backend is None:
            self._groq_backend = GroqBackend()
        return self._groq_backend

    def _get_cf_stt_backend(self) -> CloudflareSTTBackend:
        if self._cf_stt_backend is None:
            self._cf_stt_backend = CloudflareSTTBackend()
        return self._cf_stt_backend

    async def warm_up_cloud(self) -> bool:
        """Prime the cloud connection pool(s). Never raises."""
        results = []
        # Warm up Cloudflare STT first (primary)
        if settings.cf_stt_enabled:
            results.append(await self._get_cf_stt_backend().warm_up())
        # Warm up Groq (fallback)
        if settings.groq_api_key:
            results.append(await self._get_groq_backend().warm_up())
        return any(results)  # True if at least one succeeded

    @property
    def is_ready(self) -> bool:
        # Always ready since we only use cloud backends
        return True

    async def shutdown(self) -> None:
        """Release all backends."""
        for backend in (self._groq_backend, self._cf_stt_backend):
            if backend is None:
                continue
            closer = getattr(backend, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception as exc:  # noqa: BLE001 - releasing must never raise
                    log.warning("error closing backend: %s", exc)
        self._groq_backend = None
        self._cf_stt_backend = None

    async def transcribe_cloud_for_muted(
        self, audio: np.ndarray, language: str | None
    ) -> tuple[str, float]:
        """Transcribe for muted-mode keyword pass using cloud STT.

        Uses the same fallback chain as normal transcription but with a shorter
        timeout since muted segments are typically short commands.
        """
        # Silence gate - always first
        try:
            level_dbfs = audio_dbfs(audio)
            if level_dbfs < settings.silence_threshold_dbfs:
                return "", 0.0
        except Exception:
            return "", 0.0

        audio_seconds = audio.size / settings.sample_rate
        groq_timeout = breaker_timeout(audio_seconds)
        cf_stt_timeout = settings.cf_stt_timeout_sec

        # Determine fallback order
        use_cf_primary = settings.cf_stt_enabled and settings.cf_stt_use_as_primary
        groq_available = bool(settings.groq_api_key)
        cf_stt_available = settings.cf_stt_enabled and settings.cf_api_key and settings.cf_account_id

        last_error = None
        for backend_name in self._get_backend_order(use_cf_primary, groq_available, cf_stt_available):
            try:
                if backend_name == "cf_stt":
                    result = await asyncio.wait_for(
                        self._get_cf_stt_backend().transcribe_async(audio, language),
                        timeout=cf_stt_timeout,
                    )
                else:  # groq
                    result = await asyncio.wait_for(
                        self._get_groq_backend().transcribe_async(audio, language),
                        timeout=groq_timeout,
                    )

                # Success - apply hallucination filter and return
                text, confidence, filtered = _filter_hallucination(
                    result.text, result.confidence, level_dbfs
                )
                if filtered and not text:
                    return "", 0.0
                return text, confidence

            except asyncio.TimeoutError:
                last_error = f"{backend_name} timeout"
            except Exception as exc:
                last_error = f"{backend_name} error: {exc}"

        log.warning("All STT backends failed for muted pass; last error: %s", last_error)
        return "", 0.0

    async def transcribe_async(
        self, audio: np.ndarray, language: str | None
    ) -> TranscriptionResult:
        """Transcribe with Cloudflare STT primary and Groq fallback.

        Fallback chain:
        - CF_STT_ENABLED + CF_STT_USE_AS_PRIMARY: CF STT → Groq
        - CF_STT_ENABLED + not primary: Groq → CF STT
        - CF_STT disabled: Groq only

        Returns ``(text, confidence)``. Any cloud timeout/error falls back to
        the next backend, so the caller always gets a transcript.
        Buffers quieter than ``settings.silence_threshold_dbfs`` return
        immediately with empty text: no upload, no local decode, and no
        Whisper hallucination of "you" / "thank you".
        """
        # Silence gate - always first
        try:
            level_dbfs = audio_dbfs(audio)
            if level_dbfs < settings.silence_threshold_dbfs:
                log.debug(
                    "silence gate: %.1f dBFS < %.1f dBFS, skipping backends",
                    level_dbfs,
                    settings.silence_threshold_dbfs,
                )
                return TranscriptionResult(
                    text="", confidence=0.0, latency_ms=0.0, model="silence_gate"
                )
        except Exception as exc:
            log.exception("STT transcribe_async failed in silence gate")
            return TranscriptionResult(
                text="", confidence=0.0, latency_ms=0.0, model="stt_failed"
            )

        audio_seconds = audio.size / settings.sample_rate
        groq_timeout = breaker_timeout(audio_seconds)
        cf_stt_timeout = settings.cf_stt_timeout_sec

        # Determine fallback order based on settings
        use_cf_primary = settings.cf_stt_enabled and settings.cf_stt_use_as_primary
        groq_available = bool(settings.groq_api_key)  # Groq available if API key set
        cf_stt_available = settings.cf_stt_enabled and settings.cf_api_key and settings.cf_account_id

        # Try backends in order
        last_error = None
        for attempt, backend_name in enumerate(self._get_backend_order(use_cf_primary, groq_available, cf_stt_available)):
            try:
                if backend_name == "cf_stt":
                    result = await asyncio.wait_for(
                        self._get_cf_stt_backend().transcribe_async(audio, language),
                        timeout=cf_stt_timeout,
                    )
                elif backend_name == "groq":
                    result = await asyncio.wait_for(
                        self._get_groq_backend().transcribe_async(audio, language),
                        timeout=groq_timeout,
                    )

                # Success - apply hallucination filter and return
                text, confidence, filtered = _filter_hallucination(
                    result.text, result.confidence, level_dbfs
                )
                if filtered and not text:
                    log.debug(
                        "STT [%s] filtered hallucination (%.1f dBFS): %r",
                        result.model,
                        level_dbfs,
                        result.text[:60],
                    )
                    return TranscriptionResult(
                        text="", confidence=0.0, latency_ms=result.latency_ms, model=result.model
                    )

                log.debug(
                    "STT [%s] latency=%.1fms confidence=%.3f level=%.1f dBFS text=%r",
                    result.model,
                    result.latency_ms,
                    confidence,
                    level_dbfs,
                    text[:50],
                )
                return TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    latency_ms=result.latency_ms,
                    model=result.model,
                )

            except asyncio.TimeoutError:
                last_error = f"{backend_name} timeout"
                log.warning(
                    "%s STT timeout (%.1fs for %.1fs audio), trying next backend",
                    backend_name.upper(),
                    cf_stt_timeout if backend_name == "cf_stt" else groq_timeout,
                    audio_seconds,
                )
            except Exception as exc:
                last_error = f"{backend_name} error: {exc}"
                log.warning(
                    "%s STT error (%s), trying next backend",
                    backend_name.upper(),
                    exc,
                )

        # All backends failed
        log.error("All STT backends failed; last error: %s", last_error)
        return TranscriptionResult(
            text="", confidence=0.0, latency_ms=0.0, model="stt_failed"
        )

    def _get_backend_order(
        self, use_cf_primary: bool, groq_available: bool, cf_stt_available: bool
    ) -> list[str]:
        """Determine the order of STT backends to try."""
        order = []

        if use_cf_primary:
            # CF STT primary → Groq
            if cf_stt_available:
                order.append("cf_stt")
            if groq_available:
                order.append("groq")
        else:
            # Groq primary → CF STT (if available)
            if groq_available:
                order.append("groq")
            if cf_stt_available:
                order.append("cf_stt")

        return order


stt_service = STTService()
