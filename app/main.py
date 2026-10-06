"""Alvin real-time WebSocket Speech-to-Text service.

Exposes:

* ``GET  /health``         - readiness/liveness probe.
* ``WS   /ws/transcribe``     - real-time PCM audio -> JSON transcription events.

WebSocket output events:

* ``{transcript, is_partial, is_final, confidence}`` — STT result.
* ``{type: "llm_response", text, model}`` — LLM reply (when LLM is enabled).
* ``{type: "llm_error", error}`` — LLM failure (speaks transcript as fallback).
* ``{type: "tts_end", ...}`` / ``{type: "tts_error", ...}`` — synthesis events.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketException, status
from fastapi.responses import JSONResponse

from .config import settings
from .connection import Connection, manager, verify_token
from .llm import llm_service
from .intents import SUPPORTED_INTENTS
from .stt import stt_service
from .tts import warm_up as tts_warm_up
from .vad import VADStreamDetector

log = logging.getLogger("alvin")

# uvicorn only configures its own loggers, so app messages (readiness line, warm-up
# results, failover warnings) would be swallowed by the "no handler" fallback.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def cloud_primary_enabled() -> bool:
    """True when turns are attempted on Groq before the local fallback.

    ``settings.groq_fallback`` is the routing switch: while it is on, every
    segment tries the cloud first and only decodes locally on timeout/error,
    which is what keeps ``settings.fallback_model`` a fallback-only model.
    """
    return bool(settings.groq_fallback and settings.groq_api_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await stt_service.ensure_loaded()

    # Prime both providers off the critical path so the first real turn does not
    # pay TCP + TLS (STT) or the first edge-tts request (TTS). Both run in the
    # background, and every failure is logged inside the warm-up rather than
    # raised, so a slow or unreachable provider cannot delay startup.
    warmup_tasks: list[asyncio.Task] = []
    if cloud_primary_enabled() and settings.groq_warmup:
        warmup_tasks.append(asyncio.create_task(stt_service.warm_up_cloud()))
    if settings.tts_warmup:
        warmup_tasks.append(asyncio.create_task(tts_warm_up()))
    if llm_service.enabled and settings.llm_warmup:
        warmup_tasks.append(asyncio.create_task(llm_service.warm_up()))

    if cloud_primary_enabled():
        log.info(
            "Alvin STT ready: primary=%s via groq (Opus in-memory, breaker=%.1fs "
            "+ %.2fs per audio second), fallback=%s local only (int8, %d threads, "
            "preloaded=%s), silence_gate=%.0f dBFS, auth=%s, max_connections=%d",
            settings.groq_model,
            settings.groq_timeout_sec,
            settings.groq_timeout_slope_sec,
            settings.fallback_model,
            settings.cpu_threads,
            settings.preload_fallback,
            settings.silence_threshold_dbfs,
            "required" if settings.auth_required else "disabled",
            settings.max_connections,
        )
    else:
        log.info(
            "Alvin STT ready: cloud primary disabled; every turn decodes locally "
            "with %s (int8, %d threads), auth=%s, max_connections=%d",
            settings.fallback_model,
            settings.cpu_threads,
            "required" if settings.auth_required else "disabled",
            settings.max_connections,
        )
    if llm_service.enabled:
        log.info(
            "LLM brain ready: model=%s (%s)",
            settings.llm_model,
            settings.llm_base_url,
        )
    else:
        log.info("LLM brain disabled (set LLM_API_KEY to enable)")
    yield
    for task in warmup_tasks:
        if not task.done():
            task.cancel()
    if warmup_tasks:
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
    await stt_service.shutdown()
    await llm_service.shutdown()


app = FastAPI(
    title="Alvin STT Service",
    version="0.1.0",
    description="Real-time WebSocket speech-to-text ingest for the Alvin voice assistant.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> JSONResponse:
    """Report the models that actually serve requests.

    The cloud model answers the primary path and ``settings.fallback_model``
    (small.en) is used only on cloud timeout/error, so both are reported: the
    ``model`` field is the one that answers first, never the fallback.
    """
    ready = stt_service.is_ready
    cloud_primary = cloud_primary_enabled()
    return JSONResponse(
        {
            "status": "ok" if ready else "starting",
            "provider": "groq" if cloud_primary else "local",
            "model": settings.groq_model if cloud_primary else settings.fallback_model,
            "fallback_model": settings.fallback_model,
            "fallback_ready": ready,
            "ready": ready,
            "language": settings.language,
            "tts_voice": settings.tts_voice,
            "auth_required": settings.auth_required,
            "active_connections": manager.active_count,
            "max_connections": settings.max_connections,
            "llm": {
                "enabled": llm_service.enabled,
                "model": settings.llm_model if llm_service.enabled else None,
                "base_url": settings.llm_base_url if llm_service.enabled else None,
                "cf_fallback": (
                    {"model": settings.cf_model, "enabled": llm_service.cf_enabled}
                    if llm_service.cf_enabled
                    else None
                ),
            },
            "intents": {
                "enabled": settings.intent_enabled,
                "commands": list(SUPPORTED_INTENTS),
                "min_confidence": settings.intent_min_confidence,
                "wake_words": [
                    w.strip()
                    for w in (settings.wake_words or "").split(",")
                    if w.strip()
                ],
                "wake_word_required": settings.wake_word_required,
                "wake_word_window_sec": settings.wake_word_window_sec,
                "echo_mode": settings.echo_mode,
                "echo_cooldown_ms": settings.echo_cooldown_ms,
                "echo_similarity_threshold": settings.echo_similarity_threshold,
                "max_frame_size": settings.max_frame_size,
                "push_to_talk": settings.push_to_talk,
                "push_to_talk_timeout_ms": settings.push_to_talk_timeout_ms,
            },
        }
    )


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """Stream raw 16 kHz / 16-bit / mono PCM; receive JSON transcription events.

    Binary frames  -> PCM audio (little-endian int16).
    Text frames    -> JSON control messages (see ``app.connection``).

    Admission order: authenticate, then enforce ``MAX_CONNECTIONS``. Both run
    before ``accept()``, so a rejected client never holds an open socket.
    """
    if settings.auth_required and not verify_token(websocket):
        log.warning("rejecting websocket: bad or missing API key")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized"
        )

    await manager.connect(websocket, settings.max_connections)
    try:
        # A fresh VAD model keeps per-stream recurrent state isolated.
        detector = VADStreamDetector(sampling_rate=settings.sample_rate)
        connection = Connection(
            websocket=websocket,
            stt=stt_service,
            detector=detector,
            language=settings.language,
        )
        await connection.run()
    finally:
        manager.disconnect(websocket)
