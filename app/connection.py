"""Per-WebSocket streaming connection state machine.

Pipeline per connection:

    WebSocket receive loop (producer, event loop thread)
        -> PCM decode -> append to audio buffer
        -> feed 512-sample frames to the VAD
        -> on speech start/end + periodic partial interval, enqueue a segment
        -> transcription queue
        -> background consumer task (threaded Whisper via STTService)
        -> emit JSON events back on the socket

    Optional TTS path (client-driven via ``speak``/``stop_speak``, or automatic
    when ``TTS_AUTO_SPEAK`` is on):

    text -> sentence chunks -> prefetched edge-tts fetch -> trimmed PCM
        -> binary WebSocket frames -> ``tts_end`` / ``tts_error`` JSON event

    The VAD runs inline on the event-loop thread (each 32 ms frame is ~0.1 ms), so
    the producer keeps draining audio while the consumer transcribes in a worker
    thread, giving low-latency interim hypotheses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket, WebSocketException, status
from starlette.websockets import WebSocketDisconnect

from .config import settings
from .llm import MessageParam, build_messages, llm_service
from .stt import STTService
from .tts import synthesize_stream
from .vad import FRAME_SAMPLES, VADStreamDetector

log = logging.getLogger("alvin.connection")

# Sentinel values for text control messages supported by clients.
_CONTROL_CONFIG = "config"
_CONTROL_STOP = "stop"
_CONTROL_PING = "ping"
_CONTROL_SPEAK = "speak"
_CONTROL_STOP_SPEAK = "stop_speak"

_NONE_LANGS = {"auto", "none", "null", ""}


class ConnectionManager:
    """Registry of live sockets, so ``MAX_CONNECTIONS`` is actually enforced.

    The capacity check and the insert happen in the same synchronous block (no
    ``await`` in between), so two handshakes racing on the event loop cannot
    both pass the check and over-admit. Admission happens before ``accept()``:
    a rejected client gets a denied handshake instead of an accepted socket
    that is immediately closed.
    """

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def connect(self, websocket: WebSocket, max_connections: int) -> None:
        """Reserve a slot and accept, or deny the handshake with 1008."""
        if len(self._active) >= max_connections:
            log.warning(
                "rejecting websocket: %d/%d connections in use",
                len(self._active),
                max_connections,
            )
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="server at capacity"
            )
        self._active.add(websocket)
        try:
            await websocket.accept()
        except BaseException:
            self._active.discard(websocket)
            raise

    def disconnect(self, websocket: WebSocket) -> None:
        self._active.discard(websocket)


manager = ConnectionManager()


def verify_token(websocket: WebSocket) -> bool:
    """Check the shared secret from ``API_KEY``.

    The secret is read from the ``Authorization: Bearer`` or ``X-API-Key``
    header, falling back to a ``?token=`` query parameter for browser clients
    that cannot set WebSocket headers. Query strings land in access logs and
    proxy logs, so prefer the header. With no ``API_KEY`` configured the
    endpoint stays open (loopback/dev deployments).

    Comparison is constant-time, and the presented value is length-checked
    first so a wrong-length guess cannot be distinguished by timing.
    """
    expected = settings.api_key
    if not expected:
        return True

    scheme, _, value = websocket.headers.get("authorization", "").partition(" ")
    presented = (
        value.strip()
        if scheme.lower() == "bearer"
        else websocket.headers.get("x-api-key", "")
    )
    if not presented:
        presented = websocket.query_params.get("token", "")
    if not presented:
        return False

    expected_bytes = expected.encode()
    presented_bytes = presented.encode()
    if len(expected_bytes) != len(presented_bytes):
        return False
    return secrets.compare_digest(presented_bytes, expected_bytes)


@dataclass
class _PendingSegment:
    """A speech slice awaiting transcription."""

    audio: np.ndarray
    is_partial: bool
    is_final: bool
    utterance_id: int = 0


class Connection:
    """Real-time streaming STT connection state."""

    def __init__(
        self,
        websocket: WebSocket,
        stt: STTService,
        detector: VADStreamDetector,
        language: str | None,
    ) -> None:
        self.ws = websocket
        self.stt = stt
        self.detector = detector
        self.language = language

        # Audio accumulator + VAD read pointer. The buffer is reset whenever a
        # finalized segment completes so memory stays bounded to ~one utterance.
        self._audio_chunks: list[np.ndarray] = []
        self._audio_length = 0
        self.vad_pointer = 0
        self._audio_cache: np.ndarray | None = None

        # Pending (in-progress) utterance bookkeeping.
        self.seg_start: int | None = None
        self.last_partial_at = 0

        # Derived thresholds.
        sr = settings.sample_rate
        self.partial_interval_samples = max(
            1, int(settings.partial_interval_ms / 1000 * sr)
        )
        self.min_segment_samples = settings.min_segment_samples
        self.max_segment_samples = settings.max_segment_samples

        self.queue: asyncio.Queue[_PendingSegment | None] = asyncio.Queue()
        self.consumer: asyncio.Task | None = None
        self.stopping = False
        self._utterance_id: int = 0

        # Speech output: at most one synthesis or LLM stream runs at a time,
        # so a new request cancels the previous one instead of interleaving.
        self.tts_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        self.tts_voice = settings.tts_voice
        self.send_lock = asyncio.Lock()

        # Conversation memory: system prompt is pinned at index 0, followed by
        # alternating user/assistant turns.  Trimmed to MAX_LLM_HISTORY messages
        # before each LLM call to bound latency and context length.
        self.history: list[MessageParam] = [
            {"role": "system", "content": settings.llm_system_prompt}
        ]

    def _get_audio(self) -> np.ndarray:
        """Get the full audio buffer, concatenating chunks lazily."""
        if (
            self._audio_cache is not None
            and self._audio_cache.size == self._audio_length
        ):
            return self._audio_cache
        if not self._audio_chunks:
            self._audio_cache = np.empty(0, dtype=np.float32)
        else:
            self._audio_cache = np.concatenate(self._audio_chunks)
        return self._audio_cache

    async def run(self) -> None:
        """Main entry: start the consumer, drain the socket, then flush."""
        self.consumer = asyncio.create_task(self._consume())
        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.exception("receive loop error")
            await self._send_error(exc)
        finally:
            await self._flush_pending()
            await self.queue.put(None)
            await self._cancel_ongoing_turn()
            if self.consumer is not None:
                try:
                    await self.consumer
                except Exception:
                    log.exception("consumer task error")

    # ------------------------------------------------------------------ #
    # Receiving
    # ------------------------------------------------------------------ #
    async def _receive_loop(self) -> None:
        while not self.stopping:
            try:
                msg = await self.ws.receive()
            except WebSocketDisconnect:
                break
            mtype = msg.get("type")
            if mtype == "websocket.receive":
                if msg.get("bytes"):
                    await self._on_audio(msg["bytes"])
                elif msg.get("text"):
                    await self._on_text(msg["text"])
            elif mtype == "websocket.disconnect":
                break

    async def _on_audio(self, data: bytes) -> None:
        if not data:
            return
        if len(data) > settings.max_frame_size:
            log.warning(
                "rejecting oversized audio frame: %d bytes > %d bytes",
                len(data),
                settings.max_frame_size,
            )
            raise WebSocketException(
                code=1009,
                reason=f"frame too large (max {settings.max_frame_size} bytes)",
            )
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return
        self._audio_chunks.append(samples)
        self._audio_length += samples.size
        await self._feed_vad()

    async def _on_text(self, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("malformed JSON from client: %s", text[:200])
            return
        ctype = data.get("type")
        if ctype == _CONTROL_CONFIG:
            lang = data.get("language")
            if lang is not None:
                self.language = _resolve_language(lang)
            voice = data.get("voice")
            if isinstance(voice, str) and voice.strip():
                self.tts_voice = voice.strip()
        elif ctype == _CONTROL_STOP:
            self.stopping = True
        elif ctype == _CONTROL_PING:
            await self._send_raw({"type": "pong"})
        elif ctype == _CONTROL_SPEAK:
            speak_text = data.get("text")
            if isinstance(speak_text, str) and speak_text.strip():
                await self._cancel_ongoing_turn()
                await self._start_tts(speak_text, data.get("voice"))
        elif ctype == _CONTROL_STOP_SPEAK:
            await self._cancel_ongoing_turn()
        # Unknown control messages are ignored for forward compatibility.

    # ------------------------------------------------------------------ #
    # VAD + endpointing
    # ------------------------------------------------------------------ #
    async def _feed_vad(self) -> None:
        """Feed every complete 512-sample frame accumulated since last call."""
        audio = self._get_audio()
        while self.vad_pointer + FRAME_SAMPLES <= audio.size:
            frame = np.ascontiguousarray(
                audio[self.vad_pointer : self.vad_pointer + FRAME_SAMPLES],
                dtype=np.float32,
            )
            self.vad_pointer += FRAME_SAMPLES
            event = self.detector.process_frame(frame)
            if event is None:
                await self._maybe_partial()
                continue

            if "start" in event:
                self.seg_start = int(event["start"])
                self.last_partial_at = self.vad_pointer
                # Barge-in: cancel any in-progress LLM generation or TTS when
                # the user starts speaking again.
                await self._cancel_ongoing_turn()
            elif "end" in event:
                await self._finalize(int(event["end"]))
                # _finalize may reset the audio buffer (via _reset_segment),
                # so re-fetch to avoid reprocessing stale cached audio.
                audio = self._get_audio()
                continue

            await self._maybe_partial()

        # Enforce a hard cap on utterance length (bounds memory + latency).
        if (
            self.seg_start is not None
            and self.vad_pointer - self.seg_start >= self.max_segment_samples
        ):
            await self._finalize(self.vad_pointer)

    async def _maybe_partial(self) -> None:
        """Emit an interim hypothesis for ongoing speech on a time interval."""
        if self.seg_start is None:
            return
        if self.vad_pointer - self.last_partial_at < self.partial_interval_samples:
            return
        audio = self._get_audio()
        seg = audio[self.seg_start : self.vad_pointer]
        if seg.size < self.min_segment_samples:
            return
        self.last_partial_at = self.vad_pointer
        await self.queue.put(
            _PendingSegment(seg.copy(), True, False, self._utterance_id)
        )

    async def _finalize(self, end_sample: int) -> None:
        """Endpoint an utterance and enqueue its final transcription."""
        if self.seg_start is None:
            return
        start = self.seg_start
        audio = self._get_audio()
        end = max(start, min(end_sample, self.vad_pointer, audio.size))
        if end - start < self.min_segment_samples:
            # Likely a VAD false-positive; discard without transcribing.
            self._reset_segment()
            return
        seg = audio[start:end].copy()
        await self.queue.put(_PendingSegment(seg, False, True, self._utterance_id))
        self._utterance_id += 1
        self._reset_segment()

    async def _flush_pending(self) -> None:
        """Flush any in-progress utterance as a final on shutdown/stop."""
        if self.seg_start is None or self.vad_pointer <= self.seg_start:
            return
        start = self.seg_start
        audio = self._get_audio()
        # Use actual audio length, not vad_pointer, to capture sub-frame samples
        end = audio.size
        if end - start < self.min_segment_samples:
            self._reset_segment()
            return
        seg = audio[start:end].copy()
        await self.queue.put(_PendingSegment(seg, False, True, self._utterance_id))
        self._utterance_id += 1
        self._reset_segment()

    def _reset_segment(self) -> None:
        """Begin a fresh VAD lifetime (new utterance)."""
        self.detector.reset()
        self._audio_chunks.clear()
        self._audio_length = 0
        self._audio_cache = None
        self.vad_pointer = 0
        self.seg_start = None
        self.last_partial_at = 0

    # ------------------------------------------------------------------ #
    # Transcription consumer
    # ------------------------------------------------------------------ #
    async def _consume(self) -> None:
        current_utterance_id = 0
        while True:
            item = await self.queue.get()
            if item is None:
                break
            if not isinstance(item, _PendingSegment):
                continue
            # Skip stale partials from previous utterances
            if item.is_partial and item.utterance_id < current_utterance_id:
                continue
            try:
                text, confidence = await self.stt.transcribe_async(
                    item.audio, self.language
                )
            except Exception as exc:
                log.exception("transcription failed")
                await self._send_error(exc)
                continue
            await self._send_event(text, item.is_partial, item.is_final, confidence)
            if item.is_final:
                current_utterance_id = item.utterance_id + 1
                await self._handle_final_utterance(text)

    # ------------------------------------------------------------------ #
    # LLM + final utterance handling
    # ------------------------------------------------------------------ #
    async def _handle_final_utterance(self, text: str) -> None:
        """After a final transcript: run it through the LLM brain.

        Pipeline: transcript -> LLM (streamed sentences) -> TTS per sentence.
        Uses streaming so the first sentence reaches the speaker before the
        LLM finishes writing the full reply (lower Time-To-First-Audio).

        When the LLM is disabled (no API key), falls back to the previous
        behaviour of speaking the raw transcript when ``TTS_AUTO_SPEAK`` is on.
        """
        if not text.strip():
            return

        # Don't process LLM if the connection is already closing/stopping.
        if self.stopping:
            return

        # Build the message list: system prompt + trimmed history + new user turn.
        messages = build_messages(
            self.history, text, max_history=settings.max_llm_history
        )
        # Keep history in sync (replaces any stale trimming).
        self.history = messages

        if llm_service.enabled:
            self._spawn_llm_turn(messages, fallback_text=text)
        else:
            if settings.tts_auto_speak:
                await self._start_tts(text)

    def _spawn_llm_turn(self, messages: list[MessageParam], fallback_text: str) -> None:
        """Spawn a background task that streams the LLM reply to TTS.

        The task iterates over ``llm_service.stream_response``, feeding each
        completed sentence to ``synthesize_stream`` for immediate audio output.
        The task reference is stored in ``self.llm_task`` so a barge-in
        (VAD speech-start) can cancel it mid-generation.
        """
        self.llm_task = asyncio.create_task(
            self._stream_llm_turn(messages, fallback_text)
        )

    async def _cancel_ongoing_turn(self) -> None:
        """Abort any in-progress LLM generation and TTS playback.

        Called when the user starts speaking again (barge-in) so the assistant
        does not keep talking over the user.
        """
        # Cancel the LLM streaming task.
        task, self.llm_task = self.llm_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("LLM task failed while cancelling")
        # Cancel any in-flight TTS.
        await self._cancel_tts()

    async def _stream_llm_turn(self, messages: list[dict], fallback_text: str) -> None:
        """Stream LLM sentences to TTS, appending the reply to history.

        Each completed sentence is sent to the client as an ``llm_response``
        event and simultaneously fed to ``synthesize_stream`` for speech output.
        If the LLM fails entirely, the fallback text (raw transcript) is spoken.
        """
        started = time.perf_counter()
        total_bytes = 0
        errors: list[dict] = []
        full_response_parts: list[str] = []

        try:
            async with contextlib.aclosing(
                llm_service.stream_response(
                    messages, fallback_text=fallback_text
                )
            ) as agen:
                async for sentence, model in agen:
                    full_response_parts.append(sentence)
                    await self._send_raw(
                        {
                            "type": "llm_response",
                            "text": sentence,
                            "model": model,
                        }
                    )
                    # Feed this sentence directly to TTS for immediate playback.
                    async for item in synthesize_stream(sentence, self.tts_voice):
                        if isinstance(item, dict):
                            event = {**item, "type": "tts_error"}
                            errors.append(event)
                            await self._send_raw(event)
                            continue
                        total_bytes += len(item)
                        await self._send_bytes(item)
        except asyncio.CancelledError:
            # Barge-in: discard the partial response, do not record it in history.
            log.debug("LLM stream cancelled (barge-in)")
            return
        except Exception as exc:  # noqa: BLE001 — fallback to TTS with transcript
            log.warning(
                "LLM streaming failed (%s: %s); speaking transcript",
                type(exc).__name__,
                exc,
            )
            await self._send_raw({"type": "llm_error", "error": str(exc)})
            # Speak the raw transcript as a fallback.
            async for item in synthesize_stream(fallback_text, self.tts_voice):
                if isinstance(item, dict):
                    event = {**item, "type": "tts_error"}
                    errors.append(event)
                    await self._send_raw(event)
                else:
                    total_bytes += len(item)
                    await self._send_bytes(item)
            full_response_parts.append(fallback_text)

        # Record the assistant turn in conversation history.
        if full_response_parts:
            full_response = " ".join(full_response_parts)
            assistant_msg: MessageParam = {
                "role": "assistant",
                "content": full_response,
            }
            self.history.append(assistant_msg)
            # Trim to the configured window, preserving the system prompt.
            if len(self.history) > settings.max_llm_history:
                self.history = [self.history[0]] + self.history[
                    -(settings.max_llm_history - 1) :
                ]

        elapsed_ms = (time.perf_counter() - started) * 1000
        await self._send_raw(
            {
                "type": "tts_end",
                "voice": self.tts_voice,
                "bytes": total_bytes,
                "seconds": round(total_bytes / (settings.sample_rate * 2), 3),
                "elapsed_ms": round(elapsed_ms, 1),
                "errors": errors,
            }
        )

    # ------------------------------------------------------------------ #
    # Speech output
    # ------------------------------------------------------------------ #
    async def _start_tts(self, text: str, voice: str | None = None) -> None:
        """(Re)start speech synthesis, cancelling anything already playing."""
        await self._cancel_ongoing_turn()
        self.tts_task = asyncio.create_task(self._speak(text, voice or self.tts_voice))

    async def _cancel_tts(self) -> None:
        task, self.tts_task = self.tts_task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("TTS task failed while cancelling")

    async def _speak(self, text: str, voice: str) -> None:
        """Stream synthesized PCM as binary frames, then a completion event.

        Synthesis failures are reported per sentence as ``tts_error`` events so
        the client can retry, fall back to another voice, or play a filler.
        """
        started = time.perf_counter()
        total_bytes = 0
        errors: list[dict] = []

        async for item in synthesize_stream(text, voice):
            if isinstance(item, dict):
                event = {**item, "type": "tts_error"}
                errors.append(event)
                await self._send_raw(event)
                continue
            total_bytes += len(item)
            await self._send_bytes(item)

        elapsed_ms = (time.perf_counter() - started) * 1000
        await self._send_raw(
            {
                "type": "tts_end",
                "voice": voice,
                "bytes": total_bytes,
                "seconds": round(total_bytes / (settings.sample_rate * 2), 3),
                "elapsed_ms": round(elapsed_ms, 1),
                "errors": errors,
            }
        )

    async def _send_event(
        self, transcript: str, is_partial: bool, is_final: bool, confidence: float
    ) -> None:
        event: dict = {
            "transcript": transcript,
            "is_partial": is_partial,
            "is_final": is_final,
            "confidence": round(float(confidence), 4),
        }
        if is_final and llm_service.enabled:
            event["llm_enabled"] = True
        await self._send_raw(event)

    async def _send_error(self, exc: BaseException) -> None:
        await self._send_raw({"type": "error", "error": str(exc)})

    async def _send_raw(self, payload: dict) -> None:
        try:
            async with self.send_lock:
                await self.ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            # Client gone; nothing left to do.
            pass

    async def _send_bytes(self, payload: bytes) -> bool:
        try:
            async with self.send_lock:
                await self.ws.send_bytes(payload)
        except (WebSocketDisconnect, RuntimeError):
            return False
        return True


def _resolve_language(value) -> str | None:
    if isinstance(value, str):
        return None if value.strip().lower() in _NONE_LANGS else value.strip()
    return value  # allow explicit None -> auto
