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
from .echo import EchoGuard
from .intents import (
    ALWAYS_CONFIRM_INTENTS,
    ASSISTANT_SIDE_EFFECT_INTENTS,
    MIC_CONTROL_INTENTS,
    IntentMatch,
    route_command,
)
from .llm import ConversationHistory, MessageParam, WEB_SEARCH_TOOL_SCHEMA, build_messages, llm_service
from .stt import STTService
from .tts import synthesize_stream
from .vad import FRAME_SAMPLES, VADStreamDetector

# Initial capacity of the per-connection audio accumulator, in samples
# (~2 s at 16 kHz). It doubles on demand up to one utterance's worth, so a
# short utterance never allocates a 20 s buffer while a long one stops
# reallocating after the first couple of seconds.
_INITIAL_BUFFER_SAMPLES = 32_000

# How long to let the transcription consumer drain on shutdown before we stop
# waiting for it. A local Whisper decode of a long segment can take a couple of
# seconds; past this the socket is closed anyway.
_CONSUMER_SHUTDOWN_TIMEOUT_SEC = 5.0

log = logging.getLogger("alvin.connection")

# Sentinel values for text control messages supported by clients.
_CONTROL_CONFIG = "config"
_CONTROL_STOP = "stop"
_CONTROL_PING = "ping"
_CONTROL_SPEAK = "speak"
_CONTROL_STOP_SPEAK = "stop_speak"
_CONTROL_TALK = "talk"
_CONTROL_CONTROL = "control"

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
        # ``_buffer`` is a fixed-capacity growable buffer with an explicit fill length:
        # appending is O(new samples) and slicing for the VAD/segments never copies
        # the whole utterance.
        self._buffer = np.empty(_INITIAL_BUFFER_SAMPLES, dtype=np.float32)
        self._audio_length = 0
        self.vad_pointer = 0

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
        # Hard cap on how much audio we will ever hold for one utterance. Sized
        # from MAX_SEGMENT_SECONDS plus one VAD frame of slack so the growable
        # buffer stops reallocating in the steady state.
        self._max_buffer_samples = self.max_segment_samples + FRAME_SAMPLES * 2

        self.queue: asyncio.Queue[_PendingSegment | None] = asyncio.Queue(maxsize=settings.transcription_queue_maxsize)
        self.consumer: asyncio.Task | None = None
        self.stopping = False
        self._utterance_id: int = 0

        # Speech output: at most one synthesis or LLM stream runs at a time,
        # so a new request cancels the previous one instead of interleaving.
        self.tts_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        self.tts_voice = settings.tts_voice
        self.send_lock = asyncio.Lock()

        # Mic-control audio gating: when the user mutes the microphone,
        # incoming segments run a local-only keyword pass (never the cloud)
        # instead of the full STT -> intent/LLM pipeline.
        self.is_muted = False

        # Echo guard: tracks playback state and history for self-echo filtering.
        self.echo = EchoGuard()

        # Push-to-talk: when active, audio is processed; when inactive, audio
        # is discarded unless the user is holding the hotkey.
        self.ptt_active = False
        self._ptt_timeout_handle: asyncio.Task | None = None

        # Conversation memory: system prompt is pinned at index 0, followed by
        # alternating user/assistant turns. ConversationHistory enforces the
        # MAX_LLM_HISTORY cap on every append, so no code path can grow the
        # context without bound.
        self.history: ConversationHistory = ConversationHistory(
            [{"role": "system", "content": settings.llm_system_prompt}],
            max_history=settings.max_llm_history,
        )

    def _get_audio(self) -> np.ndarray:
        """The filled prefix of the audio accumulator (no copy)."""
        return self._buffer[: self._audio_length]

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
            self._cancel_ptt_timeout()
            await self._flush_pending()
            # Never block on a full queue during shutdown: the consumer is
            # bounded by a drain timeout, so dropping the sentinel here is safe.
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            await self._cancel_ongoing_turn()
            if self.consumer is not None:
                try:
                    await asyncio.wait_for(self.consumer, timeout=_CONSUMER_SHUTDOWN_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    log.warning(
                        "consumer did not drain within %.1fs; cancelling",
                        _CONSUMER_SHUTDOWN_TIMEOUT_SEC,
                    )
                    self.consumer.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
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
        if settings.push_to_talk and not self.ptt_active:
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
        self._append_audio(samples)
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
        elif ctype == _CONTROL_TALK:
            active = data.get("active")
            if active is not None:
                await self._set_ptt_active(bool(active))
        elif ctype == _CONTROL_CONTROL:
            # Client-side keyword spotter (or equivalent) heard a command
            # while the mic was muted; the client stays the one deciding what
            # to listen for, the server just applies the state change.
            action = data.get("action")
            if action == "unmute_request" and self.is_muted:
                log.info("unmute requested via control frame; re-enabling STT")
                match = route_command("unmute", allowed=MIC_CONTROL_INTENTS)
                if match is not None:
                    await self._dispatch_command(match)
        # Unknown control messages are ignored for forward compatibility.

    async def _set_ptt_active(self, active: bool) -> None:
        if self.ptt_active == active:
            return
        self.ptt_active = active
        if active:
            self._reset_ptt_timeout()
            if self.llm_task is not None or self.tts_task is not None:
                asyncio.create_task(self._cancel_ongoing_turn())
        else:
            self._cancel_ptt_timeout()
        # Notify client of PTT state change
        await self._send_raw({"type": "ptt_state", "active": active})

    def _reset_ptt_timeout(self) -> None:
        self._cancel_ptt_timeout()
        if settings.push_to_talk_timeout_ms > 0:
            self._ptt_timeout_handle = asyncio.create_task(
                asyncio.sleep(settings.push_to_talk_timeout_ms / 1000.0)
            )

            def _timeout_cb(t):
                if not t.cancelled():
                    asyncio.create_task(self._on_ptt_timeout())

            self._ptt_timeout_handle.add_done_callback(_timeout_cb)

    def _cancel_ptt_timeout(self) -> None:
        if self._ptt_timeout_handle is not None:
            self._ptt_timeout_handle.cancel()
            self._ptt_timeout_handle = None

    async def _on_ptt_timeout(self) -> None:
        self.ptt_active = False
        self._ptt_timeout_handle = None
        log.info("PTT timeout expired; auto-disabling push-to-talk")
        await self._send_raw({"type": "ptt_state", "active": False})

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
                # Auto-disable PTT when speech ends (single-press mode)
                if settings.push_to_talk and self.ptt_active:
                    await self._set_ptt_active(False)
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


    def _append_audio(self, samples: np.ndarray) -> None:
        """Append samples to the accumulator, growing it geometrically if needed.

        The buffer is capped at ``_max_buffer_samples`` (one utterance's worth).
        A single oversized frame that would exceed the cap is clipped to it --
        ``_feed_vad`` will finalize the segment at that point and call
        ``_reset_segment``, which starts fresh.
        """
        needed = self._audio_length + samples.size
        if needed > self._buffer.size:
            # Grow geometrically, but never past what one utterance can hold:
            # past that point MAX_SEGMENT_SECONDS ends the utterance anyway.
            capacity = min(max(needed, self._buffer.size * 2), self._max_buffer_samples)
            grown = np.empty(capacity, dtype=np.float32)
            grown[: self._audio_length] = self._buffer[: self._audio_length]
            self._buffer = grown
        # Clip the write to the hard cap so an oversized frame cannot grow the
        # buffer beyond ``_max_buffer_samples`` (a previous ``max(capacity, needed)``
        # line defeated the cap here, allowing unbounded growth from a large frame).
        write_end = min(needed, self._buffer.size)
        copy_len = write_end - self._audio_length
        if copy_len > 0:
            self._buffer[self._audio_length : write_end] = samples[:copy_len]
        self._audio_length = write_end

    def _reset_segment(self) -> None:
        """Begin a fresh VAD lifetime (new utterance)."""
        self.detector.reset()
        self._audio_length = 0
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
    async def _handle_final_utterance(self, text: str, confidence: float = 1.0) -> None:
        """After a final transcript: try deterministic intents first, then LLM.

        Fixed device actions ("stop", "mute", "volume up", "turn off the
        lights", ...) are matched on the ITN-normalized transcript by
        :func:`app.intents.route_command` and never pay an LLM round-trip;
        ambiguous or open-ended utterances fall through to the LLM brain.
        """
        if not text.strip():
            return

        # Don't process LLM if the connection is already closing/stopping.
        if self.stopping:
            return

        # Self-echo: if this final transcript is what we just said, it is our
        # own voice leaking back in, not a user request. Dropping it before
        # the intent router keeps the assistant from obeying itself.
        if self.echo.is_echo(text):
            # The real final event was already emitted by the consumer; this
            # utterance is our own playback leaking back, so drop it here
            # without a second event (a duplicate is_final would make clients
            # treat one utterance as two turns).
            log.info("echo guard: dropped self-referential transcript %r", text[:60])
            return

        # Try deterministic intent routing first.
        match = route_command(text, confidence)
        if match is not None:
            await self._dispatch_command(match)
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

    async def _dispatch_command(self, match: IntentMatch) -> None:
        """Execute a deterministic device command without an LLM round-trip.

        Assistant-side effects (stop/pause) cancel any in-progress LLM stream
        and TTS first so the assistant does not keep talking. Every command is
        relayed to the client as a ``command`` event carrying the intent and
        extracted slots; the owning device acts on it. Commands never enter
        the LLM conversation history.

        Mic-control commands additionally flip ``self.is_muted`` (audio
        gating), and their hardcoded confirmation is always spoken -- it is
        the user's only feedback, and after a mute it explains how to get
        the assistant back.
        """
        if match.intent in ASSISTANT_SIDE_EFFECT_INTENTS:
            await self._cancel_ongoing_turn()

        # A bare wake word is not a device command: it is a local "I'm here".
        # Acknowledge it in-band (no ``command`` event, no client action), arm
        # the wake window if one is configured, and keep it out of history so
        # the next utterance is the start of a fresh conversation turn.
        if match.intent == "wake":
            if settings.wake_word_window_sec > 0:
                self._wake_armed_until = (
                    time.monotonic() + settings.wake_word_window_sec
                )
            log.debug("wake word %r acknowledged", match.wake_word)
            if settings.wake_speak_ack and match.confirmation:
                await self._start_tts(match.confirmation)
            return

        if match.intent == "mute_mic":
            self.is_muted = True
        elif match.intent == "unmute_mic":
            self.is_muted = False

        speak = match.confirmation and (
            settings.intent_speak_confirmation or match.intent in ALWAYS_CONFIRM_INTENTS
        )

        event: dict = {
            "type": "command",
            "intent": match.intent,
            "slots": match.slots,
            "raw": match.raw,
            "normalized": match.normalized,
        }
        if match.action:
            event["action"] = match.action
        if match.wake_word:
            # Lets a client render a "heard" indicator without re-running STT.
            event["wake_word"] = match.wake_word
        if speak:
            event["speak"] = match.confirmation
        await self._send_raw(event)

        if speak:
            await self._start_tts(match.confirmation)

    async def _handle_muted_segment(self, item: _PendingSegment) -> None:
        """Muted-mode transcription: local keyword pass only, never the cloud.

        While the mic is muted, final segments are decoded on the local INT8
        model and matched against the mic-control intents only, so "unmute"
        (or "alvin unmute") still works without paying for a full STT/LLM
        turn. Partial hypotheses are ignored, and anything that is not a
        mic-control command is dropped silently.
        """
        if not item.is_final:
            return
        try:
            text, confidence = await self.stt.transcribe_local_async(
                item.audio, self.language
            )
        except Exception as exc:  # noqa: BLE001 - gating must not kill the loop
            log.warning("muted keyword pass failed (%s: %s)", type(exc).__name__, exc)
            return
        if not text.strip():
            return
        log.debug("muted keyword pass: %r (conf=%.2f)", text[:60], confidence)
        match = route_command(text, confidence, allowed=MIC_CONTROL_INTENTS)
        if match is not None:
            await self._dispatch_command(match)

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
        When a tool call is detected, a short filler line is spoken before the
        search runs.
        """
        started = time.perf_counter()
        total_bytes = 0
        errors: list[dict] = []
        full_response_parts: list[str] = []

        tools = None
        if llm_service.enabled:
            tools = [WEB_SEARCH_TOOL_SCHEMA]

        try:
            async with contextlib.aclosing(
                llm_service.stream_response(
                    messages, fallback_text=fallback_text, tools=tools, tool_choice="auto"
                )
            ) as agen:
                async for sentence, model in agen:
                    if sentence is None and isinstance(model, str) and model.startswith("tool_call:"):
                        tool_name = model.split(":", 1)[1]
                        filler = "Let me check that." if tool_name == "web_search" else "Let me check."
                        async for item in synthesize_stream(filler, self.tts_voice):
                            if isinstance(item, dict):
                                event = {**item, "type": "tts_error"}
                                errors.append(event)
                                await self._send_raw(event)
                                continue
                            total_bytes += len(item)
                            await self._send_bytes(item)
                        continue

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
        if settings.push_to_talk:
            await self._set_ptt_active(False)
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
            "type": "transcript",
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
