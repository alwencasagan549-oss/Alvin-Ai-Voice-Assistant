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
from collections import deque
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
from .llm import (
    ConversationHistory,
    MessageParam,
    WEB_SEARCH_TOOL_SCHEMA,
    build_messages,
    build_system_prompt,
    llm_service,
)
from .youtube import fallback_search_url, search_youtube
from .speaker import get_speaker_service
from .stt import STTService, audio_dbfs
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
_PARTIAL_QUEUE_TIMEOUT_SEC = 0.5
_FINAL_QUEUE_TIMEOUT_SEC = 2.0

# Consecutive turns that must go unidentified before the speaker context is
# dropped. One unknown turn is usually a cough or a mis-heard syllable, so the
# personalization is kept until the evidence repeats.
_UNKNOWN_SPEAKER_RESET_STREAK = 2

log = logging.getLogger("alvin.connection")

# Sentinel values for text control messages supported by clients.
_CONTROL_CONFIG = "config"
_CONTROL_STOP = "stop"
_CONTROL_PING = "ping"
_CONTROL_SPEAK = "speak"
_CONTROL_STOP_SPEAK = "stop_speak"
_CONTROL_TALK = "talk"
_CONTROL_CONTROL = "control"
_CONTROL_RESET = "reset"

_NONE_LANGS = {"auto", "none", "null", ""}


class ConnectionManager:
    """Registry of live sockets, so ``MAX_CONNECTIONS`` is actually enforced.

    The capacity check and the insert happen in the same synchronous block (no
    ``await`` in between), so two handshakes racing on the event loop cannot
    both pass the check and over-admit. Admission happens before ``accept()``:
    a rejected client gets a denied handshake instead of an accepted socket
    that is immediately closed. Starlette rejects a pre-accept close as an
    HTTP 403 response; browser WebSocket APIs do not receive its internal code.
    """

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def connect(self, websocket: WebSocket, max_connections: int) -> None:
        """Reserve a slot and accept, or reject the handshake before acceptance."""
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
        self.tts_enabled = True  # Per-connection TTS enable/disable
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
        self.ptt_explicitly_used = False
        self._ptt_timeout_handle: asyncio.Task | None = None
        # Wake-word arming window: commands are only honored within this many
        # seconds after a wake word, if WAKE_WORD_WINDOW_SEC > 0.
        self._wake_armed_until: float = 0.0

        # Conversation memory: system prompt is pinned at index 0, followed by
        # alternating user/assistant turns. ConversationHistory enforces the
        # MAX_LLM_HISTORY cap on every append, so no code path can grow the
        # context without bound.
        self.history: ConversationHistory = ConversationHistory(
            [{"role": "system", "content": settings.llm_system_prompt}],
            max_history=settings.max_llm_history,
        )

        # Speaker identification: track current speaker for personalization
        self.speaker_service = None
        self.current_speaker_id: str | None = None
        self.current_speaker_name: str | None = None
        self.speaker_audio_buffer: deque[np.ndarray] = deque()
        self._speaker_buffer_samples = 0
        self.speaker_identification_pending = False
        # Consecutive turns that did not match the current speaker. Used to
        # drop a stale personalization once the unknown speaker repeats.
        self._unknown_speaker_streak = 0

    def _get_audio(self) -> np.ndarray:
        """The filled prefix of the audio accumulator (no copy)."""
        return self._buffer[: self._audio_length]

    async def run(self) -> None:
        """Main entry: start the consumer, drain the socket, then flush."""
        # Initialize speaker service if enabled
        await self._init_speaker_service()

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
        # Echo gate: in ``gate`` mode, drop mic audio while the assistant is
        # speaking so its own playback cannot re-enter the pipeline. In
        # ``filter`` mode the gate is a no-op (barge-in stays enabled) and the
        # similarity check in ``_handle_final_utterance`` does the filtering.
        if not self.echo.should_accept_audio():
            return
        # PTT gating: drop audio when not active if either global PTT is enabled
        # or the client has explicitly used PTT via talk frames.
        ptt_enabled = settings.push_to_talk or self.ptt_explicitly_used
        if ptt_enabled and not self.ptt_active:
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return

        self._append_audio(samples)
        await self._feed_vad()

        # VAD updates seg_start while processing this frame. Only retain samples
        # once speech is active, so microphone silence cannot dilute embeddings.
        if settings.speaker_enabled and self.seg_start is not None:
            self._buffer_audio_for_speaker_id(samples)

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
            voice_enabled = data.get("voice_enabled")
            if isinstance(voice_enabled, bool):
                self.tts_enabled = voice_enabled
                if not voice_enabled:
                    await self._cancel_tts()
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
                if active is False and not self.ptt_active:
                    # Explicit "PTT is off" from a client that isn't using PTT:
                    # stop treating this connection as PTT-gated.
                    self.ptt_explicitly_used = False
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
        elif ctype == _CONTROL_RESET:
            # Client is starting a fresh mic session. Drop state that would
            # otherwise leak across Stop/Start on the same WebSocket.
            log.info("reset control frame; clearing session state")
            self.is_muted = False
            self.ptt_explicitly_used = False
            self.ptt_active = False
            self._wake_armed_until = 0.0
            self.echo.reset_cooldown()
            self._reset_segment()      # drains the buffer + VAD
            self._cancel_ptt_timeout()
        # Unknown control messages are ignored for forward compatibility.

    async def _set_ptt_active(self, active: bool) -> None:
        if self.ptt_active == active:
            return
        self.ptt_active = active
        self.ptt_explicitly_used = True  # Client explicitly used PTT
        if active:
            self._reset_ptt_timeout()
            if self.llm_task is not None or self.tts_task is not None:
                cancel_task = asyncio.create_task(self._cancel_ongoing_turn())
                cancel_task.add_done_callback(
                    lambda task: self._observe_background_task(
                        task, "PTT turn cancellation"
                    )
                )
        else:
            self._cancel_ptt_timeout()
        # Notify client of PTT state change
        await self._send_raw({"type": "ptt_state", "active": active})

    @staticmethod
    def _observe_background_task(task: asyncio.Task, label: str) -> None:
        """Retrieve and log failures from connection-owned fire-and-forget tasks."""
        if task.cancelled():
            return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("%s failed", label)

    def _reset_ptt_timeout(self) -> None:
        self._cancel_ptt_timeout()
        if settings.push_to_talk_timeout_ms > 0:
            self._ptt_timeout_handle = asyncio.create_task(
                asyncio.sleep(settings.push_to_talk_timeout_ms / 1000.0)
            )

            def _timeout_cb(t):
                if not t.cancelled():
                    timeout_task = asyncio.create_task(self._on_ptt_timeout())
                    timeout_task.add_done_callback(
                        lambda task: self._observe_background_task(
                            task, "PTT timeout handler"
                        )
                    )

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
        try:
            await asyncio.wait_for(
                self.queue.put(
                    _PendingSegment(seg.copy(), True, False, self._utterance_id)
                ),
                timeout=_PARTIAL_QUEUE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.debug("transcription queue full; dropping partial")

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

        # Reject VAD false-positives early: measure the segment's RMS before
        # it costs a Whisper decode. A "silent room" after browser AGC sits
        # around -45 to -38 dBFS, and Whisper will confidently hallucinate
        # "Thank you." / "Please subscribe." on that. Dropping here means the
        # consumer never sees the segment.
        level_dbfs = audio_dbfs(seg)
        if level_dbfs < settings.silence_threshold_dbfs:
            log.debug(
                "finalize: dropping quiet segment (%.1f dBFS < %.1f dBFS)",
                level_dbfs,
                settings.silence_threshold_dbfs,
            )
            self._reset_segment()
            return

        try:
            await asyncio.wait_for(
                self.queue.put(
                    _PendingSegment(seg, False, True, self._utterance_id)
                ),
                timeout=_FINAL_QUEUE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning(
                "transcription queue full for %.1fs; dropping final segment",
                _FINAL_QUEUE_TIMEOUT_SEC,
            )
            self._reset_segment()
            return
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
        try:
            self.queue.put_nowait(
                _PendingSegment(seg, False, True, self._utterance_id)
            )
        except asyncio.QueueFull:
            log.warning("transcription queue full during shutdown; dropping final segment")
            self._reset_segment()
            return
        self._utterance_id += 1
        self._reset_segment()


    def _append_audio(self, samples: np.ndarray) -> None:
        """Append samples to the accumulator, growing it geometrically if needed.

        The buffer is capped at ``_max_buffer_samples`` (one utterance's worth).
        A single oversized frame that would exceed the cap is clipped to it --
        ``_feed_vad`` will finalize the segment at that point and call
        ``_reset_segment``, which starts fresh.
        """
        # If we're not in an active utterance and the buffer is about to overflow,
        # drop the accumulated silence. Otherwise a long silent session (mic on,
        # user not talking) fills the buffer with silence; the cap check in
        # _feed_vad is gated on ``seg_start is not None``, so it never fires and
        # the buffer never resets. Once full, every future frame is silently
        # dropped -- this is the "works first time, can't be heard after Stop/Start"
        # bug, because Connection (and its buffer) survives across mic sessions.
        if (
            self.seg_start is None
            and self._audio_length + samples.size > self._max_buffer_samples
        ):
            self._reset_segment()

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
            # Muted mode never sends partials or finals to the cloud STT path.
            if self.is_muted:
                if item.is_final:
                    await self._handle_muted_segment(item)
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
    # Speaker Identification
    # ------------------------------------------------------------------ #
    async def _init_speaker_service(self) -> None:
        """Initialize the speaker service if enabled."""
        if settings.speaker_enabled:
            try:
                self.speaker_service = await get_speaker_service()
                log.info("Speaker identification enabled")
            except Exception as exc:
                log.warning("Failed to initialize speaker service: %s", exc)
                self.speaker_service = None

    async def _identify_speaker(self) -> None:
        """Identify the current speaker from buffered audio."""
        if not self.speaker_service or not self.speaker_audio_buffer:
            return

        # Combine buffered audio
        audio = np.concatenate(tuple(self.speaker_audio_buffer))
        self.speaker_audio_buffer.clear()
        self._speaker_buffer_samples = 0

        # Check minimum duration
        duration = audio.size / settings.sample_rate
        min_duration = settings.speaker_min_identify_sec
        if duration < min_duration:
            log.debug("Speaker ID skipped: audio too short (%.1fs < %.1fs)", duration, min_duration)
            return

        # Run identification
        try:
            result = await self.speaker_service.identify(audio)
            if result.is_known:
                # A known speaker clears any pending "not this person" streak,
                # even when it is the same one as last turn.
                self._unknown_speaker_streak = 0
                if result.speaker_id != self.current_speaker_id:
                    self.current_speaker_id = result.speaker_id
                    self.current_speaker_name = result.name
                    log.info("Speaker identified: %s (%.3f)", result.name, result.similarity)
                    # Notify client of speaker change
                    await self._send_raw({
                        "type": "speaker_change",
                        "speaker_id": result.speaker_id,
                        "name": result.name,
                        "similarity": round(result.similarity, 4),
                    })
            else:
                self._unknown_speaker_streak += 1
                log.debug(
                    "Unknown speaker (best similarity: %.3f, streak=%d)",
                    result.similarity,
                    self._unknown_speaker_streak,
                )
                if self._unknown_speaker_streak >= _UNKNOWN_SPEAKER_RESET_STREAK:
                    self._unknown_speaker_streak = 0
                    if self.current_speaker_name is not None:
                        # The voice that last set the personalization is not
                        # talking any more. Drop it so the next turn is not
                        # answered as the wrong person.
                        log.info(
                            "Speaker context cleared (unknown for %d consecutive turns)",
                            _UNKNOWN_SPEAKER_RESET_STREAK,
                        )
                        self.current_speaker_id = None
                        self.current_speaker_name = None
                        await self._send_raw({
                            "type": "speaker_change",
                            "speaker_id": None,
                            "name": None,
                            "similarity": round(result.similarity, 4),
                        })
        except Exception as exc:
            log.warning("Speaker identification failed: %s", exc)

    def _buffer_audio_for_speaker_id(self, audio: np.ndarray) -> None:
        """Buffer audio for speaker identification."""
        if self.speaker_service is None:
            return
        self.speaker_audio_buffer.append(audio.copy())
        self._speaker_buffer_samples += audio.size
        max_samples = int(
            settings.speaker_max_identify_sec * settings.sample_rate
        )
        while (
            self._speaker_buffer_samples > max_samples
            and self.speaker_audio_buffer
        ):
            removed = self.speaker_audio_buffer.popleft()
            self._speaker_buffer_samples -= removed.size

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

        # Speaker ID updates state for the next turn. Keep model inference off
        # this turn's intent, LLM, and TTS path.
        if (
            settings.speaker_enabled
            and self.speaker_service is not None
            and not self.speaker_identification_pending
        ):
            self.speaker_identification_pending = True
            task = asyncio.create_task(self._identify_speaker())
            task.add_done_callback(
                lambda completed: self._observe_background_task(
                    completed, "speaker identification"
                )
            )
            task.add_done_callback(
                lambda _completed: setattr(
                    self, "speaker_identification_pending", False
                )
            )

        # Try deterministic intent routing first.
        now = time.monotonic()
        wake_ok = now < self._wake_armed_until
        match = route_command(text, confidence, wake_word_ok=wake_ok)
        if match is not None:
            await self._dispatch_command(match)
            return

        # Build the message list: system prompt + trimmed history + new user turn.
        # Include speaker context if available
        if self.current_speaker_name:
            # Augment the FULL system prompt (today's date + TTS contract + search
            # rules) with speaker identity. Previously this used
            # ``settings.llm_system_prompt`` directly, which silently dropped the
            # date and all TTS rules the moment a speaker was identified.
            contextualized_prompt = (
                f"{build_system_prompt()} The current user is {self.current_speaker_name}."
            )
            messages = build_messages(
                self.history, text, max_history=settings.max_llm_history,
                system_prompt=contextualized_prompt
            )
        else:
            messages = build_messages(
                self.history, text, max_history=settings.max_llm_history
            )
        # Keep the self-trimming container after build_messages returns a list.
        self.history = ConversationHistory(
            messages, max_history=settings.max_llm_history
        )

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

        if match.intent == "play_youtube":
            await self._dispatch_youtube_play(match)
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

    async def _dispatch_youtube_play(self, match: IntentMatch) -> None:
        """Resolve a YouTube search and emit a ``play_youtube`` command.

        The search runs on the server so the browser receives a ready-to-play
        ``watch?v=...`` URL. If the search fails (network, rate-limit, no
        results, yt-dlp not installed) we fall back to the search-results URL
        so the user still lands somewhere useful.

        TTS is deliberately **not** spoken here: the browser navigates a
        pre-opened player tab to the video, and any speech would overlap with
        the song's audio.
        """
        query = (match.slots.get("query") or "").strip()
        if not query:
            return

        # If the assistant is currently talking, stop so the video doesn't
        # compete with its voice.
        await self._cancel_ongoing_turn()

        yt = await search_youtube(query)

        event: dict = {
            "type": "command",
            "intent": "play_youtube",
            "slots": {
                "query": query,
                "video_id": yt.video_id if yt else None,
                "title": yt.title if yt else None,
                "url": yt.url if yt else fallback_search_url(query),
                "resolved": bool(yt),
            },
            "raw": match.raw,
            "normalized": match.normalized,
        }
        if match.wake_word:
            event["wake_word"] = match.wake_word
        await self._send_raw(event)

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
        """Abort both the in-progress LLM stream task and any in-flight TTS."""
        had_turn = (self.tts_task is not None and not self.tts_task.done()) or (
            self.llm_task is not None and not self.llm_task.done()
        )
        # Cancel TTS task
        await self._cancel_tts()
        # Cancel LLM task
        task, self.llm_task = self.llm_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("LLM task failed while cancelling")
        if had_turn:
            # The user interrupted us: clear the post-playback cooldown so their
            # own voice is not gated away. (The echo history is left intact -- a
            # partial barge-in is not a reason to forget what we just said.)
            self.echo.reset_cooldown()

    async def _cancel_tts(self) -> None:
        """Cancel any in-flight TTS synthesis."""
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

    async def _stream_llm_turn(
        self, messages: list[MessageParam], fallback_text: str
    ) -> None:
        """Stream LLM sentences to TTS, appending the reply to history.

        Each completed sentence is sent to the client as an ``llm_response``
        event and simultaneously queued for speech output. A background TTS
        consumer task pulls from the queue, overlapping the edge-tts fetch of
        sentence N+1 with audio playback of sentence N so the LLM stream is
        never blocked waiting for synthesis.
        """
        started = time.perf_counter()
        total_bytes = 0
        errors: list[dict] = []
        full_response_parts: list[str] = []

        tools = None
        if llm_service.enabled:
            tools = [WEB_SEARCH_TOOL_SCHEMA]

        # Queue-based decoupling: the LLM producer pushes sentences onto this
        # queue and never awaits TTS playback. The TTS consumer pulls batches
        # (up to TTS_PREFETCH) and runs synthesize_stream concurrently with the
        # next LLM sentence being produced.
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_consumer() -> None:
            """Pull sentence batches from the queue and synthesize them.

            A None sentinel signals shutdown. Batches are formed with a short
            150 ms deadline so a slow LLM does not hold a single sentence
            indefinitely (latency ≈ first audio byte of *that* sentence).
            """
            while True:
                first = await tts_queue.get()
                if first is None:
                    return
                # Batch waiting in the queue so synthesize_stream can prefetch
                # multiple sentences concurrently.
                batch: list[str] = [first]
                deadline = time.perf_counter() + 0.15
                while len(batch) < settings.tts_prefetch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(
                            tts_queue.get(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    if nxt is None:
                        await play_text(" ".join(batch))
                        return
                    batch.append(nxt)
                await play_text(" ".join(batch))

        async def play_text(text: str) -> None:
            """Synthesize and stream a single text to audio, tracking bytes."""
            nonlocal total_bytes
            self.echo.begin_speech(text)
            try:
                async with contextlib.aclosing(
                    synthesize_stream(text, self.tts_voice)
                ) as audio_stream:
                    async for item in audio_stream:
                        if not self.tts_enabled:
                            break
                        if isinstance(item, dict):
                            event = {**item, "type": "tts_error"}
                            errors.append(event)
                            await self._send_raw(event)
                            continue
                        total_bytes += len(item)
                        if not await self._send_bytes(item):
                            return
            finally:
                self.echo.end_speech()

        tts_task = asyncio.create_task(tts_consumer())

        try:
            async with contextlib.aclosing(
                llm_service.stream_response(
                    messages,
                    fallback_text=fallback_text,
                    tools=tools,
                    tool_choice="auto",
                )
            ) as agen:
                async for sentence, model in agen:
                    # tool_call sentinel — flush queued sentences first so
                    # prefetch order is preserved, then speak filler.
                    if (
                        sentence is None
                        and isinstance(model, str)
                        and model.startswith("tool_call:")
                    ):
                        # Drain the queue: put a None sentinel so the consumer
                        # finishes the current batch, then send the filler.
                        await tts_queue.put(None)
                        await tts_task
                        tts_task = asyncio.create_task(tts_consumer())
                        if not self.tts_enabled:
                            continue
                        tool_name = model.split(":", 1)[1]
                        filler = (
                            "Let me check that."
                            if tool_name == "web_search"
                            else "Let me check."
                        )
                        await play_text(filler)
                        continue

                    full_response_parts.append(sentence)
                    await self._send_raw(
                        {
                            "type": "llm_response",
                            "text": sentence,
                            "model": model,
                        }
                    )
                    if self.tts_enabled:
                        await tts_queue.put(sentence)

            # Empty-stream guard: some models — notably CF llama-3.2-3b when
            # offered a tools schema — return an empty completion. Without this
            # check the turn produces no llm_response event, no audio, and no
            # history entry: a fully silent dead turn.
            if not full_response_parts:
                log.warning(
                    "LLM stream produced no sentences (empty stream); "
                    "speaking fallback transcript so the turn is not silent"
                )
                await self._send_raw({
                    "type": "llm_response",
                    "text": fallback_text,
                    "model": "fallback:empty-stream",
                })
                if self.tts_enabled:
                    await tts_queue.put(fallback_text)
                full_response_parts.append(fallback_text)

            # Signal the consumer to drain and finish.
            await tts_queue.put(None)
            await tts_task

        except asyncio.CancelledError:
            # Barge-in: discard the partial response, do not record it in history.
            tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tts_task
            log.debug("LLM stream cancelled (barge-in)")
            return
        except Exception as exc:  # noqa: BLE001 — fallback to TTS with transcript
            tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tts_task
            log.warning(
                "LLM streaming failed (%s: %s); speaking transcript",
                type(exc).__name__,
                exc,
            )
            await self._send_raw({"type": "llm_error", "error": str(exc)})
            # Speak the raw transcript as a fallback.
            if self.tts_enabled:
                await play_text(fallback_text)
            full_response_parts.append(fallback_text)

        # Record the assistant turn in conversation history.
        if full_response_parts:
            full_response = " ".join(full_response_parts)
            assistant_msg: MessageParam = {
                "role": "assistant",
                "content": full_response,
            }
            self.history.append(assistant_msg)

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
        if not self.tts_enabled:
            return
        if settings.push_to_talk:
            await self._set_ptt_active(False)
        await self._cancel_ongoing_turn()
        self.tts_task = asyncio.create_task(self._speak_tts(text, voice or self.tts_voice))

    async def _speak_tts(self, text: str, voice: str) -> None:
        """Stream synthesized PCM as binary frames, then a completion event.

        Synthesis failures are reported per sentence as ``tts_error`` events so
        the client can retry, fall back to another voice, or play a filler.
        """
        started = time.perf_counter()
        total_bytes = 0
        errors: list[dict] = []

        # Remember what is being said so the echo guard can recognize its
        # own playback when it leaks back through the mic.
        self.echo.begin_speech(text)
        try:
            async with contextlib.aclosing(synthesize_stream(text, voice)) as audio_stream:
                async for item in audio_stream:
                    if isinstance(item, dict):
                        event = {**item, "type": "tts_error"}
                        errors.append(event)
                        await self._send_raw(event)
                        continue
                    total_bytes += len(item)
                    if not await self._send_bytes(item):
                        log.debug("client disconnected during TTS; stopping synthesis")
                        return
        finally:
            self.echo.end_speech()

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
