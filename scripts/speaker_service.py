"""Standalone Speaker Service for the Alvin voice assistant.

Consumes binary PCM audio frames and JSON control events from the Alvin
WebSocket server (``/ws/transcribe``) and plays them to audio hardware with
low-latency pre-buffering and barge-in support.

Usage::

    python scripts/speaker_service.py
    python scripts/speaker_service.py --url ws://localhost:8000/ws/transcribe --token YOUR_KEY
    python scripts/speaker_service.py --device 0 --prebuffer-ms 100

Audio output: 16 kHz / 16-bit signed integer (s16le) / mono — matching the
backend TTS pipeline in ``app/tts.py``.

Barge-in:
    When the assistant starts a new response while audio from the previous turn
    is still queued or playing (i.e. ``tts_end`` was not received), the local
    buffer is flushed and the hardware stream is aborted so speech cuts off
    instantaneously.  A final STT transcript arriving mid-turn is treated the
    same way — the user just spoke, so the assistant should stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when run as ``python scripts/speaker_service.py``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sounddevice as sd
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

# --- Audio specifications (must match backend TTS pipeline) ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BYTES_PER_SAMPLE = 2  # 16-bit signed integer
BYTES_PER_MS = (SAMPLE_RATE / 1000) * BYTES_PER_SAMPLE  # 32 bytes per ms

# Default pre-buffer: ~50 ms of audio (~1 600 bytes) to absorb network jitter
# and prevent clicks / pops on the first write of each turn.
DEFAULT_PREBUFFER_MS = 50

# Threshold below which the queue is considered "empty enough" for barge-in
# purposes — avoids flushing on tiny residual amounts.
BARGE_IN_QUEUE_THRESHOLD = 0  # any queued bytes triggers flush

log = logging.getLogger("speaker")


class SpeakerService:
    """Decoupled WebSocket audio consumer with hardware playback and barge-in.

    Architecture::

        WebSocket receive loop  →  asyncio.Queue  →  sounddevice playback
        (event-loop thread)       (thread-safe)     (separate coroutine,
                                                      blocking I/O in a thread)

    The receive loop only calls the synchronous handlers ``_on_audio`` and
    ``_on_text``, which never block on I/O.  Blocking ``stream.write()`` calls
    run inside ``asyncio.to_thread`` so the WebSocket event loop stays free to
    process incoming frames.

    Turn lifecycle:

    * A turn starts when the first ``llm_response`` or binary PCM frame arrives
      after an idle state or a completed turn (``tts_end``).
    * A turn ends when ``tts_end`` is received.
    * Barge-in is detected when a *new* turn's first event arrives while the
      previous turn hasn't ended — audio already queued or playing is flushed.
    """

    def __init__(
        self,
        ws_url: str,
        token: str | None = None,
        device: int | str | None = None,
        prebuffer_ms: int = DEFAULT_PREBUFFER_MS,
    ) -> None:
        self.ws_url = ws_url
        self.token = token
        self.device = device
        self.prebuffer_bytes = max(1, int(BYTES_PER_MS * prebuffer_ms))

        # Thread-safe queue between the WebSocket receive loop and the playback loop.
        # Items are ``bytes`` PCM chunks or ``None`` (sentinel to stop the loop).
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        # --- Pre-buffering state ---
        self._prebuffer: bytearray = bytearray()
        self._prebuffer_done: bool = False

        # --- Hardware stream ---
        self._stream: sd.OutputStream | None = None
        self._stream_lock = threading.Lock()

        # --- Turn tracking for barge-in detection ---
        self._turn_active: bool = False
        self._turn_completed: bool = True  # tts_end received since last turn start

        self._ws: Any = None
        self._playback_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clear_buffer(self) -> None:
        """Purge all queued audio and abort hardware playback.

        Safe to call from any thread.  ``sounddevice``'s ``abort()`` is
        thread-safe, and the ``threading.Lock`` prevents races with the
        playback thread's ``write()`` calls.
        """
        self._flush_audio()

    async def speak(self, text: str, voice: str | None = None) -> bool:
        """Send a ``speak`` command to the backend for immediate synthesis.

        Returns ``True`` if the command was sent, ``False`` if not connected.
        """
        if self._ws is None:
            log.warning("speak ignored: not connected")
            return False
        payload: dict = {"type": "speak", "text": text}
        if voice is not None:
            payload["voice"] = voice
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            log.error("failed to send speak command: %s", exc)
            return False
        return True

    async def ping(self) -> bool:
        """Send a ``ping`` and wait for ``pong``."""
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps({"type": "ping"}))
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("ping failed: %s", exc)
            return False

    async def run(self) -> None:
        """Connect to the backend and run the speaker service until disconnected."""
        url = self._build_url()
        log.info("speaker service connecting to %s", self._safe_url(url))

        self._playback_task = asyncio.create_task(self._playback_loop())

        try:
            async with connect(url) as ws:
                self._ws = ws
                log.info("speaker service connected")
                await self._receive_loop(ws)
        except ConnectionClosedOK:
            log.info("connection closed (normal)")
        except ConnectionClosed as exc:
            log.warning("connection closed: %s (code %d)", exc.reason, exc.code)
        finally:
            self._ws = None
            await self.audio_queue.put(None)
            if self._playback_task is not None:
                await asyncio.gather(self._playback_task, return_exceptions=True)
                self._playback_task = None
            self._stream = None
            log.info("speaker service stopped")

    # ------------------------------------------------------------------ #
    # WebSocket receive loop
    # ------------------------------------------------------------------ #
    async def _receive_loop(self, websocket: Any) -> None:
        """Process incoming WebSocket frames until the connection closes."""
        async for message in websocket:
            if isinstance(message, bytes):
                self._on_audio(message)
            elif isinstance(message, str):
                self._on_text(message)

    # ------------------------------------------------------------------ #
    # Message handlers (synchronous for testability)
    # ------------------------------------------------------------------ #
    def _on_audio(self, data: bytes) -> None:
        """Handle a binary PCM frame from the server."""
        if not data:
            return

        # Detect new turn: binary arriving when idle or after tts_end.
        if not self._turn_active or self._turn_completed:
            self._start_new_turn()

        self._queue_audio(data)

    def _on_text(self, text: str) -> None:
        """Dispatch a JSON control event to the appropriate handler."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("malformed JSON from server: %s", text[:200])
            return

        if not isinstance(data, dict):
            return

        event_type = data.get("type")

        if event_type == "llm_response":
            self._on_llm_response(data)
        elif event_type == "llm_error":
            self._on_llm_error(data)
        elif event_type == "tts_end":
            self._on_tts_end(data)
        elif event_type == "tts_error":
            self._on_tts_error(data)
        elif event_type == "error":
            self._on_error(data)
        elif event_type == "pong":
            log.debug("pong")
        elif "transcript" in data:
            self._on_transcript(data)
        else:
            log.debug("unhandled event type: %r", event_type)

    def _on_llm_response(self, data: dict) -> None:
        """Handle a streamed LLM response sentence.

        Barge-in detection: if the previous turn is still active (``tts_end``
        not yet received) and audio is still queued, the backend has started
        a new response — flush the old audio and begin pre-buffering for the
        new one.
        """
        text = data.get("text", "")
        model = data.get("model", "")
        if model:
            log.info("assistant [%s]: %s", model, text)
        else:
            log.info("assistant: %s", text)

        if self._turn_active and not self._turn_completed:
            # Mid-turn: this could be a continuation sentence (same turn)
            # or the start of a new turn after barge-in.  If audio is still
            # queued, the previous content hasn't been fully consumed — treat
            # it as barge-in and flush.
            if not self.audio_queue.empty():
                log.info("barge-in: new llm_response during active turn")
                self._flush_audio()
        else:
            # New turn (from idle or after tts_end): reset pre-buffering.
            self._start_new_turn()

    def _on_llm_error(self, data: dict) -> None:
        """Handle an LLM failure event."""
        error = data.get("error", "unknown error")
        log.error("LLM error: %s", error)
        # LLM failure means the assistant's response is dead.  If audio is
        # queued, the server may fall back to speaking the raw transcript —
        # let that audio play through.

    def _on_tts_end(self, data: dict) -> None:
        """Handle TTS turn completion."""
        self._turn_completed = True
        byte_count = data.get("bytes", 0)
        seconds = data.get("seconds", 0.0)
        errors = data.get("errors", [])
        if errors:
            log.warning(
                "tts_end: %d bytes (%.2fs) with %d error(s)",
                byte_count,
                seconds,
                len(errors),
            )
        else:
            log.info("tts_end: %d bytes (%.2fs)", byte_count, seconds)

        # Flush any pre-buffered data so it starts playing immediately.
        if not self._prebuffer_done and self._prebuffer:
            self.audio_queue.put_nowait(bytes(self._prebuffer))
            self._prebuffer = bytearray()
        self._prebuffer_done = True

    def _on_tts_error(self, data: dict) -> None:
        """Handle a per-sentence TTS synthesis error."""
        stage = data.get("stage", "tts")
        message = data.get("message", "unknown error")
        sentence_index = data.get("sentence_index", -1)
        fatal = data.get("fatal", False)
        log.error(
            "tts error (sentence %d, %s, fatal=%s): %s",
            sentence_index,
            stage,
            fatal,
            message,
        )
        # Non-fatal errors don't interrupt the stream; the server continues
        # with the next sentence.  Fatal errors are followed by tts_end.

    def _on_error(self, data: dict) -> None:
        """Handle a server-level error event."""
        error = data.get("error", "unknown error")
        log.error("server error: %s", error)

    def _on_transcript(self, data: dict) -> None:
        """Handle a transcription event (if on the same connection as the mic).

        A final transcript arriving while the assistant is speaking is the
        strongest barge-in signal: the user just spoke, so the backend should
        cut off the assistant's response.
        """
        transcript = data.get("transcript", "")
        confidence = data.get("confidence", 0.0)
        is_partial = data.get("is_partial", False)
        is_final = data.get("is_final", False)

        if is_final:
            kind = "FINAL"
        elif is_partial:
            kind = "partial"
        else:
            kind = "transcript"

        log.info("[%s conf=%.2f] %s", kind, confidence, transcript)

        if (
            is_final
            and self._turn_active
            and not self._turn_completed
            and not self.audio_queue.empty()
        ):
            log.info("barge-in: final transcript during active turn")
            self._flush_audio()

    # ------------------------------------------------------------------ #
    # Turn / pre-buffer state management
    # ------------------------------------------------------------------ #
    def _start_new_turn(self) -> None:
        """Reset state for a new TTS turn, flushing any leftover audio."""
        # Flush any residual audio from the previous turn.
        if not self.audio_queue.empty() or self._prebuffer:
            self._flush_audio()
        self._turn_active = True
        self._turn_completed = False
        self._prebuffer_done = False
        self._prebuffer = bytearray()

    def _queue_audio(self, data: bytes) -> None:
        """Queue audio data, applying pre-buffering for new turns.

        Until ``prebuffer_bytes`` has accumulated, data is held in
        ``self._prebuffer``.  Once the threshold is reached (or the turn
        ends), the accumulated data is flushed to ``self.audio_queue`` in a
        single burst so the playback loop can begin.
        """
        if not self._prebuffer_done:
            self._prebuffer.extend(data)
            if len(self._prebuffer) >= self.prebuffer_bytes:
                self.audio_queue.put_nowait(bytes(self._prebuffer))
                self._prebuffer = bytearray()
                self._prebuffer_done = True
            # Don't queue yet — wait for pre-buffer threshold.
            return

        self.audio_queue.put_nowait(data)

    def _flush_audio(self) -> None:
        """Purge all queued audio and abort hardware playback.

        Called on every barge-in to cut off speech instantaneously.
        Also invoked when a new turn starts and residual audio from the
        previous turn remains.
        """
        # Clear the pre-buffer accumulator.
        self._prebuffer.clear()

        # Drain the asyncio queue.
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Abort hardware playback immediately.
        if self._stream is not None:
            with self._stream_lock:
                try:
                    self._stream.abort()
                    self._stream.start()  # restart so the next write succeeds
                except Exception as exc:  # noqa: BLE001
                    log.debug("stream abort error (may be expected): %s", exc)

        self._prebuffer_done = False
        log.info("audio buffer flushed (barge-in)")

    # ------------------------------------------------------------------ #
    # Playback loop
    # ------------------------------------------------------------------ #
    async def _playback_loop(self) -> None:
        """Pull PCM chunks from the queue and write to the speaker device.

        Blocking ``stream.write()`` calls run inside ``asyncio.to_thread``
        so the event loop stays free for the WebSocket receive loop.  The
        ``threading.Lock`` in ``_safe_write`` / ``_flush_audio`` prevents
        concurrent access to the stream during barge-in.
        """
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self.device,
            blocksize=1024,
            latency="low",
        ) as stream:
            self._stream = stream
            log.info(
                "audio device ready: %d Hz / %s / %d ch (prebuffer=%d bytes)",
                SAMPLE_RATE,
                DTYPE,
                CHANNELS,
                self.prebuffer_bytes,
            )

            while True:
                chunk = await self.audio_queue.get()
                if chunk is None:
                    break
                await asyncio.to_thread(self._safe_write, stream, chunk)
                self.audio_queue.task_done()

    def _safe_write(self, stream: sd.OutputStream, data: bytes) -> None:
        """Write raw PCM to the stream, recovering from abort/restart races.

        Called from a worker thread via ``asyncio.to_thread``.  The
        ``threading.Lock`` ensures ``_flush_audio`` can't abort the stream
        while a write is in flight.
        """
        if not data:
            return
        with self._stream_lock:
            try:
                stream.write(data)
            except Exception:  # noqa: BLE001
                # Stream may have been aborted by barge-in; restart and retry.
                try:
                    stream.start()
                    stream.write(data)
                except Exception as exc:  # noqa: BLE001
                    log.warning("audio write failed (recovered): %s", exc)

    # ------------------------------------------------------------------ #
    # URL helpers
    # ------------------------------------------------------------------ #
    def _build_url(self) -> str:
        """Append the API token to the WebSocket URL as a query parameter."""
        url = self.ws_url
        if self.token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={self.token}"
        return url

    def _safe_url(self, url: str) -> str:
        """Mask the token in the URL for logging."""
        if not self.token:
            return url
        return url.replace(self.token, "***")


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alvin standalone Speaker Service — plays TTS audio to speakers.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("SPEAKER_WS_URL", "ws://localhost:8000/ws/transcribe"),
        help="WebSocket URL of the Alvin backend (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("API_KEY", os.getenv("SERVICE_AUTH_TOKEN", "")),
        help="API key for auth (default: env API_KEY). Empty = no auth.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice output device index (default: system default)",
    )
    parser.add_argument(
        "--prebuffer-ms",
        type=int,
        default=DEFAULT_PREBUFFER_MS,
        help="pre-buffer duration in ms (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="log verbosity (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    service = SpeakerService(
        ws_url=args.url,
        token=args.token or None,
        device=args.device,
        prebuffer_ms=args.prebuffer_ms,
    )

    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        log.info("interrupted by user")


if __name__ == "__main__":
    main()
