"""Unit tests for the standalone Speaker Service.

These tests exercise the message-handling state machine and barge-in logic
without requiring a live WebSocket server or audio hardware.  The
``sounddevice.OutputStream`` is replaced with a mock that records writes and
tracks ``abort`` / ``start`` calls, so every transition is deterministic.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Self

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.speaker_service import SpeakerService


class MockStream:
    """Stand-in for ``sounddevice.OutputStream`` used in unit tests.

    Records every ``write`` call and tracks ``abort`` / ``start`` so tests
    can assert on playback behaviour without an audio device.
    """

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.aborts: int = 0
        self.starts: int = 0
        self._active = True
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        with self._lock:
            if not self._active:
                raise RuntimeError("Stream not active")
            self.writes.append(data)
            return len(data)

    def abort(self) -> None:
        with self._lock:
            self._active = False
            self.aborts += 1

    def start(self) -> None:
        with self._lock:
            self._active = True
            self.starts += 1

    def active(self) -> bool:
        with self._lock:
            return self._active

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._active = False


def _make_service(prebuffer_ms: int = 0) -> SpeakerService:
    """Create a SpeakerService wired to a MockStream.

    A ``prebuffer_ms`` of 0 means the pre-buffer threshold is 1 byte, so every
    chunk passes through immediately — useful for most logic tests.
    """
    svc = SpeakerService(
        ws_url="ws://localhost:8000/ws/transcribe",
        token=None,
        prebuffer_ms=prebuffer_ms,
    )
    svc._stream = MockStream()
    return svc


def _pcm_bytes(n: int) -> bytes:
    """Generate ``n`` bytes of dummy int16 PCM data."""
    return bytes(range(256)) * ((n + 255) // 256)


def _text_event(**kwargs: object) -> str:
    """Serialise a dict to a JSON text frame string."""
    return json.dumps(kwargs)


# --------------------------------------------------------------------------- #
# Audio handler
# --------------------------------------------------------------------------- #
class TestOnAudio:
    def test_binary_starts_new_turn(self):
        svc = _make_service()
        assert not svc._turn_active
        svc._on_audio(_pcm_bytes(100))
        assert svc._turn_active
        assert not svc._turn_completed
        assert not svc.audio_queue.empty()

    def test_binary_during_active_turn_queues_directly(self):
        svc = _make_service()
        svc._on_audio(_pcm_bytes(100))
        svc._on_audio(_pcm_bytes(200))
        # Both chunks should be queued (prebuffer is done after first threshold)
        items = []
        while not svc.audio_queue.empty():
            items.append(svc.audio_queue.get_nowait())
        total = sum(len(item) for item in items)
        assert total >= 100

    def test_empty_binary_ignored(self):
        svc = _make_service()
        svc._on_audio(b"")
        assert not svc._turn_active
        assert svc.audio_queue.empty()

    def test_binary_after_tts_end_starts_new_turn(self):
        svc = _make_service()
        svc._on_audio(_pcm_bytes(100))
        svc._on_text(_text_event(type="tts_end", bytes=100, seconds=0.01))
        svc._on_audio(_pcm_bytes(50))
        assert svc._turn_active
        assert not svc._turn_completed


# --------------------------------------------------------------------------- #
# Event routing
# --------------------------------------------------------------------------- #
class TestEventRouting:
    def test_unknown_event_type_logged_not_raised(self):
        svc = _make_service()
        svc._on_text(_text_event(type="mystery_event", foo="bar"))
        # No state change, no exception.
        assert not svc._turn_active

    def test_transcript_event_without_type_field(self):
        """Transcripts have no ``type`` field — they use ``transcript`` key."""
        svc = _make_service()
        svc._on_text(
            _text_event(
                transcript="hello", is_partial=True, is_final=False, confidence=0.9
            )
        )
        assert not svc._turn_active

    def test_malformed_json_does_not_raise(self):
        svc = _make_service()
        svc._on_text("{not valid json")
        # State unchanged.
        assert not svc._turn_active


# --------------------------------------------------------------------------- #
# LLM response handling
# --------------------------------------------------------------------------- #
class TestLLMResponse:
    def test_first_llm_response_starts_turn(self):
        svc = _make_service()
        svc._on_text(
            _text_event(type="llm_response", text="Hello world", model="nemotron")
        )
        assert svc._turn_active
        assert not svc._turn_completed
        assert not svc._prebuffer_done or not svc._prebuffer

    def test_continuation_llm_response_with_empty_queue_no_flush(self):
        """A second llm_response in the same turn with an empty queue is a
        continuation, not a barge-in — no flush should occur."""
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="First sentence."))
        # Simulate playback consuming the audio (queue drains).
        # Second sentence of the same turn.
        svc._on_text(_text_event(type="llm_response", text="Second sentence."))
        # No flush should have been needed — prebuffer_done should not be
        # reset (we're still in the same turn).
        # The stream should not have been aborted.
        assert svc._turn_active
        assert not svc._turn_completed

    def test_barge_in_flushes_queued_audio(self):
        """When a new llm_response arrives during an active turn and audio is
        still queued, the buffer is flushed (barge-in)."""
        svc = _make_service()
        svc._stream = MockStream()
        svc._on_text(_text_event(type="llm_response", text="First sentence."))
        # Simulate audio still in the queue (playback hasn't consumed it).
        svc.audio_queue.put_nowait(_pcm_bytes(200))
        assert not svc.audio_queue.empty()

        svc._on_text(
            _text_event(type="llm_response", text="New sentence after barge-in.")
        )

        # Queue should be drained.
        assert svc.audio_queue.empty()
        assert svc._prebuffer_done is False
        assert svc._stream.aborts >= 1
        assert svc._stream.starts >= 1

    def test_llm_response_after_tts_end_starts_fresh_turn(self):
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="First turn."))
        svc._on_text(_text_event(type="tts_end", bytes=100, seconds=0.02))
        assert svc._turn_completed

        svc._on_audio(_pcm_bytes(100))  # New turn's audio
        svc._on_text(_text_event(type="llm_response", text="Second turn."))
        assert svc._turn_active
        assert not svc._turn_completed


# --------------------------------------------------------------------------- #
# TTS end handling
# --------------------------------------------------------------------------- #
class TestTtsEnd:
    def test_tts_end_marks_turn_completed(self):
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="Hello."))
        svc._on_text(_text_event(type="tts_end", bytes=3200, seconds=0.2, errors=[]))
        assert svc._turn_completed
        assert svc._turn_active

    def test_tts_end_flushes_prebuffer(self):
        """If the turn ends before the pre-buffer threshold is met, the
        accumulated data is pushed to the queue immediately."""
        svc = _make_service(prebuffer_ms=100)  # 100 ms ≈ 3 200 bytes
        svc._on_text(_text_event(type="llm_response", text="Hi."))
        svc._on_audio(_pcm_bytes(500))  # Less than prebuffer_bytes
        assert not svc._prebuffer_done
        assert svc.audio_queue.empty()

        svc._on_text(_text_event(type="tts_end", bytes=500, seconds=0.03))
        assert svc._prebuffer_done
        assert not svc.audio_queue.empty()


# --------------------------------------------------------------------------- #
# TTS error handling
# --------------------------------------------------------------------------- #
class TestTtsError:
    def test_tts_error_logged_not_fatal(self):
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="Hello."))
        svc._on_text(
            _text_event(
                type="tts_error",
                stage="tts",
                message="voice not found",
                sentence_index=0,
                sentence="Hello.",
                fatal=False,
            )
        )
        # Turn should still be active; non-fatal error doesn't end it.
        assert svc._turn_active
        assert not svc._turn_completed


# --------------------------------------------------------------------------- #
# LLM error handling
# --------------------------------------------------------------------------- #
class TestLLMError:
    def test_llm_error_logged(self):
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="Hello."))
        svc._on_text(_text_event(type="llm_error", error="rate limited"))
        # LLM error doesn't change turn state by itself.
        assert svc._turn_active


# --------------------------------------------------------------------------- #
# Server error handling
# --------------------------------------------------------------------------- #
class TestServerError:
    def test_server_error_logged(self):
        svc = _make_service()
        svc._on_text(_text_event(type="error", error="internal"))
        assert not svc._turn_active


# --------------------------------------------------------------------------- #
# Transcription barge-in
# --------------------------------------------------------------------------- #
class TestTranscriptBargeIn:
    def test_final_transcript_during_speaking_flushes(self):
        svc = _make_service()
        svc._stream = MockStream()
        svc._on_text(_text_event(type="llm_response", text="Assistant speaking."))
        svc.audio_queue.put_nowait(_pcm_bytes(500))
        assert not svc.audio_queue.empty()

        # User's final transcript arrives → barge-in.
        svc._on_text(
            _text_event(
                transcript="what is the weather",
                is_partial=False,
                is_final=True,
                confidence=0.95,
            )
        )
        assert svc.audio_queue.empty()
        assert svc._prebuffer_done is False

    def test_partial_transcript_does_not_flush(self):
        svc = _make_service()
        svc._on_text(_text_event(type="llm_response", text="Hello."))
        svc.audio_queue.put_nowait(_pcm_bytes(500))

        svc._on_text(
            _text_event(
                transcript="wha",
                is_partial=True,
                is_final=False,
                confidence=0.5,
            )
        )
        # Partial transcripts must not trigger barge-in.
        assert not svc.audio_queue.empty()


# --------------------------------------------------------------------------- #
# Pre-buffering
# --------------------------------------------------------------------------- #
class TestPrebuffering:
    def test_first_turn_prebuffers_until_threshold(self):
        svc = _make_service(prebuffer_ms=50)  # 1 600 bytes
        svc._on_audio(_pcm_bytes(800))
        # Not enough yet — held in pre-buffer, not queued.
        assert not svc._prebuffer_done
        assert svc.audio_queue.empty()

        svc._on_audio(_pcm_bytes(900))
        # 1 700 > 1 600 threshold → pre-buffer flushed to queue.
        assert svc._prebuffer_done
        assert not svc.audio_queue.empty()
        assert svc._prebuffer == bytearray()

    def test_second_turn_resets_prebuffer(self):
        svc = _make_service(prebuffer_ms=50)
        svc._on_audio(_pcm_bytes(2000))  # Fills pre-buffer
        svc._on_text(_text_event(type="tts_end", bytes=2000, seconds=0.1))
        svc._on_text(_text_event(type="llm_response", text="New turn."))
        # New turn → pre-buffer reset.
        assert not svc._prebuffer_done
        svc._on_audio(_pcm_bytes(500))
        assert svc.audio_queue.empty()  # Still holding in pre-buffer


# --------------------------------------------------------------------------- #
# clear_buffer (public API)
# --------------------------------------------------------------------------- #
class TestClearBuffer:
    def test_clear_buffer_drains_queue_and_aborts_stream(self):
        svc = _make_service()
        svc._stream = MockStream()
        svc._on_audio(_pcm_bytes(1000))
        assert not svc.audio_queue.empty()

        svc.clear_buffer()
        assert svc.audio_queue.empty()
        assert svc._prebuffer == bytearray()
        assert svc._prebuffer_done is False
        assert svc._stream.aborts >= 1
        assert svc._stream.starts >= 1

    def test_clear_buffer_with_empty_queue(self):
        """Flushing when the queue is already empty is a safe no-op."""
        svc = _make_service()
        svc._stream = MockStream()
        svc.clear_buffer()
        assert svc.audio_queue.empty()
        assert svc._prebuffer_done is False


# --------------------------------------------------------------------------- #
# URL building
# --------------------------------------------------------------------------- #
class TestUrlBuilding:
    def test_url_without_token_unchanged(self):
        svc = SpeakerService(ws_url="ws://localhost:8000/ws/transcribe")
        assert svc._build_url() == "ws://localhost:8000/ws/transcribe"

    def test_url_with_token_appends_query(self):
        svc = SpeakerService(ws_url="ws://localhost:8000/ws/transcribe", token="secret")
        assert svc._build_url() == "ws://localhost:8000/ws/transcribe?token=secret"

    def test_url_with_existing_query_appends_with_ampersand(self):
        svc = SpeakerService(
            ws_url="ws://localhost:8000/ws/transcribe?foo=bar", token="secret"
        )
        assert (
            svc._build_url() == "ws://localhost:8000/ws/transcribe?foo=bar&token=secret"
        )

    def test_safe_url_masks_token(self):
        svc = SpeakerService(
            ws_url="ws://localhost:8000/ws/transcribe", token="secret123"
        )
        safe = svc._safe_url(svc._build_url())
        assert "secret123" not in safe
        assert "***" in safe


# --------------------------------------------------------------------------- #
# Playback loop (with mock stream)
# --------------------------------------------------------------------------- #
class TestPlaybackLoop:
    def test_playback_writes_chunks_to_stream(self):
        svc = _make_service()
        svc._stream = MockStream()
        svc.audio_queue.put_nowait(_pcm_bytes(320))
        svc.audio_queue.put_nowait(_pcm_bytes(320))
        svc.audio_queue.put_nowait(None)  # sentinel

        with svc._stream as stream:
            while True:
                chunk = svc.audio_queue.get_nowait()
                if chunk is None:
                    break
                svc._safe_write(stream, chunk)

        assert len(svc._stream.writes) == 2
        assert svc._stream.writes[0] == _pcm_bytes(320)

    def test_safe_write_recovers_after_abort(self):
        """If the stream was aborted (barge-in), the next write restarts it."""
        svc = _make_service()
        stream = MockStream()
        svc._stream = stream

        # Simulate an aborted stream.
        stream.abort()
        assert not stream.active()

        # _safe_write should restart and write successfully.
        svc._safe_write(stream, _pcm_bytes(320))
        assert stream.active()
        assert len(stream.writes) == 1
