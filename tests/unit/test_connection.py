"""Unit tests for the Connection class (WebSocket state machine)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from app.connection import Connection

# Upper bound for a shutdown that must not block. The production drain timeout
# is 5 s, so anything near this means we are waiting on the queue backlog.
_SHUTDOWN_TEST_TIMEOUT = 8.0


class MockWebSocket:
    """Mock WebSocket for testing Connection logic."""

    def __init__(self) -> None:
        self.received_frames: list[bytes | dict] = []
        self.errors: list[Exception] = []
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self._send_bytes_task = asyncio.Event()

    async def accept(self) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        self.received_frames.append(data)

    async def send_json(self, data: dict) -> None:
        self.received_frames.append(json.dumps(data).encode())

    async def receive_bytes(self) -> bytes:
        raise RuntimeError("WebSocket not ready for receive_bytes in this test")

    async def receive_text(self) -> str:
        return ""

    async def receive_json(self) -> dict:
        return {}

    def iter_text(self):
        return iter([])

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def wait_closed(self) -> None:
        pass


@pytest.mark.asyncio
async def test_connection_initialization() -> None:
    """Connection should initialize with correct defaults."""
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_stt.transcribe_async = AsyncMock(return_value=("", 0.9))
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    assert conn.language == "en"
    assert conn.is_muted is False
    assert conn.tts_task is None
    assert conn.llm_task is None


@pytest.mark.asyncio
async def test_connection_muted_state() -> None:
    """Connection should track muted state correctly."""
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    assert conn.is_muted is False

    # Mute arrives as a transcript that the intent router recognizes; the
    # server then flips the gating state itself.
    from app.intents import route_command

    mute = route_command("mute")
    assert mute is not None
    await conn._dispatch_command(mute)
    assert conn.is_muted is True

    # "unmute" via the client-side keyword spotter control frame.
    await conn._on_text(json.dumps({"type": "control", "action": "unmute_request"}))
    assert conn.is_muted is False


@pytest.mark.asyncio
async def test_connection_transcription_queue_bounded() -> None:
    """Transcription queue should have a maxsize from settings."""
    from app.config import settings
    from app.connection import Connection

    # Verify settings has the new setting
    assert hasattr(settings, "transcription_queue_maxsize")
    assert settings.transcription_queue_maxsize == 100

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    # Queue should be bounded
    assert conn.queue.maxsize == settings.transcription_queue_maxsize


@pytest.mark.asyncio
async def test_connection_history_capped() -> None:
    """LLM history should be capped at max_llm_history."""
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    # Add more messages than the cap
    from app.config import settings

    for i in range(settings.max_llm_history + 10):
        conn.history.append({"role": "user", "content": f"message {i}"})

    # History should be trimmed to max_llm_history entries
    # (first message is system prompt, so max_llm_history total)
    assert len(conn.history) <= settings.max_llm_history


@pytest.mark.asyncio
async def test_connection_partial_frames_dropped() -> None:
    """Connection should drop partial audio frames that are too small."""
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    # A tiny frame of near-silence is buffered and fed to the VAD, but it must
    # never reach the transcription backend on its own.
    small_frame = b"\x00" * 100  # way below max_frame_size
    await conn._on_audio(small_frame)

    # STT should not have been called for small frames
    mock_stt.transcribe_async.assert_not_called()


@pytest.mark.asyncio
async def test_connection_handles_barge_in() -> None:
    """Connection should cancel ongoing LLM/TTS on barge-in (stop command)."""
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    # Start a fake TTS task
    conn.tts_task = asyncio.create_task(asyncio.sleep(10))
    assert conn.tts_task is not None
    assert not conn.tts_task.done()

    # The assistant-side-effect "stop" command must cancel it (barge-in).
    from app.intents import route_command

    stop = route_command("stop")
    assert stop is not None
    await conn._dispatch_command(stop)

    # TTS task should be cancelled
    assert conn.tts_task is None or conn.tts_task.done()


@pytest.mark.asyncio
async def test_connection_queue_backpressure() -> None:
    """When queue is full, new segments should be dropped or handled gracefully."""
    from app.config import settings
    from app.connection import Connection

    mock_ws = MockWebSocket()
    mock_stt = AsyncMock()
    mock_vad = MagicMock()

    conn = Connection(
        websocket=mock_ws,
        stt=mock_stt,
        detector=mock_vad,
        language="en",
    )

    # Fill the queue
    for i in range(settings.transcription_queue_maxsize):
        conn.queue.put_nowait(True)  # Fill with True values

    # Verify queue is full
    assert conn.queue.full()

    # Next put_nowait should raise or be handled
    with pytest.raises(asyncio.QueueFull):
        conn.queue.put_nowait(True)


# --------------------------------------------------------------------------- #
# Audio buffer, shutdown, and frame-size guard
# --------------------------------------------------------------------------- #


def _make_conn() -> Connection:
    from app.connection import Connection

    return Connection(
        websocket=MockWebSocket(),
        stt=AsyncMock(),
        detector=MagicMock(),
        language="en",
    )


def test_audio_buffer_appends_in_order():
    """The accumulator must preserve sample order and report its length."""
    import numpy as np

    conn = _make_conn()
    first = np.arange(10, dtype=np.float32) / 100.0
    second = np.arange(10, 20, dtype=np.float32) / 100.0
    conn._append_audio(first)
    conn._append_audio(second)

    audio = conn._get_audio()
    assert audio.size == 20
    assert conn._audio_length == 20
    assert np.allclose(audio[:10], first)
    assert np.allclose(audio[10:], second)


def test_audio_buffer_grows_beyond_initial_capacity():
    """Appending past the initial capacity must grow without data loss."""
    import numpy as np

    conn = _make_conn()
    initial = conn._buffer.size
    total = initial + 5000
    block = np.arange(total, dtype=np.float32)

    conn._append_audio(block)

    assert conn._buffer.size >= total
    assert conn._audio_length == total
    assert np.array_equal(conn._get_audio(), block)


def test_audio_buffer_reset_keeps_capacity_and_clears_length():
    """Resetting an utterance reuses the buffer instead of reallocating."""
    import numpy as np

    conn = _make_conn()
    conn._append_audio(np.ones(1000, dtype=np.float32))
    grown = conn._buffer

    conn._reset_segment()

    assert conn._get_audio().size == 0
    assert conn._audio_length == 0
    assert conn.vad_pointer == 0
    assert conn.seg_start is None
    # The same array object is reused, so steady-state streaming does not churn.
    assert conn._buffer is grown


def test_audio_buffer_append_is_linear_not_quadratic():
    """Regression: appending must not re-copy the whole buffer every frame.

    The previous list-of-chunks design re-concatenated the entire utterance on
    every incoming frame, which is O(n^2): ~178 ms of wasted CPU per 20 s
    utterance. Appending into a preallocated buffer is O(new samples).
    """
    import time

    import numpy as np

    conn = _make_conn()
    frame = np.zeros(320, dtype=np.float32)  # 20 ms at 16 kHz
    frames = 1000  # 20 s of audio

    start = time.perf_counter()
    for _ in range(frames):
        conn._append_audio(frame)
        # Reading the audio (as the VAD loop does) must stay O(1).
        _ = conn._get_audio().size
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert conn._audio_length == frames * 320
    # Generous ceiling: the O(n^2) version needed ~180 ms for this workload.
    assert elapsed_ms < 50, f"buffer appends took {elapsed_ms:.1f}ms (too slow)"


@pytest.mark.asyncio
async def test_oversized_frame_closes_socket_with_1009():
    """An oversized frame must close the socket with 1009, not hang."""
    from fastapi import WebSocketException

    from app.config import settings

    conn = _make_conn()
    conn.ptt_active = True  # Enable PTT so frame size check runs
    monkey_max = settings.max_frame_size
    try:
        settings.max_frame_size = 16
        with pytest.raises(WebSocketException) as exc_info:
            await conn._on_audio(b"\x00" * 64)
        assert exc_info.value.code == 1009
    finally:
        settings.max_frame_size = monkey_max


@pytest.mark.asyncio
async def test_shutdown_does_not_deadlock_on_full_queue():
    """``run()`` must terminate even when the transcription queue is full.

    Regression: the shutdown path did ``await queue.put(None)``, which blocks
    while the queue is full. If the consumer is stuck on a slow transcription
    (the normal case during a long utterance), every connection shutdown blocked
    behind it and the socket was never released. The shutdown path now never
    awaits the queue and bounds the drain with a timeout.
    """
    import asyncio

    import numpy as np

    from app.config import settings
    from app.connection import _PendingSegment

    conn = _make_conn()

    # A transcription backend that never returns, standing in for a slow local
    # Whisper decode: the consumer will block on the very first item.
    async def _hanging_transcribe(audio, language):
        await asyncio.sleep(30)
        return "", 0.0

    conn.stt.transcribe_async = _hanging_transcribe

    # Fill the queue with real segments so the consumer actually blocks.
    segment = _PendingSegment(np.zeros(1600, dtype=np.float32), False, True, 0)
    for _ in range(settings.transcription_queue_maxsize):
        conn.queue.put_nowait(segment)
    assert conn.queue.full()

    # Simulate a client that says "stop": the receive loop should exit and run()
    # should tear down without waiting for the backlog.
    conn.stopping = True

    try:
        # If shutdown blocked on the queue, this would never return.
        await asyncio.wait_for(conn.run(), timeout=_SHUTDOWN_TEST_TIMEOUT)
    except asyncio.TimeoutError:
        pytest.fail("run() blocked on a full transcription queue during shutdown")
    finally:
        if conn.consumer is not None and not conn.consumer.done():
            conn.consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conn.consumer


@pytest.mark.asyncio
async def test_speaker_buffer_only_collects_frames_after_vad_marks_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from app.config import settings

    conn = _make_conn()
    conn.speaker_service = object()
    monkeypatch.setattr(settings, "speaker_enabled", True)

    vad_states = iter((None, 0))

    async def feed_vad() -> None:
        conn.seg_start = next(vad_states)

    monkeypatch.setattr(conn, "_feed_vad", feed_vad)
    frame = np.full(512, 8192, dtype="<i2").tobytes()

    await conn._on_audio(frame)
    assert not conn.speaker_audio_buffer

    await conn._on_audio(frame)
    assert len(conn.speaker_audio_buffer) == 1
    assert np.allclose(conn.speaker_audio_buffer[0], 0.25)


def test_speaker_audio_buffer_tracks_a_bounded_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from app.config import settings

    conn = _make_conn()
    conn.speaker_service = object()
    monkeypatch.setattr(settings, "speaker_enabled", True)
    monkeypatch.setattr(settings, "speaker_max_identify_sec", 0.01)

    for _ in range(5):
        conn._buffer_audio_for_speaker_id(np.ones(100, dtype=np.float32))

    assert conn._speaker_buffer_samples <= 160
    assert conn._speaker_buffer_samples == sum(
        chunk.size for chunk in conn.speaker_audio_buffer
    )
    assert len(conn.speaker_audio_buffer) == 1


@pytest.mark.asyncio
async def test_speaker_identification_clears_buffer_and_sample_count() -> None:
    import numpy as np

    from app.config import settings

    conn = _make_conn()
    conn.speaker_audio_buffer.append(
        np.ones(settings.sample_rate, dtype=np.float32)
    )
    conn._speaker_buffer_samples = settings.sample_rate

    class UnknownSpeaker:
        async def identify(self, audio):
            return SimpleNamespace(is_known=False, similarity=0.0)

    conn.speaker_service = UnknownSpeaker()
    await conn._identify_speaker()

    assert not conn.speaker_audio_buffer
    assert conn._speaker_buffer_samples == 0


@pytest.mark.asyncio
async def test_final_utterance_does_not_wait_for_speaker_identification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from app.config import settings

    conn = _make_conn()
    monkeypatch.setattr(settings, "speaker_enabled", True)
    monkeypatch.setattr(settings, "echo_mode", "off")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")
    conn.speaker_audio_buffer.append(np.ones(settings.sample_rate, dtype=np.float32))
    conn._speaker_buffer_samples = settings.sample_rate
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowSpeaker:
        async def identify(self, audio):
            started.set()
            await finish.wait()
            return SimpleNamespace(
                is_known=False,
                speaker_id=None,
                name=None,
                similarity=0.0,
            )

    conn.speaker_service = SlowSpeaker()

    try:
        await asyncio.wait_for(
            conn._handle_final_utterance("tell me something interesting"),
            timeout=0.05,
        )
        await asyncio.wait_for(started.wait(), timeout=0.05)
        assert conn.speaker_identification_pending
    finally:
        finish.set()
        await asyncio.sleep(0)


def _known_speaker(speaker_id: str = "alice", name: str = "Alice", similarity: float = 0.91):
    """An identification result for an enrolled speaker."""
    return SimpleNamespace(
        is_known=True,
        speaker_id=speaker_id,
        name=name,
        similarity=similarity,
    )


def _unknown_speaker(similarity: float = 0.31):
    """An identification result below the similarity threshold."""
    return SimpleNamespace(
        is_known=False,
        speaker_id=None,
        name=None,
        similarity=similarity,
    )


class _StubSpeaker:
    """Speaker service stub that returns queued identification results."""

    def __init__(self, *results) -> None:
        self._results = list(results)
        self.calls = 0

    async def identify(self, audio):
        self.calls += 1
        index = min(self.calls - 1, len(self._results) - 1)
        return self._results[index]


def _seed_speaker_buffer(conn: Connection) -> None:
    """Put one second of speech in the speaker buffer so ID runs."""
    import numpy as np

    from app.config import settings

    conn.speaker_audio_buffer.append(np.ones(settings.sample_rate, dtype=np.float32))
    conn._speaker_buffer_samples = settings.sample_rate


def _speaker_events(conn: Connection) -> list[dict]:
    """Decode the JSON events the connection has sent to the client."""
    events = []
    for frame in conn.ws.received_frames:
        if isinstance(frame, (bytes, bytearray)):
            payload = json.loads(frame)
            if isinstance(payload, dict):
                events.append(payload)
    return events


@pytest.mark.asyncio
async def test_a_single_unknown_turn_keeps_the_current_speaker_context() -> None:
    """One unknown turn is not enough evidence to drop the speaker context."""
    conn = _make_conn()
    conn.speaker_service = _StubSpeaker(_known_speaker(), _unknown_speaker())

    for _ in range(2):
        _seed_speaker_buffer(conn)
        await conn._identify_speaker()

    assert conn.current_speaker_id == "alice"
    assert conn.current_speaker_name == "Alice"


@pytest.mark.asyncio
async def test_two_consecutive_unknown_turns_clear_the_stale_speaker_context() -> None:
    """Repeated unknown speakers must not be greeted as the last known one."""
    conn = _make_conn()
    conn.speaker_service = _StubSpeaker(
        _known_speaker(), _unknown_speaker(), _unknown_speaker()
    )

    for _ in range(2):
        _seed_speaker_buffer(conn)
        await conn._identify_speaker()
    assert conn.current_speaker_name == "Alice"

    _seed_speaker_buffer(conn)
    await conn._identify_speaker()

    assert conn.current_speaker_id is None
    assert conn.current_speaker_name is None
    assert _speaker_events(conn)[-1] == {
        "type": "speaker_change",
        "speaker_id": None,
        "name": None,
        "similarity": 0.31,
    }


@pytest.mark.asyncio
async def test_a_known_speaker_resets_the_unknown_streak() -> None:
    """Re-identifying a known speaker restarts the unknown-speaker count."""
    conn = _make_conn()
    conn.speaker_service = _StubSpeaker(
        _known_speaker(), _unknown_speaker(), _known_speaker(), _unknown_speaker()
    )

    for _ in range(4):
        _seed_speaker_buffer(conn)
        await conn._identify_speaker()

    assert conn.current_speaker_name == "Alice"
