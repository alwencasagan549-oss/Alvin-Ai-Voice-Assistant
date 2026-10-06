"""Connection-level wiring for deterministic intent routing.

Drives ``Connection._handle_final_utterance`` directly with a fake socket and
stubbed LLM/TTS, so no Whisper model, edge-tts, or network is needed. The
guards here:

* a recognized device command emits a ``command`` event and never reaches the
  LLM;
* an open-ended utterance is routed to the LLM exactly as before;
* ``stop`` cancels any in-progress LLM/TTS turn;
* low-confidence transcripts are not trusted enough to execute a command;
* ``mute the mic`` flips the connection's audio-gating state, signals the
  client with ``disable_stt``/``enable_stt``, and always speaks the
  hardcoded confirmation;
* while muted, final segments run the local-only keyword pass (never the
  cloud pipeline) and only mic-control matches are dispatched.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import connection as connection_mod
from app.config import settings
from app.connection import Connection, _PendingSegment
from app.intents import MUTE_RESPONSE_TEXT, UNMUTE_RESPONSE_TEXT
from app.llm import llm_service


class FakeWebSocket:
    """Records every outbound message instead of sending it anywhere."""

    def __init__(self) -> None:
        self.json: list[dict] = []
        self.binary: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.binary.append(payload)


class FakeDetector:
    """The routing path never feeds the VAD; only storage is exercised."""

    def reset(self) -> None:
        pass

    def process_frame(self, frame):
        return None


def _connection() -> tuple[FakeWebSocket, Connection]:
    ws = FakeWebSocket()
    conn = Connection(
        websocket=ws,
        stt=SimpleNamespace(),
        detector=FakeDetector(),
        language="en",
    )
    return ws, conn


def _disable_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "tts_auto_speak", False)


def _fake_tts() -> object:
    async def fake_tts(text, voice=None, prefetch=None):
        yield b"\x00\x00"

    return fake_tts


def test_command_emits_event_and_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    ws, conn = _connection()

    async def _boom(*_args, **_kwargs):
        raise AssertionError("LLM path was taken for a deterministic command")

    monkeypatch.setattr(conn, "_spawn_llm_turn", _boom)

    asyncio.run(conn._handle_final_utterance("turn off the lights", 0.95))

    commands = [event for event in ws.json if event.get("type") == "command"]
    assert len(commands) == 1, f"events={ws.json}"
    assert commands[0]["intent"] == "lights_off"
    assert commands[0]["slots"] == {}
    assert commands[0]["normalized"] == "turn off the lights"
    assert conn.llm_task is None
    assert conn.tts_task is None


def test_set_volume_carries_the_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    ws, conn = _connection()

    asyncio.run(conn._handle_final_utterance("set the volume to forty", 0.95))

    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "set_volume"
    assert commands[0]["slots"] == {"volume": 40}


def test_stop_cancels_in_progress_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    ws, conn = _connection()

    async def scenario() -> asyncio.Task:
        # The Connection is muted (no llm) in this test, so _spawn_llm_turn is never called.
        # Use a plain sync sleep for the task so we can cancel it.
        task = asyncio.create_task(asyncio.sleep(30))
        conn.llm_task = task
        await conn._handle_final_utterance("stop", 0.95)
        await asyncio.sleep(0.05)
        return task

    task = asyncio.run(scenario())

    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "stop"
    assert task.cancelled(), "an in-progress turn must be aborted by 'stop'"


def test_open_ended_utterance_reaches_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    seen: list[object] = []

    async def fake_stream(
        messages, *, max_tokens=None, temperature=None, fallback_text=None, tools=None, tool_choice=None
    ):
        seen.append(messages)
        yield ("Sure. I will check the weather for you.", "fake-model")

    # Temporarily disable the echo guard because the fake TTS doesn't invoke it.
    monkeypatch.setattr(settings, "echo_mode", "off")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(llm_service, "stream_response", fake_stream)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())

    ws, conn = _connection()

    async def scenario() -> None:
        await conn._handle_final_utterance("what is the weather like", 0.95)
        if conn.llm_task is not None:
            await conn.llm_task

    asyncio.run(scenario())

    assert seen, "the LLM must handle open-ended utterances"
    assert not [event for event in ws.json if event.get("type") == "command"]
    responses = [event for event in ws.json if event.get("type") == "llm_response"]
    assert responses, f"events={ws.json}"


def test_low_confidence_transcript_falls_back_to_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "intent_min_confidence", 0.5)
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    seen: list[object] = []

    async def fake_stream(
        messages, *, max_tokens=None, temperature=None, fallback_text=None, tools=None, tool_choice=None
    ):
        seen.append(messages)
        yield ("Sure.", "fake-model")

    # Temporarily disable the echo guard because the fake TTS doesn't invoke it.
    monkeypatch.setattr(settings, "echo_mode", "off")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(llm_service, "stream_response", fake_stream)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())

    ws, conn = _connection()

    async def scenario() -> None:
        await conn._handle_final_utterance("stop", 0.2)
        if conn.llm_task is not None:
            await conn.llm_task

    asyncio.run(scenario())

    assert seen, "a low-confidence transcript must not execute a device command"
    assert not [event for event in ws.json if event.get("type") == "command"]


def test_confirmation_is_spoken_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(settings, "intent_speak_confirmation", True)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())
    ws, conn = _connection()

    async def scenario() -> None:
        await conn._handle_final_utterance("mute the speaker", 0.95)
        if conn.tts_task is not None:
            await conn.tts_task

    asyncio.run(scenario())

    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "mute"
    assert ws.binary, "the confirmation must be spoken as PCM frames"
    assert [event for event in ws.json if event.get("type") == "tts_end"]


# --------------------------------------------------------------------------- #
# Mic control / audio gating
# --------------------------------------------------------------------------- #
def test_mute_mic_flips_state_signals_client_and_spoke_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(settings, "intent_speak_confirmation", False)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())
    ws, conn = _connection()

    async def _boom(*_args, **_kwargs):
        raise AssertionError("LLM path was taken for a mic-control command")

    monkeypatch.setattr(conn, "_spawn_llm_turn", _boom)

    async def scenario() -> None:
        await conn._handle_final_utterance("mute the mic", 0.95)
        if conn.tts_task is not None:
            await conn.tts_task

    asyncio.run(scenario())

    assert conn.is_muted is True
    commands = [event for event in ws.json if event.get("type") == "command"]
    assert len(commands) == 1, f"events={ws.json}"
    assert commands[0]["intent"] == "mute_mic"
    assert commands[0]["action"] == "disable_stt"
    # Always-confirmed: the hardcoded acknowledgement is spoken even with
    # INTENT_SPEAK_CONFIRMATION off, and shipped to the client in the event.
    assert commands[0]["speak"] == MUTE_RESPONSE_TEXT
    assert ws.binary, "the hardcoded confirmation must be synthesized"
    assert [event for event in ws.json if event.get("type") == "tts_end"]


def test_unmute_mic_restores_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())
    ws, conn = _connection()
    conn.is_muted = True

    async def scenario() -> None:
        await conn._handle_final_utterance("unmute", 0.95)
        if conn.tts_task is not None:
            await conn.tts_task

    asyncio.run(scenario())

    assert conn.is_muted is False
    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "unmute_mic"
    assert commands[0]["action"] == "enable_stt"
    assert commands[0]["speak"] == UNMUTE_RESPONSE_TEXT


def test_unmute_request_control_frame_resumes_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Option A: the client's local keyword spotter says the user unmuted."""
    _disable_llm(monkeypatch)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())
    ws, conn = _connection()
    conn.is_muted = True

    async def scenario() -> None:
        await conn._on_text(json.dumps({"type": "control", "action": "unmute_request"}))
        if conn.tts_task is not None:
            await conn.tts_task

    asyncio.run(scenario())

    assert conn.is_muted is False
    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "unmute_mic"
    assert commands[0]["action"] == "enable_stt"


def test_unmute_request_control_frame_is_ignored_when_not_muted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_llm(monkeypatch)
    ws, conn = _connection()

    asyncio.run(
        conn._on_text(json.dumps({"type": "control", "action": "unmute_request"}))
    )

    assert conn.is_muted is False
    assert not [event for event in ws.json if event.get("type") == "command"]


def _fake_muted_stt(text: str) -> SimpleNamespace:
    """STT stub whose local-only pass answers ``text``; the full pipeline is
    armed to fail the moment the muted path touches it."""

    async def _local(audio, language):
        return text, 0.9

    async def _boom(*_args, **_kwargs):
        raise AssertionError("the muted path must never run the full STT pipeline")

    return SimpleNamespace(transcribe_local_async=_local, transcribe_async=_boom)


def _muted_segment() -> _PendingSegment:
    return _PendingSegment(np.zeros(16000, dtype=np.float32), False, True, 7)


def test_muted_final_segment_runs_local_only_keyword_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setattr(connection_mod, "synthesize_stream", _fake_tts())
    ws, conn = _connection()
    conn.stt = _fake_muted_stt("unmute")
    conn.is_muted = True

    async def scenario() -> None:
        await conn._handle_muted_segment(_muted_segment())
        if conn.tts_task is not None:
            await conn.tts_task

    asyncio.run(scenario())

    assert conn.is_muted is False
    commands = [event for event in ws.json if event.get("type") == "command"]
    assert commands and commands[0]["intent"] == "unmute_mic"
    assert commands[0]["action"] == "enable_stt"


def test_muted_final_segment_drops_non_command_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anything that is not a mic-control command is dropped silently."""
    _disable_llm(monkeypatch)
    ws, conn = _connection()
    conn.stt = _fake_muted_stt("what time is it")
    conn.is_muted = True

    asyncio.run(conn._handle_muted_segment(_muted_segment()))

    assert conn.is_muted is True
    assert not [event for event in ws.json if event.get("type") == "command"]


def test_muted_partial_segments_never_reach_the_local_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_llm(monkeypatch)
    ws, conn = _connection()
    conn.stt = _fake_muted_stt("unmute")
    conn.is_muted = True

    segment = _PendingSegment(np.zeros(16000, dtype=np.float32), True, False, 7)
    asyncio.run(conn._handle_muted_segment(segment))

    assert conn.is_muted is True
    assert not [event for event in ws.json if event.get("type") == "command"]


# --------------------------------------------------------------------------- #
# Wake window (connection-level state machine)
# --------------------------------------------------------------------------- #
def _wake_router(window_sec: float, required: bool = True):
    """A minimal stand-in for Connection._handle_final_utterance's wake logic."""
    import time

    from app.intents import route_command

    state = {"until": 0.0}

    def handle(text: str):
        match = route_command(text, 0.9)
        source = "direct"
        if match is None and state["until"] and time.monotonic() < state["until"]:
            match = route_command(text, 0.9, wake_word_ok=True)
            source = "window"
        if match is not None and match.wake_word and window_sec > 0:
            state["until"] = time.monotonic() + window_sec
        return (match.intent, source) if match else None

    return handle, state


def test_wake_window_allows_a_bare_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "wake_word_required", True)
    monkeypatch.setattr(settings, "wake_word_window_sec", 6.0)
    handle, _state = _wake_router(6.0)

    assert handle("alvin turn on the lights") == ("lights_on", "direct")
    # Armed by the previous turn: no wake word needed now.
    assert handle("turn off the lights") == ("lights_off", "window")
    assert handle("set the volume to 30") == ("set_volume", "window")
    # A fresh wake word works too.
    assert handle("alvin stop") == ("stop", "direct")


def test_wake_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "wake_word_required", True)
    monkeypatch.setattr(settings, "wake_word_window_sec", 6.0)
    handle, state = _wake_router(6.0)

    handle("alvin stop")
    state["until"] = time_minus_one()  # window already elapsed
    assert handle("turn on the lights") is None


def test_no_wake_window_means_every_command_needs_the_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "wake_word_required", True)
    monkeypatch.setattr(settings, "wake_word_window_sec", 0.0)
    handle, _state = _wake_router(0.0)

    assert handle("alvin stop") == ("stop", "direct")
    assert handle("turn on the lights") is None


def time_minus_one() -> float:
    import time

    return time.monotonic() - 1.0
