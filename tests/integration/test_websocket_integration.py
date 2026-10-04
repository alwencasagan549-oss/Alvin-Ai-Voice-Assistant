"""End-to-end WebSocket integration tests against a live uvicorn server.

Streams the speech fixture as raw int16 PCM using the ``websockets`` client and
asserts the documented JSON event schema is emitted. A fast ``tiny.en`` model
is used (see conftest env overrides).
"""

from __future__ import annotations

import asyncio
import json
import urllib.request

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

FRAME_SEC = 0.2


def _stream_events(url: str, raw: bytes, sr: int) -> list[dict]:
    """Stream ``raw`` int16 PCM to the server and collect JSON events."""

    async def run() -> list[dict]:
        events: list[dict] = []
        async with connect(url) as sock:
            frame_bytes = int(sr * FRAME_SEC) * 2
            for i in range(0, len(raw), frame_bytes):
                await sock.send(raw[i : i + frame_bytes])
                await asyncio.sleep(FRAME_SEC / 5)
            await sock.send(json.dumps({"type": "stop"}))
            try:
                while True:
                    msg = await asyncio.wait_for(sock.recv(), timeout=25)
                    if isinstance(msg, bytes):
                        continue
                    try:
                        events.append(json.loads(msg))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
        return events

    return asyncio.run(run())


def test_health_endpoint(server_url: str):
    port = server_url.split(":")[2].split("/")[0]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["ready"] is True
    # conftest pins FALLBACK_MODEL=tiny.en and disables the cloud, so the model
    # serving requests here is the local fallback.
    assert body["provider"] == "local"
    assert body["model"] == "tiny.en"
    assert body["fallback_model"] == "tiny.en"


def test_streaming_transcribes_speech(
    server_url: str, speech_int16_bytes: tuple[bytes, int]
):
    raw, sr = speech_int16_bytes
    events = _stream_events(server_url, raw, sr)

    assert events, "no events received"
    finals = [e for e in events if e.get("is_final")]
    assert finals, f"no final event; events={events}"

    final = finals[0]
    for key in ("transcript", "is_partial", "is_final", "confidence"):
        assert key in final, f"missing key {key} in {final}"
    assert final["is_final"] is True
    assert final["is_partial"] is False
    assert isinstance(final["transcript"], str) and final["transcript"].strip()
    assert 0.0 <= float(final["confidence"]) <= 1.0

    text = final["transcript"].lower()
    assert any(
        token in text for token in ("alvin", "assistant", "speech", "text", "test")
    ), text

    # Interim hypotheses should be emitted during streaming.
    assert any(e.get("is_partial") and not e.get("is_final") for e in events), [
        e.get("transcript") for e in events
    ]


def test_ping_pong(server_url: str):
    async def run() -> dict:
        async with connect(server_url) as sock:
            await sock.send(json.dumps({"type": "ping"}))
            return json.loads(await asyncio.wait_for(sock.recv(), timeout=10))

    msg = asyncio.run(run())
    assert msg == {"type": "pong"}


def test_speak_streams_pcm_then_tts_end(server_url: str):
    """``speak`` returns binary 16 kHz PCM frames followed by ``tts_end``."""

    async def run() -> tuple[bytes, list[dict]]:
        audio = bytearray()
        events: list[dict] = []
        async with connect(server_url) as sock:
            await sock.send(
                json.dumps({"type": "speak", "text": "Sure. Setting your timer now."})
            )
            try:
                while True:
                    msg = await asyncio.wait_for(sock.recv(), timeout=40)
                    if isinstance(msg, bytes):
                        audio.extend(msg)
                        continue
                    event = json.loads(msg)
                    events.append(event)
                    if event.get("type") == "tts_end":
                        break
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
        return bytes(audio), events

    audio, events = asyncio.run(run())

    assert audio, "no binary audio frames received"
    assert len(audio) % 2 == 0, "PCM frames must be whole int16 samples"

    ends = [e for e in events if e.get("type") == "tts_end"]
    assert ends, f"no tts_end event; events={events}"
    end = ends[0]
    assert end["bytes"] == len(audio)
    assert end["seconds"] > 0.2
    assert end["errors"] == []
    assert not [e for e in events if e.get("type") == "tts_error"]


def test_speak_reports_failure_for_invalid_voice(server_url: str):
    """A bad voice yields a ``tts_error`` event instead of killing the socket."""

    async def run() -> list[dict]:
        events: list[dict] = []
        async with connect(server_url) as sock:
            await sock.send(
                json.dumps({"type": "speak", "text": "Hello.", "voice": "not-a-voice"})
            )
            try:
                while True:
                    msg = await asyncio.wait_for(sock.recv(), timeout=40)
                    if isinstance(msg, bytes):
                        continue
                    event = json.loads(msg)
                    events.append(event)
                    if event.get("type") == "tts_end":
                        break
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
        return events

    events = asyncio.run(run())
    errors = [e for e in events if e.get("type") == "tts_error"]
    assert errors, f"expected a tts_error event; events={events}"
    assert errors[0]["stage"] == "tts"
    assert errors[0]["sentence_index"] == 0
    assert any(e.get("type") == "tts_end" for e in events), events


def test_stop_speak_cancels_playback(server_url: str):
    """``stop_speak`` interrupts an in-flight synthesis without closing."""

    async def run() -> dict:
        async with connect(server_url) as sock:
            await sock.send(
                json.dumps(
                    {
                        "type": "speak",
                        "text": "This is a very long sentence that will not finish "
                        "before the cancel arrives, not by a long way indeed.",
                    }
                )
            )
            await asyncio.sleep(0.05)
            await sock.send(json.dumps({"type": "stop_speak"}))
            await sock.send(json.dumps({"type": "ping"}))
            return json.loads(await asyncio.wait_for(sock.recv(), timeout=20))

    msg = asyncio.run(run())
    assert msg == {"type": "pong"}, "socket must stay usable after stop_speak"
