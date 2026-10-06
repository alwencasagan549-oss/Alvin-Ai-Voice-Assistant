"""End-to-end proof that the self-trigger loop is broken.

Real leaked speech is fed back in as microphone input while the assistant is
speaking, exactly as a leaky room or a virtual cable does.

The assertion is on *which* turns the assistant answers, not on how many events
were emitted: a dropped echo still emits an (empty) final event by design, so
counting events would not detect the loop. Instead the transcript is checked --
if the assistant answers its own words, the transcript of the second round is
the reply it just spoke.
"""

from __future__ import annotations

import asyncio
import json
import wave

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# The assistant's reply. The leaked audio is the sample utterance, whose
# transcript is "Hello Alvin. How are you today?" -- so the guard has something
# real to compare against rather than a hand-waved match.
REPLY = "Hello Alvin. How are you today?"


@pytest.fixture(scope="module")
def leaked_speech() -> bytes:
    """The sample utterance, attenuated as it would be after air/speaker loss."""
    with wave.open("tests/data/sample_speech.wav", "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return (pcm * 0.55).astype("<i2").tobytes()


async def _drive(ws, audio: bytes, silence_frames: int = 55) -> dict:
    """Send one utterance + trailing silence; collect what the assistant did.

    An echo is *recognized* (so a transcript event is still emitted -- the client
    sees what the recognizer heard) but must never produce an ``llm_response`` or
    a ``command`` event. Those are what make it "talk to itself".
    """
    transcripts: list[str] = []
    answered: list[dict] = []
    for i in range(0, len(audio), 640):
        await ws.send(audio[i : i + 640])
        await asyncio.sleep(0.004)
    for _ in range(silence_frames):
        await ws.send(b"\x00" * 640)
        await asyncio.sleep(0.01)
    deadline = asyncio.get_running_loop().time() + 8.0
    while asyncio.get_running_loop().time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            break
        if isinstance(msg, bytes):
            continue
        data = json.loads(msg)
        kind = data.get("type")
        if kind in {"llm_response", "command", "tts_start"}:
            answered.append(data)
        elif data.get("is_final") and data.get("transcript"):
            transcripts.append(data["transcript"])
    return {"transcripts": transcripts, "answered": answered}


@pytest.mark.asyncio
async def test_assistant_does_not_answer_its_own_voice(
    server_url, leaked_speech
) -> None:
    """Playback leaked into the mic must not come back as a fresh request."""
    import websockets

    async with websockets.connect(server_url, max_size=8 * 1024 * 1024) as ws:
        # Round 1 -- a genuine user turn. Nothing has been spoken yet, so there
        # is no history and nothing can be an echo.
        first = await _drive(ws, leaked_speech)
        assert first["transcripts"], "the first utterance should reach the assistant"

        # The assistant answers; its voice now leaks back through the mic.
        await ws.send(json.dumps({"type": "speak", "text": REPLY}))
        await asyncio.sleep(2.0)  # synthesis + cooldown

        # Round 2 -- the assistant hears its own reply.
        echoed = await _drive(ws, leaked_speech)

    # The recognizer still hears the leaked audio, which is expected and
    # harmless. What must not happen is the assistant *acting* on it: no LLM
    # reply, no command, no new playback. That is the self-talk loop.
    assert echoed["answered"] == [], (
        f"assistant answered its own playback: {echoed['answered']!r}"
    )


@pytest.mark.asyncio
async def test_speak_is_accepted_while_a_connection_is_open(server_url) -> None:
    """Regression guard: the echo guard must not wedge the speak path."""
    import websockets

    async with websockets.connect(server_url) as ws:
        await ws.send(json.dumps({"type": "speak", "text": "one two three"}))
        saw_end = False
        deadline = asyncio.get_running_loop().time() + 25.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                continue
            if json.loads(msg).get("type") == "tts_end":
                saw_end = True
                break
        assert saw_end, "tts_end was never sent: the echo guard deadlocked speak"
