"""Minimal smoke check: stream the fixture to an already-running server.

Prints partial/final events until the first final. Manual use only.

Usage:
    python tests/manual/quick_smoke.py [ws://127.0.0.1:8000/ws/transcribe]
"""

import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_speech.wav"
URI = "ws://127.0.0.1:8000/ws/transcribe"


async def smoke(url: str = URI, wav: Path = FIXTURE) -> None:
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
        sr = w.getframerate()

    print(f"Connecting to {url}...")
    async with websockets.connect(url) as sock:
        frame_bytes = int(sr * 0.2) * 2
        for i in range(0, len(raw), frame_bytes):
            await sock.send(raw[i : i + frame_bytes])
            await asyncio.sleep(0.04)
        await sock.send(json.dumps({"type": "stop"}))

        try:
            while True:
                msg = await asyncio.wait_for(sock.recv(), timeout=10)
                event = json.loads(msg)
                if event.get("is_final"):
                    print(f"STT final: {event}")
                    break
                if event.get("is_partial"):
                    print(f"  STT partial: {event.get('transcript')}")
        except Exception as exc:  # noqa: BLE001 - smoke script, report and stop
            print(f"Done: {exc}")


if __name__ == "__main__":
    asyncio.run(smoke(sys.argv[1] if len(sys.argv) > 1 else URI))
