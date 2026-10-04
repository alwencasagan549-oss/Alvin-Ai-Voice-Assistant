"""Manual end-to-end client: stream a WAV file to a running server.

Prints every JSON transcription event. Run it by hand against a live server
(``python run.py``); it is not part of the pytest suite.

Usage:
    python tests/manual/stream_wav.py tests/data/sample_speech.wav
    python tests/manual/stream_wav.py tests/data/sample_speech.wav --realtime
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_speech.wav"


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.astype("<i2"), sr


async def main() -> int:
    parser = argparse.ArgumentParser(description="Alvin STT test client")
    parser.add_argument(
        "wav", nargs="?", default=str(FIXTURE), help="16 kHz mono WAV file to stream"
    )
    parser.add_argument(
        "--url", default="ws://localhost:8000/ws/transcribe", help="WebSocket URL"
    )
    parser.add_argument("--chunk-ms", type=int, default=20, help="chunk size in ms")
    parser.add_argument(
        "--realtime", action="store_true", help="sleep to simulate real-time"
    )
    parser.add_argument("--language", default=None, help="override language e.g. en")
    args = parser.parse_args()

    pcm, sr = load_wav(args.wav)
    chunk_size = max(1, int(sr * args.chunk_ms / 1000))
    print(f"Streaming {args.wav}: {len(pcm)} samples ({len(pcm) / sr:.2f}s) @ {sr}Hz")

    async with connect(args.url) as sock:
        if args.language:
            await sock.send(json.dumps({"type": "config", "language": args.language}))
        sent = 0
        frame = 0
        while sent < len(pcm):
            end = min(sent + chunk_size, len(pcm))
            await sock.send(pcm[sent:end].tobytes())
            sent = end
            frame += 1
            if args.realtime:
                await asyncio.sleep(args.chunk_ms / 1000.0)
        await sock.send(json.dumps({"type": "stop"}))

        while True:
            try:
                msg = await sock.recv()
            except ConnectionClosed:
                print("(connection closed)")
                break
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                print("non-json:", msg)
                continue
            kind = "FINAL" if obj.get("is_final") else "partial"
            print(
                f"[{kind:>7} conf={obj.get('confidence'):.3f}] {obj.get('transcript')!r}"
            )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
