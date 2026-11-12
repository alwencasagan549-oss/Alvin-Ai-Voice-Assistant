"""Reproduce the "freezes on 2nd question" bug on a persistent connection.

The real UI keeps the WebSocket open across turns (only `stop` closes it).
Here we stream audio, wait for the final transcript + llm_response + tts_end,
then immediately ask a second question on the same socket and watch for the
freeze.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_speech.wav"


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype="<i2")
    return arr.astype("<i2"), sr


async def stream_audio(sock, pcm, sr) -> None:
    chunk_size = max(1, int(sr * 0.02))
    sent = 0
    while sent < len(pcm):
        end = min(sent + chunk_size, len(pcm))
        await sock.send(pcm[sent:end].tobytes())
        sent = end
        await asyncio.sleep(0.01)


async def run_turn(sock, pcm, sr, label: str) -> bool:
    """Stream one utterance and wait for the full turn to complete.

    Returns True if the turn completed (tts_end seen), False on freeze/timeout.
    """
    got_final = asyncio.Event()
    got_llm = asyncio.Event()
    got_tts_end = asyncio.Event()
    frozen = asyncio.Event()

    async def recv_loop():
        deadline = time.monotonic() + 90.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"[{label}] *** RECV TIMEOUT (freeze!) ***")
                frozen.set()
                return
            try:
                msg = await asyncio.wait_for(sock.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                print(f"[{label}] *** RECV TIMEOUT (freeze!) ***")
                frozen.set()
                return
            except ConnectionClosed:
                print(f"[{label}] (connection closed)")
                frozen.set()
                return
            ts = time.strftime("%H:%M:%S")
            if isinstance(msg, bytes):
                # binary PCM frame
                if len(msg) > 0:
                    print(f"[{label}] {ts} binary frame ({len(msg)} bytes)")
                continue
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                print(f"[{label}] {ts} non-json: {msg[:80]!r}")
                continue
            if not isinstance(obj, dict):
                continue
            t = obj.get("type", "")
            if obj.get("is_final"):
                got_final.set()
            if t == "llm_response":
                print(f"[{label}] {ts} LLM response: {obj.get('text','')!r}")
                got_llm.set()
            elif t == "llm_error":
                print(f"[{label}] {ts} LLM error: {obj.get('error')}")
                got_llm.set()
            elif t == "tts_end":
                print(f"[{label}] {ts} tts_end bytes={obj.get('bytes')}")
                got_tts_end.set()
                return
            elif t == "transcript":
                kind = "FINAL" if obj.get("is_final") else "partial"
                print(f"[{label}] {ts} {kind} {obj.get('transcript')!r}")
            else:
                print(f"[{label}] {ts} {t} {obj}")
        # unreachable

    recv_task = asyncio.create_task(recv_loop())
    try:
        await stream_audio(sock, pcm, sr)
        # Wait for the turn to finish.
        await asyncio.wait_for(got_tts_end.wait(), timeout=90.0)
        return True
    except asyncio.TimeoutError:
        print(f"[{label}] *** TURN TIMEOUT (freeze!) ***")
        return False
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8001/ws/transcribe")
    parser.add_argument("--turns", type=int, default=2)
    args = parser.parse_args()

    pcm, sr = load_wav(str(FIXTURE))
    print(f"fixture: {len(pcm)} samples ({len(pcm)/sr:.2f}s) @ {sr}Hz")

    async with connect(args.url) as sock:
        await sock.send(json.dumps({"type": "config", "language": "en"}))
        for turn in range(1, args.turns + 1):
            label = f"turn{turn}"
            ok = await run_turn(sock, pcm, sr, label)
            print(f"[{label}] completed={ok}")
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))