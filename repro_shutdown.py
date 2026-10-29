"""Reproduction script: connect WebSocket, send real speech, capture server logs."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

# Start the backend server, capturing its output
proc = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "app.main:app",
     "--host", "127.0.0.1", "--port", "8000",
     "--ws", "auto", "--log-level", "debug"],
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

server_output = []


def read_server_output():
    """Non-blocking read of server output."""
    lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
            server_output.append(line.rstrip())
    except Exception:
        pass
    return lines


try:
    # Wait for health
    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("ready"):
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        print("Server failed to become ready")
        proc.terminate()
        sys.exit(1)

    print("Server ready. Testing WebSocket with real speech...")

    import websockets

    def load_wav(path: str):
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
        import numpy as np
        arr = np.frombuffer(raw, dtype="<i2")
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return arr.astype("<i2"), sr

    pcm, sr = load_wav(os.path.join(ROOT, "tests", "data", "sample_speech.wav"))
    print(f"Loaded {len(pcm)} samples ({len(pcm) / sr:.2f}s) @ {sr}Hz")

    async def test():
        uri = "ws://127.0.0.1:8000/ws/transcribe"
        async with websockets.connect(uri) as ws:
            print("WebSocket connected")
            await ws.send(json.dumps({"type": "config", "language": "en"}))

            frame_size = 512
            sent = 0
            frame_count = 0
            while sent < len(pcm):
                end = min(sent + frame_size, len(pcm))
                frame_bytes = pcm[sent:end].tobytes()
                if len(frame_bytes) > 0:
                    await ws.send(frame_bytes)
                    frame_count += 1
                sent = end
                await asyncio.sleep(0.032)
            print(f"Sent {frame_count} audio frames")

            await ws.send(json.dumps({"type": "stop"}))
            print("Sent stop command")

            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    if isinstance(msg, bytes):
                        print(f"[AUDIO] {len(msg)} bytes PCM")
                        continue
                    obj = json.loads(msg)
                    kind = "FINAL" if obj.get("is_final") else "partial"
                    conf = obj.get('confidence') or 0.0
                    print(f"[{kind:>7} conf={conf:.3f}] "
                          f"type={obj.get('type', 'transcript')} "
                          f"{obj.get('transcript', obj.get('text', ''))!r}")
                    if obj.get("type") in ("tts_end", "command", "llm_error"):
                        print(f"  -> full event: {obj}")
            except asyncio.TimeoutError:
                print("No more events (timeout)")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")

            await asyncio.sleep(3)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    print(f"Server still up after full pipeline: {data.get('ready')}")
            except Exception as e:
                print(f"SERVER IS DOWN after full pipeline: {e}")

    asyncio.run(test())

    print("Done.")


finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    read_server_output()
    print("\n--- SERVER OUTPUT (last 80 lines) ---")
    for line in server_output[-80:]:
        print(line)
