"""Manual end-to-end run: boot a throwaway tiny.en server, stream the fixture,
print events, then shut the server down.

The pytest suite does the same thing automatically (tests/conftest.py
``server_url`` fixture); use this only for eyeballing raw event output.

Usage:
    python tests/manual/run_with_server.py
"""

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_speech.wav"

# Force the smallest/fastest path for this throwaway server: FALLBACK_MODEL is
# the only local model the service loads, and GROQ_FALLBACK=false keeps the run
# offline and deterministic.
os.environ.update(
    {
        "FALLBACK_MODEL": "tiny.en",
        "GROQ_FALLBACK": "false",
        "GROQ_WARMUP": "false",
        "TTS_WARMUP": "false",
        "WHISPER_LANGUAGE": "en",
        "PARTIAL_INTERVAL_MS": "1000",
        "MIN_SEGMENT_SECONDS": "0.3",
        "MAX_SEGMENT_SECONDS": "20",
    }
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_health(port: int, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with (
            contextlib.suppress(Exception),
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as resp,
        ):
            if json.loads(resp.read().decode()).get("ready"):
                return True
        time.sleep(0.3)
    return False


async def stream_file(uri: str) -> None:
    import websockets

    async with websockets.connect(uri) as websocket:
        print("Connected to server. Streaming audio...")

        async def receive_results():
            try:
                while True:
                    print("Received:", json.loads(await websocket.recv()))
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")

        listener = asyncio.create_task(receive_results())

        # Open the test file and stream raw PCM
        with FIXTURE.open("rb") as f:
            f.read(44)  # skip WAV header
            while chunk := f.read(4096):
                await websocket.send(chunk)
                await asyncio.sleep(0.1)

        print("Finished sending audio. Waiting for final transcription...")
        await asyncio.sleep(2)
        listener.cancel()


def main() -> int:
    port = _free_port()
    print(f"Starting server on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ws",
            "websockets",
            "--log-level",
            "warning",
        ],
        env=os.environ.copy(),
        cwd=str(ROOT),
    )
    try:
        if not wait_for_health(port):
            print("Server failed to become ready.")
            return 1
        print("Server ready.")
        asyncio.run(stream_file(f"ws://127.0.0.1:{port}/ws/transcribe"))
    finally:
        print("Stopping server...")
        proc.terminate()
    print("Test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
