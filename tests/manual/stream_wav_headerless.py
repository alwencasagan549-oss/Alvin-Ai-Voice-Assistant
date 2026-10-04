"""Manual client: stream a WAV file as headerless raw PCM frames.

Useful for verifying that the server tolerates arbitrary frame sizes (it sends
raw PCM, skipping the 44-byte WAV header). Manual use against ``python run.py``.

Usage:
    python tests/manual/stream_wav_headerless.py [path/to.wav]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_speech.wav"
URI = "ws://localhost:8000/ws/transcribe"


async def stream_file(wav: Path = FIXTURE, uri: str = URI) -> None:
    async with websockets.connect(uri, max_size=None) as websocket:
        print("Connected to server. Streaming audio...", flush=True)

        t0 = time.time()
        with wav.open("rb") as f:
            f.read(44)  # Skip the standard 44-byte WAV header to send pure PCM
            while chunk := f.read(4096):
                await websocket.send(chunk)
                await asyncio.sleep(0.1)  # Artificially delay to mimic real-time speed

        print(
            f"Finished sending audio in {time.time() - t0:.1f}s. "
            "Waiting for final transcription...",
            flush=True,
        )

        # Wait for events until we see a final transcription (up to 120s for
        # larger models such as distil-large-v3 running on CPU).
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=10)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.", flush=True)
                break
            data = json.loads(msg)
            print("Received:", data, flush=True)
            if data.get("is_final"):
                print("Got final transcription. Done.", flush=True)
                break


if __name__ == "__main__":
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    asyncio.run(stream_file(wav))
