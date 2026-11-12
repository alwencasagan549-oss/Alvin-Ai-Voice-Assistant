r"""Manual benchmark: pipe the speech fixture through live STT, then TTS.

Not collected by pytest. Start the server first (``python run.py``), then run
``python scripts\measure_stt_tts_latency.py`` from the project root.
"""

import asyncio
import json
import sys
import time
import traceback
import wave
from pathlib import Path

import websockets

from app.tts import synthesize_pcm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "data" / "sample_speech.wav"

FRAME_SEC = 0.2


async def stream_stt_and_tts(url: str, raw: bytes, sr: int) -> dict:
    """Stream audio to STT, pipe final transcript to TTS, measure latency."""

    tts_start = None
    tts_end = None

    events = []
    final_transcript = None

    # Send at real-time speed: 0.2s frames every 0.2s
    frame_sec = 0.2
    frame_bytes = int(sr * frame_sec) * 2  # int16 = 2 bytes
    frames = [raw[i : i + frame_bytes] for i in range(0, len(raw), frame_bytes)]

    async with websockets.connect(url) as sock:
        # Stream audio frames at real-time rate
        for frame in frames:
            await sock.send(frame)
            await asyncio.sleep(frame_sec)  # Real-time pacing

        # Signal end and start timing
        stt_start = time.perf_counter()
        await sock.send(json.dumps({"type": "stop"}))

        # Collect events
        try:
            while True:
                msg = await asyncio.wait_for(sock.recv(), timeout=25)
                try:
                    event = json.loads(msg)
                    events.append(event)

                    if event.get("is_final"):
                        final_transcript = event.get("transcript", "")
                        stt_end = time.perf_counter()
                        stt_latency = stt_end - stt_start
                        print(
                            f"STT final: '{final_transcript}' (latency: {stt_latency * 1000:.1f}ms)"
                        )
                        break
                    elif event.get("is_partial"):
                        print(f"  STT partial: '{event.get('transcript', '')}'")
                except json.JSONDecodeError:
                    continue
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    if not final_transcript:
        return {"error": "No final transcript received"}

    # Now synthesize with TTS
    print(f"\nSynthesizing TTS for: '{final_transcript}'")
    tts_start = time.perf_counter()

    total_tts_bytes = 0
    first_chunk_time = None

    async for pcm_chunk in synthesize_pcm(final_transcript):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
            ttfa = (first_chunk_time - tts_start) * 1000
            print(f"  TTS TTFA (Time to First Audio): {ttfa:.1f}ms")
        total_tts_bytes += len(pcm_chunk)

    tts_end = time.perf_counter()
    tts_total_latency = (tts_end - tts_start) * 1000
    tts_duration = total_tts_bytes / (16000 * 2)

    total_latency = (tts_end - stt_start) * 1000

    return {
        "final_transcript": final_transcript,
        "stt_latency_ms": round((stt_end - stt_start) * 1000, 1),
        "tts_ttfa_ms": round(ttfa, 1) if first_chunk_time else None,
        "tts_total_ms": round(tts_total_latency, 1),
        "tts_audio_duration_sec": round(tts_duration, 2),
        "tts_bytes": total_tts_bytes,
        "end_to_end_ms": round(total_latency, 1),
        "num_stt_events": len(events),
    }


async def main():
    # Load audio
    with wave.open(str(FIXTURE), "rb") as w:
        raw = w.readframes(w.getnframes())
        sr = w.getframerate()

    print(f"Loaded audio: {len(raw)} bytes, {sr}Hz, {len(raw) / (sr * 2):.2f}s")

    # Find running server or start one
    # For this test, we'll try to connect to the default port
    url = "ws://127.0.0.1:8000/ws/transcribe"

    print(f"Connecting to {url}...")

    try:
        results = await stream_stt_and_tts(url, raw, sr)

        print("\n=== RESULTS ===")
        for key, value in results.items():
            print(f"  {key}: {value}")

        # Performance assessment
        if results.get("end_to_end_ms"):
            e2e = results["end_to_end_ms"]
            if e2e < 2000:
                print(f"\n[EXCELLENT] End-to-end: {e2e}ms (< 2s)")
            elif e2e < 5000:
                print(f"\n[GOOD] End-to-end: {e2e}ms (< 5s)")
            elif e2e < 10000:
                print(f"\n[ACCEPTABLE] End-to-end: {e2e}ms (< 10s)")
            else:
                print(f"\n[SLOW] End-to-end: {e2e}ms (>= 10s)")

        if results.get("tts_ttfa_ms"):
            ttfa = results["tts_ttfa_ms"]
            if ttfa < 500:
                print(f"[EXCELLENT] TTS TTFA: {ttfa}ms")
            elif ttfa < 1000:
                print(f"[GOOD] TTS TTFA: {ttfa}ms")
            else:
                print(f"[SLOW] TTS TTFA: {ttfa}ms")

    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
