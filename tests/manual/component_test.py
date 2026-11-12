r"""Comprehensive component test: STT, LLM, TTS.

Run against a live server (python run.py). Not part of the pytest suite.
Usage:
    PYTHONPATH=. .venv\Scripts\python tests\manual/component_test.py
"""

import asyncio
import json
import time

import websockets

from app.llm import build_messages, llm_service
from app.tts import synthesize_pcm
from tests.manual.stream_wav import FIXTURE, load_wav

FRAME_BYTES = 1024  # 512 samples × 2 bytes/sample (int16)


def print_safe(text: str) -> None:
    """Print safely on terminals with limited encodings (e.g. Windows cp1252)."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)


async def test_llm_streaming() -> None:
    """Test LLM streaming with Cloudflare fallback."""
    print("=== LLM Streaming Test ===", flush=True)
    msgs = build_messages([], "What is the weather like today?", max_history=6)
    start = time.perf_counter()
    count = 0
    async for sentence, _model in llm_service.stream_response(
        msgs, fallback_text="fallback"
    ):
        count += 1
        elapsed = (time.perf_counter() - start) * 1000
        print_safe(f"  [{elapsed:.0f}ms] sentence {count}: {sentence[:60]}")
    print_safe(f"  Total: {count} sentences in {time.perf_counter() - start:.2f}s")
    print_safe("")


async def test_tts() -> None:
    """Test TTS synthesis."""
    print_safe("=== TTS Test ===")
    text = "Hello Alvin how are you today"
    start = time.perf_counter()
    try:
        total_bytes = 0
        async for chunk in synthesize_pcm(text, voice="en-US-AndrewNeural"):
            total_bytes += len(chunk)
        elapsed = (time.perf_counter() - start) * 1000
        duration = total_bytes / (16000 * 2)
        print_safe(f"  Text: {text}")
        print_safe(f"  Audio: {total_bytes} bytes ({duration:.2f}s) in {elapsed:.0f}ms")
    except Exception as e:  # noqa: BLE001
        print(f"  Error: {e}", flush=True)
    print(flush=True)


async def test_full_pipeline() -> None:
    """Test the full pipeline via WebSocket: STT -> LLM -> TTS."""
    print_safe("=== Full Pipeline Test (WebSocket) ===")
    print_safe("Streaming WAV file at real-time speed to trigger VAD endpointing...")
    print_safe("")

    url = "ws://localhost:8000/ws/transcribe"
    pcm, sr = load_wav(str(FIXTURE))
    chunk_size = int(sr * 0.02)  # 20ms chunks

    try:
        async with websockets.connect(url, max_size=None) as ws:
            sent = 0
            while sent < len(pcm):
                end = min(sent + chunk_size, len(pcm))
                await ws.send(pcm[sent:end].tobytes())
                sent = end
                await asyncio.sleep(0.02)

            # Stream trailing silence (512-sample frames) so the server's VAD
            # detects endpointing naturally (MIN_SILENCE_DURATION_MS=500).
            # We do NOT send {"type": "stop"} because that sets
            # self.stopping=True on the server, which prevents LLM processing
            # for the final utterance.
            silence_frame = bytes(FRAME_BYTES)  # 1024 bytes of silence
            for _ in range(50):  # ~1.6 s of VAD-processable silence
                await ws.send(silence_frame)
                await asyncio.sleep(0.032)  # real-time: 32 ms per frame

            print_safe("Waiting for events (STT -> LLM -> TTS)...")
            audio_frames = 0
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        obj = json.loads(msg)
                        if "transcript" in obj:
                            kind = "FINAL" if obj.get("is_final") else "partial"
                            llm_flag = " [LLM]" if obj.get("llm_enabled") else ""
                            print_safe(f"  [{kind}] {obj['transcript']!r}{llm_flag}")
                        elif obj.get("type") in (
                            "llm_response",
                            "llm_error",
                            "tts_end",
                        ):
                            print_safe(f"  [{obj['type']}] {json.dumps(obj)[:80]}")
                            if obj.get("type") == "tts_end":
                                break
                    elif isinstance(msg, bytes):
                        audio_frames += 1
                        if audio_frames == 1:
                            print_safe("  [TTS-AUDIO] streaming PCM audio...")
                        if audio_frames and audio_frames % 20 == 0:
                            print_safe(
                                f"  ...still streaming TTS ({audio_frames} frames)"
                            )
                        if audio_frames > 1000:
                            print_safe("  (stopping - got enough audio)")
                            break
            except websockets.exceptions.ConnectionClosed:
                print_safe(f"  Pipeline complete: {audio_frames} audio frames received")
    except Exception as e:  # noqa: BLE001
        print_safe(f"  Connection error: {e}")
        print_safe("  (Is the server running? Execute: python run.py)")


async def main() -> None:
    await test_llm_streaming()
    await test_tts()
    await test_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
