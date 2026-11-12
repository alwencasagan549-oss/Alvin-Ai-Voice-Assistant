#!/usr/bin/env python3
"""Test the Alvin WebSocket connection with a vision query."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import websockets

# Simple 1x1 pixel JPEG (base64 encoded)
TEST_JPEG_B64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/AB//2Q=="


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.astype("<i2"), sr


async def test_vision_query():
    url = "ws://127.0.0.1:8000/ws/transcribe"
    print(f"Connecting to {url}...")

    # Load the test WAV file
    wav_path = Path(__file__).parent / "tests" / "data" / "sample_speech.wav"
    pcm, sr = load_wav(str(wav_path))
    print(f"Loaded {wav_path}: {len(pcm)} samples ({len(pcm) / sr:.2f}s) @ {sr}Hz")

    async with websockets.connect(url) as ws:
        print("Connected!")

        # Send config
        await ws.send(json.dumps({"type": "config", "language": "en"}))

        # Send camera frame
        print("Sending camera frame...")
        await ws.send(json.dumps({
            "type": "camera_frame",
            "image": TEST_JPEG_B64
        }))

        await asyncio.sleep(0.5)

        # Send a vision query as text (simulating STT result)
        # We'll send the audio but also inject a vision query
        print("Sending audio (simulating 'what can you see')...")
        chunk_size = int(sr * 20 / 1000)  # 20ms chunks
        print(f"Streaming audio in {chunk_size} sample chunks...")
        sent = 0
        while sent < len(pcm):
            end = min(sent + chunk_size, len(pcm))
            await ws.send(pcm[sent:end].tobytes())
            sent = end
            await asyncio.sleep(0.02)  # 20ms real-time

        # Send stop to finalize
        await ws.send(json.dumps({"type": "stop"}))

        print("Waiting for responses...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                try:
                    data = json.loads(msg)
                    print(f"Received: {json.dumps(data, indent=2)}")

                    # Check if it's an LLM response
                    if data.get("type") == "llm_response":
                        text = data.get("text", "")
                        model = data.get("model", "")
                        print(f"\n>>> LLM Response (model: {model}): {text}")
                        if any(word in text.lower() for word in ["see", "image", "camera", "picture", "photo", "frame", "black", "white", "pixel", "dark", "bright", "holding", "hand", "object"]):
                            print(">>> SUCCESS: Vision model appears to be analyzing the image!")
                        else:
                            print(">>> WARNING: Response doesn't seem to reference the image")
                    elif data.get("type") == "llm_error":
                        print(f"\n>>> LLM Error: {data.get('error')}")
                    elif data.get("is_final"):
                        print(f"\n>>> Final transcript: {data.get('transcript')}")
                    elif data.get("type") == "request_camera_frame":
                        # Server wants a fresh frame - send one
                        print("Server requested fresh camera frame, sending...")
                        await ws.send(json.dumps({
                            "type": "camera_frame",
                            "image": TEST_JPEG_B64
                        }))

                except json.JSONDecodeError:
                    print(f"Received binary data: {len(msg)} bytes (likely TTS audio)")
        except asyncio.TimeoutError:
            print("Timeout waiting for response")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")


if __name__ == "__main__":
    asyncio.run(test_vision_query())