#!/usr/bin/env python3
"""Test the Alvin WebSocket connection with a vision query sent as text."""

from __future__ import annotations

import asyncio
import json
import base64
import websockets


async def test_vision_text_query():
    url = "ws://127.0.0.1:8000/ws/transcribe"
    print(f"Connecting to {url}...")

    # Read the image
    with open('image.png', 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f"Image size: {len(b64)} chars")

    async with websockets.connect(url) as ws:
        print("Connected!")

        # Send config
        await ws.send(json.dumps({"type": "config", "language": "en"}))

        # Send camera frame
        print("Sending camera frame...")
        await ws.send(json.dumps({
            "type": "camera_frame",
            "image": b64
        }))

        await asyncio.sleep(0.5)

        # Send a vision query as text control message (simulating final STT result)
        # This simulates what happens when STT produces "what can you see"
        print("Sending vision query via text...")
        await ws.send(json.dumps({
            "type": "transcript",
            "transcript": "What can you see in this image?",
            "is_partial": False,
            "is_final": True,
            "confidence": 0.95
        }))

        print("Waiting for responses...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                try:
                    data = json.loads(msg)
                    print(f"Received: {json.dumps(data, indent=2)}")

                    if data.get("type") == "llm_response":
                        text = data.get("text", "")
                        model = data.get("model", "")
                        print(f"\n>>> LLM Response (model: {model}): {text}")
                        if any(word in text.lower() for word in ["apple", "fruit", "red", "image", "see", "picture", "photo"]):
                            print(">>> SUCCESS: Vision model analyzed the image!")
                        else:
                            print(">>> WARNING: Response doesn't seem to reference the image")
                    elif data.get("type") == "llm_error":
                        print(f"\n>>> LLM Error: {data.get('error')}")
                    elif data.get("type") == "request_camera_frame":
                        print("Server requested fresh camera frame")
                    elif data.get("type") == "transcript":
                        print(f"Transcript: {data.get('transcript')}")

                except json.JSONDecodeError:
                    print(f"Received binary data: {len(msg)} bytes (likely TTS audio)")
        except asyncio.TimeoutError:
            print("Timeout waiting for response")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")


if __name__ == "__main__":
    asyncio.run(test_vision_text_query())