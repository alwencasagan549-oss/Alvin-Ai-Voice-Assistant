"""Manual client: stream live microphone audio to a running Alvin server.

Implements the full client-side protocol:

  Transport
  - WebSocket to ws://<host>:<port>/ws/transcribe
  - Auth via Bearer header or ?token= query parameter
  - max_size=2 MB to stay under the 1 MB frame ceiling

  Audio ingest
  - Raw 16-bit PCM, monotonic, little-endian, 16 kHz
  - 512-sample (1024-byte) aligned frames for Silero VAD
  - Mic capture runs on the sounddevice callback thread; audio is pushed
    into an asyncio.Queue so network backpressure never freezes the
    high-priority audio thread

   Barge-in / local playback
   - Server TTS PCM is played back via an sd.OutputStream
   - Mic input is **muted (not sent to the server) while TTS is playing** to
     prevent the assistant from transcribing and responding to its own voice
   - Local VAD (silero-vad) still runs on captured audio for barge-in detection:
     user speech onset during TTS instantly stops playback and unmutes the mic
   - Consecutive-frame gate (default 2 frames = 64 ms) prevents false
     positives from transient noises like throat clearing or door slams
   - NOTE: the client should NOT flush on llm_response events, because a
     single LLM turn may emit multiple llm_response events (one per
     sentence) and the server has already cancelled any prior TTS via
     _cancel_ongoing_turn.  Only local speech onset should trigger a flush.
   - As a secondary safety net, the queue is also flushed when an incoming
     llm_response event signals a new turn has started

  Heartbeats
  - Periodic {"type": "ping"} text frames every 15 s

Requires the optional ``sounddevice`` and ``silero-vad`` dependencies:
    pip install sounddevice silero-vad

Usage:
    python tests/manual/stream_mic.py [options]

Options:
    --url URL        WebSocket endpoint (default: ws://localhost:8000/ws/transcribe)
    --token TOKEN    API key; sent as Bearer header (falls back to ?token= if needed)
    --language LANG  Send {"type": "config", "language": LANG} at connect
    --voice VOICE    Send {"type": "config", "voice": VOICE} at connect
    --vad-threshold  Silero VAD speech probability threshold (default 0.5; raise
                     to 0.6+ to reduce false positives from transient noise)
    --vad-frames     Consecutive speech frames required to trigger flush
                     (default: 2, i.e. 64 ms at 32 ms/frame)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys

import numpy as np
import sounddevice as sd
import websockets
from silero_vad import VADIterator, load_silero_vad

URI = "ws://localhost:8000/ws/transcribe"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample

PING_INTERVAL = 15.0
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_FRAMES = 2
PLAYBACK_STOP_TIMEOUT = 2.0


def _build_uri(uri: str, token: str | None) -> str:
    """Append ?token= to the URL for environments without header control."""
    if not token:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}token={token}"


def _auth_headers(token: str | None) -> dict[str, str]:
    """Bearer header preferred over query-string token (avoids access logs)."""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def stream_mic(
    uri: str,
    token: str | None,
    language: str | None,
    voice: str | None,
    vad_threshold: float,
    vad_frames: int,
) -> None:
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    playback_task: asyncio.Task | None = None
    speech_onset_count = 0
    tts_playing = False

    model = load_silero_vad(onnx=True)
    vad = VADIterator(model, threshold=vad_threshold, sampling_rate=SAMPLE_RATE)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio-callback] {status}", file=sys.stderr, flush=True)
        mic_queue.put_nowait(indata.tobytes())

    async def flush_playback():
        """Drain the TTS playback queue and stop the playback task."""
        nonlocal playback_task, tts_playing
        if playback_task is None or playback_task.done():
            return
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await audio_queue.put(None)
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(playback_task, timeout=PLAYBACK_STOP_TIMEOUT)
        playback_task = None
        tts_playing = False

    async def send_audio(websocket):
        nonlocal speech_onset_count, tts_playing
        sample_buf = bytearray()
        while True:
            chunk = await mic_queue.get()
            sample_buf.extend(chunk)
            while len(sample_buf) >= FRAME_BYTES:
                frame = bytes(sample_buf[:FRAME_BYTES])
                del sample_buf[:FRAME_BYTES]

                # Run client-side VAD on every captured frame for barge-in
                # detection, regardless of whether we're sending to the server.
                samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
                result = vad(samples)  # keep feeding VAD so its state stays coherent

                # Barge-in is only trusted while Alvin is NOT speaking. While TTS
                # is playing, the microphone also picks up Alvin's own voice, and
                # counting that echo as a "speech onset" makes the client flush
                # playback and cut the assistant off mid-sentence (the "sentence
                # gets cut off" bug seen when the room is quiet). True echo
                # cancellation (AEC) would let barge-in stay live during playback;
                # without it we must suppress onset detection while speaking.
                if tts_playing:
                    speech_onset_count = 0
                elif result and "start" in result:
                    speech_onset_count += 1
                    if speech_onset_count >= vad_frames:
                        print(
                            f"[barge-in] speech onset confirmed "
                            f"({speech_onset_count} frames), flushing TTS",
                            flush=True,
                        )
                        await flush_playback()
                        speech_onset_count = 0
                elif result and "end" in result:
                    speech_onset_count = 0

                # While TTS is playing, do NOT send mic audio to the server.
                # This prevents the assistant from hearing its own TTS output
                # and responding to itself.
                if not tts_playing:
                    await websocket.send(frame)

    async def play_tts_audio():
        nonlocal tts_playing
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
        )
        stream.start()
        try:
            while True:
                data = await audio_queue.get()
                if data is None:
                    break
                stream.write(np.frombuffer(data, dtype="<i2"))
        except OSError as e:
            print(f"[playback] audio error: {e}", file=sys.stderr, flush=True)
        finally:
            if stream.active:
                stream.stop()
            stream.close()

    async def receive_loop(websocket):
        nonlocal playback_task, tts_playing, speech_onset_count
        while True:
            msg = await websocket.recv()
            if isinstance(msg, bytes):
                # First TTS audio frame: mark TTS as playing to mute mic
                if playback_task is None or playback_task.done():
                    playback_task = asyncio.create_task(play_tts_audio())
                    tts_playing = True
                    speech_onset_count = 0
                await audio_queue.put(msg)
            else:
                obj = json.loads(msg)
                msg_type = obj.get("type", "")
                if msg_type == "tts_end":
                    if playback_task is not None and playback_task.done():
                        playback_task = None
                    tts_playing = False
                elif msg_type == "tts_error":
                    tts_playing = False
                elif msg_type == "llm_response":
                    # New LLM turn incoming — make sure mic is unmuted
                    pass
                print(f"[event] {obj}", flush=True)

    async def ping_loop(websocket):
        while True:
            await asyncio.sleep(PING_INTERVAL)
            await websocket.send(json.dumps({"type": "ping"}))

    headers = _auth_headers(token)
    ws_uri = _build_uri(uri, token)

    async with websockets.connect(
        ws_uri,
        additional_headers=headers,
        max_size=2 * 1024 * 1024,
    ) as websocket:
        print("Connected! Start speaking into your microphone...", flush=True)

        config_msgs: dict = {}
        if language:
            config_msgs["language"] = language
        if voice:
            config_msgs["voice"] = voice
        if config_msgs:
            await websocket.send(json.dumps({"type": "config", **config_msgs}))
            print(f"[config] sent: {config_msgs}", flush=True)
        if vad_threshold != DEFAULT_VAD_THRESHOLD:
            print(f"[vad] threshold={vad_threshold}, frames={vad_frames}", flush=True)

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=audio_callback,
        )

        with stream:
            await asyncio.gather(
                send_audio(websocket),
                receive_loop(websocket),
                ping_loop(websocket),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alvin live microphone client")
    parser.add_argument(
        "--url",
        default=URI,
        help=f"WebSocket URL (default: {URI})",
    )
    parser.add_argument("--token", default=None, help="API key for auth")
    parser.add_argument("--language", default=None, help="Override STT language")
    parser.add_argument("--voice", default=None, help="Override TTS voice")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help="Silero VAD speech probability threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--vad-frames",
        type=int,
        default=DEFAULT_VAD_FRAMES,
        help="Consecutive speech frames to trigger flush (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            stream_mic(
                args.url,
                args.token,
                args.language,
                args.voice,
                args.vad_threshold,
                args.vad_frames,
            )
        )
    except KeyboardInterrupt:
        print("\nLive microphone test stopped.", flush=True)
