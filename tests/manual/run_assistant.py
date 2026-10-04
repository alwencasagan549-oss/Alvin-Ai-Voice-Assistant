"""Manual end-to-end run: boot the Alvin server, then stream live microphone.

One command to launch everything:
    python tests/manual/run_assistant.py [options]

The script:
  1. (Optional) Starts a uvicorn server subprocess on an auto-selected port
  2. Waits for the /health endpoint to report ready
  3. Launches the live microphone client (stream_mic)
  4. On Ctrl-C or disconnect, terminates the server and exits

If --url is provided, the script skips server startup and connects to the
existing server at that URL.

Environment:
  - LLM_API_KEY must be set in .env (or env) to enable the LLM path.
  - sounddevice and silero-vad are required for the mic client.

Options:
    --url URL        Connect to an existing server (skips server startup)
    --token TOKEN    API key; sent as Bearer header (falls back to ?token=)
    --language LANG  Send {"type": "config", "language": LANG} at connect
    --voice VOICE    Send {"type": "config", "voice": VOICE} at connect
    --vad-threshold  Silero VAD threshold (default 0.5; raise to reduce false positives)
    --vad-frames     Consecutive speech frames required to trigger flush (default: 2)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

os.environ.setdefault("WHISPER_LANGUAGE", "en")
os.environ.setdefault("LOG_LEVEL", "warning")
os.environ.setdefault("TTS_AUTO_SPEAK", "true")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
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
            data = json.loads(resp.read().decode())
            if data.get("ready"):
                return True
            llm_info = data.get("llm", {})
            if llm_info.get("enabled"):
                return True
        time.sleep(0.3)
    return False


def report_health(port: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
        health = json.loads(resp.read().decode())
    llm_info = health.get("llm", {})
    print(
        f"Server ready. LLM: {llm_info.get('model', '?')} "
        f"(enabled={llm_info.get('enabled', False)})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Alvin assistant (server + mic)")
    parser.add_argument(
        "--url", default=None, help="Connect to existing server (skip startup)"
    )
    parser.add_argument("--token", default=None, help="API key for auth")
    parser.add_argument("--language", default=None, help="Override STT language")
    parser.add_argument("--voice", default=None, help="Override TTS voice")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold (default: 0.5)",
    )
    parser.add_argument(
        "--vad-frames",
        type=int,
        default=2,
        help="Consecutive speech frames to trigger flush (default: 2)",
    )
    args = parser.parse_args()

    proc: subprocess.Popen | None = None

    def cleanup():
        if proc is not None:
            print("\nStopping server...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def signal_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.url:
        # Connect to an existing server — skip startup.
        uri = args.url
        report_health_from_url(uri)
    else:
        port = _free_port()
        uri = f"ws://127.0.0.1:{port}/ws/transcribe"
        print(f"Starting Alvin server on port {port}...", flush=True)
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
                "auto",
                "--log-level",
                os.environ.get("LOG_LEVEL", "warning"),
            ],
            env=os.environ.copy(),
            cwd=str(ROOT),
        )
        try:
            print("Waiting for server to become ready...", flush=True)
            if not wait_for_health(port):
                print("Server failed to become ready.", file=sys.stderr, flush=True)
                cleanup()
                return 1
            report_health(port)
        except (OSError, urllib.error.URLError):
            print("Server health check failed.", file=sys.stderr, flush=True)
            cleanup()
            return 1

    # Import and run the mic client inline
    sys.path.insert(0, str(ROOT))
    from tests.manual.stream_mic import stream_mic

    try:
        asyncio.run(
            stream_mic(
                uri,
                args.token,
                args.language,
                args.voice,
                args.vad_threshold,
                args.vad_frames,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

    print("Done.", flush=True)
    return 0


def report_health_from_url(uri: str) -> None:
    """Extract host:port from a ws:// URL and query /health."""
    host = uri.split("://", 1)[1].split("/")[0]
    if ":" in host:
        hostname, port = host.rsplit(":", 1)
        health_url = f"http://{hostname}:{port}/health"
    else:
        health_url = f"http://{host}:80/health"
    with urllib.request.urlopen(health_url, timeout=2) as resp:
        health = json.loads(resp.read().decode())
    llm_info = health.get("llm", {})
    print(
        f"Server ready. LLM: {llm_info.get('model', '?')} "
        f"(enabled={llm_info.get('enabled', False)})",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
