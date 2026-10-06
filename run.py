"""Convenience entrypoint: ``python run.py`` starts the uvicorn server.

Add ``--mic`` to also launch the live microphone client after the server
is ready — one command for the full voice assistant pipeline:

    python run.py --mic [--voice en-US-AndrewNeural] [--vad-threshold 0.5]

With ``--mic``, the server runs as a subprocess and the mic client attaches
to it.  Ctrl-C stops both cleanly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import uvicorn

from app.config import settings

ROOT = os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_health(port: int, timeout: float = 120) -> bool:
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
            llm = data.get("llm", {})
            if llm.get("enabled"):
                return True
        time.sleep(0.3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Alvin voice assistant server")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Also launch the live microphone client after server startup",
    )
    parser.add_argument("--token", default=None, help="API key for auth")
    parser.add_argument("--language", default=None, help="Override STT language")
    parser.add_argument("--voice", default=None, help="Override TTS voice")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold (default: 0.5; raise to reduce false positives)",
    )
    parser.add_argument(
        "--vad-frames",
        type=int,
        default=2,
        help="Consecutive speech frames to trigger barge-in flush (default: 2)",
    )
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument(
        "--ptt-mode",
        choices=["global", "console"],
        default="console",
        help="PTT hotkey mode: 'global' (Shift+Z, requires admin), 'console' (Enter key, no admin)",
    )
    args = parser.parse_args()

    if args.mic:
        port = args.port or _free_port()
        os.environ.setdefault("TTS_AUTO_SPEAK", "true")
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
            cwd=ROOT,
        )

        def cleanup(*_):
            print("\nStopping server...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        signal.signal(signal.SIGINT, cleanup)

        try:
            print("Waiting for server to become ready...", flush=True)
            if not _wait_for_health(port):
                print("Server failed to become ready.", file=sys.stderr, flush=True)
                cleanup()
                sys.exit(1)

            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as resp:
                health = json.loads(resp.read().decode())
            llm_info = health.get("llm", {})
            print(
                f"Server ready. LLM: {llm_info.get('model', '?')} "
                f"(enabled={llm_info.get('enabled', False)})",
                flush=True,
            )

            sys.path.insert(0, ROOT)
            from tests.manual.stream_mic import stream_mic

            uri = f"ws://127.0.0.1:{port}/ws/transcribe"
            asyncio_run(
                stream_mic(
                    uri,
                    args.token,
                    args.language,
                    args.voice,
                    args.vad_threshold,
                    args.vad_frames,
                    args.ptt_mode,
                )
            )
        except KeyboardInterrupt:
            pass
        finally:
            cleanup()
        print("Done.", flush=True)
    else:
        host = "127.0.0.1" if settings.host in ("0.0.0.0", "") else settings.host
        print(
            f"Starting Alvin STT/TTS/LLM server on {host}:{settings.port}",
            flush=True,
        )
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            ws="auto",
        )


def asyncio_run(coro):
    """Wrapper to run an async coroutine (Python 3.12 compatible)."""
    import asyncio

    asyncio.run(coro)


if __name__ == "__main__":
    main()
