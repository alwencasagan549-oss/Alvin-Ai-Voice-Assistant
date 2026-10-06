"""Shared pytest fixtures for the Alvin STT suite.

Tests run against a *real* uvicorn server (started in a subprocess) and connect
with the ``websockets`` client library. This exercises the actual production
codepath and sidesteps the ``TestClient`` websocket transport, which is
incompatible with the ``httpx2`` transport required by the installed
Starlette 1.x.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "data" / "sample_speech.wav"

# tests/manual/ holds run-by-hand client scripts (they connect to a live
# server at import time), so keep pytest out of that directory.
collect_ignore_glob = ["manual/*"]

# --- Force a fast Whisper model for the test subprocess (inherited via env) --
# The only local model the service loads is FALLBACK_MODEL, so that is the knob
# to shrink for tests. GROQ_FALLBACK=false keeps the subprocess offline: these
# tests must not depend on a cloud key, quota, or network reachability.
os.environ.setdefault("FALLBACK_MODEL", "tiny.en")
os.environ.setdefault("GROQ_FALLBACK", "false")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("WHISPER_LANGUAGE", "en")
os.environ.setdefault("PARTIAL_INTERVAL_MS", "1000")
os.environ.setdefault("MIN_SEGMENT_SECONDS", "0.3")
os.environ.setdefault("MAX_SEGMENT_SECONDS", "20")
# Keep the server subprocess offline: the startup warm-up is a real cloud call.
os.environ.setdefault("GROQ_WARMUP", "false")
# Explicitly disable Groq for integration tests (overrides .env)
os.environ["GROQ_FALLBACK"] = "false"
os.environ["GROQ_API_KEY"] = ""



def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_ready(port: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as resp:
                data = json.loads(resp.read().decode())
                if data.get("ready") is True:
                    return
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.3)
    raise RuntimeError(f"server on port {port} did not become ready")


@pytest.fixture(scope="module")
def server_url() -> str:
    """Start a uvicorn subprocess and yield its /ws/transcribe URL."""
    port = _free_port()
    # Explicitly disable Groq in subprocess (overrides .env)
    env = os.environ.copy()
    env["GROQ_FALLBACK"] = "false"
    env["GROQ_API_KEY"] = ""
    env["GROQ_WARMUP"] = "false"
    # Lower VAD threshold for test audio (attenuated to 55% volume)
    env["VAD_THRESHOLD"] = "0.3"
    
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
        cwd=str(ROOT),
        env=env,
    )
    try:
        _wait_ready(port)
        yield f"ws://127.0.0.1:{port}/ws/transcribe"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _load_wav() -> tuple[bytes, int]:
    with wave.open(str(FIXTURE), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


@pytest.fixture(scope="session")
def speech_array() -> tuple[np.ndarray, int]:
    raw, sr = _load_wav()
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr, sr


@pytest.fixture(scope="session")
def speech_int16_bytes() -> tuple[bytes, int]:
    return _load_wav()
