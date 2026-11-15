"""Run the Alvin backend and optionally the holographic UI.

Usage:
    python run-all.py              # backend only (port 8000)
    python run-all.py --hud       # backend + holographic UI
    python run-all.py --hud --hidden  # HUD starts hidden, appears on "Alvin, wake up"
    python run-all.py --mic       # backend + mic client

This starts:
  1. The FastAPI STT/TTS/LLM backend on http://localhost:8000
  2. Optional: PySide6 Holographic UI (launched on --hud or by "Alvin, wake up")
  3. Optional: Local mic client (run.py --mic)

Press Ctrl+C to stop all servers cleanly.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json
import socket

from app.config import settings

ROOT = os.path.dirname(os.path.abspath(__file__))
# Bug 28: mirror run.py's TTS_AUTO_SPEAK guard so the combined launcher never
# overrides a caller that has intentionally disabled auto-speak.
if os.environ.get("TTS_AUTO_SPEAK", "").lower() not in {"true", "1", "yes", "on"}:
    os.environ["TTS_AUTO_SPEAK"] = "true"
BACKEND_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alvin backend and optional holographic UI")
    parser.add_argument(
        "--hud",
        action="store_true",
        help="Start the PySide6 Holographic UI immediately (otherwise it starts on 'Alvin, wake up')",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help=(
            "Start the HUD hidden; it appears when the user says 'Alvin, wake up' "
            "(requires --hud)"
        ),
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Also start the local mic client",
    )
    return parser.parse_args()


def _find_python() -> str:
    venv_python = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python
    venv_python_unix = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_python_unix):
        return venv_python_unix
    return sys.executable


def _wait_for_url(url: str, timeout: float = 60, label: str = "service", require_ready: bool = False) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    if not require_ready:
                        return True
                    body = resp.read().decode()
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        time.sleep(0.5)
                        continue
                    if data.get("ready"):
                        return True
        except Exception:
            time.sleep(0.5)
    print(f"ERROR: {label} did not start within {timeout}s", flush=True)
    return False


def _ensure_pyside6() -> bool:
    """Check if PySide6 is installed, return True if available."""
    try:
        subprocess.run(
            [_find_python(), "-c", "import PySide6"],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    args = parse_args()
    backend_proc = None
    hud_proc = None
    mic_proc = None
    cleanup_started = False

    def cleanup(*_, exit_code=0):
        nonlocal cleanup_started
        if cleanup_started:
            raise SystemExit(exit_code)
        cleanup_started = True
        print("\n\nShutting down servers...", flush=True)
        if mic_proc and mic_proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(mic_proc.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(mic_proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if hud_proc and hud_proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(hud_proc.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(hud_proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
        print("Done.", flush=True)
        raise SystemExit(exit_code)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    bind_host = settings.host
    if bind_host in ("0.0.0.0", ""):
        bind_host = "127.0.0.1"
    if bind_host == "::":
        bind_host = "::1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_host, BACKEND_PORT))
    except OSError:
        print(f"ERROR: Port {BACKEND_PORT} is already in use", flush=True)
        sys.exit(1)
    sock.close()

    print("\n" + "=" * 60, flush=True)
    print("Starting Alvin backend (FastAPI STT/TTS/LLM)...", flush=True)
    print("=" * 60, flush=True)
    child_group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    backend_proc = subprocess.Popen(
        [_find_python(), "-m", "uvicorn", "app.main:app",
         "--host", bind_host, "--port", str(BACKEND_PORT),
         "--ws", "auto", "--log-level", "info"],
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        **child_group,
    )

    health_url = f"http://127.0.0.1:{BACKEND_PORT}/health"
    if not _wait_for_url(health_url, timeout=120, label="backend", require_ready=True):
        print("ERROR: Backend failed to start. Check the error messages above.", flush=True)
        cleanup()
    print(f"  [OK] Backend ready at http://localhost:{BACKEND_PORT}", flush=True)

    if args.hud:
        if not _ensure_pyside6():
            print("\n" + "=" * 60, flush=True)
            print("Installing PySide6 and dependencies...", flush=True)
            print("=" * 60, flush=True)
            result = subprocess.run(
                [_find_python(), "-m", "pip", "install", "PySide6", "sounddevice", "websockets"],
                cwd=ROOT,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            if result.returncode != 0:
                print("  [ERROR] Failed to install PySide6", flush=True)
                print("  Run manually: pip install PySide6 sounddevice websockets", flush=True)
            else:
                print("  [OK] Dependencies installed", flush=True)

        if _ensure_pyside6():
            print("\n" + "=" * 60, flush=True)
            print("Launching Holographic UI...", flush=True)
            print("=" * 60, flush=True)
            hud_cmd = [_find_python(), "alvin_hud.py"]
            if args.hidden:
                hud_cmd.append("--hidden")
            hud_proc = subprocess.Popen(
                hud_cmd,
                cwd=ROOT,
                stdout=sys.stdout,
                stderr=sys.stderr,
                **child_group,
            )
            time.sleep(1)
            if hud_proc.poll() is None:
                state = "hidden (say 'Alvin, wake up')" if args.hidden else "ready"
                print(f"  [OK] Holographic UI {state}", flush=True)
            else:
                print("  [ERROR] Holographic UI failed to start", flush=True)
        else:
            print("\n  [ERROR] Cannot launch Holographic UI - PySide6 not available", flush=True)

    if args.mic:
        print("\n" + "=" * 60, flush=True)
        print("Starting local mic client...", flush=True)
        print("=" * 60, flush=True)
        mic_proc = subprocess.Popen(
            [_find_python(), "-c", 
             "import sys; import asyncio; "
             "sys.path.insert(0, '.'); "
             "from tests.manual.stream_mic import stream_mic; "
             "asyncio.run(stream_mic('ws://127.0.0.1:8000/ws/transcribe'))"],
            cwd=ROOT,
            stdout=sys.stdout,
            stderr=sys.stderr,
            **child_group,
        )
        time.sleep(2)
        if mic_proc.poll() is None:
            print("  [OK] Mic client started", flush=True)
        else:
            print("  [ERROR] Mic client failed to start", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL SERVERS RUNNING", flush=True)
    print(f"  Backend : {health_url}", flush=True)
    if args.hud and args.hidden:
        print("  HUD     : HIDDEN (say 'Alvin, wake up')", flush=True)
    elif args.hud:
        print("  HUD     : ON (PySide6 Holographic UI)", flush=True)
    else:
        print("  HUD     : ON DEMAND (say 'Alvin, wake up')", flush=True)
    if args.mic:
        print("  Mic     : ON", flush=True)
    else:
        print("  Mic     : OFF (use --mic)", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    try:
        while True:
            if backend_proc.poll() is not None:
                print("ERROR: Backend exited unexpectedly!", flush=True)
                break
            if hud_proc and hud_proc.poll() is not None and args.hud:
                print("ERROR: Holographic UI exited unexpectedly!", flush=True)
                break
            if mic_proc and mic_proc.poll() is not None and args.mic:
                print("ERROR: Mic client exited unexpectedly!", flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    cleanup()


if __name__ == "__main__":
    main()