"""Run both the Alvin backend and the dashboard frontend with a single command.

Usage:
    python run-all.py

This starts:
  1. The FastAPI STT/TTS/LLM backend on http://localhost:8000
  2. The React + Vite dashboard on http://localhost:5173

Press Ctrl+C to stop both servers cleanly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import json

from app.config import settings

ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(ROOT, "alvin-ai-dashboard")
BACKEND_PORT = 8000
FRONTEND_PORT = 5173


def _ports_are_available(backend_host: str, frontend_host: str) -> bool:
    """Refuse occupied ports instead of terminating processes we do not own."""
    import socket

    for port, host, label in (
        (BACKEND_PORT, backend_host, "backend"),
        (FRONTEND_PORT, frontend_host, "frontend"),
    ):
        try:
            addresses = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
            )
        except OSError:
            _print_err(f"Cannot resolve bind host {host!r} for the {label}.")
            return False
        available = False
        for family, socktype, proto, _, address in addresses:
            test_sock = socket.socket(family, socktype, proto)
            try:
                test_sock.bind(address)
                available = True
                break
            except OSError:
                continue
            finally:
                test_sock.close()
        if not available:
            _print_err(f"Port {port} ({label}) is already in use; close its owner and retry.")
            return False
    return True


def _stop_process_tree(proc: subprocess.Popen | None) -> None:
    """Stop only a child process tree started by this launcher."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def _wait_for_url(url: str, timeout: float = 60, label: str = "service", require_ready: bool = False) -> bool:
    """Wait until an HTTP endpoint responds, polling every 0.5s.

    When ``require_ready`` is True, the endpoint must return a JSON body with
    ``ready`` set to true (used for the backend's ``/health`` endpoint).
    """
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


def _url_host(host: str, *, health_probe: bool = False) -> str:
    """Format a bind address for a local HTTP health check or display URL."""
    if health_probe and host in {"0.0.0.0", ""}:
        host = "127.0.0.1"
    elif health_probe and host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _find_npm() -> str:
    """Find npm executable path."""
    # Check common Windows locations
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "nodejs", "npm.cmd"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "nodejs", "npm.cmd"),
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fall back to PATH lookup
    return "npm"


def _find_python() -> str:
    """Find Python executable path, preferring the venv."""
    venv_python = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python
    # Try venv/bin for Unix-like
    venv_python = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable


def _print_ok(msg: str) -> None:
    """Print a success message without Unicode characters for Windows compat."""
    print(f"  [OK] {msg}", flush=True)


def _print_err(msg: str) -> None:
    """Print an error message without Unicode characters."""
    print(f"  [ERROR] {msg}", flush=True)


def _ensure_frontend_deps() -> None:
    """Run npm install if node_modules is missing."""
    node_modules = os.path.join(DASHBOARD_DIR, "node_modules")
    if not os.path.isdir(node_modules):
        print("Installing frontend dependencies (npm install)...", flush=True)
        npm_cmd = _find_npm()
        result = subprocess.run(
            [npm_cmd, "install"],
            cwd=DASHBOARD_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            _print_err("npm install failed")
            sys.exit(1)
        _print_ok("Frontend dependencies installed")


def main() -> None:
    backend_proc = None
    frontend_proc = None
    cleanup_started = False

    def cleanup(*_, exit_code=0):
        nonlocal cleanup_started
        if cleanup_started:
            raise SystemExit(exit_code)
        cleanup_started = True
        print("\n\nShutting down servers...", flush=True)
        _stop_process_tree(frontend_proc)
        _stop_process_tree(backend_proc)
        print("Done.", flush=True)
        raise SystemExit(exit_code)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    bind_host = settings.host
    # Keep the browser app loopback-only: VITE_API_KEY is shipped to the page
    # when configured and must not be exposed to other machines on the LAN.
    frontend_host = "127.0.0.1"
    exposed_hosts = {bind_host} - {"127.0.0.1", "localhost", "::1"}
    if exposed_hosts and not settings.api_key:
        _print_err(
            "A non-loopback bind host requires API_KEY. Set a key or bind to 127.0.0.1."
        )
        raise SystemExit(1)
    if not _ports_are_available(bind_host, frontend_host):
        raise SystemExit(1)

    # Install frontend deps if needed (only on first run)
    _ensure_frontend_deps()

    # ---- Start backend ----
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

    # Wait for backend health
    health_url = f"http://{_url_host(bind_host, health_probe=True)}:{BACKEND_PORT}/health"
    if not _wait_for_url(health_url, timeout=120, label="backend", require_ready=True):
        _print_err("Backend failed to start. Check the error messages above.")
        cleanup(exit_code=1)
    _print_ok(f"Backend ready at http://localhost:{BACKEND_PORT}")

    # ---- Start frontend ----
    print("\n" + "=" * 60, flush=True)
    print("Starting Alvin dashboard (React + Vite)...", flush=True)
    print("=" * 60, flush=True)
    npm_cmd = _find_npm()
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--port", str(FRONTEND_PORT), "--host", frontend_host],
        cwd=DASHBOARD_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        **child_group,
    )

    # Wait for frontend
    frontend_url = f"http://{_url_host(frontend_host, health_probe=True)}:{FRONTEND_PORT}"
    if not _wait_for_url(frontend_url, timeout=30, label="frontend"):
        _print_err("Frontend failed to start. Check the error messages above.")
        cleanup(exit_code=1)
    _print_ok(f"Frontend ready at {frontend_url}")

    print("\n" + "=" * 60, flush=True)
    print("ALL SERVERS RUNNING", flush=True)
    print(f"  Backend : {health_url}", flush=True)
    print(f"  Frontend: {frontend_url}", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    # Keep the parent process alive until interrupted
    try:
        while True:
            # Check if either process died unexpectedly
            if backend_proc.poll() is not None:
                _print_err("Backend exited unexpectedly!")
                break
            if frontend_proc.poll() is not None:
                _print_err("Frontend exited unexpectedly!")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
    cleanup(exit_code=1)


if __name__ == "__main__":
    main()
