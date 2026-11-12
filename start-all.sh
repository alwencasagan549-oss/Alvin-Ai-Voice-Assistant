#!/bin/bash
# Start both backend and frontend with a single command
# Usage: ./start-all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting Alvin AI Dashboard ==="

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    echo ""
    echo "Shutting down..."
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_in_use() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$1 )" | tail -n +2 | grep -q .
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(.25); code=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(0 if code == 0 else 1)' "$1"
    elif command -v python >/dev/null 2>&1; then
        python -c 'import socket,sys; s=socket.socket(); s.settimeout(.25); code=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(0 if code == 0 else 1)' "$1"
    else
        return 1
    fi
}

for port in 8000 5173; do
    if port_in_use "$port"; then
        echo "Port $port is already in use. Close its owner and try again."
        exit 1
    fi
done

# Bug 12: all start-all.{sh,bat,ps1} behave the same -- exec run-all.py, which
# parses HOST from .env (127.0.0.1 fallback) and owns process lifecycle.
# Usage: ./start-all.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick the right venv interpreter for the platform.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="$(command -v python3)"
fi

# Resolve HOST from .env (with 127.0.0.1 fallback) so a non-loopback bind is
# surfaced to run-all.py consistently across all launchers.
HOST="127.0.0.1"
if [ -f ".env" ]; then
    HOST="$(grep -E '^[[:space:]]*HOST=' .env | head -n1 | sed 's/^[^=]*=//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//')"
    if [ -z "$HOST" ]; then HOST="127.0.0.1"; fi
fi
export HOST

exec "$PYTHON" -B run-all.py
