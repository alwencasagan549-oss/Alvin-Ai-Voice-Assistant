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

# Pick the right venv interpreter for the platform
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="$(command -v python3)"
fi

# Start backend
echo "Starting backend..."
"$PYTHON" -m uvicorn app.main:app \
    --host 127.0.0.1 --port 8000 --ws auto --log-level info &
BACKEND_PID=$!

# Wait for backend
echo "Waiting for backend..."
backend_ready=false
for i in $(seq 1 60); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        break
    fi
    if health_json="$(curl -fsS http://127.0.0.1:8000/health 2>/dev/null)" &&
        printf '%s' "$health_json" | "$PYTHON" -c \
            'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ready") else 1)' \
            >/dev/null 2>&1; then
        backend_ready=true
        echo "  [OK] Backend ready (http://localhost:8000)"
        break
    fi
    sleep 1
done
if [ "$backend_ready" != true ]; then
    echo "Backend did not become ready. Check the error messages above."
    exit 1
fi

# Start frontend
echo "Starting frontend..."
cd alvin-ai-dashboard
./node_modules/.bin/vite --port 5173 &
FRONTEND_PID=$!
cd ..

# Wait for frontend
echo "Waiting for frontend..."
frontend_ready=false
for i in $(seq 1 15); do
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        break
    fi
    if curl -fsS http://127.0.0.1:5173 > /dev/null 2>&1; then
        frontend_ready=true
        echo "  [OK] Frontend ready (http://localhost:5173)"
        break
    fi
    sleep 1
done
if [ "$frontend_ready" != true ]; then
    echo "Frontend did not become ready. Check the error messages above."
    exit 1
fi

echo ""
echo "=== ALL SERVERS RUNNING ==="
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  Press Ctrl+C to stop."

# Wait for processes
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 1
done
echo "A server exited unexpectedly."
exit 1
