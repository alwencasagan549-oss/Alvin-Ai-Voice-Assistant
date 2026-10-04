"""Admission control against a live server: ``API_KEY`` auth and ``MAX_CONNECTIONS``.

Boots a second uvicorn subprocess with a key and a cap of 2, so both guards are
exercised over a real handshake instead of a fake socket. A rejected client must
fail during the handshake (``InvalidStatus``), never receive an accepted socket
that closes a moment later.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.request

import pytest
from conftest import ROOT, _free_port, _wait_ready
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

API_KEY = "test-admission-key"
MAX_CONNECTIONS = 2


@pytest.fixture(scope="module")
def secured_server():
    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "API_KEY": API_KEY,
            "MAX_CONNECTIONS": str(MAX_CONNECTIONS),
            "FALLBACK_MODEL": "tiny.en",
            "GROQ_FALLBACK": "false",
            "GROQ_WARMUP": "false",
            "TTS_WARMUP": "false",
        }
    )
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
        yield f"ws://127.0.0.1:{port}/ws/transcribe", f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _health(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
        return json.loads(resp.read().decode())


def _run(coro_fn):
    return asyncio.run(coro_fn())


def _assert_refused(uri: str, **kwargs) -> None:
    async def run() -> None:
        with pytest.raises(InvalidStatus):
            async with connect(uri, **kwargs):
                pass

    _run(run)


def test_health_advertises_the_guards(secured_server) -> None:
    _, base = secured_server
    data = _health(base)
    assert data["auth_required"] is True
    assert data["max_connections"] == MAX_CONNECTIONS
    assert data["active_connections"] == 0


def test_missing_token_is_refused(secured_server) -> None:
    uri, _ = secured_server
    _assert_refused(uri)


def test_wrong_bearer_token_is_refused(secured_server) -> None:
    uri, _ = secured_server
    _assert_refused(uri, additional_headers={"Authorization": "Bearer wrong"})


def test_wrong_query_token_is_refused(secured_server) -> None:
    uri, _ = secured_server
    _assert_refused(f"{uri}?token=wrong")


def test_bearer_header_is_accepted(secured_server) -> None:
    uri, _ = secured_server

    async def run() -> None:
        async with connect(
            uri, additional_headers={"Authorization": f"Bearer {API_KEY}"}
        ):
            pass

    _run(run)


def test_x_api_key_header_is_accepted(secured_server) -> None:
    uri, _ = secured_server

    async def run() -> None:
        async with connect(uri, additional_headers={"X-API-Key": API_KEY}):
            pass

    _run(run)


def test_query_token_is_accepted(secured_server) -> None:
    uri, _ = secured_server

    async def run() -> None:
        async with connect(f"{uri}?token={API_KEY}"):
            pass

    _run(run)


def test_connection_cap_is_enforced_and_released(secured_server) -> None:
    uri, base = secured_server

    async def run() -> None:
        held = [
            await connect(uri, additional_headers={"X-API-Key": API_KEY})
            for _ in range(MAX_CONNECTIONS)
        ]
        try:
            assert _health(base)["active_connections"] == MAX_CONNECTIONS
            # The next one must be turned away at the handshake.
            with pytest.raises(InvalidStatus):
                await connect(uri, additional_headers={"X-API-Key": API_KEY})
        finally:
            for ws in held:
                await ws.close()

    _run(run)

    # Slots must come back: a leaked registration would wedge the service.
    async def drain() -> None:
        for _ in range(50):
            if _health(base)["active_connections"] == 0:
                return
            await asyncio.sleep(0.1)
        raise AssertionError("connections never released")

    _run(drain)

    async def reuse() -> None:
        async with connect(uri, additional_headers={"X-API-Key": API_KEY}):
            pass

    _run(reuse)
