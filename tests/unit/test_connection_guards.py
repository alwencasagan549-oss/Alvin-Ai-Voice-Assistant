"""Admission control: ``MAX_CONNECTIONS`` enforcement and ``API_KEY`` auth.

These guards run before ``accept()``, so they are tested against a fake socket
that records handshake calls instead of a live server. ``TestClient`` websockets
are unusable here (see ``tests/conftest.py``), and a subprocess per case would
be far slower than the code under test.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import WebSocketException
from starlette.datastructures import Headers, QueryParams

from app.config import settings
from app.connection import ConnectionManager, verify_token

SECRET = "s3cr3t-token"


class FakeWebSocket:
    """Minimal stand-in for ``fastapi.WebSocket`` covering the guard surface."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
        accept_error: Exception | None = None,
    ) -> None:
        self.headers = Headers(headers or {})
        self.query_params = QueryParams(query or {})
        self.accepted = False
        self.accept_error = accept_error
        self.closed: tuple[int, str | None] | None = None

    async def accept(self) -> None:
        if self.accept_error is not None:
            raise self.accept_error
        self.accepted = True

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason)


@pytest.fixture
def secured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", SECRET)


# --------------------------------------------------------------------------- #
# verify_token
# --------------------------------------------------------------------------- #
def test_open_when_no_api_key_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "")
    assert verify_token(FakeWebSocket()) is True


def test_missing_token_is_rejected(secured: None) -> None:
    assert verify_token(FakeWebSocket()) is False


def test_wrong_token_is_rejected(secured: None) -> None:
    ws = FakeWebSocket(headers={"authorization": "Bearer nope"})
    assert verify_token(ws) is False


def test_correct_bearer_header_is_accepted(secured: None) -> None:
    ws = FakeWebSocket(headers={"Authorization": f"Bearer {SECRET}"})
    assert verify_token(ws) is True


def test_bearer_scheme_is_case_insensitive(secured: None) -> None:
    ws = FakeWebSocket(headers={"authorization": f"bearer {SECRET}"})
    assert verify_token(ws) is True


def test_x_api_key_header_is_accepted(secured: None) -> None:
    ws = FakeWebSocket(headers={"X-API-Key": SECRET})
    assert verify_token(ws) is True


def test_query_param_is_accepted(secured: None) -> None:
    ws = FakeWebSocket(query={"token": SECRET})
    assert verify_token(ws) is True


def test_non_bearer_authorization_is_not_a_token(secured: None) -> None:
    ws = FakeWebSocket(headers={"authorization": f"Basic {SECRET}"})
    assert verify_token(ws) is False


def test_non_ascii_token_does_not_raise(secured: None) -> None:
    ws = FakeWebSocket(query={"token": "tökén-\u00e9"})
    assert verify_token(ws) is False


def test_longer_token_does_not_raise(secured: None) -> None:
    ws = FakeWebSocket(query={"token": SECRET + "-extra"})
    assert verify_token(ws) is False


# --------------------------------------------------------------------------- #
# ConnectionManager
# --------------------------------------------------------------------------- #
def test_connect_accepts_and_tracks() -> None:
    manager = ConnectionManager()
    ws = FakeWebSocket()

    asyncio.run(manager.connect(ws, max_connections=2))

    assert ws.accepted is True
    assert manager.active_count == 1

    manager.disconnect(ws)
    assert manager.active_count == 0


def test_connect_denies_past_the_limit() -> None:
    manager = ConnectionManager()
    first, second, third = FakeWebSocket(), FakeWebSocket(), FakeWebSocket()
    asyncio.run(manager.connect(first, max_connections=2))
    asyncio.run(manager.connect(second, max_connections=2))

    with pytest.raises(WebSocketException) as excinfo:
        asyncio.run(manager.connect(third, max_connections=2))

    assert excinfo.value.code == 1008
    assert third.accepted is False
    assert manager.active_count == 2


def test_slot_is_freed_after_disconnect() -> None:
    manager = ConnectionManager()
    first, second = FakeWebSocket(), FakeWebSocket()
    asyncio.run(manager.connect(first, max_connections=1))

    with pytest.raises(WebSocketException):
        asyncio.run(manager.connect(second, max_connections=1))

    manager.disconnect(first)
    asyncio.run(manager.connect(second, max_connections=1))
    assert second.accepted is True
    assert manager.active_count == 1


def test_failed_handshake_releases_the_slot() -> None:
    manager = ConnectionManager()
    broken = FakeWebSocket(accept_error=RuntimeError("handshake aborted"))

    with pytest.raises(RuntimeError):
        asyncio.run(manager.connect(broken, max_connections=1))

    assert manager.active_count == 0
    asyncio.run(manager.connect(FakeWebSocket(), max_connections=1))
    assert manager.active_count == 1


def test_disconnect_is_idempotent() -> None:
    manager = ConnectionManager()
    ws = FakeWebSocket()
    asyncio.run(manager.connect(ws, max_connections=1))

    manager.disconnect(ws)
    manager.disconnect(ws)

    assert manager.active_count == 0
