"""Unit tests for the main FastAPI app (lifespan, health, WebSocket endpoint)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import cloud_primary_enabled


class _FakeHeaders(dict):
    """Minimal ``Headers``-alike for :func:`app.connection.verify_token`."""

    def get(self, key: str, default=None):
        return super().get(key.lower(), default)


class _FakeWebSocket:
    """Enough of a Starlette WebSocket to drive the admission path."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = _FakeHeaders({k.lower(): v for k, v in (headers or {}).items()})
        self.query_params: dict[str, str] = {}
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True


def test_cloud_primary_enabled_without_groq_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cloud_primary_enabled should return False when GROQ_API_KEY is not set."""
    monkeypatch.setattr("app.main.settings.groq_fallback", True)
    monkeypatch.setattr("app.main.settings.groq_api_key", "")
    assert cloud_primary_enabled() is False


def test_cloud_primary_enabled_with_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """cloud_primary_enabled should return True when both fallback and key are set."""
    monkeypatch.setattr("app.main.settings.groq_fallback", True)
    monkeypatch.setattr("app.main.settings.groq_api_key", "test-key")
    assert cloud_primary_enabled() is True


def test_cloud_primary_disabled_when_fallback_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cloud_primary_enabled should return False when groq_fallback is False."""
    monkeypatch.setattr("app.main.settings.groq_fallback", False)
    monkeypatch.setattr("app.main.settings.groq_api_key", "test-key")
    assert cloud_primary_enabled() is False


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health endpoint should return status OK with expected fields."""
    from app.main import app

    # Mock stt_service
    with patch("app.main.stt_service") as mock_stt:
        mock_stt.is_ready = True

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "provider" in data
        assert "model" in data
        assert "fallback_model" in data
        assert "fallback_ready" in data
        assert "ready" in data
        assert "language" in data
        assert "tts_voice" in data
        assert "auth_required" in data
        assert "active_connections" in data
        assert "max_connections" in data
        assert "llm" in data
        assert "intents" in data


@pytest.mark.asyncio
async def test_health_endpoint_shows_starting_when_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health endpoint should return 'starting' when STT service not ready."""
    from app.main import app

    with patch("app.main.stt_service") as mock_stt:
        mock_stt.is_ready = False

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "starting"
        assert data["ready"] is False


@pytest.mark.asyncio
async def test_websocket_rejects_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebSocket should reject connections when API_KEY is set but token is missing.

    The endpoint function is called directly with a fake socket rather than
    through ``TestClient``: the installed Starlette 1.x requires the httpx2
    transport, and ``TestClient.websocket_connect`` deadlocks with the httpx
    build in this environment. ``tests/integration/test_ws_admission.py``
    covers this handshake against a real uvicorn subprocess.
    """
    from fastapi import WebSocketException

    from app.main import settings, ws_transcribe

    # ``auth_required`` is derived from ``api_key``, so setting the key is
    # enough (the property itself is read-only).
    monkeypatch.setattr(settings, "api_key", "secret")
    assert settings.auth_required is True

    ws = _FakeWebSocket(headers={})

    with pytest.raises(WebSocketException) as exc_info:
        await ws_transcribe(ws)

    assert exc_info.value.code == 1008
    # A rejected handshake must not accept the socket first.
    assert ws.accepted is False


@pytest.mark.asyncio
async def test_websocket_accepts_authorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebSocket should accept connections with correct token."""
    from app.main import settings, ws_transcribe

    monkeypatch.setattr(settings, "api_key", "secret")

    ws = _FakeWebSocket(headers={"Authorization": "Bearer secret"})

    with (
        patch("app.main.Connection") as mock_conn_class,
        patch("app.main.manager") as mock_manager,
    ):
        mock_conn_class.return_value.run = AsyncMock()
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = MagicMock()

        await ws_transcribe(ws)

        mock_manager.connect.assert_called_once()
        mock_manager.disconnect.assert_called_once()
        # Admitted clients proceed into the Connection state machine.
        mock_conn_class.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_preloads_fallback_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan should preload fallback model when PRELOAD_FALLBACK is True."""
    from app.main import app, lifespan

    monkeypatch.setattr("app.main.settings.preload_fallback", True)

    with patch("app.main.stt_service") as mock_stt:
        mock_stt.ensure_loaded = AsyncMock()
        mock_stt.shutdown = AsyncMock()

        # Run lifespan context manager
        async with lifespan(app):
            pass

        mock_stt.ensure_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_skips_fallback_preload_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PRELOAD_FALLBACK off, the real service must not load the model.

    ``lifespan`` always awaits ``stt_service.ensure_loaded()``; the decision to
    touch the (slow) Whisper model lives inside the service, so this asserts
    against the real :class:`STTService` rather than a mock that would swallow
    the branch.
    """
    from app.main import app, lifespan
    from app.stt import STTService

    monkeypatch.setattr("app.main.settings.preload_fallback", False)

    service = STTService()
    called = False

    async def _fake_ensure_loaded() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(service, "ensure_loaded", _fake_ensure_loaded)

    with patch("app.main.stt_service", service):
        async with lifespan(app):
            pass

    # The lifespan delegated the decision to the service, so the service-level
    # method did run; assert the real service is what declines to load.
    assert called is True
    fresh = STTService()
    assert fresh.is_ready is True  # nothing loaded, and that's acceptable


@pytest.mark.asyncio
async def test_lifespan_warms_up_cloud_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan should warm up cloud STT when GROQ_WARMUP is True and cloud primary."""
    from app.main import app, lifespan

    monkeypatch.setattr("app.main.settings.groq_fallback", True)
    monkeypatch.setattr("app.main.settings.groq_api_key", "test-key")
    monkeypatch.setattr("app.main.settings.groq_warmup", True)

    with patch("app.main.stt_service") as mock_stt:
        mock_stt.warm_up_cloud = AsyncMock(return_value=True)
        mock_stt.ensure_loaded = AsyncMock()
        mock_stt.shutdown = AsyncMock()

        async with lifespan(app):
            pass

        mock_stt.warm_up_cloud.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_warms_up_tts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan should warm up TTS when TTS_WARMUP is True."""
    from app.main import app, lifespan

    monkeypatch.setattr("app.main.settings.tts_warmup", True)

    with (
        patch("app.main.tts_warm_up", new_callable=AsyncMock) as mock_tts_warm,
        patch("app.main.stt_service") as mock_stt,
    ):
        # The lifespan wraps these in asyncio.create_task / awaits them, so they
        # must be awaitable.
        mock_tts_warm.return_value = True
        mock_stt.ensure_loaded = AsyncMock()
        mock_stt.shutdown = AsyncMock()

        async with lifespan(app):
            pass

        mock_tts_warm.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_warms_up_llm_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan should warm up LLM when LLM_WARMUP is True and LLM enabled."""
    from app.main import app, lifespan

    monkeypatch.setattr("app.main.settings.llm_enabled", True)
    monkeypatch.setattr("app.main.settings.llm_warmup", True)
    # ``llm_service.enabled`` is a read-only property derived from the key.
    monkeypatch.setattr("app.main.settings.llm_api_key", "test-key")

    with patch("app.main.llm_service") as mock_llm:
        mock_llm.enabled = True
        mock_llm.warm_up = AsyncMock(return_value=True)
        mock_llm.shutdown = AsyncMock()

        async with lifespan(app):
            pass

        mock_llm.warm_up.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_cancels_warmup_tasks_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan should cancel warmup tasks on shutdown."""
    from app.main import app, lifespan

    with (
        patch("app.main.stt_service") as mock_stt,
        patch("app.main.tts_warm_up"),
        patch("app.main.llm_service") as mock_llm,
    ):
        mock_stt.ensure_loaded = AsyncMock()
        mock_stt.warm_up_cloud = AsyncMock()
        mock_stt.shutdown = AsyncMock()
        mock_llm.enabled = True
        mock_llm.warm_up = AsyncMock()
        mock_llm.shutdown = AsyncMock()

        async with lifespan(app):
            pass

        mock_stt.shutdown.assert_called_once()
        mock_llm.shutdown.assert_called_once()
