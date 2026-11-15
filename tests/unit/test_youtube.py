"""Ad-free YouTube stream resolution (:mod:`app.youtube`)."""

from __future__ import annotations

import asyncio

import pytest

from app import youtube


@pytest.fixture(autouse=True)
def _clear_stream_cache():
    youtube._STREAM_CACHE.clear()
    yield
    youtube._STREAM_CACHE.clear()


def test_resolve_stream_url_caches_positive_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_resolve(video_id: str, audio_only: bool) -> str:
        calls.append((video_id, audio_only))
        return "https://example.com/direct"

    monkeypatch.setattr(youtube, "_resolve_stream_sync", fake_resolve)

    first = asyncio.run(youtube.resolve_stream_url("abc123", audio_only=True))
    second = asyncio.run(youtube.resolve_stream_url("abc123", audio_only=True))

    assert first == second == "https://example.com/direct"
    assert calls == [("abc123", True)], "a cached URL must not be resolved twice"


def test_resolve_stream_url_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_video_id: str, _audio_only: bool) -> str:
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(youtube, "_resolve_stream_sync", boom)
    assert asyncio.run(youtube.resolve_stream_url("abc123")) is None


def test_resolve_stream_url_requires_video_id() -> None:
    assert asyncio.run(youtube.resolve_stream_url("")) is None


def test_failed_resolution_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def flaky(video_id: str, _audio_only: bool) -> str | None:
        calls.append(video_id)
        return None

    monkeypatch.setattr(youtube, "_resolve_stream_sync", flaky)

    asyncio.run(youtube.resolve_stream_url("abc123"))
    asyncio.run(youtube.resolve_stream_url("abc123"))

    assert calls == ["abc123", "abc123"], "a miss must be retried next time"
