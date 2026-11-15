"""Unit tests for the Google Tasks briefing source."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app import tasks as tasks_mod
from app.config import settings
from app.tasks import _format_due, fetch_open_tasks


def _api_date(days_from_today: int) -> str:
    due = datetime.now().date() + timedelta(days=days_from_today)
    return f"{due.isoformat()}T00:00:00.000Z"


def test_format_due_handles_upcoming_dates() -> None:
    assert _format_due(_api_date(0)) == "today"
    assert _format_due(_api_date(1)) == "tomorrow"
    assert _format_due(_api_date(-1)) == "overdue"


def test_format_due_uses_weekday_within_a_week() -> None:
    expected = (datetime.now().date() + timedelta(days=3)).strftime("%A")
    assert _format_due(_api_date(3)) == expected


def test_format_due_uses_short_date_beyond_a_week() -> None:
    due = datetime.now().date() + timedelta(days=10)
    assert _format_due(_api_date(10)) == f"{due.strftime('%b')} {due.day}"


def test_format_due_handles_missing_and_malformed_values() -> None:
    assert _format_due(None) == ""
    assert _format_due("") == ""
    assert _format_due("not-a-date") == "not-a-date"


def test_is_configured_checks_for_the_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = tmp_path / "google_tasks_token.json"
    monkeypatch.setattr(settings, "briefing_tasks_token_path", str(token))
    assert tasks_mod.is_configured() is False

    token.write_text("{}", encoding="utf-8")
    assert tasks_mod.is_configured() is True


def test_fetch_open_tasks_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "briefing_tasks_enabled", False)

    assert asyncio.run(fetch_open_tasks()) == []


def test_fetch_sync_explains_the_missing_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "google_tasks_token.json"
    monkeypatch.setattr(settings, "briefing_tasks_token_path", str(missing))

    with pytest.raises(RuntimeError, match="google_tasks_auth.py"):
        tasks_mod._fetch_sync(5)
