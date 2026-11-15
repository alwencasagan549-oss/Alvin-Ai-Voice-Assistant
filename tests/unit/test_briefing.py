"""Unit tests for the wake-up briefing composer.

The source fetchers are monkeypatched, so these tests never touch the network,
IMAP, or Google's API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app import briefing as briefing_mod
from app.briefing import (
    BriefingData,
    BriefingService,
    compose_template,
    greeting_for,
)
from app.config import settings
from app.email_source import EmailItem
from app.llm import llm_service
from app.tasks import TaskItem
from app.weather import WeatherReport


def _set_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    weather: bool,
    tasks: bool,
    email: bool,
) -> None:
    monkeypatch.setattr(settings, "briefing_weather_enabled", weather)
    monkeypatch.setattr(settings, "briefing_tasks_enabled", tasks)
    monkeypatch.setattr(settings, "briefing_email_enabled", email)


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (5, "Good morning"),
        (11, "Good morning"),
        (12, "Good afternoon"),
        (17, "Good afternoon"),
        (18, "Good evening"),
        (4, "Good evening"),
    ],
)
def test_greeting_tracks_the_time_of_day(hour: int, expected: str) -> None:
    text = greeting_for(datetime(2026, 11, 15, hour, 30))
    assert text.startswith(expected)
    assert "Today is Sunday, November 15" in text


def test_greeting_uses_a_twelve_hour_clock() -> None:
    assert "2:35 PM" in greeting_for(datetime(2026, 11, 15, 14, 35))
    assert "9:05 AM" in greeting_for(datetime(2026, 11, 15, 9, 5))
    assert "12:00 AM" in greeting_for(datetime(2026, 11, 15, 0, 0))


def test_compose_template_reads_every_enabled_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=True, tasks=True, email=True)
    data = BriefingData(
        greeting="Good afternoon, sir.",
        weather=WeatherReport(
            location="Cagayan de Oro",
            temperature_c=30.6,
            condition="partly cloudy",
        ),
        tasks=[
            TaskItem(title="Submit the quarterly report"),
            TaskItem(title="Call the bank"),
        ],
        emails=[
            EmailItem(sender="Alice", subject="Project update"),
            EmailItem(sender="Bob", subject="Lunch on Friday"),
        ],
    )

    text = compose_template(data)

    assert text.startswith("Good afternoon, sir.")
    assert "The weather in Cagayan de Oro is 31 degrees Celsius, partly cloudy." in text
    assert "You have 2 open tasks: Submit the quarterly report; Call the bank." in text
    assert "Your latest email is from Alice, about Project update." in text
    assert "You have 1 more unread message." in text
    assert text.endswith("What's on your mind today, sir?")


def test_compose_template_degrades_per_failed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=True, tasks=True, email=True)
    data = BriefingData(
        greeting="Good afternoon, sir.",
        errors=["weather", "tasks", "email"],
    )

    text = compose_template(data)

    assert "I could not check the weather right now, sir." in text
    assert "I could not reach your task list right now, sir." in text
    assert "I could not reach your inbox right now, sir." in text
    assert text.endswith("What's on your mind today, sir?")


def test_compose_template_reports_empty_tasks_and_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=False, tasks=True, email=True)
    data = BriefingData(greeting="Good afternoon, sir.")

    text = compose_template(data)

    assert "You don't have any available tasks right now, sir." in text
    assert "You don't have any new email right now, sir." in text


def test_compose_template_ignores_disabled_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=False, tasks=False, email=False)
    data = BriefingData(
        greeting="Good afternoon, sir.",
        weather=WeatherReport(
            location="Cagayan de Oro", temperature_c=30.0, condition="clear sky"
        ),
        tasks=[TaskItem(title="A task that should stay silent")],
        emails=[EmailItem(sender="Alice", subject="An email that should stay silent")],
    )

    assert (
        compose_template(data)
        == "Good afternoon, sir. What's on your mind today, sir?"
    )


def test_build_gathers_enabled_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_sources(monkeypatch, weather=True, tasks=True, email=True)
    monkeypatch.setattr(settings, "briefing_llm_polish", False)

    async def fake_weather():
        return WeatherReport(
            location="Cagayan de Oro", temperature_c=28.0, condition="light rain"
        )

    async def fake_tasks():
        return [TaskItem(title="Water the plants")]

    async def fake_email():
        return [EmailItem(sender="Alice", subject="Hello")]

    monkeypatch.setattr(briefing_mod, "fetch_weather", fake_weather)
    monkeypatch.setattr(briefing_mod, "fetch_open_tasks", fake_tasks)
    monkeypatch.setattr(briefing_mod, "fetch_latest_emails", fake_email)
    monkeypatch.setattr(briefing_mod, "tasks_configured", lambda: True)
    monkeypatch.setattr(briefing_mod, "email_configured", lambda: True)

    data = asyncio.run(BriefingService().build())

    assert data.weather is not None
    assert data.weather.condition == "light rain"
    assert [item.title for item in data.tasks] == ["Water the plants"]
    assert [item.subject for item in data.emails] == ["Hello"]
    assert data.errors == []
    assert "light rain" in data.spoken_text


def test_build_keeps_going_when_one_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=True, tasks=False, email=False)
    monkeypatch.setattr(settings, "briefing_llm_polish", False)

    async def failing_weather():
        raise RuntimeError("provider down")

    monkeypatch.setattr(briefing_mod, "fetch_weather", failing_weather)

    data = asyncio.run(BriefingService().build())

    assert data.errors == ["weather"]
    assert "I could not check the weather right now, sir." in data.spoken_text


def test_build_reports_unconfigured_tasks_and_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=False, tasks=True, email=True)
    monkeypatch.setattr(settings, "briefing_llm_polish", False)
    monkeypatch.setattr(briefing_mod, "tasks_configured", lambda: False)
    monkeypatch.setattr(briefing_mod, "email_configured", lambda: False)

    data = asyncio.run(BriefingService().build())

    assert data.errors == ["tasks_unconfigured", "email_unconfigured"]
    assert "Your task list is not connected yet, sir." in data.spoken_text
    assert "Your email is not connected yet, sir." in data.spoken_text


def test_weather_results_are_cached_between_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=True, tasks=False, email=False)
    monkeypatch.setattr(settings, "briefing_llm_polish", False)
    calls: list[bool] = []

    async def fake_weather():
        calls.append(True)
        return WeatherReport(
            location="Cagayan de Oro", temperature_c=28.0, condition="clear sky"
        )

    monkeypatch.setattr(briefing_mod, "fetch_weather", fake_weather)
    service = BriefingService()

    asyncio.run(service.build())
    asyncio.run(service.build())

    assert len(calls) == 1


def test_llm_polish_rewrites_the_template(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_sources(monkeypatch, weather=False, tasks=False, email=False)
    monkeypatch.setattr(settings, "briefing_llm_polish", True)
    monkeypatch.setattr(type(llm_service), "enabled", property(lambda self: True))
    captured: dict = {}

    async def fake_generate(text, *, max_tokens=None, temperature=None, messages=None):
        captured["text"] = text
        captured["max_tokens"] = max_tokens
        captured["messages"] = messages
        return "Good afternoon, sir. All systems are ready."

    monkeypatch.setattr(llm_service, "generate_response", fake_generate)

    data = asyncio.run(BriefingService().build())

    assert data.spoken_text == "Good afternoon, sir. All systems are ready."
    assert captured["max_tokens"] == settings.briefing_llm_max_tokens
    assert captured["messages"][0]["content"] == briefing_mod.BRIEFING_POLISH_PROMPT
    assert captured["messages"][1]["content"] == (
        data.greeting + " What's on your mind today, sir?"
    )


def test_llm_polish_failure_falls_back_to_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, weather=False, tasks=False, email=False)
    monkeypatch.setattr(settings, "briefing_llm_polish", True)
    monkeypatch.setattr(type(llm_service), "enabled", property(lambda self: True))

    async def failing_generate(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(llm_service, "generate_response", failing_generate)

    data = asyncio.run(BriefingService().build())

    assert data.spoken_text.endswith("What's on your mind today, sir?")


def test_to_event_carries_structured_sources() -> None:
    data = BriefingData(
        greeting="Good afternoon, sir.",
        weather=WeatherReport(
            location="Cagayan de Oro", temperature_c=30.0, condition="clear sky"
        ),
        tasks=[TaskItem(title="Task")],
        emails=[EmailItem(sender="Alice", subject="Subject")],
        errors=["email"],
    )

    event = data.to_event()

    assert event["type"] == "briefing"
    assert event["greeting"] == "Good afternoon, sir."
    assert event["weather"] == {
        "location": "Cagayan de Oro",
        "temperature_c": 30.0,
        "condition": "clear sky",
        "apparent_c": None,
    }
    assert event["tasks"] == [{"title": "Task", "due": "", "notes": ""}]
    assert event["emails"] == [
        {"sender": "Alice", "subject": "Subject", "date": "", "unread": True}
    ]
    assert event["errors"] == ["email"]
