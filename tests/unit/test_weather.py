"""Unit tests for the MET Norway weather source."""

from __future__ import annotations

import asyncio

import pytest

from app import weather as weather_mod
from app.config import settings
from app.weather import WeatherReport, _describe_symbol, _fetch_sync, fetch_weather


def test_describe_symbol_maps_met_codes() -> None:
    assert _describe_symbol("clearsky_day") == "clear sky"
    assert _describe_symbol("partlycloudy_night") == "partly cloudy"
    assert _describe_symbol("rainandthunder") == "a thunderstorm with rain"
    assert _describe_symbol("weirdcode") == "unsettled weather"


def test_fetch_sync_geocodes_and_reads_current_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_get_json(url: str, timeout: float) -> dict:
        calls.append(url)
        if "geocoding" in url:
            return {
                "results": [
                    {
                        "latitude": 8.4542,
                        "longitude": 124.6319,
                        "name": "Cagayan de Oro",
                    }
                ]
            }
        return {
            "properties": {
                "timeseries": [
                    {
                        "data": {
                            "instant": {"details": {"air_temperature": 30.6}},
                            "next_1_hours": {
                                "summary": {"symbol_code": "partlycloudy_day"}
                            },
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    monkeypatch.setattr(settings, "briefing_weather_location", "Cagayan de Oro")

    report = _fetch_sync()

    assert report == WeatherReport(
        location="Cagayan de Oro",
        temperature_c=30.6,
        condition="partly cloudy",
        apparent_c=None,
    )
    assert "geocoding-api.open-meteo.com" in calls[0]
    assert "api.met.no" in calls[1]
    assert "lat=8.4542" in calls[1]
    assert "lon=124.6319" in calls[1]


def test_fetch_sync_rejects_an_unknown_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weather_mod, "_get_json", lambda url, timeout: {"results": []}
    )
    monkeypatch.setattr(settings, "briefing_weather_location", "Nowhere At All")

    with pytest.raises(RuntimeError, match="could not find weather location"):
        _fetch_sync()


def test_fetch_sync_rejects_a_missing_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_json(url: str, timeout: float) -> dict:
        if "geocoding" in url:
            return {"results": [{"latitude": 1.0, "longitude": 2.0, "name": "X"}]}
        return {"properties": {"timeseries": [{"data": {"instant": {"details": {}}}}]}}

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    monkeypatch.setattr(settings, "briefing_weather_location", "X")

    with pytest.raises(RuntimeError, match="missing the temperature"):
        _fetch_sync()


def test_fetch_sync_rejects_an_empty_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_json(url: str, timeout: float) -> dict:
        if "geocoding" in url:
            return {"results": [{"latitude": 1.0, "longitude": 2.0, "name": "X"}]}
        return {"properties": {"timeseries": []}}

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    monkeypatch.setattr(settings, "briefing_weather_location", "X")

    with pytest.raises(RuntimeError, match="missing the forecast"):
        _fetch_sync()


def test_fetch_weather_returns_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "briefing_weather_enabled", False)

    assert asyncio.run(fetch_weather()) is None
