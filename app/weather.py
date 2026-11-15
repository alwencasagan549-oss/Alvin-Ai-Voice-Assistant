"""Current weather for the wake-up briefing.

The place name is resolved with Open-Meteo's geocoding endpoint (free, no API
key and no signup). Current conditions come from MET Norway's Locationforecast
API, which is also free and keyless. Only the current temperature and the
next-hour condition symbol are read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .config import settings

log = logging.getLogger("alvin.weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

# MET Norway condition symbols, minus the _day/_night/_polartwilight suffix.
_MET_SYMBOLS = {
    "clearsky": "clear sky",
    "fair": "mainly clear",
    "partlycloudy": "partly cloudy",
    "cloudy": "overcast",
    "fog": "foggy",
    "lightrain": "light rain",
    "lightrainshowers": "light rain showers",
    "rain": "rain",
    "rainshowers": "rain showers",
    "heavyrain": "heavy rain",
    "heavyrainshowers": "heavy rain showers",
    "lightsleet": "light sleet",
    "lightsleetshowers": "light sleet showers",
    "sleet": "sleet",
    "sleetshowers": "sleet showers",
    "heavysleet": "heavy sleet",
    "heavysleetshowers": "heavy sleet showers",
    "lightsnow": "light snow",
    "lightsnowshowers": "light snow showers",
    "snow": "snow",
    "snowshowers": "snow showers",
    "heavysnow": "heavy snow",
    "heavysnowshowers": "heavy snow showers",
    "rainandthunder": "a thunderstorm with rain",
    "lightrainandthunder": "a thunderstorm with light rain",
    "heavyrainandthunder": "a thunderstorm with heavy rain",
    "rainshowersandthunder": "a thunderstorm with rain showers",
    "sleetandthunder": "a thunderstorm with sleet",
    "snowandthunder": "a thunderstorm with snow",
}


@dataclass(frozen=True)
class WeatherReport:
    """One current-conditions reading, already shaped for speech."""

    location: str
    temperature_c: float
    condition: str
    apparent_c: float | None = None


def _describe_symbol(symbol: str) -> str:
    key = (symbol or "").strip().lower()
    for suffix in ("_polartwilight", "_day", "_night"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return _MET_SYMBOLS.get(key, "unsettled weather")


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AlvinVoiceAssistant/1.1 (personal use)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _geocode(location: str, timeout: float) -> tuple[float, float, str]:
    query = urllib.parse.urlencode(
        {"name": location, "count": 1, "language": "en", "format": "json"}
    )
    data = _get_json(f"{_GEOCODE_URL}?{query}", timeout)
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"could not find weather location {location!r}")
    place = results[0]
    try:
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("weather geocoding response was malformed") from exc
    return latitude, longitude, str(place.get("name") or location)


def _fetch_sync() -> WeatherReport:
    location = settings.briefing_weather_location.strip()
    if not location:
        raise RuntimeError("BRIEFING_WEATHER_LOCATION is empty")

    timeout = settings.briefing_weather_timeout_sec
    latitude, longitude, label = _geocode(location, timeout)
    query = urllib.parse.urlencode({"lat": latitude, "lon": longitude})
    data = _get_json(f"{_FORECAST_URL}?{query}", timeout)

    properties = data.get("properties") or {}
    timeseries = properties.get("timeseries") or []
    if not timeseries:
        raise RuntimeError("weather response is missing the forecast")
    entry = timeseries[0] if isinstance(timeseries[0], dict) else {}
    current = entry.get("data") or {}
    details = ((current.get("instant") or {}).get("details") or {})

    temperature = details.get("air_temperature")
    if temperature is None:
        raise RuntimeError("weather response is missing the temperature")
    try:
        temperature_c = float(temperature)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("weather response was malformed") from exc

    summary = ((current.get("next_1_hours") or {}).get("summary") or {})
    symbol = summary.get("symbol_code") or ""
    if not symbol:
        summary = ((current.get("next_6_hours") or {}).get("summary") or {})
        symbol = summary.get("symbol_code") or ""

    return WeatherReport(
        location=label,
        temperature_c=temperature_c,
        condition=_describe_symbol(symbol),
    )


async def fetch_weather() -> WeatherReport | None:
    """Return the current weather for the configured location.

    Returns:
        The report, or ``None`` when weather is disabled.

    Raises:
        RuntimeError: when the location cannot be resolved or the response is
            malformed.
        asyncio.TimeoutError: when the request exceeds the configured timeout.
    """
    if not settings.briefing_weather_enabled:
        return None
    return await asyncio.wait_for(
        asyncio.to_thread(_fetch_sync),
        timeout=settings.briefing_weather_timeout_sec,
    )
