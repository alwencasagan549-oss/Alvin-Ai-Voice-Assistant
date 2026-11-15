"""Wake-up briefing: greeting, weather, tasks, and email.

The briefing is deliberately assembled outside the LLM. Each source is fetched
independently, the spoken text is templated, and an optional LLM pass can
rewrite it for variety. This keeps the feature working with small/free models,
keeps the order stable, and lets one failed source degrade gracefully without
losing the rest of the briefing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import settings
from .email_source import EmailItem, fetch_latest_emails
from .email_source import is_configured as email_configured
from .llm import llm_service
from .tasks import TaskItem, fetch_open_tasks
from .tasks import is_configured as tasks_configured
from .weather import WeatherReport, fetch_weather

log = logging.getLogger("alvin.briefing")

BRIEFING_POLISH_PROMPT = (
    "You are Alvin, a concise voice assistant. Rewrite the briefing below as "
    "one natural spoken paragraph. Preserve every fact, name, and number. Do "
    "not add or invent facts. Do not use bullets, markdown, emojis, or stage "
    "directions. Keep it under 110 words and suitable for text-to-speech."
)


@dataclass
class BriefingData:
    """Everything the briefing gathered, plus the final spoken text."""

    greeting: str
    weather: WeatherReport | None = None
    tasks: list[TaskItem] = field(default_factory=list)
    emails: list[EmailItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    spoken_text: str = ""

    def to_event(self) -> dict:
        """Structured event for the HUD/dashboard, plus the text being spoken."""
        return {
            "type": "briefing",
            "text": self.spoken_text,
            "greeting": self.greeting,
            "weather": asdict(self.weather) if self.weather else None,
            "tasks": [asdict(item) for item in self.tasks],
            "emails": [asdict(item) for item in self.emails],
            "errors": list(self.errors),
        }


def greeting_for(now: datetime | None = None) -> str:
    """Return a time-aware greeting with today's date and time."""
    current = now or datetime.now()
    hour = current.hour
    if 5 <= hour < 12:
        part = "Good morning"
    elif 12 <= hour < 18:
        part = "Good afternoon"
    else:
        part = "Good evening"
    clock = current.strftime("%I:%M %p").lstrip("0")
    return (
        f"{part}, sir. Today is {current.strftime('%A, %B')} "
        f"{current.day}, {clock}."
    )


def _inline(value: str, limit: int = 140) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text


def compose_template(data: BriefingData) -> str:
    """Build the spoken briefing without an LLM."""
    parts = [data.greeting]

    if settings.briefing_weather_enabled:
        if data.weather:
            weather = data.weather
            temperature = round(weather.temperature_c)
            parts.append(
                f"The weather in {weather.location} is {temperature} degrees "
                f"Celsius, {weather.condition}."
            )
        elif "weather" in data.errors:
            parts.append("I could not check the weather right now, sir.")

    if settings.briefing_tasks_enabled:
        if data.tasks:
            titles = "; ".join(_inline(item.title, 90) for item in data.tasks)
            count = len(data.tasks)
            noun = "task" if count == 1 else "tasks"
            parts.append(f"You have {count} open {noun}: {titles}.")
        elif "tasks_unconfigured" in data.errors:
            parts.append("Your task list is not connected yet, sir.")
        elif "tasks" in data.errors:
            parts.append("I could not reach your task list right now, sir.")
        else:
            parts.append("You don't have any available tasks right now, sir.")

    if settings.briefing_email_enabled:
        if data.emails:
            first = data.emails[0]
            sender = _inline(first.sender, 60) or "someone"
            subject = _inline(first.subject, 110) or "no subject"
            parts.append(f"Your latest email is from {sender}, about {subject}.")
            remaining = len(data.emails) - 1
            if remaining > 0:
                noun = "message" if remaining == 1 else "messages"
                parts.append(f"You have {remaining} more unread {noun}.")
        elif "email_unconfigured" in data.errors:
            parts.append("Your email is not connected yet, sir.")
        elif "email" in data.errors:
            parts.append("I could not reach your inbox right now, sir.")
        else:
            parts.append("You don't have any new email right now, sir.")

    parts.append("What's on your mind today, sir?")
    return " ".join(parts)


class BriefingService:
    """Fetch the briefing sources and compose the spoken result."""

    def __init__(self) -> None:
        self._weather_cache: tuple[float, WeatherReport] | None = None
        self._email_cache: tuple[float, list[EmailItem]] | None = None

    async def warm_up(self) -> None:
        """Prefetch the network-backed sources so the first wake-up is fast."""
        if not settings.briefing_enabled:
            return
        pending = []
        if settings.briefing_weather_enabled:
            pending.append(self._weather())
        if settings.briefing_email_enabled and email_configured():
            pending.append(self._emails())
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)

    async def build(self) -> BriefingData:
        """Collect every enabled source and return the spoken briefing."""
        data = BriefingData(greeting=greeting_for())

        sources: list[tuple[str, object]] = []
        if settings.briefing_weather_enabled:
            sources.append(("weather", self._weather()))
        if settings.briefing_tasks_enabled:
            if tasks_configured():
                sources.append(("tasks", self._tasks()))
            else:
                data.errors.append("tasks_unconfigured")
        if settings.briefing_email_enabled:
            if email_configured():
                sources.append(("email", self._emails()))
            else:
                data.errors.append("email_unconfigured")

        if sources:
            results = await asyncio.gather(
                *(coro for _, coro in sources),
                return_exceptions=True,
            )
            for (label, _), result in zip(sources, results):
                if isinstance(result, BaseException):
                    log.warning("briefing source %s failed: %s", label, result)
                    data.errors.append(label)
                elif label == "weather":
                    data.weather = result  # type: ignore[assignment]
                elif label == "tasks":
                    data.tasks = result  # type: ignore[assignment]
                else:
                    data.emails = result  # type: ignore[assignment]

        data.spoken_text = compose_template(data)
        if settings.briefing_llm_polish and llm_service.enabled:
            data.spoken_text = await self._polish(data.spoken_text)
        return data

    async def _weather(self) -> WeatherReport | None:
        cached = self._weather_cache
        if cached and time.monotonic() - cached[0] < settings.briefing_weather_cache_sec:
            return cached[1]
        try:
            report = await fetch_weather()
        except Exception:
            if cached is not None:
                log.warning("weather fetch failed; using the cached reading")
                return cached[1]
            raise
        if report is not None:
            self._weather_cache = (time.monotonic(), report)
        return report

    async def _emails(self) -> list[EmailItem]:
        cached = self._email_cache
        if cached and time.monotonic() - cached[0] < settings.briefing_email_cache_sec:
            return cached[1]
        try:
            items = await fetch_latest_emails()
        except Exception:
            if cached is not None:
                log.warning("email fetch failed; using cached messages")
                return cached[1]
            raise
        self._email_cache = (time.monotonic(), items)
        return items

    async def _tasks(self) -> list[TaskItem]:
        # Tasks change often and the API call is cheap, so no cache.
        return await fetch_open_tasks()

    async def _polish(self, text: str) -> str:
        messages = [
            {"role": "system", "content": BRIEFING_POLISH_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            polished = await asyncio.wait_for(
                llm_service.generate_response(
                    "",
                    messages=messages,
                    max_tokens=settings.briefing_llm_max_tokens,
                    temperature=0.3,
                ),
                timeout=settings.briefing_llm_timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("briefing LLM polish failed; using template: %s", exc)
            return text
        cleaned = (polished or "").strip()
        return cleaned or text


briefing_service = BriefingService()
