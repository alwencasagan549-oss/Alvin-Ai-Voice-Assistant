"""Read open tasks from Google Tasks for the wake-up briefing.

Google Tasks uses OAuth, so the first run requires
``python scripts/google_tasks_auth.py`` to create a token file. The runtime
only needs read access (``tasks.readonly``) and never modifies the task list.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from .config import settings

log = logging.getLogger("alvin.tasks")

TASKS_SCOPE = "https://www.googleapis.com/auth/tasks.readonly"


@dataclass(frozen=True)
class TaskItem:
    title: str
    due: str = ""
    notes: str = ""


def is_configured() -> bool:
    """True when the OAuth token created by google_tasks_auth.py exists."""
    path = settings.briefing_tasks_token_path
    return bool(path) and os.path.isfile(path)


def _format_due(value: str | None) -> str:
    """Turn the API's RFC3339 midnight timestamp into a spoken date."""
    if not value:
        return ""
    try:
        due = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:10]
    delta = (due.date() - datetime.now().date()).days
    if delta < 0:
        return "overdue"
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta < 7:
        return due.strftime("%A")
    return f"{due.strftime('%b')} {due.day}"


def _fetch_sync(limit: int) -> list[TaskItem]:
    token_path = settings.briefing_tasks_token_path
    if not os.path.isfile(token_path):
        raise RuntimeError(
            f"Google Tasks token not found at {token_path!r}; run "
            "python scripts/google_tasks_auth.py first"
        )

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Tasks support requires google-api-python-client and "
            "google-auth-oauthlib; reinstall requirements.txt"
        ) from exc

    credentials = Credentials.from_authorized_user_file(token_path)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise RuntimeError(
                "Google Tasks credentials are invalid; re-run "
                "python scripts/google_tasks_auth.py"
            )

    service = build("tasks", "v1", credentials=credentials, cache_discovery=False)
    result = (
        service.tasks()
        .list(
            tasklist=settings.briefing_tasks_list or "@default",
            showCompleted=False,
            showHidden=False,
            maxResults=limit,
        )
        .execute()
    )

    items: list[TaskItem] = []
    for task in result.get("items", []):
        if task.get("status") == "completed":
            continue
        title = str(task.get("title") or "").strip()
        if not title:
            continue
        items.append(
            TaskItem(
                title=title,
                due=_format_due(task.get("due")),
                notes=str(task.get("notes") or "").strip()[:200],
            )
        )
    return items


async def fetch_open_tasks(limit: int | None = None) -> list[TaskItem]:
    """Return open tasks from the configured Google Tasks list.

    Raises:
        RuntimeError: when the token is missing, Google libraries are absent,
            or the credentials cannot be refreshed.
        asyncio.TimeoutError: when the API exceeds the configured timeout.
    """
    if not settings.briefing_tasks_enabled:
        return []

    count = max(
        1,
        int(limit if limit is not None else settings.briefing_tasks_count),
    )
    return await asyncio.wait_for(
        asyncio.to_thread(_fetch_sync, count),
        timeout=settings.briefing_tasks_timeout_sec,
    )
