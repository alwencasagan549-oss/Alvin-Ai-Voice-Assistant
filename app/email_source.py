"""Read the newest Gmail headers over IMAP for the wake-up briefing.

The mailbox is opened read-only and every fetch uses ``BODY.PEEK`` so reading
the briefing never marks a message as read. Only the From, Subject, and Date
headers are downloaded -- the message body is never fetched.
"""

from __future__ import annotations

import asyncio
import imaplib
import logging
import re
from dataclasses import dataclass
from email import message_from_bytes
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime

from .config import settings

log = logging.getLogger("alvin.email")

_WS_RE = re.compile(r"\s+")
_SENSITIVE_RE = re.compile(
    r"\b(?:otp|one[- ]time\s+(?:code|password)|verification\s+code|"
    r"security\s+code|passcode|password\s+reset|two[- ]factor|2fa)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmailItem:
    """A message header, safe to display and speak."""

    sender: str
    subject: str
    date: str = ""
    unread: bool = True


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (LookupError, UnicodeDecodeError, ValueError):
        return str(value)


def _inline(value: str, limit: int = 120) -> str:
    text = _WS_RE.sub(" ", value or "").strip()
    if len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _display_sender(value: str | None) -> str:
    decoded = _decode_header_value(value)
    name, address = parseaddr(decoded)
    return _inline(name or address, 70)


def _format_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.strftime("%a %H:%M")
    except (TypeError, ValueError, OverflowError):
        return ""


def _is_sensitive(item: EmailItem) -> bool:
    return bool(_SENSITIVE_RE.search(item.subject))


def is_configured() -> bool:
    """True when both the IMAP user and App Password are set."""
    return bool(
        settings.briefing_email_imap_user.strip()
        and settings.briefing_email_imap_password
    )


def parse_header_bytes(raw: bytes, *, unread: bool = True) -> EmailItem:
    """Turn a fetched header block into an :class:`EmailItem`.

    Exposed separately so the parsing rules can be unit-tested without a live
    IMAP connection.
    """
    message = message_from_bytes(raw)
    return EmailItem(
        sender=_display_sender(message.get("From")),
        subject=_inline(_decode_header_value(message.get("Subject")), 140),
        date=_format_date(message.get("Date")),
        unread=unread,
    )


def _fetch_sync(limit: int) -> list[EmailItem]:
    host = settings.briefing_email_imap_host
    port = settings.briefing_email_imap_port
    user = settings.briefing_email_imap_user.strip()
    password = settings.briefing_email_imap_password
    folder = settings.briefing_email_folder or "INBOX"

    connection = imaplib.IMAP4_SSL(
        host,
        port,
        timeout=settings.briefing_email_timeout_sec,
    )
    try:
        connection.login(user, password)
        status, _ = connection.select(folder, readonly=True)
        if status != "OK":
            raise RuntimeError(f"IMAP could not open folder {folder!r}")

        criteria = "UNSEEN" if settings.briefing_email_unread_only else "ALL"
        status, data = connection.search(None, criteria)
        if status != "OK":
            raise RuntimeError("IMAP search failed")

        message_ids = (data[0] or b"").split()
        unread = settings.briefing_email_unread_only
        if not message_ids and unread:
            # No unread mail: fall back to the newest messages so the briefing
            # still reports something useful.
            status, data = connection.search(None, "ALL")
            if status == "OK":
                message_ids = (data[0] or b"").split()
                unread = False

        items: list[EmailItem] = []
        for message_id in reversed(message_ids[-limit:]):
            status, message_data = connection.fetch(
                message_id,
                "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])",
            )
            if status != "OK":
                continue
            raw = b"".join(
                part[1]
                for part in (message_data or [])
                if isinstance(part, tuple) and part[1]
            )
            if not raw:
                continue
            item = parse_header_bytes(raw, unread=unread)
            if settings.briefing_email_skip_sensitive and _is_sensitive(item):
                log.debug("briefing skipped a sensitive email subject")
                continue
            items.append(item)
        return items
    finally:
        try:
            connection.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


async def fetch_latest_emails(limit: int | None = None) -> list[EmailItem]:
    """Return the newest inbox headers.

    Raises:
        RuntimeError: when IMAP credentials are not configured.
        asyncio.TimeoutError: when IMAP exceeds ``BRIEFING_EMAIL_TIMEOUT_SEC``.
    """
    if not settings.briefing_email_enabled:
        return []
    if not settings.briefing_email_imap_user.strip() or not settings.briefing_email_imap_password:
        raise RuntimeError(
            "Gmail IMAP is not configured; set BRIEFING_EMAIL_IMAP_USER and "
            "BRIEFING_EMAIL_IMAP_PASSWORD (an App Password) in .env"
        )

    count = max(
        1,
        int(limit if limit is not None else settings.briefing_email_count),
    )
    return await asyncio.wait_for(
        asyncio.to_thread(_fetch_sync, count),
        timeout=settings.briefing_email_timeout_sec,
    )
