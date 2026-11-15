"""Unit tests for the read-only Gmail IMAP briefing source."""

from __future__ import annotations

import asyncio

import pytest

from app import email_source
from app.config import settings
from app.email_source import EmailItem, fetch_latest_emails, parse_header_bytes


class FakeIMAP:
    """Minimal ``imaplib.IMAP4_SSL`` stand-in that records every call."""

    instances: list["FakeIMAP"] = []
    unread_ids = b""
    all_ids = b""
    headers: dict[bytes, bytes] = {}

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.calls: list[tuple] = []
        type(self).instances.append(self)

    def login(self, user: str, password: str):
        self.calls.append(("login", user, password))
        return "OK", [b""]

    def select(self, folder: str, readonly: bool = False):
        self.calls.append(("select", folder, readonly))
        return "OK", [b"1"]

    def search(self, charset, criteria: str):
        self.calls.append(("search", criteria))
        ids = type(self).unread_ids if criteria == "UNSEEN" else type(self).all_ids
        return "OK", [ids]

    def fetch(self, message_id: bytes, spec: str):
        self.calls.append(("fetch", message_id, spec))
        raw = type(self).headers.get(message_id, b"")
        return "OK", [
            (b"1 (BODY[HEADER.FIELDS (FROM SUBJECT DATE)] {%d}" % len(raw), raw),
            b")",
        ]

    def logout(self):
        self.calls.append(("logout",))
        return "BYE", [b""]


def _header(sender: str, subject: str) -> bytes:
    return (
        f"From: {sender}\r\n"
        f"Subject: {subject}\r\n"
        "Date: Sat, 15 Nov 2026 09:00:00 +0000\r\n"
        "\r\n"
    ).encode()


@pytest.fixture
def fake_imap(monkeypatch: pytest.MonkeyPatch) -> type[FakeIMAP]:
    FakeIMAP.instances = []
    FakeIMAP.unread_ids = b""
    FakeIMAP.all_ids = b""
    FakeIMAP.headers = {}
    monkeypatch.setattr(email_source.imaplib, "IMAP4_SSL", FakeIMAP)
    monkeypatch.setattr(settings, "briefing_email_imap_host", "imap.gmail.com")
    monkeypatch.setattr(settings, "briefing_email_imap_port", 993)
    monkeypatch.setattr(settings, "briefing_email_imap_user", "me@example.com")
    monkeypatch.setattr(settings, "briefing_email_imap_password", "app-password")
    monkeypatch.setattr(settings, "briefing_email_folder", "INBOX")
    monkeypatch.setattr(settings, "briefing_email_unread_only", True)
    monkeypatch.setattr(settings, "briefing_email_skip_sensitive", True)
    return FakeIMAP


def test_parse_header_decodes_subject_and_display_name() -> None:
    item = parse_header_bytes(
        b"From: Alice Example <alice@example.com>\r\n"
        b"Subject: =?utf-8?q?Project_update?=\r\n"
        b"Date: Sat, 15 Nov 2026 09:00:00 +0000\r\n"
        b"\r\n"
    )

    assert item.sender == "Alice Example"
    assert item.subject == "Project update"
    assert item.date
    assert item.unread is True


def test_parse_header_accepts_a_bare_address() -> None:
    item = parse_header_bytes(
        b"From: alice@example.com\r\nSubject: Hello\r\n\r\n",
        unread=False,
    )

    assert item.sender == "alice@example.com"
    assert item.unread is False


def test_sensitive_subjects_are_detected() -> None:
    assert email_source._is_sensitive(
        EmailItem(sender="Bank", subject="Your one-time code is 123456")
    )
    assert email_source._is_sensitive(
        EmailItem(sender="Service", subject="Password reset requested")
    )
    assert not email_source._is_sensitive(
        EmailItem(sender="Alice", subject="Project update")
    )


def test_fetch_uses_read_only_select_and_peek(fake_imap: type[FakeIMAP]) -> None:
    fake_imap.unread_ids = b"1 2 3"
    fake_imap.headers = {
        b"1": _header("Alice <alice@example.com>", "Project update"),
        b"2": _header("Bob <bob@example.com>", "Lunch on Friday"),
        b"3": _header("Bank <security@bank.example>", "Your verification code is 123456"),
    }

    items = email_source._fetch_sync(3)

    # Newest first, with the one-time-code subject filtered out.
    assert [item.sender for item in items] == ["Bob", "Alice"]

    instance = fake_imap.instances[0]
    assert instance.host == "imap.gmail.com"
    assert instance.port == 993
    assert ("login", "me@example.com", "app-password") in instance.calls
    assert ("select", "INBOX", True) in instance.calls

    fetches = [call for call in instance.calls if call[0] == "fetch"]
    assert fetches, "expected at least one header fetch"
    for _, _, spec in fetches:
        assert spec.startswith("(BODY.PEEK[HEADER.FIELDS")

    # The briefing must never mutate the mailbox.
    methods = {call[0] for call in instance.calls}
    assert methods <= {"login", "select", "search", "fetch", "logout"}


def test_fetch_falls_back_to_all_when_nothing_is_unread(
    fake_imap: type[FakeIMAP],
) -> None:
    fake_imap.unread_ids = b""
    fake_imap.all_ids = b"1"
    fake_imap.headers = {b"1": _header("Alice <alice@example.com>", "Project update")}

    items = email_source._fetch_sync(3)

    assert [item.sender for item in items] == ["Alice"]
    assert items[0].unread is False
    searches = [call for call in fake_imap.instances[0].calls if call[0] == "search"]
    assert [criteria for _, criteria in searches] == ["UNSEEN", "ALL"]


def test_is_configured_requires_user_and_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "briefing_email_imap_user", "")
    monkeypatch.setattr(settings, "briefing_email_imap_password", "")
    assert email_source.is_configured() is False

    monkeypatch.setattr(settings, "briefing_email_imap_user", "me@example.com")
    assert email_source.is_configured() is False

    monkeypatch.setattr(settings, "briefing_email_imap_password", "app-password")
    assert email_source.is_configured() is True


def test_fetch_requires_credentials(
    fake_imap: type[FakeIMAP],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "briefing_email_enabled", True)
    monkeypatch.setattr(settings, "briefing_email_imap_user", "")
    monkeypatch.setattr(settings, "briefing_email_imap_password", "")

    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(fetch_latest_emails())
