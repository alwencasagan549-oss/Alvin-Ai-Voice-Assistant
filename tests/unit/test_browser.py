"""Unit tests for :mod:`app.browser` decision logic.

These tests run offline and never touch the desktop: the Win32 interop helpers
are exercised only through injectable fakes, so the suite is safe on CI and on
a developer machine with a live browser open.
"""

from __future__ import annotations

import sys
import time

import pytest

import app.browser as browser
from app.browser import (
    _close_foreground_tab_and_verify,
    _is_interactive_session,
    _poll_until,
)


class TestInteractiveSessionGuard:
    """``_is_interactive_session`` gates the whole command on SESSIONNAME."""

    def test_console_session_is_interactive(self, monkeypatch):
        monkeypatch.setenv("SESSIONNAME", "Console")
        assert _is_interactive_session() is True

    def test_services_session_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SESSIONNAME", "Services")
        assert _is_interactive_session() is False

    def test_missing_sessionname_is_rejected(self, monkeypatch):
        monkeypatch.delenv("SESSIONNAME", raising=False)
        assert _is_interactive_session() is False

    def test_rdp_session_is_interactive(self, monkeypatch):
        monkeypatch.setenv("SESSIONNAME", "RDP-Tcp#0")
        assert _is_interactive_session() is True


class TestPollUntil:
    """``_poll_until`` re-checks a predicate until it holds or time runs out."""

    def test_returns_true_immediately_when_predicate_holds(self):
        calls: list[int] = []

        def predicate() -> bool:
            calls.append(1)
            return True

        assert _poll_until(predicate, timeout=0.05, interval=0.01) is True
        assert len(calls) == 1

    def test_returns_true_once_predicate_flips(self):
        state = {"n": 0}

        def predicate() -> bool:
            state["n"] += 1
            return state["n"] >= 3

        assert _poll_until(predicate, timeout=1.0, interval=0.01) is True
        assert state["n"] == 3

    def test_returns_false_after_timeout_when_never_true(self):
        started = time.monotonic()
        assert _poll_until(lambda: False, timeout=0.05, interval=0.01) is False
        assert time.monotonic() - started < 1.0


class TestCloseForegroundTabAndVerify:
    """``_close_foreground_tab_and_verify`` must not report success on faith.

    Regression guard for the "says it worked, didn't" bug: the old flow sent
    Ctrl+W and returned True unconditionally, so ``closed_youtube_tab: true``
    reached the client even when the keystroke was swallowed.
    """

    def test_returns_true_when_tab_leaves_foreground(self):
        state = {"title": "YouTube - My Song - Google Chrome"}
        sent: list[int] = []

        def send_ctrl_w() -> None:
            sent.append(1)
            state["title"] = "New Tab - Google Chrome"

        result = _close_foreground_tab_and_verify(
            send_ctrl_w, lambda: state["title"], timeout=0.2, interval=0.01
        )
        assert result is True
        assert sent == [1]

    def test_returns_false_when_tab_stays_foreground(self):
        state = {"title": "YouTube - My Song - Google Chrome"}
        sent: list[int] = []

        def send_ctrl_w() -> None:
            sent.append(1)

        result = _close_foreground_tab_and_verify(
            send_ctrl_w, lambda: state["title"], timeout=0.1, interval=0.01
        )
        assert result is False
        assert sent == [1]

    def test_warns_when_keystroke_is_ignored(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="alvin.browser"):
            _close_foreground_tab_and_verify(
                lambda: None,
                lambda: "YouTube - Still Playing - Google Chrome",
                timeout=0.1,
                interval=0.01,
            )
        assert any("still foreground" in r.message for r in caplog.records)


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="flow wires win32 helpers"
)
class TestCloseYoutubeTabFlow:
    """``close_youtube_tab`` flow: guards, path selection, honest result.

    The win32 helpers are monkeypatched with recording fakes so the flow runs
    without touching the desktop. Each test pins one decision the flow makes.
    """

    YOUTUBE_TITLE = "YouTube - My Song - Google Chrome"
    CHROME_TITLE = "Google Chrome"
    FIREFOX_TITLE = "Mozilla Firefox"

    @pytest.fixture(autouse=True)
    def console_session(self, monkeypatch):
        monkeypatch.setenv("SESSIONNAME", "Console")

    @pytest.fixture(autouse=True)
    def default_wiring(self, monkeypatch):
        """One focused Chrome window whose foreground title is not YouTube."""
        monkeypatch.setattr(
            browser,
            "_find_browser_windows",
            lambda preferred: [(4242, self.CHROME_TITLE)],
        )
        monkeypatch.setattr(browser, "_focus", lambda hwnd: True)
        monkeypatch.setattr(browser, "_foreground_title", lambda: self.CHROME_TITLE)
        monkeypatch.setattr(browser._user32, "GetForegroundWindow", lambda: 4242)

    def test_bails_out_in_services_session(self, monkeypatch):
        monkeypatch.setenv("SESSIONNAME", "Services")
        calls: list[str] = []

        def _find(preferred):
            calls.append("find")
            return [(4242, "Google Chrome")]

        monkeypatch.setattr(browser, "_find_browser_windows", _find)
        assert browser.close_youtube_tab() is False
        assert calls == []

    def test_returns_false_when_no_browser_window_found(self, monkeypatch):
        monkeypatch.setattr(browser, "_find_browser_windows", lambda preferred: [])
        assert browser.close_youtube_tab() is False

    def test_returns_false_when_focus_fails(self, monkeypatch):
        monkeypatch.setattr(browser, "_focus", lambda hwnd: False)
        assert browser.close_youtube_tab() is False

    def test_returns_false_when_foreground_verification_fails(self, monkeypatch):
        """_focus can succeed while the window still does not own foreground."""
        monkeypatch.setattr(browser._user32, "GetForegroundWindow", lambda: 9999)
        assert browser.close_youtube_tab() is False

    def test_youtube_foreground_tab_close_failure_propagates(self, monkeypatch):
        """THE bug: Ctrl+W swallowed must yield False, not a lying True."""
        monkeypatch.setattr(browser, "_foreground_title", lambda: self.YOUTUBE_TITLE)
        close_calls: list[tuple] = []

        def _verify(send, get_title, **kwargs):
            close_calls.append((send, get_title))
            return False

        monkeypatch.setattr(browser, "_close_foreground_tab_and_verify", _verify)
        assert browser.close_youtube_tab() is False
        assert len(close_calls) == 1
        assert close_calls[0][0] is browser._send_ctrl_w

    def test_youtube_foreground_tab_close_success_propagates(self, monkeypatch):
        monkeypatch.setattr(browser, "_foreground_title", lambda: self.YOUTUBE_TITLE)
        monkeypatch.setattr(
            browser, "_close_foreground_tab_and_verify", lambda send, get, **k: True
        )
        assert browser.close_youtube_tab() is True

    def test_tab_search_failure_skips_ctrl_w(self, monkeypatch):
        close_calls: list[tuple] = []
        monkeypatch.setattr(
            browser,
            "_close_foreground_tab_and_verify",
            lambda send, get, **k: close_calls.append(1) or True,
        )
        monkeypatch.setattr(browser, "_tab_search_youtube", lambda hwnd: False)
        assert browser.close_youtube_tab() is False
        assert close_calls == []

    def test_tab_search_success_closes_tab(self, monkeypatch):
        monkeypatch.setattr(browser, "_tab_search_youtube", lambda hwnd: True)
        monkeypatch.setattr(
            browser, "_close_foreground_tab_and_verify", lambda send, get, **k: True
        )
        assert browser.close_youtube_tab() is True

    def test_firefox_without_youtube_tab_bails_out(self, monkeypatch):
        monkeypatch.setattr(
            browser,
            "_find_browser_windows",
            lambda preferred: [(4242, self.FIREFOX_TITLE)],
        )
        tab_search_calls: list[int] = []
        monkeypatch.setattr(
            browser,
            "_tab_search_youtube",
            lambda hwnd: tab_search_calls.append(1) or True,
        )
        assert browser.close_youtube_tab() is False
        assert tab_search_calls == []

    def test_tries_every_window_instead_of_giving_up_after_first(self, monkeypatch):
        """Problem A regression: a failed first window must not abort the flow.

        The first browser window rejects foreground ownership; the second one
        owns it and shows YouTube as the active tab. The flow must skip the
        first and close the YouTube tab in the second.
        """
        monkeypatch.setattr(
            browser,
            "_find_browser_windows",
            lambda preferred: [
                (1111, self.CHROME_TITLE),
                (2222, self.YOUTUBE_TITLE),
            ],
        )
        owner = {"hwnd": None}

        def fake_focus(hwnd: int) -> bool:
            # The first window's focus request is refused by Windows; the
            # second window actually comes to the foreground.
            if hwnd == 2222:
                owner["hwnd"] = hwnd
                return True
            return False

        monkeypatch.setattr(browser, "_focus", fake_focus)
        monkeypatch.setattr(
            browser._user32, "GetForegroundWindow", lambda: owner["hwnd"]
        )
        monkeypatch.setattr(
            browser,
            "_foreground_title",
            lambda: self.YOUTUBE_TITLE if owner["hwnd"] == 2222 else self.CHROME_TITLE,
        )
        closed: list[int] = []
        monkeypatch.setattr(
            browser,
            "_close_foreground_tab_and_verify",
            lambda send, get, **k: closed.append(1) or True,
        )
        assert browser.close_youtube_tab() is True
        assert closed == [1]
