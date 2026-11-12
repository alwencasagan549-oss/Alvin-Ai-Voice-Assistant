"""Close the browser tab that's playing YouTube. Windows-only, stdlib-only.

Strategy:
  1. Require an interactive Windows desktop session.
  2. Find ALL visible browser windows, not just the first one.
  3. Prioritize windows whose active tab is already YouTube.
  4. Focus each browser window and verify foreground ownership.
  5. For Chrome/Edge, search background tabs using Ctrl+Shift+A.
  6. Send Ctrl+W only after verifying YouTube is the active tab.
  7. Verify the tab actually closed before returning True.

Returns True only when YouTube has been verified closed.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from typing import Callable

log = logging.getLogger("alvin.browser")

_WINDOWS = sys.platform == "win32"


def _is_interactive_session() -> bool:
    """Check that this process runs in the user's interactive desktop."""
    return os.environ.get("SESSIONNAME", "") not in ("", "Services")


def _poll_until(
    predicate: Callable[[], bool],
    timeout: float,
    interval: float = 0.05,
) -> bool:
    """Poll until a condition becomes true or timeout expires."""
    deadline = time.monotonic() + timeout

    while True:
        if predicate():
            return True

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            return False

        time.sleep(min(interval, remaining))


def _close_foreground_tab_and_verify(
    send_ctrl_w: Callable[[], None],
    foreground_title: Callable[[], str],
    timeout: float = 1.5,
    interval: float = 0.05,
) -> bool:
    """Close the active YouTube tab and verify it disappeared."""

    send_ctrl_w()

    gone = _poll_until(
        lambda: "youtube" not in foreground_title().lower(),
        timeout=timeout,
        interval=interval,
    )

    if not gone:
        log.warning(
            "close_youtube_tab: Ctrl+W sent but YouTube still foreground (%r)",
            foreground_title(),
        )
        return False

    log.info("close_youtube_tab: tab closed")
    return True


if _WINDOWS:
    import ctypes.wintypes as wt

    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    _EnumWindowsProc = ctypes.WINFUNCTYPE(
        wt.BOOL,
        wt.HWND,
        wt.LPARAM,
    )

    # Win32 function signatures.

    _user32.EnumWindows.argtypes = [_EnumWindowsProc, wt.LPARAM]
    _user32.EnumWindows.restype = wt.BOOL

    _user32.IsWindowVisible.argtypes = [wt.HWND]
    _user32.IsWindowVisible.restype = wt.BOOL

    _user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int

    _user32.GetWindowTextW.argtypes = [
        wt.HWND,
        wt.LPWSTR,
        ctypes.c_int,
    ]
    _user32.GetWindowTextW.restype = ctypes.c_int

    _user32.IsIconic.argtypes = [wt.HWND]
    _user32.IsIconic.restype = wt.BOOL

    _user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wt.BOOL

    _user32.SetForegroundWindow.argtypes = [wt.HWND]
    _user32.SetForegroundWindow.restype = wt.BOOL

    _user32.SendInput.argtypes = [
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _user32.SendInput.restype = ctypes.c_uint

    _user32.GetWindowThreadProcessId.argtypes = [
        wt.HWND,
        ctypes.POINTER(wt.DWORD),
    ]
    _user32.GetWindowThreadProcessId.restype = wt.DWORD

    _user32.AttachThreadInput.argtypes = [
        wt.DWORD,
        wt.DWORD,
        wt.BOOL,
    ]
    _user32.AttachThreadInput.restype = wt.BOOL

    _user32.GetForegroundWindow.argtypes = []
    _user32.GetForegroundWindow.restype = wt.HWND

    _user32.SetWindowPos.argtypes = [
        wt.HWND,
        wt.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wt.UINT,
    ]
    _user32.SetWindowPos.restype = wt.BOOL

    _user32.BringWindowToTop.argtypes = [wt.HWND]
    _user32.BringWindowToTop.restype = wt.BOOL

    # Keyboard structures.

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wt.WORD),
            ("wScan", wt.WORD),
            ("dwFlags", wt.DWORD),
            ("time", wt.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wt.LONG),
            ("dy", wt.LONG),
            ("mouseData", wt.DWORD),
            ("dwFlags", wt.DWORD),
            ("time", wt.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [
            ("ki", _KEYBDINPUT),
            ("mi", _MOUSEINPUT),
        ]

    class _INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wt.DWORD),
            ("u", _INPUTUNION),
        ]

    # Constants.

    _INPUT_KEYBOARD = 1

    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_UNICODE = 0x0004

    _VK_CONTROL = 0x11
    _VK_SHIFT = 0x10
    _VK_MENU = 0x12
    _VK_A = 0x41
    _VK_W = 0x57
    _VK_RETURN = 0x0D

    _SW_RESTORE = 9

    _HWND_TOP = 0
    _SWP_NOMOVE = 0x0002
    _SWP_NOSIZE = 0x0001
    _SWP_SHOWWINDOW = 0x0040

    # Supported browsers.

    _BROWSER_HINTS = (
        "Google Chrome",
        "Microsoft Edge",
        "Mozilla Firefox",
        "Brave",
        "Vivaldi",
        "Opera",
        "Arc",
    )

    # Background-tab search via Ctrl+Shift+A is supported by all
    # Chromium-based browsers: Chrome, Edge, Brave, Vivaldi, Opera, Arc.

    _TAB_SEARCH_HINTS = (
        "Google Chrome",
        "Microsoft Edge",
        "Brave",
        "Vivaldi",
        "Opera",
        "Arc",
    )

    # ------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------

    def _visible_titled_windows() -> list[tuple[int, str]]:
        """Return all visible windows with titles."""

        out: list[tuple[int, str]] = []

        def _cb(hwnd: int, _lp: int) -> bool:
            if not _user32.IsWindowVisible(hwnd):
                return True

            n = _user32.GetWindowTextLengthW(hwnd)

            if n <= 0:
                return True

            buf = ctypes.create_unicode_buffer(n + 1)

            _user32.GetWindowTextW(
                hwnd,
                buf,
                n + 1,
            )

            out.append((hwnd, buf.value))

            return True

        _user32.EnumWindows(
            _EnumWindowsProc(_cb),
            0,
        )

        return out

    def _title(hwnd: int) -> str:
        """Get a window title."""

        n = _user32.GetWindowTextLengthW(hwnd)

        if n <= 0:
            return ""

        buf = ctypes.create_unicode_buffer(n + 1)

        _user32.GetWindowTextW(
            hwnd,
            buf,
            n + 1,
        )

        return buf.value

    def _find_browser_windows(
        preferred: str | None = None,
    ) -> list[tuple[int, str]]:
        """Find every browser window, prioritizing active YouTube tabs."""

        windows = _visible_titled_windows()

        candidates = [
            (hwnd, title)
            for hwnd, title in windows
            if any(h.lower() in title.lower() for h in _BROWSER_HINTS)
        ]

        if not candidates:
            if windows:
                log.warning(
                    "close_youtube_tab: no browser window matched. Visible windows: %s",
                    [(h, t[:60]) for h, t in windows],
                )
            else:
                log.warning("close_youtube_tab: no visible titled windows found")

            return []

        preferred_lower = (preferred or "").lower()

        def _rank(
            item: tuple[int, str],
        ) -> tuple[int, int, int]:

            _hwnd, title = item

            lower = title.lower()

            active_youtube_first = 0 if "youtube" in lower else 1

            preferred_first = 0 if preferred_lower and preferred_lower in lower else 1

            return (
                active_youtube_first,
                preferred_first,
                windows.index(item),
            )

        return sorted(
            candidates,
            key=_rank,
        )

    # ------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------

    def _key(
        vk: int,
        up: bool = False,
    ) -> None:
        """Send a keyboard input."""

        inp = _INPUT(type=_INPUT_KEYBOARD)

        inp.u.ki = _KEYBDINPUT(
            wVk=vk,
            wScan=0,
            dwFlags=_KEYEVENTF_KEYUP if up else 0,
            time=0,
            dwExtraInfo=0,
        )

        sent = _user32.SendInput(
            1,
            ctypes.byref(inp),
            ctypes.sizeof(_INPUT),
        )

        if sent != 1:
            log.warning(
                "close_youtube_tab: SendInput failed "
                "for key vk=%#x up=%s (winerror=%s)",
                vk,
                up,
                ctypes.get_last_error(),
            )

    def _combo(*vks: int) -> None:
        """Send a keyboard combination."""

        for vk in vks:
            _key(vk)
            time.sleep(0.02)

        for vk in reversed(vks):
            _key(vk, up=True)
            time.sleep(0.02)

    def _send_unicode(ch: str) -> None:
        """Send Unicode keyboard input."""

        def _one(
            scan: int,
            flags: int,
        ) -> None:

            inp = _INPUT(type=_INPUT_KEYBOARD)

            inp.u.ki = _KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=flags,
            )

            sent = _user32.SendInput(
                1,
                ctypes.byref(inp),
                ctypes.sizeof(_INPUT),
            )

            if sent != 1:
                log.warning(
                    "close_youtube_tab: SendInput failed "
                    "for unicode character %r (winerror=%s)",
                    ch,
                    ctypes.get_last_error(),
                )

        _one(ord(ch), _KEYEVENTF_UNICODE)

        _one(
            ord(ch),
            _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP,
        )

    def _type_text(text: str) -> None:
        """Type text using Unicode input."""

        for ch in text:
            _send_unicode(ch)
            time.sleep(0.01)

    # ------------------------------------------------------------
    # Browser focus
    # ------------------------------------------------------------

    def _focus(hwnd: int) -> bool:
        """Bring a browser window to the foreground and verify."""

        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(
                hwnd,
                _SW_RESTORE,
            )

        fg = _user32.GetForegroundWindow()

        target_tid = _user32.GetWindowThreadProcessId(
            hwnd,
            None,
        )

        fg_tid = _user32.GetWindowThreadProcessId(fg, None) if fg else 0

        attached = False

        if fg_tid and fg_tid != target_tid:
            attached = bool(
                _user32.AttachThreadInput(
                    fg_tid,
                    target_tid,
                    True,
                )
            )

        try:
            _user32.SetForegroundWindow(hwnd)
            _user32.BringWindowToTop(hwnd)

        finally:
            if attached:
                _user32.AttachThreadInput(
                    fg_tid,
                    target_tid,
                    False,
                )

        if _user32.GetForegroundWindow() == hwnd:
            return True

        # Retry foreground promotion.

        try:
            _user32.SetWindowPos(
                hwnd,
                _HWND_TOP,
                0,
                0,
                0,
                0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_SHOWWINDOW,
            )

            _user32.BringWindowToTop(hwnd)

            _key(_VK_MENU)
            _key(_VK_MENU, up=True)

            _user32.SetForegroundWindow(hwnd)

        except Exception:
            pass

        return _user32.GetForegroundWindow() == hwnd

    # ------------------------------------------------------------
    # Browser actions
    # ------------------------------------------------------------

    def _foreground_title() -> str:
        hwnd = _user32.GetForegroundWindow()

        return _title(hwnd) if hwnd else ""

    def _send_ctrl_w() -> None:
        """Close the currently active browser tab."""

        _combo(
            _VK_CONTROL,
            _VK_W,
        )

    def _tab_search_youtube(hwnd: int) -> bool:
        """Find a background YouTube tab using Chrome/Edge tab search.

        Ctrl+Shift+A opens Search Tabs in supported Chrome/Edge versions.
        Verify browser focus before sending text or Enter.
        """

        if _user32.GetForegroundWindow() != hwnd:
            log.warning("close_youtube_tab: browser lost foreground before tab search")
            return False

        _combo(
            _VK_CONTROL,
            _VK_SHIFT,
            _VK_A,
        )

        time.sleep(0.4)

        if _user32.GetForegroundWindow() != hwnd:
            log.warning("close_youtube_tab: focus changed while opening tab search")
            return False

        _type_text("youtube")

        time.sleep(0.3)

        if _user32.GetForegroundWindow() != hwnd:
            log.warning("close_youtube_tab: focus changed while searching tabs")
            return False

        _key(_VK_RETURN)
        _key(_VK_RETURN, up=True)

        landed = _poll_until(
            lambda: (
                _user32.GetForegroundWindow() == hwnd
                and "youtube" in _foreground_title().lower()
            ),
            timeout=2.0,
        )

        if not landed:
            log.warning(
                "close_youtube_tab: tab search did not land on YouTube (foreground=%r)",
                _foreground_title(),
            )

        return landed


# ------------------------------------------------------------
# Public function
# ------------------------------------------------------------


def close_youtube_tab(
    preferred_browser: str | None = None,
) -> bool:
    """Close a YouTube tab in a visible browser window.

    All visible browser windows are checked because a background YouTube tab
    cannot be identified from the window title alone.

    Direct close works for any recognized browser when YouTube
    is already the active tab.

    Background-tab search is supported only for Chrome and Edge.
    """

    if not _WINDOWS:
        log.error(
            "close_youtube_tab: Windows-only operation; platform=%s",
            sys.platform,
        )
        return False

    if not _is_interactive_session():
        log.error(
            "close_youtube_tab: not in an interactive desktop session "
            "(SESSIONNAME=%r). Run the server from the logged-in "
            "user's terminal.",
            os.environ.get("SESSIONNAME"),
        )
        return False

    candidates = _find_browser_windows(preferred_browser)

    if not candidates:
        return False

    # Try every detected browser window.

    for hwnd, title in candidates:
        log.info(
            "close_youtube_tab: trying hwnd=%r title=%r",
            hwnd,
            title,
        )

        if not _focus(hwnd):
            log.warning(
                "close_youtube_tab: could not focus browser window %r",
                title,
            )
            continue

        if _user32.GetForegroundWindow() != hwnd:
            log.warning(
                "close_youtube_tab: foreground verification failed for %r",
                title,
            )
            continue

        time.sleep(0.2)

        active_title = _foreground_title()

        # Case 1: YouTube is already the active tab.

        if "youtube" in active_title.lower():
            if _close_foreground_tab_and_verify(
                _send_ctrl_w,
                _foreground_title,
            ):
                return True

            continue

        # Case 2: YouTube is a background tab.
        # Only use tab search in Chrome and Edge.

        if not any(h.lower() in title.lower() for h in _TAB_SEARCH_HINTS):
            log.info(
                "close_youtube_tab: YouTube is not the active tab in %r; "
                "background-tab search is unsupported for this browser",
                title,
            )

            continue

        # Find the YouTube tab.

        if not _tab_search_youtube(hwnd):
            continue

        # Close only after the YouTube tab is confirmed active.

        if _close_foreground_tab_and_verify(
            _send_ctrl_w,
            _foreground_title,
        ):
            return True

    log.warning("close_youtube_tab: no YouTube tab was verified closed")

    return False
