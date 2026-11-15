"""HUD window behavior (no Qt event loop needed)."""

from __future__ import annotations

import pytest

hud = pytest.importorskip("alvin_hud")


class FakeWindow:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def showFullScreen(self) -> None:
        self.calls.append("fullscreen")

    def showNormal(self) -> None:
        self.calls.append("normal")

    def raise_(self) -> None:
        self.calls.append("raise")

    def activateWindow(self) -> None:
        self.calls.append("activate")

    def setFocus(self) -> None:
        self.calls.append("focus")

    def reload_if_stale(self) -> None:
        self.calls.append("reload")


def test_reveal_opens_the_hud_fullscreen() -> None:
    window = FakeWindow()

    hud.AlvinWindow.reveal(window)

    assert window.calls[0] == "fullscreen"
    assert "normal" not in window.calls


def test_wake_up_reveals_the_hud() -> None:
    window = FakeWindow()
    revealed: list[bool] = []
    window.reveal = lambda: revealed.append(True)  # type: ignore[method-assign]

    hud.AlvinWindow._on_command(window, "wake_up", {})

    assert revealed == [True]


def test_existing_system_window_is_restored_fullscreen() -> None:
    window = FakeWindow()
    window.system_window = FakeWindow()  # type: ignore[attr-defined]

    hud.AlvinWindow._show_system_window(window, request_start=False)

    assert window.system_window.calls[0] == "fullscreen"


def test_existing_camera_window_is_restored_fullscreen() -> None:
    window = FakeWindow()
    window.camera_window = FakeWindow()  # type: ignore[attr-defined]

    hud.AlvinWindow.open_camera_window(window)

    assert window.camera_window.calls[0] == "fullscreen"
