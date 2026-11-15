"""HUD ad-free playback wiring (no Qt event loop needed)."""

from __future__ import annotations

import pytest

hud = pytest.importorskip("alvin_hud")


class FakeWorker:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class DummyWindow:
    def __init__(self) -> None:
        self._youtube_url = None
        self._youtube_stream_url = None
        self._youtube_audio_only = True
        self._media_proc = None
        self.played: list[tuple[str, bool]] = []
        self.worker = FakeWorker()

    def _play_media(self, stream_url: str, audio_only: bool = True) -> bool:
        self.played.append((stream_url, audio_only))
        return True

    def stop_local_media(self) -> None:
        self._media_proc = None

    def _find_media_player(self) -> str:
        return r"C:\VLC\vlc.exe"


def test_play_youtube_prefers_the_adfree_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(hud.webbrowser, "open", lambda url, new=1: opened.append(url))

    window = DummyWindow()
    hud.AlvinWindow.play_youtube(
        window,
        {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "stream_url": "https://example.com/direct",
            "audio_only": True,
        },
    )

    assert window.played == [("https://example.com/direct", True)]
    assert opened == [], "the browser must not open when ad-free playback works"
    assert window.worker.sent == [{"type": "youtube_playback", "local": True}]


def test_play_youtube_falls_back_to_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(hud.webbrowser, "open", lambda url, new=1: opened.append(url))

    window = DummyWindow()
    hud.AlvinWindow.play_youtube(
        window,
        {"video_id": "abc123", "url": "", "stream_url": None},
    )

    assert window.played == []
    assert opened and "watch?v=abc123" in opened[0]
    assert window.worker.sent == [{"type": "youtube_playback", "local": False}]


def test_play_media_builds_a_vlc_audio_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> None:
            pass

    def fake_popen(args, **kwargs):
        calls.append(args)
        return FakeProc()

    monkeypatch.setattr(hud.subprocess, "Popen", fake_popen)

    window = DummyWindow()
    ok = hud.AlvinWindow._play_media(window, "https://example.com/direct", True)

    assert ok is True
    assert calls and calls[0][-1] == "https://example.com/direct"
    assert "--no-video" in calls[0]
    assert "--intf" in calls[0] and "dummy" in calls[0], (
        "audio playback must run in the background with no VLC window"
    )
    assert "--no-one-instance" in calls[0], (
        "background playback must not attach to a visible VLC window"
    )


def test_stop_command_terminates_background_player() -> None:
    window = DummyWindow()
    stopped: list[bool] = []
    window.stop_local_media = lambda: stopped.append(True)

    hud.AlvinWindow._on_command(window, "stop", {})

    assert stopped == [True], "'stop the music' must terminate the local player"


def test_video_playback_is_fullscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> None:
            pass

    def fake_popen(args, **kwargs):
        calls.append(args)
        return FakeProc()

    monkeypatch.setattr(hud.subprocess, "Popen", fake_popen)

    window = DummyWindow()
    ok = hud.AlvinWindow._play_media(window, "https://example.com/direct", False)

    assert ok is True
    assert "--fullscreen" in calls[0]
    assert "--no-video" not in calls[0]


class FakeSignal:
    def __init__(self) -> None:
        self.emitted: list[tuple] = []

    def emit(self, *args) -> None:
        self.emitted.append(args)


class FakeSpeechWorker:
    def __init__(self, speaking: bool = True) -> None:
        self.is_listening = True
        self.is_speaking = speaking
        self.status_changed = FakeSignal()


def test_tts_stop_reopens_the_microphone() -> None:
    """A cancelled TTS turn sends tts_stop, never tts_end."""
    worker = FakeSpeechWorker(speaking=True)

    hud.BackendWorker._handle_json(worker, {"type": "tts_stop"})

    assert worker.is_speaking is False, (
        "tts_stop must reopen the mic gate or the HUD goes deaf"
    )
    assert worker.status_changed.emitted == [("LISTENING", "#00e5ff")]


def test_resume_youtube_replays_the_adfree_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(hud.webbrowser, "open", lambda url, new=1: opened.append(url))

    window = DummyWindow()
    window._youtube_url = "https://www.youtube.com/watch?v=abc123"
    window._youtube_stream_url = "https://example.com/direct"
    window._youtube_audio_only = True

    hud.AlvinWindow.resume_youtube(window)

    assert window.played == [("https://example.com/direct", True)]
    assert opened == []
    assert window.worker.sent == [{"type": "youtube_playback", "local": True}]
