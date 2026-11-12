"""Unit tests for echo / self-trigger suppression (:mod:`app.echo`).

The regression these guard against is the assistant hearing its own playback
and replying to itself in a loop.
"""

from __future__ import annotations

import time

import pytest

from app.config import settings
from app.echo import EchoGuard, looks_like_echo, similarity


@pytest.fixture(autouse=True)
def _echo_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "echo_mode", "filter")
    monkeypatch.setattr(settings, "echo_similarity_threshold", 0.75)
    monkeypatch.setattr(settings, "echo_cooldown_ms", 600)
    monkeypatch.setattr(settings, "echo_history_sec", 30.0)


# --------------------------------------------------------------------------- #
# similarity
# --------------------------------------------------------------------------- #
def test_identical_text_is_full_similarity() -> None:
    assert similarity("the weather is nice today", "The weather is nice today") == 1.0


def test_filler_words_do_not_affect_similarity() -> None:
    assert similarity("it is the weather here", "the weather") == 1.0


def test_unrelated_text_scores_low() -> None:
    assert similarity("turn on the kitchen lights", "the capital of france") < 0.5


def test_empty_and_fillerless_inputs() -> None:
    assert similarity("", "") == 1.0
    assert similarity("the a of", "stop") == 0.0
    assert similarity("stop", "") == 0.0


# --------------------------------------------------------------------------- #
# looks_like_echo
# --------------------------------------------------------------------------- #
def test_verbatim_echo_is_detected() -> None:
    spoken = ["Sure, the kitchen lights are now on."]
    assert looks_like_echo("sure the kitchen lights are now on", spoken, 0.75)


def test_partial_echo_is_detected() -> None:
    """STT rarely transcribes leaked playback perfectly."""
    spoken = ["Sure, the kitchen lights are now on."]
    assert looks_like_echo("the kitchen lights are now on", spoken, 0.75)


def test_genuine_command_is_not_echo() -> None:
    """A real user command shares almost no text with the assistant's reply."""
    spoken = ["Sure, the kitchen lights are now on."]
    for text in ("stop", "turn on the lights", "set the volume to 30"):
        assert not looks_like_echo(text, spoken, 0.75), f"{text!r} misread as echo"


def test_echo_matches_any_recent_sentence_not_only_the_last() -> None:
    """A loop can re-enter several turns later."""
    spoken = ["the first thing I said", "a later sentence", "the last thing I said"]
    assert looks_like_echo("the first thing I said", spoken, 0.75)


def test_no_history_means_no_echo() -> None:
    assert not looks_like_echo("anything at all", [], 0.75)


def test_empty_transcript_is_never_an_echo() -> None:
    assert not looks_like_echo("", ["some spoken text"], 0.75)


# --------------------------------------------------------------------------- #
# EchoGuard
# --------------------------------------------------------------------------- #
def test_filter_mode_allows_audio_while_speaking() -> None:
    """Barge-in must keep working in the default mode."""
    guard = EchoGuard()
    guard.begin_speech("here is your answer")
    assert guard.speaking is True
    assert guard.should_accept_audio() is True


def test_gate_mode_drops_audio_while_speaking() -> None:
    guard = EchoGuard()
    settings.echo_mode = "gate"
    guard.begin_speech("here is your answer")
    assert guard.should_accept_audio() is False


def test_cooldown_blocks_audio_after_speech_ends() -> None:
    guard = EchoGuard()
    guard.begin_speech("hello")
    guard.end_speech()
    # Cooldown is active right after playback stops.
    assert guard.speaking is False
    assert guard.should_accept_audio() is False


def test_cooldown_expires() -> None:
    guard = EchoGuard()
    guard.begin_speech("hello")
    guard.end_speech()
    guard._cooldown_until = time.monotonic() - 1.0
    assert guard.should_accept_audio() is True


def test_zero_cooldown_allows_audio_immediately() -> None:
    guard = EchoGuard()
    settings.echo_cooldown_ms = 0
    guard.begin_speech("hello")
    guard.end_speech()
    assert guard.should_accept_audio() is True


def test_is_echo_disabled_in_gate_mode() -> None:
    guard = EchoGuard()
    settings.echo_mode = "gate"
    guard.begin_speech("the kitchen lights are on")
    assert guard.is_echo("the kitchen lights are on") is False


def test_is_echo_disabled_when_off() -> None:
    guard = EchoGuard()
    settings.echo_mode = "off"
    guard.begin_speech("the kitchen lights are on")
    assert guard.is_echo("the kitchen lights are on") is False


def test_history_expires() -> None:
    guard = EchoGuard()
    settings.echo_history_sec = 0.0
    guard.begin_speech("the kitchen lights are on")
    assert guard.recent_speech() == []
    assert guard.is_echo("the kitchen lights are on") is False


def test_guard_remembers_multiple_sentences() -> None:
    guard = EchoGuard()
    guard.begin_speech("first sentence here")
    guard.end_speech()
    guard.begin_speech("second sentence here")
    assert len(guard.recent_speech()) == 2
