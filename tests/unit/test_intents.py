"""Unit tests for deterministic intent routing (:mod:`app.intents`).

Fixed device actions must match on the raw (ITN-normalized) transcript with
zero LLM involvement; conversational utterances containing command words must
fall through to the LLM path.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.intents import SUPPORTED_INTENTS, route_command


def route(text: str, confidence: float | None = 0.95):
    return route_command(text, confidence)


@pytest.mark.parametrize(
    ("utterance", "intent", "slots"),
    [
        # playback control
        ("stop", "stop", {}),
        ("stop it", "stop", {}),
        ("stop the music", "stop", {}),
        ("shut up", "stop", {}),
        ("pause", "pause", {}),
        ("resume", "resume", {}),
        # mic control (audio gating)
        ("mute", "mute_mic", {}),
        ("mute the mic", "mute_mic", {}),
        ("mute the microphone", "mute_mic", {}),
        ("alvin mute", "mute_mic", {}),
        ("alvin mute the microphone", "mute_mic", {}),
        ("unmute", "unmute_mic", {}),
        ("unmute the mic", "unmute_mic", {}),
        ("alvin unmute the microphone", "unmute_mic", {}),
        # device (speaker) mute — qualified forms fall through
        ("mute the speaker", "mute", {}),
        ("mute the music", "mute", {}),
        ("unmute the speaker", "unmute", {}),
        # volume
        ("volume up", "volume_up", {}),
        ("louder", "volume_up", {}),
        ("turn the volume up", "volume_up", {}),
        ("turn the volume up a bit", "volume_up", {}),
        ("volume down", "volume_down", {}),
        ("quieter", "volume_down", {}),
        ("turn the music down a little", "volume_down", {}),
        ("set the volume to 50", "set_volume", {"volume": 50}),
        ("set the volume to fifty", "set_volume", {"volume": 50}),  # via ITN
        ("volume to forty", "set_volume", {"volume": 40}),
        ("volume 60 percent", "set_volume", {"volume": 60}),
        # lights
        ("turn on the lights", "lights_on", {}),
        ("lights on", "lights_on", {}),
        ("switch on the lights", "lights_on", {}),
        ("turn off the lights", "lights_off", {}),
        ("lights off", "lights_off", {}),
        ("please turn off the lights", "lights_off", {}),
        ("alvin turn off the lights", "lights_off", {}),
        ("alvin please stop the music", "stop", {}),
        # media
        ("next track", "next", {}),
        ("skip to the next", "next", {}),
        ("previous song", "previous", {}),
        ("play the music", "play", {}),
        # device query
        ("what time is it", "query_time", {}),
        ("what's the time", "query_time", {}),
        # STT engines append sentence punctuation to short commands
        ("mute.", "mute_mic", {}),
        ("Alvin Mute.", "mute_mic", {}),
        ("alvin, mute", "mute_mic", {}),
        ("Mute the mic.", "mute_mic", {}),
        ("Unmute the mic.", "unmute_mic", {}),
        ("stop it!", "stop", {}),
        ("turn off the lights.", "lights_off", {}),
        ("what's the time?", "query_time", {}),
    ],
)
def test_command_recognized(utterance: str, intent: str, slots: dict) -> None:
    match = route(utterance)
    assert match is not None, f"{utterance!r} should route to {intent}"
    assert match.intent == intent
    assert match.slots == slots
    assert match.raw == utterance


@pytest.mark.parametrize(
    "utterance",
    [
        # command word embedded in a longer sentence
        "how do I stop my car from rolling",
        "don't stop the music",
        "stop the music and play jazz",
        "turn the lights off when you leave the room",
        "how loud is the volume right now",
        "what time did the game start",
        "I would like to turn off the lights at the party tonight",
        # open-ended questions
        "tell me a joke",
        "what is the difference between volume and loudness",
        "set a timer for the pasta",
    ],
)
def test_conversational_utterance_is_not_a_command(utterance: str) -> None:
    assert route(utterance) is None, f"{utterance!r} must fall through to the LLM"


def test_set_volume_slot_is_an_int() -> None:
    match = route("set the volume to 42")
    assert match is not None
    assert match.slots["volume"] == 42
    assert isinstance(match.slots["volume"], int)


def test_min_confidence_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "intent_min_confidence", 0.5)
    assert route("stop", 0.3) is None, "below the floor the LLM decides"
    assert route("stop", 0.5) is not None
    assert route("stop", 0.9) is not None
    # confidence=None means "unknown": the gate is skipped, not applied.
    assert route("stop", None) is not None


def test_disabled_router_never_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "intent_enabled", False)
    assert route("stop") is None
    assert route("turn off the lights") is None


def test_mic_control_intents_carry_the_stt_action() -> None:
    mute = route("mute the mic")
    assert mute is not None and mute.intent == "mute_mic"
    assert mute.action == "disable_stt"
    assert mute.confirmation  # the hardcoded confirmation is always spoken

    unmute = route("alvin unmute")
    assert unmute is not None and unmute.intent == "unmute_mic"
    assert unmute.action == "enable_stt"
    assert unmute.confirmation


def test_supported_intents_cover_the_registry() -> None:
    for intent in (
        "stop",
        "pause",
        "mute",
        "mute_mic",
        "unmute_mic",
        "set_volume",
        "lights_on",
        "lights_off",
    ):
        assert intent in SUPPORTED_INTENTS


# --------------------------------------------------------------------------- #
# Natural phrasing (regression battery)
#
# Every phrase here failed against the first implementation: only a leading
# "please"/"Alvin" prefix was stripped, polite request wrappers and trailing
# "please" were not handled, and bare/short commands such as "next" or "make
# it louder" were absent from the word lists.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        # trailing politeness
        ("turn the lights off please", "lights_off"),
        ("turn on the lights please", "lights_on"),
        ("next song please", "next"),
        # polite request wrappers
        ("can you turn on the lights", "lights_on"),
        ("could you turn off the lights please", "lights_off"),
        ("would you play the music", "play"),
        # greetings + wake word
        ("hey alvin turn on the lights", "lights_on"),
        ("ok alvin stop", "stop"),
        ("alvin please stop", "stop"),
        # relative volume, short forms
        ("turn up the volume", "volume_up"),
        ("turn down the volume", "volume_down"),
        ("make it louder", "volume_up"),
        ("make it quieter", "volume_down"),
        ("turn it all the way up", "volume_up"),
        ("turn the volume up a bit", "volume_up"),
        ("turn the music up", "volume_up"),
        ("turn the music down", "volume_down"),
        # bare media commands
        ("next", "next"),
        ("skip this track", "next"),
        ("play some music", "play"),
        ("previous", "previous"),
    ],
)
def test_natural_phrasings_are_recognized(text: str, expected: str) -> None:
    match = route(text)
    assert match is not None, f"{text!r} was not recognized"
    assert match.intent == expected


@pytest.mark.parametrize(
    "text,volume",
    [
        ("set the volume to 50", 50),
        ("volume 70", 70),
        ("set the volume to 30 percent", 30),
        ("set volume to max", 100),
        ("set the volume to full", 100),
        ("set volume to half", 50),
        ("set the volume to minimum", 0),
        ("set volume to mute", 0),
    ],
)
def test_named_and_numeric_volume_levels(text: str, volume: int) -> None:
    match = route(text)
    assert match is not None, f"{text!r} was not recognized"
    assert match.intent == "set_volume"
    assert match.slots["volume"] == volume


def test_relative_volume_with_target_is_a_set() -> None:
    """'turn the volume up to 50' carries a number, so it is not a nudge."""
    match = route("turn the volume up to 50")
    assert match is not None
    assert match.intent == "set_volume"
    assert match.slots["volume"] == 50
    assert match.slots["direction"] == "up"


# --------------------------------------------------------------------------- #
# Wake word
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,wake,rest",
    [
        ("alvin turn on the lights", "alvin", "turn on the lights"),
        ("Alvin, stop.", "alvin", "stop"),
        ("hey alvin, please turn off the lights", "alvin", "turn off the lights"),
        ("ok alvin stop", "alvin", "stop"),
        ("turn on the lights", "", "turn on the lights"),
    ],
)
def test_detect_wake_word(text: str, wake: str, rest: str) -> None:
    from app.intents import detect_wake_word

    assert detect_wake_word(text) == (wake, rest)


def test_wake_word_required_gates_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "wake_word_required", True)
    # Without the wake word the utterance falls through to the LLM.
    assert route("turn on the lights") is None
    assert route("volume up") is None
    # With it, the command executes.
    assert route("alvin turn on the lights").intent == "lights_on"
    assert route("hey alvin, please turn off the lights").intent == "lights_off"


def test_wake_word_reported_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "wake_word_required", True)
    match = route("alvin stop")
    assert match.wake_word == "alvin"
    # No wake word needed -> the field is empty, not None.
    monkeypatch.setattr(settings, "wake_word_required", False)
    assert route("stop").wake_word == ""


def test_custom_wake_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "wake_words", "jarvis, computer")
    monkeypatch.setattr(settings, "wake_word_required", True)
    assert route("jarvis stop").intent == "stop"
    assert route("computer stop").intent == "stop"
    assert route("alvin stop") is None


def test_wake_word_required_still_falls_through_for_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wake word alone must not execute anything."""
    monkeypatch.setattr(settings, "wake_word_required", True)
    for text in ("alvin how are you", "alvin what is the capital of france"):
        assert route(text) is None


def test_wake_word_ok_bypasses_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An armed connection may run a command with no wake word in the text."""
    monkeypatch.setattr(settings, "wake_word_required", True)
    assert route_command("turn on the lights", 0.9) is None
    assert route_command("turn on the lights", 0.9, wake_word_ok=True).intent == (
        "lights_on"
    )


# --------------------------------------------------------------------------- #
# Bare wake word
#
# Regression: "hey alvin" with no command used to fall through to the LLM,
# which answered "Hello, how can I assist you today?" -- a full round-trip
# spent on a greeting, before the user had asked anything.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "hey alvin",
        "alvin",
        "hey alvin?",
        "alvin?",
        "ok alvin",
        "alvin are you there",
        "alvin you there",
        "alvin still there",
        "alvin yes",
        "alvin ready",
        "alvin listening",
    ],
)
def test_bare_wake_word_is_not_an_llm_turn(text: str) -> None:
    match = route(text)
    assert match is not None, f"{text!r} should be handled locally"
    assert match.intent == "wake"
    # A short acknowledgement, not a conversational reply.
    assert match.confirmation == "Yes?"
    assert match.wake_word


def test_wake_ack_is_short_enough_to_speak() -> None:
    """Long greetings defeat the point -- it should be one short syllable."""
    match = route("hey alvin")
    assert len(match.confirmation.split()) <= 3


@pytest.mark.parametrize(
    "text,expected",
    [
        # A wake word plus an actual command is still a command.
        ("alvin turn on the lights", "lights_on"),
        ("alvin stop", "stop"),
        ("alvin set the volume to 30", "set_volume"),
        # A wake word plus a real question still reaches the LLM.
        ("alvin how are you", None),
        ("alvin what is the capital of france", None),
        ("alvin tell me a joke", None),
    ],
)
def test_wake_word_plus_content_is_not_swallowed(text: str, expected) -> None:
    match = route(text)
    assert (match.intent if match else None) == expected


def test_wake_word_is_a_registered_intent() -> None:
    assert "wake" in SUPPORTED_INTENTS


def test_bare_wake_word_respects_confidence_gate() -> None:
    """A low-confidence 'alvin' must not trigger the ack."""
    assert route_command("hey alvin", 0.1) is None
