"""Unit tests for the mic-control matcher (mute / unmute the microphone).

Covers :func:`app.intents.match_mic_control_intents`, the deterministic
subset of the intent registry that drives the connection's audio-gating
state (``disable_stt`` / ``enable_stt``).
"""

from __future__ import annotations

import pytest

from app.intents import match_mic_control_intents


@pytest.mark.parametrize(
    "phrase",
    [
        "mute",
        "mute the mic",
        "mute the microphone",
        "alvin mute",
        "alvin mute the microphone",
    ],
)
def test_mute_intent_variations(phrase: str) -> None:
    res = match_mic_control_intents(phrase)
    assert res is not None, f"{phrase!r} should match the mic-control set"
    assert res["intent"] == "mute_mic"
    assert res["confidence"] == 1.0


@pytest.mark.parametrize(
    "phrase",
    [
        "unmute",
        "unmute the mic",
        "unmute the microphone",
        "alvin unmute",
        "alvin unmute the microphone",
    ],
)
def test_unmute_intent_variations(phrase: str) -> None:
    res = match_mic_control_intents(phrase)
    assert res is not None, f"{phrase!r} should match the mic-control set"
    assert res["intent"] == "unmute_mic"
    assert res["confidence"] == 1.0


@pytest.mark.parametrize(
    "phrase",
    [
        "mute the speaker",  # device intent, not mic control
        "turn off the lights",
        "stop the music",
        "volume up",
        "what time is it",
        "hello",
        "",
    ],
)
def test_non_mic_phrases_do_not_match(phrase: str) -> None:
    assert match_mic_control_intents(phrase) is None


def test_matching_is_case_insensitive() -> None:
    assert match_mic_control_intents("MUTE THE MIC")["intent"] == "mute_mic"
    assert match_mic_control_intents("Alvin Unmute")["intent"] == "unmute_mic"


# --------------------------------------------------------------------------- #
# The wake ack must not defeat muting.
#
# Regression: the muted-mode keyword pass restricts matching to
# MIC_CONTROL_INTENTS. The bare-wake-word branch was added outside that filter,
# so saying "hey alvin" while muted produced a ``wake`` match -- the assistant
# would speak, which is exactly what muting exists to prevent.
# --------------------------------------------------------------------------- #
def test_wake_word_is_ignored_while_muted() -> None:
    from app.connection import MIC_CONTROL_INTENTS
    from app.intents import route_command

    match = route_command("hey alvin", 0.9, allowed=MIC_CONTROL_INTENTS)
    assert match is None, "a bare wake word must not make a muted assistant speak"


def test_muted_keyword_pass_still_handles_unmute() -> None:
    """Guards the fix above from over-correcting and breaking muting."""
    from app.connection import MIC_CONTROL_INTENTS
    from app.intents import route_command

    for text in ("unmute", "unmute the mic", "alvin unmute"):
        match = route_command(text, 0.9, allowed=MIC_CONTROL_INTENTS)
        assert match is not None, f"{text!r} must still unmute"
        assert match.intent == "unmute_mic"


def test_muted_keyword_pass_ignores_other_commands() -> None:
    from app.connection import MIC_CONTROL_INTENTS
    from app.intents import route_command

    for text in ("turn on the lights", "volume up", "stop"):
        assert route_command(text, 0.9, allowed=MIC_CONTROL_INTENTS) is None
