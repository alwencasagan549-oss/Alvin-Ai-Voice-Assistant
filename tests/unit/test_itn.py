"""Unit tests for inverse text normalization (:mod:`app.itn`).

No model, no network: pure text rewrites that make spoken phrasing
("device two", "fifty percent") match the written form command parsers
expect ("device 2", "50%").
"""

from __future__ import annotations

import pytest

from app.itn import normalize_email, normalize_transcript


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # spoken number words -> digits
        ("device two", "device 2"),
        ("the answer is twenty one", "the answer is 21"),
        ("set the volume to fifty", "set the volume to 50"),
        ("order number one thousand two hundred", "order number 1200"),
        ("the timer is one hundred fifty", "the timer is 150"),
        ("one thousand and one", "1001"),
        ("a hundred", "100"),
        ("zero", "0"),
        # ordinals
        ("twenty first street", "21st street"),
        ("third", "3rd"),
        ("fiftieth", "50th"),
        ("thirteenth", "13th"),
        # percent / currency
        ("fifty percent", "50%"),
        ("set the volume to fifty percent", "set the volume to 50%"),
        ("twenty five dollars", "$25"),
        ("three pounds", "\u00a33"),
        ("one hundred euros", "\u20ac100"),
        # time
        ("three thirty pm", "3:30 PM"),
        ("eleven fifteen am", "11:15 AM"),
        ("three o'clock", "3:00"),
        # spoken email
        ("test at domain dot org", "test@domain.org"),
        ("reach me at alvin at gmail dot com", "reach me at alvin@gmail.com"),
        # pass-through: no number words, nothing to rewrite
        ("volume up", "volume up"),
        ("stop the music", "stop the music"),
        ("hello alvin how are you", "hello alvin how are you"),
        ("", ""),
    ],
)
def test_normalize_transcript(spoken: str, expected: str) -> None:
    assert normalize_transcript(spoken) == expected


def test_bare_article_is_not_digitized() -> None:
    """A lone "a" is the article, not the number 1."""
    assert normalize_transcript("a cat") == "a cat"
    assert normalize_transcript("give it to a friend") == "give it to a friend"


def test_already_normalized_text_is_stable() -> None:
    """Running the normalizer twice must not change the first result."""
    once = normalize_transcript("set the volume to fifty percent at three thirty pm")
    assert normalize_transcript(once) == once
    assert once == "set the volume to 50% at 3:30 PM"


def test_whitespace_is_collapsed() -> None:
    assert normalize_transcript("  stop   it  ") == "stop it"


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("test at domain dot org", "test@domain.org"),
        ("alvin at example dot com", "alvin@example.com"),
        ("no address here", "no address here"),
    ],
)
def test_normalize_email(spoken: str, expected: str) -> None:
    assert normalize_email(spoken) == expected
