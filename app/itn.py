"""Inverse Text Normalization (ITN) for Alvin transcripts.

Speech-to-text engines emit raw spoken phrasing: "device two", "fifty
percent", "twenty first street", "test at domain dot org". Deterministic
intent matching (see :mod:`app.intents`) needs consistent written
representations, so every final transcript is normalized *before* command
matching:

    "set the volume to fifty"            -> "set the volume to 50"
    "turn the volume up to fifty percent"-> "turn the volume up to 50%"
    "twenty first street"                -> "21st street"
    "three thirty pm"                    -> "3:30 PM"
    "twenty five dollars"                -> "$25"
    "test at domain dot org"             -> "test@domain.org"

The normalizer is deliberately conservative: only well-known spoken forms are
rewritten, and already-normalized text passes through unchanged. It is a pure
text function with no I/O, so it stays off the hot path and is trivially
unit-testable.
"""

from __future__ import annotations

import re

# --- spoken number words -------------------------------------------------- #

_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
# Ordinal words are spelled out explicitly (a derived table would be wrong).
_ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}

# A word that can start or extend a spoken-number run. "a"/"an" are included
# only because they lead number phrases like "a hundred"; a lone "a" is the
# article and is left alone (see _parse_number_words).
_NUMBER_WORDS: frozenset[str] = frozenset(
    set(_ONES) | set(_TENS) | set(_SCALES) | set(_ORDINALS) | {"a", "an", "and"}
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*", re.UNICODE)

# Clock times spoken as "three thirty pm": the hour and minute words are
# converted before the generic number pass, because "three thirty" would
# otherwise digitize as the number 33 instead of 3:30.
_TIME_WORDS_HOUR: dict[str, int] = {
    word: value
    for value, word in enumerate(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ],
        start=1,
    )
}
_TIME_WORDS_MINUTE: dict[str, int] = {
    "zero": 0,
    "five": 5,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "thirty five": 35,
    "forty": 40,
    "forty five": 45,
    "fifty": 50,
    "fifty five": 55,
}
_TIME_OF_DAY_RE = re.compile(
    r"\b(?P<hour>"
    + "|".join(_TIME_WORDS_HOUR)
    + r")\s+(?P<minute>"
    + "|".join(sorted(_TIME_WORDS_MINUTE, key=len, reverse=True))
    + r")\s+(?P<meridiem>am|pm)\b",
    re.IGNORECASE,
)

# --- post-pass patterns (applied after number words are digitized) -------- #

_PERCENT_RE = re.compile(r"\b(\d{1,4})\s+percent\b")
_CURRENCY_RES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(\d+)\s+dollars?\b"), "$"),
    (re.compile(r"\b(\d+)\s+pounds?\b"), "\u00a3"),
    (re.compile(r"\b(\d+)\s+euros?\b"), "\u20ac"),
    (re.compile(r"\b(\d+)\s+pesos?\b"), "\u20b1"),
)
_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})\s+(\d{2})\s+(am|pm)\b", re.IGNORECASE)
_OCLOCK_RE = re.compile(r"\b(\d{1,2})\s+(?:o'?clock|oclock)\b", re.IGNORECASE)


def _ordinal_suffix(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _parse_number_words(words: list[str]) -> str | None:
    """Parse a spoken-number run ("one hundred twenty three") into a string.

    Returns ``None`` when the run is not a real number (e.g. a lone article
    "a"), so the caller leaves the original text untouched.
    """
    if not words:
        return None
    is_ordinal = False
    saw_number = False
    total = 0
    current = 0
    last = len(words) - 1
    for index, word in enumerate(words):
        if word == "and":
            continue  # British filler: "one thousand and one"
        if word in ("a", "an"):
            if index == last:
                return None  # a bare "a" / "an" is the article, not a number
            current += 1
            continue
        saw_number = True
        if index == last and word in _ORDINALS:
            current += _ORDINALS[word]
            is_ordinal = True
        elif word in _ONES or word in _TENS:
            current += _ONES[word] if word in _ONES else _TENS[word]
        elif word in _SCALES:
            current = (current or 1) * _SCALES[word]
            total += current
            current = 0
        else:
            return None
    total += current
    if not saw_number:
        return None
    if total == 0:
        return "0"
    if is_ordinal:
        return f"{total}{_ordinal_suffix(total)}"
    return str(total)


def _digitize_number_words(text: str) -> str:
    """Rewrite maximal runs of spoken number words into digits/ordinals."""
    tokens = list(_WORD_RE.finditer(text))
    if not tokens:
        return text
    out: list[str] = []
    last_end = 0
    i = 0
    while i < len(tokens):
        start = tokens[i]
        if start.group().lower() in _NUMBER_WORDS:
            # Extend the run over contiguous, adjacent number words.
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if nxt.start() - tokens[j - 1].end() > 1:
                    break
                if nxt.group().lower() not in _NUMBER_WORDS:
                    break
                j += 1
            run = [tokens[k].group().lower() for k in range(i, j)]
            rendered = _parse_number_words(run)
            if rendered is not None:
                out.append(text[last_end : start.start()])
                out.append(rendered)
                last_end = tokens[j - 1].end()
            i = j
            continue
        i += 1
    out.append(text[last_end:])
    return "".join(out)


def normalize_email(text: str) -> str:
    """Rewrite spoken email addresses: "test at domain dot org" -> "test@domain.org"."""
    email_pattern = re.compile(
        r"\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+\s+dot\s+[a-zA-Z]{2,})\b",
        re.IGNORECASE,
    )

    def replace_email(match: re.Match) -> str:
        local_part = match.group(1)
        domain_part = match.group(2).replace(" dot ", ".")
        return f"{local_part}@{domain_part}"

    return email_pattern.sub(replace_email, text)


def _replace_time_of_day(match: re.Match) -> str:
    hour = _TIME_WORDS_HOUR[match.group("hour").lower()]
    minute = _TIME_WORDS_MINUTE[match.group("minute").lower()]
    meridiem = "AM" if match.group("meridiem").lower().startswith("a") else "PM"
    return f"{hour}:{minute:02d} {meridiem}"


def _replace_ampm(match: re.Match) -> str:
    hour = int(match.group(1))
    minute = match.group(2)
    meridiem = "AM" if match.group(3).lower().startswith("a") else "PM"
    return f"{hour}:{minute} {meridiem}"


def normalize_transcript(text: str) -> str:
    """ITN a raw transcript into the written form intent parsers expect.

    Order matters: emails are normalized first (their domain part still reads
    "dot ..."), then spoken clock times ("three thirty pm"), then the remaining
    spoken number words become digits, and finally percent/currency/time
    patterns are applied to the digitized text.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_email(text)
    text = _TIME_OF_DAY_RE.sub(_replace_time_of_day, text)
    text = _digitize_number_words(text)
    text = _PERCENT_RE.sub(r"\1%", text)
    for pattern, symbol in _CURRENCY_RES:
        text = pattern.sub(rf"{symbol}\1", text)
    text = _TIME_AMPM_RE.sub(_replace_ampm, text)
    text = _OCLOCK_RE.sub(r"\1:00", text)
    return text
