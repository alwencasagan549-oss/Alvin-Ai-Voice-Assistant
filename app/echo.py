"""Echo / self-trigger suppression.

The assistant speaks through the same room (or the same virtual cable) that its
microphone listens to, so playback leaks into the capture stream. Without
intervention the pipeline hears itself:

    TTS -> speakers -> mic -> VAD -> STT -> LLM -> TTS -> ...

which is what users describe as the assistant "talking to itself".

Three independent defenses, all cheap:

1. **Playback gate** -- while audio is being produced, incoming mic audio is
   dropped entirely (``ECHO_MODE=gate``). Strictly half-duplex: it cannot loop,
   but it also disables barge-in.
2. **Self-echo filter** -- the default. Audio is still transcribed, so barge-in
   keeps working, but a *final* transcript that closely resembles what the
   assistant recently said is dropped before it reaches the LLM.
3. **Cooldown** -- after playback stops, mic audio is ignored for
   ``ECHO_COOLDOWN_MS``. Room reverb and headset output latency mean the tail of
   the assistant's own voice arrives *after* ``tts_end``.

The similarity check deliberately compares against *all* recently spoken
sentences, not just the most recent one, because a loop can span several turns.
"""

from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher

from .config import settings

# Words carrying no discriminative signal. Two utterances that differ only in
# these are considered the same for echo purposes.
_FILLER = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "so",
    "if",
    "then",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "me",
    "my",
    "your",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "should",
    "there",
    "here",
    "not",
    "no",
    "yes",
    "okay",
    "ok",
    "um",
    "uh",
    "like",
    "just",
    "very",
    "really",
}

_WS = re.compile(r"\s+")

# A fragment shorter than this is not compared by containment: a bare "stop" or
# "yes" is contained in almost any reply and would be silently swallowed.
_MIN_CONTAINMENT_TOKENS = 3


def _sentences(text: str) -> list[str]:
    """Split into sentence-like chunks, keeping the whole text as a fallback.

    Echo arrives fragmented: VAD endpoints on the pause between sentences, so
    "Hello Alvin. How are you today?" can come back as two separate utterances,
    each matching only half of what was spoken. Comparing against individual
    sentences (as well as the full text) catches those fragments.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _containment(needle: str, haystack: str) -> float:
    """How much of ``needle`` is covered by ``haystack``, ignoring order.

    Fragmented echo ("How are you today?") scores poorly on a strict sequence
    ratio against the full utterance, but every one of its words is present in
    what we said. This asymmetric measure catches that case without matching
    short unrelated commands ("stop") against long replies, because a minimum
    token count is enforced by the caller.
    """
    n, h = set(_tokens(needle)), set(_tokens(haystack))
    if not n or not h:
        return 0.0
    return len(n & h) / len(n)


def _tokens(text: str) -> list[str]:
    """Lowercase, de-punctuate, and drop fillers for a stable comparison."""
    cleaned = _WS.sub(" ", re.sub(r"[^\w\s%]", " ", text.lower())).strip()
    return [w for w in cleaned.split() if w not in _FILLER]


def similarity(a: str, b: str) -> float:
    """Ratio (0-1) of how much two utterances say the same thing.

    Returns ``1.0`` for two empty strings so an empty transcript never looks
    like an echo of empty speech, and ``0.0`` when either side has no
    content words left after filler removal.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def looks_like_echo(text: str, spoken: list[str], threshold: float) -> bool:
    """True when ``text`` closely matches anything in ``spoken``.

    Three comparisons are tried, because leaked playback reaches the recognizer
    in several shapes:

    * full-text ratio (verbatim echo)
    * per-sentence ratio (echo split on the VAD pause between sentences)
    * word containment, with a minimum-length guard so a one-word command
      ("stop") is never mistaken for an echo of a long reply

    A match against *any* recent sentence counts, so a loop that re-enters after
    several turns is still caught.
    """
    if not spoken:
        return False
    text_tokens = _tokens(text)
    if not text_tokens:
        return False
    allow_containment = len(text_tokens) >= _MIN_CONTAINMENT_TOKENS
    for line in spoken:
        candidates = [line, *_sentences(line)]
        for candidate in candidates:
            if similarity(text, candidate) >= threshold:
                return True
        if allow_containment:
            for candidate in candidates:
                if _containment(text, candidate) >= threshold:
                    return True
    return False


class EchoGuard:
    """Per-connection echo state: playback flag, cooldown, spoken-text history."""

    def __init__(self) -> None:
        # Circular buffer of (monotonic timestamp, text) for the similarity check.
        self._spoken: deque[tuple[float, str]] = deque(maxlen=32)
        self._speaking = False
        self._cooldown_until = 0.0

    # -- playback state ------------------------------------------------- #
    @property
    def speaking(self) -> bool:
        return self._speaking

    def begin_speech(self, text: str) -> None:
        """Mark playback as active and remember what is being said."""
        self._speaking = True
        if text.strip():
            self._spoken.append((time.monotonic(), text))

    def end_speech(self) -> None:
        """Mark playback as finished and start the post-TTS cooldown."""
        self._speaking = False
        self._cooldown_until = time.monotonic() + settings.echo_cooldown_ms / 1000.0

    def should_accept_audio(self) -> bool:
        """Whether an incoming mic frame may be fed to the VAD.

        ``False`` during playback in ``gate`` mode and during the cooldown. In
        ``filter`` mode playback is allowed through (so barge-in works) and only
        the cooldown applies.
        """
        if time.monotonic() < self._cooldown_until:
            return False
        return not (self._speaking and settings.echo_mode == "gate")

    def reset_cooldown(self) -> None:
        """Reset the post-TTS cooldown for immediate barge-in.

        Called when the user explicitly interrupts (e.g., presses push-to-talk
        hotkey) so the mic becomes active immediately without waiting for the
        cooldown period.
        """
        self._cooldown_until = 0.0

    # -- self-echo filter ------------------------------------------------ #
    def recent_speech(self) -> list[str]:
        """Spoken sentences still inside ``ECHO_HISTORY_SEC``."""
        cutoff = time.monotonic() - settings.echo_history_sec
        return [t for ts, t in self._spoken if ts >= cutoff]

    def is_echo(self, text: str) -> bool:
        """True when ``text`` is (probably) the assistant hearing itself."""
        if settings.echo_mode != "filter":
            return False
        return looks_like_echo(
            text, self.recent_speech(), settings.echo_similarity_threshold
        )
