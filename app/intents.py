"""Deterministic intent recognition and command routing for Alvin.

Fixed device actions ("stop", "mute", "volume up", "turn off the lights") are
matched with compiled regexes and slot extraction directly on the
ITN-normalized final transcript, so they never pay an LLM round-trip.
Ambiguous, multi-step, or open-ended utterances do not match and fall through
to the LLM path in :mod:`app.connection`.

Matching rules:

* The transcript is normalized with :func:`app.itn.normalize_transcript`
  first, so "set the volume to fifty" matches the same parser as
  "set the volume to 50".
* The *whole* utterance must be the command (optionally prefixed with
  "please" and the "Alvin" wake-word). "Stop the music and play jazz" stays
  an LLM turn; "stop the music" is a device command.
* Sentence punctuation STT engines append to short commands ("Mute.",
  "Alvin, mute.", "stop it!") is ignored for matching.
* The STT confidence floor (``INTENT_MIN_CONFIDENCE``) guards against
  mis-transcriptions executing hardware.

A matched command is relayed to the client as a ``command`` JSON event
(``intent`` + ``slots``) and, for assistant-side effects (stop/pause),
cancels any in-progress LLM/TTS turn. Commands never enter the LLM
conversation history: they are out-of-band device control.

**Mic control / audio gating:** ``mute_mic`` / ``unmute_mic`` ("mute",
"unmute the mic", "alvin mute", ...) flip the connection's muted state. The
``command`` event carries ``"action": "disable_stt"`` / ``"enable_stt"`` so
the client can stop streaming PCM (saving bandwidth + STT cost) and fall
back to a local keyword listener for "unmute". While muted, the server
itself runs a local-only, cloud-free keyword pass on any incoming audio so
"unmute" still works even if the client keeps streaming. These confirmations
are always spoken (``ALWAYS_CONFIRM_INTENTS``), never gated on
``INTENT_SPEAK_CONFIRMATION``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .itn import normalize_transcript

log = logging.getLogger("alvin.intents")

MUTE_RESPONSE_TEXT = (
    "Done, I successfully muted myself sir. Just say unmute so I can hear you."
)
UNMUTE_RESPONSE_TEXT = "Successfully unmuted the microphone sir."


@dataclass(frozen=True)
class IntentMatch:
    """A deterministic command match, ready to dispatch."""

    intent: str
    raw: str
    normalized: str
    slots: dict[str, Any] = field(default_factory=dict, compare=False)
    confirmation: str = ""
    action: str = ""
    #: The wake word that armed the command, when one was spoken ("" if the
    #: router is open, or the utterance matched without one).
    wake_word: str = ""


def _word_list(*words: str) -> str:
    """A non-capturing alternation of literal phrases.

    Wrapped in ``(?:...)`` so the alternation keeps its grouping when embedded
    in a larger pattern — without the group, ``^`` would bind only to the
    first alternative and ``$`` only to the last.
    """
    return "(?:" + "|".join(words) + ")"


# Whole-utterance patterns. ``^...$`` anchoring is the false-positive guard:
# a command word embedded in a longer sentence ("how do I stop my car") must
# not fire a device action.
_STOP_RE = re.compile(
    rf"^{_word_list('stop', 'stop it', 'stop speaking', 'stop talking', 'stop now', 'shut up', 'be quiet')}"
    r"(?:\s+(?:the\s+)?\w+)?$",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"^pause(?:\s+(?:the\s+)?\w+)?$", re.IGNORECASE)
_RESUME_RE = re.compile(
    r"^resum(?:e|e it|e playing)(?:\s+(?:the\s+)?\w+)?$", re.IGNORECASE
)
# Mic-control commands: "mute" / "unmute" (optionally addressed as "Alvin"
# and/or qualified with "the mic(ropophone)"). These flip the connection's
# audio-gating state on the server and instruct the client whether to keep
# streaming PCM (see Connection.is_muted / the "disable_stt"/"enable_stt"
# command actions).
_MUTE_MIC_RE = re.compile(
    r"^(?:alvin\s+)?mute(?:\s+the)?(?:\s+(?:mic|microphone))?$", re.IGNORECASE
)
_UNMUTE_MIC_RE = re.compile(
    r"^(?:alvin\s+)?unmute(?:\s+the)?(?:\s+(?:mic|microphone))?$", re.IGNORECASE
)
_MUTE_RE = re.compile(r"^mute(?:\s+(?:the\s+)?\w*)?$", re.IGNORECASE)
_UNMUTE_RE = re.compile(r"^unmute(?:\s+(?:the\s+)?\w*)?$", re.IGNORECASE)
_VOLUME_UP_WORDS = _word_list(
    "volume up",
    "louder",
    "speak up",
    "turn it up",
    "turn that up",
    "turn up the volume",
    "turn the volume up",
    "turn up volume",
    "turn up the sound",
    "turn up the music",
    "turn up the speaker",
    "turn it all the way up",
    "turn that all the way up",
    "turn the volume all the way up",
    "make it louder",
    "make it louder",
    "make everything louder",
    "increase the volume",
    "raise the volume",
    "raise the volume up",
    "volume please up",
    "turn the volume up more",
    "turn up the music",
    "turn up the sound",
    "turn up the speaker",
    "turn up the system",
    "turn the music up",
    "turn the sound up",
    "turn the speaker up",
    "turn the system up",
)
_VOLUME_UP_RE = re.compile(
    rf"^{_VOLUME_UP_WORDS}(?:\s+(?:a bit|a little|some|more|way))*$",
    re.IGNORECASE,
)
_VOLUME_DOWN_WORDS = _word_list(
    "volume down",
    "quieter",
    "softer",
    "turn it down",
    "turn that down",
    "turn down the volume",
    "turn the volume down",
    "turn down volume",
    "turn down the sound",
    "turn down the music",
    "turn down the speaker",
    "turn it all the way down",
    "turn that all the way down",
    "turn the volume all the way down",
    "make it quieter",
    "make it softer",
    "make everything quieter",
    "decrease the volume",
    "lower the volume",
    "lower the volume down",
    "turn the volume down more",
    "turn down the music",
    "turn down the sound",
    "turn down the speaker",
    "turn down the system",
    "turn the music down",
    "turn the sound down",
    "turn the speaker down",
    "turn the system down",
)
_VOLUME_DOWN_RE = re.compile(
    rf"^{_VOLUME_DOWN_WORDS}(?:\s+(?:a bit|a little|some|more|way))*$",
    re.IGNORECASE,
)
# A numeric volume target: "set the volume to 50", "volume 30", "set it to 70 percent".
_SET_VOLUME_RE = re.compile(
    r"^(?:set\s+)?(?:the\s+|it\s+)?(?:volume|sound|music|audio|speaker)"
    r"\s*(?:to|at)?\s*(?P<volume>\d{1,3})\s*(?:%|percent|points?)?$",
    re.IGNORECASE,
)
# Named levels. "max"/"full" map to 100, "min"/"mute"/"zero" to 0, "half" to 50.
_SET_VOLUME_NAMED: dict[str, int] = {
    "max": 100,
    "maximum": 100,
    "full": 100,
    "loudest": 100,
    "min": 0,
    "minimum": 0,
    "zero": 0,
    "mute": 0,
    "off": 0,
    "half": 50,
    "middle": 50,
}
_SET_VOLUME_NAMED_RE = re.compile(
    r"^(?:set\s+)?(?:the\s+|it\s+)?(?:volume|sound|music|audio|speaker)"
    r"\s*(?:to|at)?\s*(?P<level>"
    + "|".join(sorted(_SET_VOLUME_NAMED, key=len, reverse=True))
    + r")\s*(?:%|percent)?$",
    re.IGNORECASE,
)
# Relative volume with an explicit target: "turn the volume up to 50".
_VOLUME_SET_RE = re.compile(
    r"^turn\s+(?:the\s+)?(?:volume|sound|music|audio|speaker)\s+"
    r"(?P<direction>up|down)\s+(?:to\s+)?(?P<volume>\d{1,3})\s*(?:%|percent)?$",
    re.IGNORECASE,
)
_LIGHTS_ON_RE = re.compile(
    r"^(?:(?:turn|switch)\s+on\s+(?:the\s+)?lights?"
    r"|turn\s+(?:the\s+)?lights?\s+on"
    r"|lights?\s+on)$",
    re.IGNORECASE,
)
_LIGHTS_OFF_RE = re.compile(
    r"^(?:(?:turn|switch)\s+off\s+(?:the\s+)?lights?"
    r"|turn\s+(?:the\s+)?lights?\s+off"
    r"|lights?\s+off)$",
    re.IGNORECASE,
)
_NEXT_RE = re.compile(
    rf"^{_word_list('next', 'next track', 'next song', 'next channel', 'next page', 'next video', 'next episode', 'next item', 'skip', 'skip this', 'skip it', 'skip that', 'skip this track', 'skip this song', 'skip to next', 'skip to the next')}$",
    re.IGNORECASE,
)
_PREVIOUS_RE = re.compile(
    rf"^{_word_list('previous', 'previous track', 'previous song', 'previous channel', 'previous page', 'previous video', 'previous episode', 'previous item', 'go back', 'back', 'last track', 'last song')}$",
    re.IGNORECASE,
)
_PLAY_RE = re.compile(
    rf"^{_word_list('play', 'play it', 'resume playback', 'play music', 'play the music', 'play some music', 'play sound', 'play the sound', 'play audio', 'play the audio', 'play song', 'play the song', 'start the music', 'start music', 'put on some music', 'put music on')}$",
    re.IGNORECASE,
)
_QUERY_TIME_RE = re.compile(
    r"^(?:what'?s?\s+the\s+time|what\s+time\s+is\s+it|tell\s+me\s+(?:the\s+)?time"
    r"|current\s+time)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Command:
    intent: str
    pattern: re.Pattern
    confirmation: str = ""
    action: str = ""


# Order matters only for readability: every pattern is whole-utterance, so two
# commands cannot both match the same string.
_COMMANDS: tuple[_Command, ...] = (
    _Command("stop", _STOP_RE),
    _Command("pause", _PAUSE_RE),
    _Command("resume", _RESUME_RE),
    # Mic-control first: bare "mute"/"unmute" are mic commands; the qualified
    # forms ("mute the speaker/music/...") fall through to the device intents.
    _Command(
        "mute_mic",
        _MUTE_MIC_RE,
        MUTE_RESPONSE_TEXT,
        action="disable_stt",
    ),
    _Command(
        "unmute_mic",
        _UNMUTE_MIC_RE,
        UNMUTE_RESPONSE_TEXT,
        action="enable_stt",
    ),
    _Command("mute", _MUTE_RE, "Muting."),
    _Command("unmute", _UNMUTE_RE, "Unmuting."),
    _Command("set_volume", _SET_VOLUME_RE, "Volume set."),
    # Named levels ("set volume to max") are resolved to a number in slots.
    _Command("set_volume", _SET_VOLUME_NAMED_RE, "Volume set."),
    # "turn the volume up to 50" carries both a direction and a target.
    _Command("set_volume", _VOLUME_SET_RE, "Volume set."),
    _Command("volume_up", _VOLUME_UP_RE, "Turning the volume up."),
    _Command("volume_down", _VOLUME_DOWN_RE, "Turning the volume down."),
    _Command("lights_on", _LIGHTS_ON_RE, "Turning the lights on."),
    _Command("lights_off", _LIGHTS_OFF_RE, "Turning the lights off."),
    _Command("next", _NEXT_RE, "Next."),
    _Command("previous", _PREVIOUS_RE, "Previous."),
    _Command("play", _PLAY_RE, "Playing."),
    _Command("query_time", _QUERY_TIME_RE),
)

#: Intents that act on the assistant's own output and must cancel any
#: in-progress LLM stream / TTS before the event is emitted.
ASSISTANT_SIDE_EFFECT_INTENTS: frozenset[str] = frozenset({"stop", "pause"})

#: Mic-control intents. They flip the connection's audio-gating state, so the
#: confirmation is always spoken (it is the only feedback the user gets, and
#: after a mute the user must hear how to unmute again).
MIC_CONTROL_INTENTS: frozenset[str] = frozenset({"mute_mic", "unmute_mic"})
ALWAYS_CONFIRM_INTENTS: frozenset[str] = frozenset(MIC_CONTROL_INTENTS)

# A wake word on its own ("hey alvin", "alvin?") means "I'm here, go ahead".
# It is not a question: routing it to the LLM burns a full round-trip on the
# greeting and produces "Hello, how can I assist you today?" before the user has
# said anything. It is answered with a short acknowledgement instead.
_WAKE_ONLY_RE = re.compile(
    r"^(?:(?:are\s+you\s+there|you\s+there|still\s+there|there)\s*[?.!]*"
    r"|(?:yes|ya|yeah|yep|ok|okay|here|listening|ready)\s*[?.!]*)$",
    re.IGNORECASE,
)

#: Spoken when the user says only the wake word. Short on purpose: it confirms
#: the assistant is listening without starting a conversation the user has not
#: asked for.
WAKE_CONFIRMATION = "Yes?"

#: Intent names, in registry order, for ``/health`` and diagnostics.
SUPPORTED_INTENTS: tuple[str, ...] = (
    "wake",
    *tuple(dict.fromkeys(command.intent for command in _COMMANDS)),
)

# Addressing/affirmation noise stripped from either end of the utterance:
# "please", "the" wake-word, and polite request wrappers, in any combination
# ("please", "alvin", "hey alvin", "can you please", ...).
#
# Trailing "please" is stripped too: STT output is frequently
# "turn the lights off please", which used to fail the whole-utterance anchor.
# Greeting/filler tokens allowed *before* the wake word ("hey alvin", "ok alvin").
_GREETING_RE = re.compile(r"^(?:(?:hey|hi|hello|ok|okay|yo)\s+)*", re.IGNORECASE)
_POLITE_LEAD_RE = re.compile(
    r"^(?:(?:please|now|alvin)\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:(?:i\s+)?(?:want|wanna|need)\s+(?:to\s+)?|"
    r"(?:i'?d\s+like\s+to\s+)|please\s+)*",
    re.IGNORECASE,
)
_POLITE_TAIL_RE = re.compile(
    r"(?:\s+(?:please|now|thanks|thank you|ok|okay))+$", re.IGNORECASE
)

# Sentence punctuation that STT engines append to short commands ("Mute.",
# "Alvin, mute"). Replaced with spaces so the whole-utterance patterns can
# match; in-word apostrophes ("what's") are untouched.
_PUNCT_RE = re.compile(r"[.,!?;:]+")


def _wake_word_pattern() -> tuple[re.Pattern, tuple[str, ...]]:
    """Compile the configured wake words into a leading-prefix pattern."""
    words = tuple(
        w.strip().lower() for w in (settings.wake_words or "").split(",") if w.strip()
    )
    if not words:
        return re.compile(r"(?!)"), ()  # never matches
    # NOTE: the pattern is deliberately NOT anchored with "^". It is matched with
    # an explicit ``pos`` after the greeting prefix, and "``^``" in Python only
    # ever matches at the true start of the string, never at ``pos``.
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"({alternation})\b[\s,]*", re.IGNORECASE), words


def _strip_polite(text: str) -> str:
    """Drop trailing/leading politeness noise and collapse whitespace.

    Applied after the wake word has been split off, so the wake word itself is
    never consumed as an address prefix.
    """
    normalized = _POLITE_TAIL_RE.sub("", text)
    normalized = _POLITE_LEAD_RE.sub("", normalized)
    return re.sub(r"\s+", " ", normalized).lower().strip()


def _normalize_after_wake(text: str) -> tuple[str, str]:
    """Split a leading wake word off an utterance and strip politeness.

    Returns ``(wake_word, command_remainder)``. The remainder is exactly the
    string :func:`route_command` matches its command patterns against, so
    :func:`normalize_for_matching` and :func:`detect_wake_word` always agree on
    the same normalized text.

    The wake word is detected *before* politeness stripping, otherwise the
    ``alvin`` token is consumed as an address prefix and never reported.
    """
    # Punctuation is normalized *before* the greeting/wake-word search so that
    # "hey alvin, stop" is seen as "hey alvin stop" (a comma would otherwise
    # stop the greeting prefix from matching).
    normalized = _PUNCT_RE.sub(" ", normalize_transcript(text))
    pattern, _words = _wake_word_pattern()
    # Allow a greeting in front: "hey alvin", "ok alvin".
    greeting = _GREETING_RE.match(normalized)
    search_from = greeting.end() if greeting else 0
    match = pattern.match(normalized, search_from)
    if match is None:
        return "", _strip_polite(normalized)
    remainder = normalized[match.end() :]
    return match.group(1).lower(), _strip_polite(remainder)


def normalize_for_matching(text: str) -> str:
    """ITN + punctuation + politeness + wake-word stripping.

    This is exactly the string :func:`route_command` matches its command
    patterns against, so callers that want to pre-normalize a transcript for
    the same patterns get a consistent result -- including the wake word and
    leading greetings being stripped. "hey alvin, please turn off the lights"
    therefore normalizes to "turn off the lights" rather than leaving the
    greeting in place, which would match nothing.
    """
    _, remainder = _normalize_after_wake(text)
    return remainder


def detect_wake_word(text: str) -> tuple[str, str]:
    """Split a leading wake word off an utterance.

    Returns ``(wake_word, remainder)``. ``remainder`` is the command text with
    the wake word (and any politeness noise) removed, so "hey alvin, please turn
    the lights off" yields ``("alvin", "turn the lights off")``. When no wake
    word is present the original normalized text is returned with ``""``.

    The wake word must be detected *before* politeness stripping, otherwise the
    ``alvin`` token is consumed as an address prefix and never reported.
    """
    return _normalize_after_wake(text)


def route_command(
    text: str,
    confidence: float | None = None,
    allowed: frozenset[str] | None = None,
    wake_word_ok: bool = False,
) -> IntentMatch | None:
    """Match a final transcript against the deterministic command registry.

    Returns an :class:`IntentMatch` when the whole utterance is a recognized
    device command, or ``None`` when it should be handled by the LLM. The
    STT ``confidence`` gate is applied only when a value is provided, so
    callers without a confidence estimate are not silently skipped.
    ``allowed`` restricts matching to a subset of intents (used by the
    muted-mode keyword pass, which listens for mic-control only).

    **Wake word:** when ``WAKE_WORD_REQUIRED`` is on, the command only executes
    if the utterance is armed by a configured wake word, or ``wake_word_ok`` is
    passed (the connection already holds an unexpired wake window). Otherwise
    ``None`` is returned and the utterance falls through to the LLM.
    """
    if not settings.intent_enabled:
        return None
    if confidence is not None and confidence < settings.intent_min_confidence:
        return None

    raw = text.strip()
    if not raw:
        return None

    wake_word, normalized = detect_wake_word(text)

    #: A wake word with nothing after it ("hey alvin", "alvin?") is a "go ahead",
    # not a question. Answer it locally with a short acknowledgement instead of
    # spending an LLM round-trip on the greeting.
    #
    # ``allowed`` must be respected here too: the muted-mode keyword pass passes
    # MIC_CONTROL_INTENTS, and "wake" is not in it. Without this check a bare
    # "hey alvin" spoken while muted would make the assistant talk, which is
    # exactly what muting is meant to prevent. This only disables the wake
    # branch below -- the command loop still honours ``allowed`` normally.
    wake_allowed = allowed is None or "wake" in allowed
    if wake_allowed and wake_word and not normalized:
        return IntentMatch(
            intent="wake",
            raw=raw,
            normalized=normalized,
            confirmation=WAKE_CONFIRMATION,
            wake_word=wake_word,
        )
    if wake_allowed and wake_word and _WAKE_ONLY_RE.match(normalized):
        return IntentMatch(
            intent="wake",
            raw=raw,
            normalized=normalized,
            confirmation=WAKE_CONFIRMATION,
            wake_word=wake_word,
        )
    if not normalized:
        return None

    if settings.wake_word_required and not wake_word and not wake_word_ok:
        log.debug("wake-word required; ignoring %r", raw[:60])
        return None

    for command in _COMMANDS:
        if allowed is not None and command.intent not in allowed:
            continue
        match = command.pattern.search(normalized)
        if match is None:
            continue
        slots: dict[str, Any] = {
            name: value for name, value in match.groupdict().items() if value
        }
        # Named levels ("set volume to max") resolve to a number here so
        # downstream consumers only ever see an int.
        if "level" in slots:
            slots["volume"] = _SET_VOLUME_NAMED.get(slots.pop("level").lower(), 0)
        if "volume" in slots:
            try:
                slots["volume"] = int(slots["volume"])
            except (TypeError, ValueError):
                return None
        # "turn the volume up to 50" is a set, not a relative nudge.
        if "direction" in slots:
            slots.setdefault("mode", "set")
        return IntentMatch(
            intent=command.intent,
            raw=raw,
            normalized=normalized,
            slots=slots,
            confirmation=command.confirmation,
            action=command.action,
            wake_word=wake_word,
        )
    return None


def match_mic_control_intents(normalized_text: str) -> dict[str, Any] | None:
    """Match a phrase against the mic-control set only (mute / unmute).

    Returns ``{"intent": "mute_mic" | "unmute_mic", "confidence": 1.0}`` —
    the confidence is 1.0 by construction (deterministic regex match), not an
    STT estimate. ``None`` when the phrase is not a mic-control command.
    """
    match = route_command(normalized_text, allowed=MIC_CONTROL_INTENTS)
    if match is None:
        return None
    return {"intent": match.intent, "confidence": 1.0}
