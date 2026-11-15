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

SHUTDOWN_RESPONSE_TEXT = "At your service, sir. I will shut down all my system."

LAPTOP_SHUTDOWN_RESPONSE_TEXT = (
    "Shutting down your laptop now, sir. Goodbye."
)

OPEN_APPS_RESPONSE_TEXT = "Done sir, I opened {app} for you."

OPEN_VS_CODE_RESPONSE_TEXT = "Done sir, I setup VS Code and open the kilo on the terminal sir just tell what next i will do sir"

OPEN_CAMERA_RESPONSE_TEXT = "Camera is open sir."
CLOSE_CAMERA_RESPONSE_TEXT = "Done sir, I close the camera."
WHAT_DO_YOU_SEE_RESPONSE_TEXT = "I am looking at the camera feed now sir."

OPEN_SYSTEM_RESPONSE_TEXT = "Opening system settings, sir."


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
# A stop that references music / YouTube / a song / a video means the user
# wants the media source closed, not just the assistant to be quiet. Bare
# "stop" / "stop it" do not qualify here.
_YOUTUBE_STOP_RE = re.compile(
    r"\b(stop|shut|close)\b.*\b(music|youtube|song|video|videos)\b",
    re.IGNORECASE,
)

def is_youtube_music_stop(text: str) -> bool:
    """True when a ``stop`` utterance actually asks to close music/YouTube."""
    return bool(_YOUTUBE_STOP_RE.search(text))

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

# Open apps command: "alvin open chrome", "open notepad", "open spotify", etc.
# Captures the app name as a slot for the server to launch via the Start menu.
_OPEN_APPS_RE = re.compile(
    r"^(?:alvin\s+)?open\s+(?P<app>.+)$", re.IGNORECASE
)
_QUERY_TIME_RE = re.compile(
    r"^(?:what'?s?\s+the\s+time|what\s+time\s+is\s+it|tell\s+me\s+(?:the\s+)?time"
    r"|current\s+time)$",
    re.IGNORECASE,
)

# Show dashboard command: "show dashboard", "show the dashboard", "open dashboard",
# "display dashboard", "alvin show dashboard", "alvin show your dashboard",
# "open your dashboard", "open up your dashboard"
_SHOW_DASHBOARD_RE = re.compile(
    r"^(?:(?:alvin\s+)?show\s+(?:the\s+|your\s+)?dashboard"
    r"|(?:alvin\s+)?open\s+(?:the\s+|your\s+)?dashboard"
    r"|(?:alvin\s+)?open\s+up\s+(?:the\s+|your\s+)?dashboard"
    r"|display\s+(?:the\s+|your\s+)?dashboard"
    r"|dashboard\s+(?:show|open|display))$",
    re.IGNORECASE,
)

# System settings command: "open your system", "open system settings",
# "open your settings", "show settings", "control panel". This is the
# dashboard-as-settings entry point; the HUD opens its embedded system
# window when it receives the matching ``command`` event. Registered before
# ``open_apps`` so "open your system" is not captured as an app launch.
_OPEN_SYSTEM_RE = re.compile(
    r"^(?:(?:open|show|display|launch)\s+"
    r"(?:your\s+|the\s+|my\s+)?"
    r"(?:system(?:\s+(?:settings|configuration|control\s+panel))?"
    r"|settings|configuration|control\s+panel)"
    r"|(?:system\s+)?(?:settings|configuration))$",
    re.IGNORECASE,
)

# "open vs code" / "open visual studio code" / "i want to code right now"
# Opens Visual Studio Code and types "kilo" in its integrated terminal.
# After wake-word + politeness stripping, "alvin i want to code right now"
# normalizes to "code right now", "i want to code" normalizes to "code".
_OPEN_VS_CODE_RE = re.compile(
    r"^(?:code(?:\s+right(?:\s+now)?)?$"
    r"|open\s+(?:visual\s+studio\s+code|vs?\.?\s*code)$"
    r"|start\s+(?:visual\s+studio\s+code|vs?\.?\s*code)$"
    r"|launch\s+(?:visual\s+studio\s+code|vs?\.?\s*code)$)",
    re.IGNORECASE,
)

# "open the camera" / "open camera" / "start camera" / "launch camera"
# Opens the laptop built-in camera and captures a frame for analysis.
_OPEN_CAMERA_RE = re.compile(
    r"^(?:open\s+(?:the\s+)?camera$|start\s+(?:the\s+)?camera$|launch\s+(?:the\s+)?camera$)",
    re.IGNORECASE,
)

# "close the camera" / "close camera" / "stop camera" / "turn off the camera"
# Releases the built-in camera and tells the client to close the camera popup.
_CLOSE_CAMERA_RE = re.compile(
    r"^(?:close\s+(?:the\s+)?camera$|stop\s+(?:the\s+)?camera$|turn\s+off\s+(?:the\s+)?camera$|shut\s+(?:the\s+)?camera$)",
    re.IGNORECASE,
)

# "what do you see" / "what can you see" / "describe what you see"
# / "what's in front of you" / "analyze the image" / "look at the camera"
# Asks the LLM to analyze the most recent camera frame.
_WHAT_DO_YOU_SEE_RE = re.compile(
    r"^(?:what\s+do\s+you\s+see$|what\s+can\s+you\s+see$"
    r"|describe\s+what\s+you\s+see$|whats\s+in\s+front\s+of\s+you$"
    r"|analyze\s+(?:the\s+)?(?:image|picture|frame|photo)$"
    r"|look\s+at\s+(?:the\s+)?camera$)",
    re.IGNORECASE,
)

# Shutdown command: "goodbye", "okay goodbye", "thank you alvin goodbye".
# The wake word ("alvin") is stripped by _normalize_after_wake, and "okay"/"ok"
# is a greeting prefix that is consumed when it appears before the wake word.
# After normalization the matched remainders are:
#   "goodbye", "okay goodbye", "ok goodbye",
#   "thank you alvin goodbye", "thank you goodbye"
_SHUTDOWN_RE = re.compile(
    rf"^(?:(?:{_word_list('okay', 'ok')}\s+)?goodbye"
    r"|thank\s+you\s+(?:alvin\s+)?goodbye)$",
    re.IGNORECASE,
)

# Fallback: a sentence that merely *contains* "goodbye" also shuts the
# assistant down ("I think it is time to say goodbye"). This is deliberately
# checked after the command registry so whole-utterance commands win first --
# "play goodbye on youtube" still plays the song. Matches STT's split "good bye"
# spelling too, but not words like "goodbyes".
_GOODBYE_ANYWHERE_RE = re.compile(r"\bgood\s*bye\b", re.IGNORECASE)

# Power off the local machine. This is deliberately a *whole-utterance* match:
# a stray "turn off my laptop" inside a longer conversation must never power
# the computer off. Wake word and politeness are stripped first, so
# "Alvin, please turn off my laptop" matches.
_LAPTOP_SHUTDOWN_RE = re.compile(
    r"^(?:(?:turn|switch|power|shut)\s+(?:off|down)\s+"
    r"(?:my\s+|the\s+)?(?:laptop|computer|pc)"
    r"|shut\s+(?:my\s+|the\s+)?(?:laptop|computer|pc)\s+down"
    r"|shutdown\s+(?:my\s+|the\s+)?(?:laptop|computer|pc)"
    r"|(?:turn|switch|power)\s+(?:my\s+|the\s+)?(?:laptop|computer|pc)\s+off)"
    r"(?:\s+(?:right\s+)?now)?$",
    re.IGNORECASE,
)

# Any utterance containing "wake up" reveals the HUD ("alvin wake up",
# "wake up now", "wake up alvin"). Like the "goodbye" fallback, the phrase is
# deliberately matched anywhere in the sentence rather than as a whole-command.
_WAKE_UP_RE = re.compile(
    r"\bwake\s+up\b",
    re.IGNORECASE,
)

# Return to the assistant's own main window ("go back to homepage"). This is
# the HUD, not the React dashboard: it closes the settings/camera pop-ups and
# raises the main window. Whole-utterance match only.
_SHOW_HOME_RE = re.compile(
    r"^(?:"
    # "go (back) (to) (the) homepage" / "get me back home" / "return home"
    r"(?:go|get|bring|take|come|return|navigate)\s+(?:me\s+)?"
    r"(?:back\s+)?(?:to\s+)?(?:the\s+)?"
    r"(?:home\s*(?:page|screen)|home|main\s*(?:page|screen|window))"
    # "back to homepage" / "back home"
    r"|(?:back|home)\s+(?:to\s+(?:the\s+)?)?"
    r"(?:home\s*(?:page|screen)|home|main\s*(?:page|screen|window))"
    # "show/open/focus (the) homepage"
    r"|(?:show|open|focus|bring\s+up|bring\s+back)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:home\s*(?:page|screen)|home|main\s*(?:page|screen|window))"
    # "focus the app/window/hud"
    r"|focus\s+(?:the\s+)?(?:app|application|window|hud|assistant)"
    r")$",
    re.IGNORECASE,
)

# ── YouTube playback ─────────────────────────────────────────────────────
# "play {song} on youtube" / "play {song} on yt"
_PLAY_ON_YOUTUBE_RE = re.compile(
    r"^(?:play|open)\s+(?P<query>.+?)\s+on\s+(?:youtube|yt)$",
    re.IGNORECASE,
)

# "play youtube {song}" / "open youtube and play {song}" / "search youtube for {song}"
_YOUTUBE_SEARCH_RE = re.compile(
    r"^(?:play|open|search)\s+(?:on\s+)?youtube"
    r"\s+(?:for\s+|and\s+play\s+)?(?P<query>.+)$",
    re.IGNORECASE,
)
# "youtube {song}"
_YOUTUBE_BARE_RE = re.compile(r"^youtube\s+(?P<query>.+)$", re.IGNORECASE)
# "play the song {song}" / "play song {song}" / "play track {song}"
_PLAY_SONG_QUALIFIED_RE = re.compile(
    r"^play\s+(?:the\s+)?(?:song|track|music\s+video|video)"
    r"\s+(?:called\s+|named\s+)?(?P<query>.+)$",
    re.IGNORECASE,
)
# Bare "play {song}" — aggressive fallback; denied-list rejects filler queries.
_PLAY_ANY_SONG_RE = re.compile(r"^play\s+(?P<query>.+)$", re.IGNORECASE)

# Queries that are playback controls, not a specific song. These are rejected
# for the bare "play X" pattern so "play music" / "play a song" stay generic.
_PLAY_GENERIC_QUERIES: frozenset[str] = frozenset({
    "it", "that", "this", "something", "anything", "nice", "again",
    "music", "the music", "some music", "a song", "the song", "song",
    "the track", "track", "the audio", "audio", "the sound", "sound",
    "the video", "video", "the game", "game", "a game", "with me", "dead",
    "loud", "louder", "more",
})


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
    # YouTube — most specific first
    _Command("play_youtube", _PLAY_ON_YOUTUBE_RE,     "Playing on YouTube."),
    _Command("play_youtube", _YOUTUBE_SEARCH_RE,      "Playing on YouTube."),
    _Command("play_youtube", _YOUTUBE_BARE_RE,        "Playing on YouTube."),
    _Command("play_youtube", _PLAY_SONG_QUALIFIED_RE, "Playing on YouTube."),
    _Command("play_youtube", _PLAY_ANY_SONG_RE,       "Playing on YouTube."),
    _Command("query_time", _QUERY_TIME_RE),
    _Command("wake_up", _WAKE_UP_RE, "Waking up for you, sir."),
    _Command("show_home", _SHOW_HOME_RE, action="focus_hud"),
    _Command("show_dashboard", _SHOW_DASHBOARD_RE, "Opening your dashboard."),
    _Command("open_system", _OPEN_SYSTEM_RE, OPEN_SYSTEM_RESPONSE_TEXT),
    _Command("open_vs_code", _OPEN_VS_CODE_RE, OPEN_VS_CODE_RESPONSE_TEXT),
    _Command("open_camera", _OPEN_CAMERA_RE, OPEN_CAMERA_RESPONSE_TEXT),
    _Command("close_camera", _CLOSE_CAMERA_RE, CLOSE_CAMERA_RESPONSE_TEXT),
    _Command("what_do_you_see", _WHAT_DO_YOU_SEE_RE, WHAT_DO_YOU_SEE_RESPONSE_TEXT),
    _Command("open_apps", _OPEN_APPS_RE, OPEN_APPS_RESPONSE_TEXT),
    _Command(
        "shutdown_laptop",
        _LAPTOP_SHUTDOWN_RE,
        LAPTOP_SHUTDOWN_RESPONSE_TEXT,
        action="laptop_shutdown",
    ),
    _Command("shutdown", _SHUTDOWN_RE, SHUTDOWN_RESPONSE_TEXT, action="shutdown"),
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
        # Bare "play X" is only a YouTube search if X looks like a song title.
        # "play music" / "play a song" / "play something" fall through to the
        # generic `play` intent above or the LLM.
        if command.intent == "play_youtube" and "query" in slots:
            q = str(slots["query"]).strip().lower()
            if len(q) < 2 or q in _PLAY_GENERIC_QUERIES:
                continue
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

    # "Goodbye" anywhere in the sentence shuts the assistant down even when it
    # is not the whole utterance. Keeping this after the registry means a real
    # command in the same sentence ("play goodbye on youtube") still wins.
    if (allowed is None or "shutdown" in allowed) and _GOODBYE_ANYWHERE_RE.search(
        normalized
    ):
        return IntentMatch(
            intent="shutdown",
            raw=raw,
            normalized=normalized,
            confirmation=SHUTDOWN_RESPONSE_TEXT,
            action="shutdown",
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
