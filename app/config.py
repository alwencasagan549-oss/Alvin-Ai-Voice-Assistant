"""Configuration for the Alvin STT service.

All settings are read from environment variables (loaded from a ``.env`` file
when present) with sensible defaults that match the service specification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(value: str, default: str) -> str:
    return os.getenv(value, default)


def _env_int(value: str, default: int) -> int:
    raw = os.getenv(value)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"environment variable {value}={raw!r} is not a valid integer"
        ) from None


def _env_float(value: str, default: float) -> float:
    raw = os.getenv(value)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"environment variable {value}={raw!r} is not a valid float"
        ) from None


def _env_bool(value: str, default: bool) -> bool:
    raw = os.getenv(value)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# All environment variables the service reads, with legacy aliases grouped by "|".
# Used by tests to EnvironmentVarGuard and reset the environment completely.
ENV_VAR_NAMES = [
    "HOST",
    "PORT",
    "LOG_LEVEL",
    "WHISPER_LANGUAGE",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_LANGUAGE",
    "GROQ_FALLBACK",
    "GROQ_TIMEOUT_SEC",
    "GROQ_TIMEOUT_BASE_SEC",
    "GROQ_TIMEOUT_SLOPE_SEC",
    "GROQ_INITIAL_PROMPT",
    "GROQ_WARMUP",
    "FALLBACK_MODEL",
    "PRELOAD_FALLBACK",
    "CPU_THREADS",
    "SILENCE_THRESHOLD_DBFS",
    "VAD_THRESHOLD",
    "MIN_SILENCE_DURATION_MS",
    "SPEECH_PAD_MS",
    "SAMPLE_RATE",
    "PARTIAL_INTERVAL_MS",
    "MIN_SEGMENT_SECONDS",
    "MAX_SEGMENT_SECONDS",
    "TTS_VOICE",
    "TTS_PREFETCH",
    "TTS_SILENCE_THRESHOLD_DBFS",
    "TTS_TRIM_LEADING_MS",
    "TTS_TRIM_TRAILING_MS",
    "TTS_SENTENCE_GAP_MS",
    "TTS_DECODE_INTERVAL_BYTES",
    "TTS_RETRIES",
    "TTS_RETRY_BACKOFF_MS",
    "TTS_TIMEOUT_SEC",
    "TTS_AUTO_SPEAK",
    "TTS_WARMUP",
    "MAX_CONNECTIONS",
    "MAX_FRAME_SIZE",
    "TRANSCRIPTION_QUEUE_MAXSIZE",
    "INTENT_ENABLED",
    "INTENT_SPEAK_CONFIRMATION",
    "INTENT_MIN_CONFIDENCE",
    "WAKE_WORDS",
    "WAKE_WORD_REQUIRED",
    "WAKE_WORD_WINDOW_SEC",
    "WAKE_SPEAK_ACK",
    "PUSH_TO_TALK",
    "PUSH_TO_TALK_TIMEOUT_MS",
    "PUSH_TO_TALK_ECHO_GATE",
    "API_KEY|SERVICE_AUTH_TOKEN",
    "LLM_API_KEY|NVIDIA_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_ENABLED",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_SYSTEM_PROMPT",
    "LLM_TIMEOUT_SEC",
    "LLM_WARMUP",
    "CF_API_KEY",
    "CF_ACCOUNT_ID",
    "CF_BASE_URL",
    "CF_MODEL",
    "CF_ENABLED",
    "CF_TIMEOUT_SEC",
    "MAX_LLM_HISTORY",
    "LLM_MAX_HISTORY_TOKENS",
    "ECHO_MODE",
    "ECHO_SIMILARITY_THRESHOLD",
    "ECHO_HISTORY_SEC",
    "ECHO_COOLDOWN_MS",
    "TAVILY_API_KEY",
    "DEBUG",
    "LLM_USE_CF_AS_PRIMARY",
    "SPEAKER_ENABLED",
    "SPEAKER_SIMILARITY_THRESHOLD",
    "SPEAKER_MIN_IDENTIFY_SEC",
    "SPEAKER_MAX_IDENTIFY_SEC",
    "SPEAKER_PROFILES_PATH",
]

_NONE = {"", "auto", "none", "null"}

# Default system prompt steering Alvin's persona. Kept short and TTS-friendly.
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You are Alvin, a helpful, concise AI voice assistant. "
    "Keep responses short and natural, suitable for text-to-speech. "
    "Do not use emojis or markdown. Answer in plain text only."
)

# Steers Whisper toward the local vocabulary and the punctuation style the
# assistant cares about (proper nouns, alphanumeric IDs, email addresses).
DEFAULT_INITIAL_PROMPT = (
    "Cagayan de Oro City, order IDs like 4829-B, "
    "email addresses like test@domain.org. "
    "The speaker is talking to a voice assistant. "
    "Ignore silence and background noise."
)


@dataclass
class Settings:
    """Runtime configuration shared across the service."""

    # --- server ------------------------------------------------------- #
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))

    # --- language -------------------------------------------------------- #
    # The only language knob: the session default handed to every Connection.
    # The local backend hardcodes cpu/int8/beam 1 and never conditions on
    # previous text, so there are no model/decoding knobs to tune here.
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "en"))

    # --- groq (cloud primary) ------------------------------------------ #
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "whisper-large-v3-turbo"))
    groq_language: str = field(default_factory=lambda: _env("GROQ_LANGUAGE", "en"))
    # When enabled a Groq timeout/error falls back to the local model.
    groq_fallback: bool = field(default_factory=lambda: _env_bool("GROQ_FALLBACK", True))
    # Circuit breaker: floor for the cloud attempt, then fall back to local.
    groq_timeout_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_SEC", 1.5))
    # Long utterances need more than the floor: Opus encoding plus upload grows
    # with the audio, so the breaker becomes
    # ``max(groq_timeout_sec, groq_timeout_base_sec + duration * slope)``.
    groq_timeout_base_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_BASE_SEC", 1.0))
    groq_timeout_slope_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_SLOPE_SEC", 0.12))
    # Vocabulary/formatting steer for Whisper. Empty disables it.
    groq_initial_prompt: str = field(default_factory=lambda: _env("GROQ_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT))
    # Open (and discard) one authenticated cloud call at startup so the TLS
    # handshake is not billed to the first real turn.
    groq_warmup: bool = field(default_factory=lambda: _env_bool("GROQ_WARMUP", True))

    # --- local fallback --------------------------------------------------- #
    # The ONLY local model. It loads only when the cloud attempt times out or
    # errors (see STTService.transcribe_async), never as a primary path.
    # Benchmark: small.en ~2.2s, base.en ~690ms, tiny.en ~350ms (CPU, int8, 4 threads).
    # If accuracy on outage matters: retain small.en (2.8s outage budget).
    # If outage latency matters: consider base.en or tiny.en (faster but less accurate).
    fallback_model: str = field(default_factory=lambda: _env("FALLBACK_MODEL", "base.en"))
    # Load the fallback into RAM at startup instead of on the first failover.
    preload_fallback: bool = field(default_factory=lambda: _env_bool("PRELOAD_FALLBACK", True))
    cpu_threads: int = field(default_factory=lambda: _env_int("CPU_THREADS", 4))

    # --- silence gate --------------------------------------------------- #
    # Segments quieter than this (dBFS RMS) never reach a backend: Whisper
    # hallucinates short phrases on near-silence ("you", "thank you").
    # Was -55 dBFS. After browser AGC, a genuinely silent room sits around
    # -45 to -38 dBFS, so -55 never fired. -42 catches room tone on a laptop
    # fan or AC hum while still accepting soft speech (whispering is ~-40).
    silence_threshold_dbfs: float = field(default_factory=lambda: _env_float("SILENCE_THRESHOLD_DBFS", -42.0))

    # --- vad ----------------------------------------------------------- #
    # Was 0.5 (Silero's own default). On room tone with browser AGC that is
    # too eager. 0.65 requires a little more confidence, which barely affects
    # real speech but eliminates most false starts.
    vad_threshold: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD", 0.65))
    # Silence that ends an utterance (end-of-turn).
    # Was 500 ms. 500 ms of "non-speech" endpoinds too eagerly on brief noise
    # gaps; 700 ms is the empirical sweet spot.
    min_silence_duration_ms: int = field(default_factory=lambda: _env_int("MIN_SILENCE_DURATION_MS", 700))
    # Padding kept on both sides of a detected speech region.
    speech_pad_ms: int = field(default_factory=lambda: _env_int("SPEECH_PAD_MS", 30))
    # --- echo / self-trigger suppression ---------------------------------- #
    # "gate" = drop mic audio during playback, "filter" = drop echoes, "off" = disable
    echo_mode: str = field(default_factory=lambda: _env("ECHO_MODE", "filter").lower())
    # Similarity threshold for self-echo detection.
    echo_similarity_threshold: float = field(default_factory=lambda: _env_float("ECHO_SIMILARITY_THRESHOLD", 0.75))
    # Seconds of spoken history to compare against.
    echo_history_sec: float = field(default_factory=lambda: _env_float("ECHO_HISTORY_SEC", 30.0))
    # Cooldown after playback ends (ms).
    echo_cooldown_ms: int = field(default_factory=lambda: _env_int("ECHO_COOLDOWN_MS", 600))

    # --- streaming ----------------------------------------------------- #
    sample_rate: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", 16000))
    # Interim hypothesis cadence, measured in received speech.
    partial_interval_ms: int = field(default_factory=lambda: _env_int("PARTIAL_INTERVAL_MS", 1000))
    # Shorter segments are VAD noise and are dropped.
    # Was 0.5 s. A 0.5 s blip of room tone was passing as an "utterance". 0.7 s
    # still accepts "yes" / "no" / "stop" but rejects VAD misfires.
    min_segment_seconds: float = field(default_factory=lambda: _env_float("MIN_SEGMENT_SECONDS", 0.7))
    # Hard cap on utterance length (bounds memory and per-turn latency).
    max_segment_seconds: float = field(default_factory=lambda: _env_float("MAX_SEGMENT_SECONDS", 20.0))

    # --- tts ----------------------------------------------------------- #
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "en-US-AndrewNeural"))
    # Sentences fetched from edge-tts ahead of the consumer.
    tts_prefetch: int = field(default_factory=lambda: _env_int("TTS_PREFETCH", 4))
    # Edge silence quieter than this (dBFS) is trimmed from each sentence.
    tts_silence_threshold_dbfs: float = field(default_factory=lambda: _env_float("TTS_SILENCE_THRESHOLD_DBFS", -45.0))
    # Audio held back at the START of a sentence to allow leading-silence removal.
    # This is cheap (just delays the first audio slightly).
    tts_trim_leading_ms: int = field(default_factory=lambda: _env_int("TTS_TRIM_LEADING_MS", 10))
    # Audio held back at the END of a sentence to allow trailing-silence removal.
    # This needs to be larger (e.g., 60ms) to detect real end-of-sentence silence.
    tts_trim_trailing_ms: int = field(default_factory=lambda: _env_int("TTS_TRIM_TRAILING_MS", 60))
    # Digital silence inserted between trimmed sentences.
    tts_sentence_gap_ms: int = field(default_factory=lambda: _env_int("TTS_SENTENCE_GAP_MS", 40))
    # New MP3 bytes buffered before re-decoding during streaming. Higher reduces
    # the O(n^2) re-decode cost; 8192 holds the first ~1 s of MP3 before decoding.
    tts_decode_interval_bytes: int = field(default_factory=lambda: _env_int("TTS_DECODE_INTERVAL_BYTES", 8192))
    # Transient edge-tts fetches fail (429 / network blip); a failed sentence is
    # otherwise dropped silently, which reads as "a word that never gets spoken".
    # Retry the fetch a few times before giving up on a sentence.
    tts_retries: int = field(default_factory=lambda: _env_int("TTS_RETRIES", 2))
    tts_retry_backoff_ms: int = field(default_factory=lambda: _env_int("TTS_RETRY_BACKOFF_MS", 150))
    # Timeout for a single edge-tts sentence fetch (seconds).
    tts_timeout_sec: float = field(default_factory=lambda: _env_float("TTS_TIMEOUT_SEC", 6.0))
    # Speak every final transcript back to the client automatically.
    tts_auto_speak: bool = field(default_factory=lambda: _env_bool("TTS_AUTO_SPEAK", False))
    # Synthesize and discard one short phrase at startup so the first speak
    # does not pay the edge-tts cold start.
    tts_warmup: bool = field(default_factory=lambda: _env_bool("TTS_WARMUP", True))

    max_connections: int = field(default_factory=lambda: _env_int("MAX_CONNECTIONS", 64))
    # Maximum size of a single WebSocket binary frame (bytes). Prevents DoS.
    max_frame_size: int = field(default_factory=lambda: _env_int("MAX_FRAME_SIZE", 1048576))  # 1 MB default
    # Cap the per-connection transcription queue so a slow consumer can't
    # exhaust memory. Each item is a small audio segment (~0.5-1s).
    transcription_queue_maxsize: int = field(default_factory=lambda: _env_int("TRANSCRIPTION_QUEUE_MAXSIZE", 100))

    # --- intent routing -------------------------------------------------- #
    # Deterministic device-command matching (stop / mute / volume / lights / ...)
    # on the ITN-normalized final transcript. Recognized commands are relayed
    # as {"type": "command"} events and never reach the LLM.
    intent_enabled: bool = field(default_factory=lambda: _env_bool("INTENT_ENABLED", True))
    # Speak a short confirmation after executing a matched command.
    intent_speak_confirmation: bool = field(default_factory=lambda: _env_bool("INTENT_SPEAK_CONFIRMATION", False))
    # Transcripts with a lower STT confidence than this are not trusted enough
    # to execute a device command (they still reach the LLM as usual).
    intent_min_confidence: float = field(default_factory=lambda: _env_float("INTENT_MIN_CONFIDENCE", 0.5))
    # --- wake word ------------------------------------------------------ #
    # Words that arm the command router, e.g. "alvin, turn on the lights".
    # Wake words are stripped before matching and reported on the match, so a
    # client can show "heard" state.
    wake_words: str = field(default_factory=lambda: _env("WAKE_WORDS", "alvin"))
    # Require the wake word before a command executes. Recommended for a shared
    # room: it stops the TV, a passing conversation, or background noise from
    # toggling your lights. Leave false for a dedicated headset.
    wake_word_required: bool = field(default_factory=lambda: _env_bool("WAKE_WORD_REQUIRED", False))
    # Seconds of armed state after the wake word, so "alvin ... turn on the
    # lights" works as one phrase and a follow-up command needs no re-wake.
    # 0 disables the window (the wake word must be in the same utterance).
    wake_word_window_sec: float = field(default_factory=lambda: _env_float("WAKE_WORD_WINDOW_SEC", 0.0))
    # Speak a short "Yes?" when the user says only the wake word, with no
    # command. Turning this off makes the wake word silent, which is often
    # nicer once you are used to it -- the assistant then just waits. A chime
    # on the client is the zero-latency alternative.
    wake_speak_ack: bool = field(default_factory=lambda: _env_bool("WAKE_SPEAK_ACK", True))
    # --- push to talk ------------------------------------------------------ #
    # Wake-word gating state, set by {"type": "talk", "active": true|false}.
    # When PUSH_TO_TALK is on, audio is discarded unless the client is holding
    # the key: the microphone is idle by default, so nothing is captured,
    # transcribed, or sent to the LLM.
    push_to_talk: bool = field(default_factory=lambda: _env_bool("PUSH_TO_TALK", False))
    # Safety net: force listening off after this long. A client that dies
    # mid-hold (crash, stuck key, lost connection) would otherwise leave the mic
    # open indefinitely. 0 disables the timeout.
    push_to_talk_timeout_ms: int = field(default_factory=lambda: _env_int("PUSH_TO_TALK_TIMEOUT_MS", 30000))
    # Run the half-duplex playback gate during a hold, so holding a key near
    # speakers cannot pick up the assistant's own voice.
    push_to_talk_echo_gate: bool = field(default_factory=lambda: _env_bool("PUSH_TO_TALK_ECHO_GATE", True))

    # --- admission control ---------------------------------------------- #
    # Shared secret required on ``/ws/transcribe``. Empty (the default) leaves
    # the endpoint open, which is fine only for a loopback/dev deployment.
    api_key: str = field(default_factory=lambda: _env("API_KEY", _env("SERVICE_AUTH_TOKEN", "")).strip())

    # --- llm (brain) ------------------------------------------------------- #
    # Kilo AI gateway (OpenAI-compatible) as the primary conversational brain.
    # Benchmarked: nvidia/nemotron-3.5-lightning:free is the fastest working
    # free model on the gateway (~2s per turn, clean output).
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", _env("NVIDIA_API_KEY", "")).strip())
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.kilo.ai/api/gateway"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "nvidia/nemotron-3.5-lightning:free"))
    llm_enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 300))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_system_prompt: str = field(default_factory=lambda: _env("LLM_SYSTEM_PROMPT", DEFAULT_LLM_SYSTEM_PROMPT))
    # Timeout for the LLM API call (seconds).
    llm_timeout_sec: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SEC", 15.0))
    llm_warmup: bool = field(default_factory=lambda: _env_bool("LLM_WARMUP", True))
    # Use Cloudflare Workers AI as primary instead of Kilo AI.
    # When true, skips the primary Kilo AI call and goes straight to Cloudflare.
    # Defaults to False because Cloudflare's ``llama-3.2-3b-instruct`` is a weak
    # 3-billion-parameter model that confuses self-referential prompts ("Okay,
    # introduce yourself." is a classic failure mode) and produces empty/hallucinated
    # completions when it sees a tool_choice directive with no tools attached. The
    # primary NVIDIA Nemotron model is used by default; Cloudflare remains the
    # automatic fallback when the primary errors or times out.
    llm_use_cf_as_primary: bool = field(default_factory=lambda: _env_bool("LLM_USE_CF_AS_PRIMARY", False))

    # --- cloudflare fallback (secondary LLM provider) ---------------------- #
    # Cloudflare Workers AI fallback when NVIDIA NIM fails or times out.
    # Tested free @cf/ models:
    # @cf/meta/llama-3.2-3b-instruct    ~880ms avg (selected as fallback)
    # @cf/meta/llama-3.3-70b-instruct-fp8-fast  ~1130ms avg (higher quality)
    # Non-@cf models (deepseek etc.) require payment on the Cloudflare dashboard.
    cf_api_key: str = field(default_factory=lambda: _env("CF_API_KEY", "").strip())
    cf_account_id: str = field(default_factory=lambda: _env("CF_ACCOUNT_ID", "").strip())
    cf_base_url: str = field(default_factory=lambda: _env("CF_BASE_URL", "https://api.cloudflare.com/client/v4/accounts"))
    cf_model: str = field(default_factory=lambda: _env("CF_MODEL", "@cf/meta/llama-3.2-3b-instruct"))
    cf_enabled: bool = field(default_factory=lambda: _env_bool("CF_ENABLED", True))
    cf_timeout_sec: float = field(default_factory=lambda: _env_float("CF_TIMEOUT_SEC", 10.0))

    # --- conversation memory ---------------------------------------------- #
    # Maximum messages in the per-connection history (including system prompt).
    max_llm_history: int = field(default_factory=lambda: _env_int("MAX_LLM_HISTORY", 12))
    # Tokens reserved for the system prompt + history when trimming.
    llm_max_history_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_HISTORY_TOKENS", 2000))

    # --- web search -------------------------------------------------------- #
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", "").strip())
    # Debug flag for logging search decisions
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))

    # --- speaker identification -------------------------------------------- #
    # Enable speaker identification (requires speechbrain + torch)
    speaker_enabled: bool = field(default_factory=lambda: _env_bool("SPEAKER_ENABLED", False))
    # Cosine similarity threshold for speaker matching (0-1)
    speaker_similarity_threshold: float = field(default_factory=lambda: _env_float("SPEAKER_SIMILARITY_THRESHOLD", 0.75))
    # Minimum audio duration for identification (seconds)
    speaker_min_identify_sec: float = field(default_factory=lambda: _env_float("SPEAKER_MIN_IDENTIFY_SEC", 1.0))
    # Maximum audio duration for identification (seconds)
    speaker_max_identify_sec: float = field(default_factory=lambda: _env_float("SPEAKER_MAX_IDENTIFY_SEC", 5.0))
    # Path to persist speaker profiles
    speaker_profiles_path: str = field(default_factory=lambda: _env("SPEAKER_PROFILES_PATH", "speaker_profiles.pkl"))

    @property
    def language(self) -> str | None:
        """Configured language, or ``None`` to auto-detect."""
        lang = (self.whisper_language or "").strip().lower()
        return None if lang in _NONE else lang

    @property
    def auth_required(self) -> bool:
        """True when clients must present ``settings.api_key``."""
        return bool(self.api_key)

    @property
    def max_segment_samples(self) -> int:
        return int(self.max_segment_seconds * self.sample_rate)

    @property
    def min_segment_samples(self) -> int:
        return int(self.min_segment_seconds * self.sample_rate)


settings = Settings()


def build_settings(**overrides) -> Settings:
    """Build a new Settings instance from current environment and overrides.

    Constructs a fresh ``Settings`` so dataclass field defaults are read from
    the current environment, then applies explicit overrides. Use this to test
    different configuration values without affecting the global settings.

    Args:
        **overrides: Keyword arguments to override specific settings.

    Returns:
        A new Settings instance with the specified overrides applied.
    """
    s = Settings()
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(s, key):
            setattr(s, key, value)
        else:
            raise AttributeError(f"Settings has no attribute '{key}'")
    return s
