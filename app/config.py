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
    "GROQ_TIMEOUT_SEC",
    "GROQ_TIMEOUT_BASE_SEC",
    "GROQ_TIMEOUT_SLOPE_SEC",
    "GROQ_INITIAL_PROMPT",
    "GROQ_WARMUP",
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
    "PIPER_TTS_ENABLED",
    "PIPER_TTS_VOICE",
    "PIPER_TTS_MODEL_DIR",
    "PIPER_TTS_LENGTH_SCALE",
    "PIPER_TTS_NOISE_SCALE",
    "PIPER_TTS_NOISE_W",
    "MAX_CONNECTIONS",
    "MAX_FRAME_SIZE",
    "TRANSCRIPTION_QUEUE_MAXSIZE",
    "AUDIO_WORKER_THREADS",
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
    "LLM_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_BASE_URL",
    "NVIDIA_MODEL",
    "NVIDIA_VISION_MODEL",
    "NVIDIA_ENABLED",
    "NVIDIA_TIMEOUT_SEC",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_ENABLED",
    "LLM_MAX_TOKENS",
    "LLM_VISION_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_VISION_TEMPERATURE",
    "LLM_SYSTEM_PROMPT",
    "LLM_TIMEOUT_SEC",
    "LLM_VISION_TIMEOUT_SEC",
    "LLM_WARMUP",
    "LLM_USE_CF_AS_PRIMARY",
    "LLM_PRIMARY_PROVIDER",
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
    "GEMINI_API_KEY",
    "DEBUG",
    "LLM_VISION_MODEL",
    "CF_STT_ENABLED",
    "CF_STT_MODEL",
    "CF_STT_TIMEOUT_SEC",
    "CF_STT_USE_AS_PRIMARY",
    "CF_STT_WARMUP",
    "SPEAKER_ENABLED",
    "SPEAKER_SIMILARITY_THRESHOLD",
    "SPEAKER_MIN_IDENTIFY_SEC",
    "SPEAKER_MAX_IDENTIFY_SEC",
    "SPEAKER_PROFILES_PATH",
    "SHUTDOWN_RESPONSE_TEXT",
    "SHUTDOWN_DELAY_SEC",
    "LAPTOP_SHUTDOWN_RESPONSE_TEXT",
    "LAPTOP_SHUTDOWN_DELAY_SEC",
    "YOUTUBE_ADFREE",
    "YOUTUBE_AUDIO_ONLY",
    "BRIEFING_ENABLED",
    "BRIEFING_ON_WAKE_UP",
    "BRIEFING_WARMUP",
    "BRIEFING_WEATHER_ENABLED",
    "BRIEFING_WEATHER_LOCATION",
    "BRIEFING_WEATHER_TIMEOUT_SEC",
    "BRIEFING_WEATHER_CACHE_SEC",
    "BRIEFING_TASKS_ENABLED",
    "BRIEFING_TASKS_TOKEN_PATH",
    "BRIEFING_TASKS_LIST",
    "BRIEFING_TASKS_COUNT",
    "BRIEFING_TASKS_TIMEOUT_SEC",
    "BRIEFING_EMAIL_ENABLED",
    "BRIEFING_EMAIL_IMAP_HOST",
    "BRIEFING_EMAIL_IMAP_PORT",
    "BRIEFING_EMAIL_IMAP_USER",
    "BRIEFING_EMAIL_IMAP_PASSWORD",
    "BRIEFING_EMAIL_FOLDER",
    "BRIEFING_EMAIL_COUNT",
    "BRIEFING_EMAIL_UNREAD_ONLY",
    "BRIEFING_EMAIL_TIMEOUT_SEC",
    "BRIEFING_EMAIL_CACHE_SEC",
    "BRIEFING_EMAIL_SKIP_SENSITIVE",
    "BRIEFING_LLM_POLISH",
    "BRIEFING_LLM_MAX_TOKENS",
    "BRIEFING_LLM_TIMEOUT_SEC",
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

SHUTDOWN_RESPONSE_TEXT_DEFAULT = (
    "At your service, sir. I will shut down all my system."
)

LAPTOP_SHUTDOWN_RESPONSE_TEXT_DEFAULT = (
    "Shutting down your laptop now, sir. Goodbye."
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
    # Circuit breaker: floor for the cloud attempt, then fall back to Cloudflare STT.
    # Increased from 0.8s to 5.0s for more reliable cloud STT on longer audio.
    groq_timeout_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_SEC", 5.0))
    # Long utterances need more than the floor: Opus encoding plus upload grows
    # with the audio, so the breaker becomes
    # ``max(groq_timeout_sec, groq_timeout_base_sec + duration * slope)``.
    # Increased base and slope for longer timeout on longer audio.
    groq_timeout_base_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_BASE_SEC", 2.0))
    groq_timeout_slope_sec: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_SLOPE_SEC", 0.2))
    # Vocabulary/formatting steer for Whisper. Empty disables it.
    groq_initial_prompt: str = field(default_factory=lambda: _env("GROQ_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT))
    # Open (and discard) one authenticated cloud call at startup so the TLS
    # handshake is not billed to the first real turn.
    groq_warmup: bool = field(default_factory=lambda: _env_bool("GROQ_WARMUP", True))

    # --- cloudflare stt (workers ai) ------------------------------------- #
    # Use Cloudflare Workers AI @cf/openai/whisper-large-v3-turbo as STT fallback.
    # Free on Workers AI (no separate billing), same CF_API_KEY/CF_ACCOUNT_ID as LLM.
    # Latency ~1-2s, cloud accuracy. Enabled by default as Groq fallback.
    cf_stt_enabled: bool = field(default_factory=lambda: _env_bool("CF_STT_ENABLED", True))
    # Model to use on Cloudflare Workers AI.
    cf_stt_model: str = field(default_factory=lambda: _env("CF_STT_MODEL", "@cf/openai/whisper-large-v3-turbo"))
    # Timeout for Cloudflare STT call (seconds).
    cf_stt_timeout_sec: float = field(default_factory=lambda: _env_float("CF_STT_TIMEOUT_SEC", 15.0))
    # Use Cloudflare STT as primary (true = CF primary, Groq fallback; false = Groq primary, CF fallback)
    cf_stt_use_as_primary: bool = field(default_factory=lambda: _env_bool("CF_STT_USE_AS_PRIMARY", False))
    # Warm up Cloudflare STT connection at startup.
    cf_stt_warmup: bool = field(default_factory=lambda: _env_bool("CF_STT_WARMUP", True))

    # --- silence gate --------------------------------------------------- #
    # Segments quieter than this (dBFS RMS) never reach a backend: Whisper
    # hallucinates short phrases on near-silence ("you", "thank you").
    # Was -55 dBFS. After browser AGC, a genuinely silent room sits around
    # -45 to -38 dBFS, so -55 never fired. -45 catches whispering (~ -40 dBFS)
    # while still filtering room noise.
    silence_threshold_dbfs: float = field(default_factory=lambda: _env_float("SILENCE_THRESHOLD_DBFS", -45.0))

    # --- vad ----------------------------------------------------------- #
    # Was 0.5 (Silero's own default). On room tone with browser AGC that is
    # too eager. 0.65 requires a little more confidence, which barely affects
    # real speech but eliminates most false starts.
    vad_threshold: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD", 0.65))
    # Silence that ends an utterance (end-of-turn).
    # 500 ms balances responsiveness with avoiding premature endpointing on
    # brief pauses. Faster than 700ms for more responsive feel.
    min_silence_duration_ms: int = field(default_factory=lambda: _env_int("MIN_SILENCE_DURATION_MS", 500))
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
    # 500ms gives more responsive partials for better perceived latency.
    partial_interval_ms: int = field(default_factory=lambda: _env_int("PARTIAL_INTERVAL_MS", 500))
    # Shorter segments are VAD noise and are dropped.
    # 0.5s accepts short commands like "yes"/"no"/"stop" while rejecting VAD misfires.
    min_segment_seconds: float = field(default_factory=lambda: _env_float("MIN_SEGMENT_SECONDS", 0.5))
    # Hard cap on utterance length (bounds memory and per-turn latency).
    # Reduced from 20s to 15s to cap worst-case latency.
    max_segment_seconds: float = field(default_factory=lambda: _env_float("MAX_SEGMENT_SECONDS", 15.0))

    # --- tts ----------------------------------------------------------- #
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "en-US-AndrewNeural"))
    # Sentences fetched from edge-tts ahead of the consumer.
    tts_prefetch: int = field(default_factory=lambda: _env_int("TTS_PREFETCH", 4))
    # Edge silence quieter than this (dBFS) is trimmed from each sentence.
    tts_silence_threshold_dbfs: float = field(default_factory=lambda: _env_float("TTS_SILENCE_THRESHOLD_DBFS", -45.0))
    # Audio held back at the START of a sentence to allow leading-silence removal.
    # Increased from 5ms to 20ms to prevent cutting off short confirmations.
    tts_trim_leading_ms: int = field(default_factory=lambda: _env_int("TTS_TRIM_LEADING_MS", 20))
    # Audio held back at the END of a sentence to allow trailing-silence removal.
    # Increased from 30ms to 80ms to prevent cutting off ends of short sentences.
    tts_trim_trailing_ms: int = field(default_factory=lambda: _env_int("TTS_TRIM_TRAILING_MS", 80))
    # Digital silence inserted between trimmed sentences.
    tts_sentence_gap_ms: int = field(default_factory=lambda: _env_int("TTS_SENTENCE_GAP_MS", 30))
    # New MP3 bytes buffered before re-decoding during streaming. Lower decodes more
    # frequently for lower latency; 4096 decodes ~2x more often than 8192.
    tts_decode_interval_bytes: int = field(default_factory=lambda: _env_int("TTS_DECODE_INTERVAL_BYTES", 4096))
    # Transient edge-tts fetches fail (429 / network blip); a failed sentence is
    # otherwise dropped silently, which reads as "a word that never gets spoken".
    # Retry the fetch a few times before giving up on a sentence.
    # Increased from 2 to 5 for better reliability.
    tts_retries: int = field(default_factory=lambda: _env_int("TTS_RETRIES", 5))
    tts_retry_backoff_ms: int = field(default_factory=lambda: _env_int("TTS_RETRY_BACKOFF_MS", 200))
    # Timeout for a single edge-tts sentence fetch (seconds).
    # Increased from 4s to 20s for more reliable short-phrase TTS.
    tts_timeout_sec: float = field(default_factory=lambda: _env_float("TTS_TIMEOUT_SEC", 20.0))
    # Speak every final transcript back to the client automatically.
    tts_auto_speak: bool = field(default_factory=lambda: _env_bool("TTS_AUTO_SPEAK", False))
    # Synthesize and discard one short phrase at startup so the first speak
    # does not pay the edge-tts cold start.
    tts_warmup: bool = field(default_factory=lambda: _env_bool("TTS_WARMUP", True))

    # --- piper tts (fallback) ------------------------------------------ #
    # Enable Piper TTS as local fallback when edge-tts fails.
    piper_tts_enabled: bool = field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    # Piper voice model name (e.g., en_US-lessac-medium).
    piper_tts_voice: str = field(default_factory=lambda: _env("PIPER_TTS_VOICE", "en_US-ryan-high"))
    # Path to Piper voice models directory.
    piper_tts_model_dir: str = field(default_factory=lambda: _env("PIPER_TTS_MODEL_DIR", "piper_voices"))
    # Piper TTS synthesis length scale (1.0 = normal, lower = faster).
    piper_tts_length_scale: float = field(default_factory=lambda: _env_float("PIPER_TTS_LENGTH_SCALE", 1.0))
    # Piper TTS noise scale (0.0 = deterministic, higher = more variation).
    piper_tts_noise_scale: float = field(default_factory=lambda: _env_float("PIPER_TTS_NOISE_SCALE", 0.667))
    # Piper TTS noise scale for width (0.0 = deterministic).
    piper_tts_noise_w: float = field(default_factory=lambda: _env_float("PIPER_TTS_NOISE_W", 0.8))

    max_connections: int = field(default_factory=lambda: _env_int("MAX_CONNECTIONS", 64))
    # Maximum size of a single WebSocket binary frame (bytes). Prevents DoS.
    max_frame_size: int = field(default_factory=lambda: _env_int("MAX_FRAME_SIZE", 1048576))  # 1 MB default
    # Cap the per-connection transcription queue so a slow consumer can't
    # exhaust memory. Each item is a small audio segment (~0.5-1s).
    # Increased to 200 to reduce drops under load.
    transcription_queue_maxsize: int = field(default_factory=lambda: _env_int("TRANSCRIPTION_QUEUE_MAXSIZE", 200))
    # Size of the dedicated bounded thread pools for CPU-bound audio work.
    # Split by criticality in app.audio_pool: a small pool for the latency-critical
    # Opus encode and a larger pool for TTS decode / Piper synth / speaker
    # embeddings. A bounded pool keeps a burst of concurrent connections from
    # saturating the default asyncio executor (sized min(32, cpu+4)) and
    # starving other to_thread callers. Clamped to [1, 32] at use site.
    audio_worker_threads: int = field(default_factory=lambda: _env_int("AUDIO_WORKER_THREADS", 8))

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
    # Vision-capable model used when the camera is active (Nemotron 3 Nano
    # Omni). Must support image input (array content with image_url).
    llm_vision_model: str = field(default_factory=lambda: _env("LLM_VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"))
    llm_enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 200))
    llm_vision_max_tokens: int = field(default_factory=lambda: _env_int("LLM_VISION_MAX_TOKENS", 100))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_vision_temperature: float = field(default_factory=lambda: _env_float("LLM_VISION_TEMPERATURE", 0.3))
    llm_system_prompt: str = field(default_factory=lambda: _env("LLM_SYSTEM_PROMPT", DEFAULT_LLM_SYSTEM_PROMPT))
    # Timeout for the LLM API call (seconds).
    # Reduced from 15s to 8s for faster fallback detection.
    llm_timeout_sec: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SEC", 8.0))
    # Vision-turn timeout — vision models are slower, so give them a bit more
    # room but still bound the turn so the pipeline doesn't stall.
    llm_vision_timeout_sec: float = field(default_factory=lambda: _env_float("LLM_VISION_TIMEOUT_SEC", 10.0))
    llm_warmup: bool = field(default_factory=lambda: _env_bool("LLM_WARMUP", True))
    # Use Cloudflare Workers AI as primary instead of Kilo AI.
    # Default true: Cloudflare's @cf/meta/llama-3.2-3b-instruct is ~880ms (2x faster
    # than Kilo AI's ~2s). Kilo AI becomes the fallback.
    # DEPRECATED: Use LLM_PRIMARY_PROVIDER instead (cf, nvidia, kilo).
    llm_use_cf_as_primary: bool = field(default_factory=lambda: _env_bool("LLM_USE_CF_AS_PRIMARY", True))
    # Primary LLM provider to try first: "cf" (Cloudflare), "nvidia" (NVIDIA NIM), or "kilo" (Kilo AI Gateway).
    # Falls back to other enabled providers in order: cf -> nvidia -> kilo.
    llm_primary_provider: str = field(default_factory=lambda: _env("LLM_PRIMARY_PROVIDER", "cf").lower())

    # --- nvidia direct (openai-compatible) -------------------------------- #
    # Call NVIDIA NIM directly instead of through the Kilo AI gateway.
    # nvidia_api_key falls back to LLM_API_KEY so an existing gateway token
    # works against NVIDIA NIM without a separate variable.
    nvidia_api_key: str = field(
        default_factory=lambda: _env("NVIDIA_API_KEY", _env("LLM_API_KEY", "")).strip()
    )
    # NVIDIA NIM endpoint (OpenAI-compatible).
    nvidia_base_url: str = field(
        default_factory=lambda: _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    nvidia_model: str = field(
        default_factory=lambda: _env("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    )
    # Vision model to use on the NVIDIA NIM endpoint for multimodal requests.
    # Available: nvidia/nemotron-3.5-super-vl, nvidia/nemotron-3-nano-omni,
    # nvidia/llama-3.2-11b-vision-instruct, nvidia/llama-3.2-90b-vision-instruct
    # Working model tested: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
    nvidia_vision_model: str = field(
        default_factory=lambda: _env("NVIDIA_VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    )
    # Enable the NVIDIA direct provider as an LLM option.
    nvidia_enabled: bool = field(default_factory=lambda: _env_bool("NVIDIA_ENABLED", False))
    # Timeout for NVIDIA NIM API calls (seconds).
    nvidia_timeout_sec: float = field(default_factory=lambda: _env_float("NVIDIA_TIMEOUT_SEC", 8.0))

    # --- cloudflare fallback (secondary LLM provider) ---------------------- #
    # Cloudflare Workers AI fallback when primary fails or times out.
    # Tested free @cf/ models:
    # @cf/meta/llama-3.2-3b-instruct    ~880ms avg (fastest, used as primary)
    # @cf/meta/llama-3.3-70b-instruct-fp8-fast  ~1130ms avg (higher quality)
    # Non-@cf models (deepseek etc.) require payment on the Cloudflare dashboard.
    cf_api_key: str = field(default_factory=lambda: _env("CF_API_KEY", "").strip())
    cf_account_id: str = field(default_factory=lambda: _env("CF_ACCOUNT_ID", "").strip())
    cf_base_url: str = field(default_factory=lambda: _env("CF_BASE_URL", "https://api.cloudflare.com/client/v4/accounts"))
    # Using 3b model for speed (~880ms vs ~1130ms for 70b).
    cf_model: str = field(default_factory=lambda: _env("CF_MODEL", "@cf/meta/llama-3.2-3b-instruct"))
    cf_enabled: bool = field(default_factory=lambda: _env_bool("CF_ENABLED", True))
    # Reduced timeout for faster fallback detection.
    cf_timeout_sec: float = field(default_factory=lambda: _env_float("CF_TIMEOUT_SEC", 6.0))

    # --- conversation memory ---------------------------------------------- #
    # Maximum messages in the per-connection history (including system prompt).
    max_llm_history: int = field(default_factory=lambda: _env_int("MAX_LLM_HISTORY", 12))
    # Tokens reserved for the system prompt + history when trimming.
    llm_max_history_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_HISTORY_TOKENS", 2000))

    # --- web search -------------------------------------------------------- #
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", "").strip())
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

    # --- shutdown command ------------------------------------------------------ #
    # Deterministic shutdown intent triggers on any utterance containing
    # "goodbye" ("alvin goodbye", "okay goodbye", "I think it is time to say
    # goodbye"). Only the assistant's own server process is stopped -- never
    # the operating system. The response text is spoken before the server
    # receives a SIGINT for graceful shutdown.
    shutdown_response_text: str = field(
        default_factory=lambda: _env(
            "SHUTDOWN_RESPONSE_TEXT", SHUTDOWN_RESPONSE_TEXT_DEFAULT
        )
    )
    # Seconds to wait after speaking the shutdown acknowledgement before the
    # server process is signalled to stop, giving the first audio bytes time to
    # reach the client before the socket closes.
    shutdown_delay_sec: float = field(
        default_factory=lambda: _env_float("SHUTDOWN_DELAY_SEC", 3.0)
    )

    # --- laptop shutdown command ----------------------------------------------- #
    # "Alvin, turn off my laptop" powers off the local machine itself via the
    # operating system's shutdown command. Exact whole-utterance phrase only.
    laptop_shutdown_response_text: str = field(
        default_factory=lambda: _env(
            "LAPTOP_SHUTDOWN_RESPONSE_TEXT", LAPTOP_SHUTDOWN_RESPONSE_TEXT_DEFAULT
        )
    )
    # Windows ``shutdown /s /t N`` countdown. The delay doubles as the window
    # in which the user can cancel with ``shutdown /a``.
    laptop_shutdown_delay_sec: float = field(
        default_factory=lambda: _env_float("LAPTOP_SHUTDOWN_DELAY_SEC", 10.0)
    )

    # --- YouTube playback ------------------------------------------------------ #
    # Resolve a direct yt-dlp stream and play it in a local player (VLC/mpv/
    # ffplay). Direct streams carry no YouTube ads; if resolution or playback
    # fails the HUD falls back to the normal watch page in the browser.
    youtube_adfree: bool = field(
        default_factory=lambda: _env_bool("YOUTUBE_ADFREE", True)
    )
    # Audio-only playback uses less bandwidth and starts faster; disable to
    # show the video in the local player window.
    youtube_audio_only: bool = field(
        default_factory=lambda: _env_bool("YOUTUBE_AUDIO_ONLY", True)
    )

    # --- wake-up briefing ------------------------------------------------------ #
    # "Alvin, wake up" speaks a time-aware greeting, current weather, open
    # Google Tasks, and the latest Gmail message. Each source is independent:
    # a disabled or failing source is skipped instead of preventing the rest.
    briefing_enabled: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_ENABLED", True)
    )
    briefing_on_wake_up: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_ON_WAKE_UP", True)
    )
    # Fetch weather and email in the background at startup so the first
    # wake-up does not wait on the network.
    briefing_warmup: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_WARMUP", True)
    )

    # Current weather (free, no API key): Open-Meteo resolves the place name,
    # MET Norway returns the forecast.
    briefing_weather_enabled: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_WEATHER_ENABLED", True)
    )
    briefing_weather_location: str = field(
        default_factory=lambda: _env("BRIEFING_WEATHER_LOCATION", "Cagayan de Oro")
    )
    briefing_weather_timeout_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_WEATHER_TIMEOUT_SEC", 12.0)
    )
    briefing_weather_cache_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_WEATHER_CACHE_SEC", 600.0)
    )

    # Google Tasks (read-only OAuth). On by default; run
    # scripts/google_tasks_auth.py once to create the token.
    briefing_tasks_enabled: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_TASKS_ENABLED", True)
    )
    briefing_tasks_token_path: str = field(
        default_factory=lambda: _env(
            "BRIEFING_TASKS_TOKEN_PATH", "credentials/google_tasks_token.json"
        )
    )
    briefing_tasks_list: str = field(
        default_factory=lambda: _env("BRIEFING_TASKS_LIST", "@default")
    )
    briefing_tasks_count: int = field(
        default_factory=lambda: _env_int("BRIEFING_TASKS_COUNT", 5)
    )
    briefing_tasks_timeout_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_TASKS_TIMEOUT_SEC", 6.0)
    )

    # Gmail over IMAP. Headers only, read-only, never marks messages as read.
    # On by default once BRIEFING_EMAIL_IMAP_USER / PASSWORD are set; Gmail
    # requires 2-Step Verification and an App Password.
    briefing_email_enabled: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_EMAIL_ENABLED", True)
    )
    briefing_email_imap_host: str = field(
        default_factory=lambda: _env("BRIEFING_EMAIL_IMAP_HOST", "imap.gmail.com")
    )
    briefing_email_imap_port: int = field(
        default_factory=lambda: _env_int("BRIEFING_EMAIL_IMAP_PORT", 993)
    )
    briefing_email_imap_user: str = field(
        default_factory=lambda: _env("BRIEFING_EMAIL_IMAP_USER", "")
    )
    briefing_email_imap_password: str = field(
        default_factory=lambda: _env("BRIEFING_EMAIL_IMAP_PASSWORD", "")
    )
    briefing_email_folder: str = field(
        default_factory=lambda: _env("BRIEFING_EMAIL_FOLDER", "INBOX")
    )
    briefing_email_count: int = field(
        default_factory=lambda: _env_int("BRIEFING_EMAIL_COUNT", 3)
    )
    briefing_email_unread_only: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_EMAIL_UNREAD_ONLY", True)
    )
    briefing_email_timeout_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_EMAIL_TIMEOUT_SEC", 8.0)
    )
    briefing_email_cache_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_EMAIL_CACHE_SEC", 120.0)
    )
    # Keep one-time codes and password-reset mail out of the spoken briefing.
    briefing_email_skip_sensitive: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_EMAIL_SKIP_SENSITIVE", True)
    )

    # Optional LLM rewrite of the templated briefing. When enabled, the
    # briefing text (including email subjects and senders) is sent to the
    # configured LLM provider.
    briefing_llm_polish: bool = field(
        default_factory=lambda: _env_bool("BRIEFING_LLM_POLISH", False)
    )
    briefing_llm_max_tokens: int = field(
        default_factory=lambda: _env_int("BRIEFING_LLM_MAX_TOKENS", 220)
    )
    briefing_llm_timeout_sec: float = field(
        default_factory=lambda: _env_float("BRIEFING_LLM_TIMEOUT_SEC", 10.0)
    )

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
