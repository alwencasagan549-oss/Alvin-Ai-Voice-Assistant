"""Configuration for the Alvin STT service.

All settings are read from environment variables (loaded from a ``.env`` file
when present) with sensible defaults that match the service specification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


_NONE = {"", "auto", "none", "null"}

# Steers Whisper toward the local vocabulary and the punctuation style the
# assistant cares about (proper nouns, alphanumeric IDs, email addresses).
DEFAULT_INITIAL_PROMPT = (
    "Cagayan de Oro City, order IDs like 4829-B, email addresses like test@domain.org."
)


@dataclass
class Settings:
    """Runtime configuration shared across the service."""

    # --- server ------------------------------------------------------- #
    host: str = _env("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)
    log_level: str = _env("LOG_LEVEL", "info")

    # --- language -------------------------------------------------------- #
    # The only language knob: the session default handed to every Connection.
    # The local backend hardcodes cpu/int8/beam 1 and never conditions on
    # previous text, so there are no model/decoding knobs to tune here.
    whisper_language: str = _env("WHISPER_LANGUAGE", "en")

    # --- groq (cloud primary) ------------------------------------------ #
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "whisper-large-v3-turbo")
    groq_language: str = _env("GROQ_LANGUAGE", "en")
    # When enabled a Groq timeout/error falls back to the local model.
    groq_fallback: bool = _env_bool("GROQ_FALLBACK", True)
    # Circuit breaker: floor for the cloud attempt, then fall back to local.
    groq_timeout_sec: float = _env_float("GROQ_TIMEOUT_SEC", 1.5)
    # Long utterances need more than the floor: Opus encoding plus upload grows
    # with the audio, so the breaker becomes
    # ``max(groq_timeout_sec, groq_timeout_base_sec + duration * slope)``.
    groq_timeout_base_sec: float = _env_float("GROQ_TIMEOUT_BASE_SEC", 1.0)
    groq_timeout_slope_sec: float = _env_float("GROQ_TIMEOUT_SLOPE_SEC", 0.12)
    # Vocabulary/formatting steer for Whisper. Empty disables it.
    groq_initial_prompt: str = _env("GROQ_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT)
    # Open (and discard) one authenticated cloud call at startup so the TLS
    # handshake is not billed to the first real turn.
    groq_warmup: bool = _env_bool("GROQ_WARMUP", True)

    # --- local fallback --------------------------------------------------- #
    # The ONLY local model. It loads only when the cloud attempt times out or
    # errors (see STTService.transcribe_async), never as a primary path.
    # Benchmark: small.en ~2.2s, base.en ~690ms, tiny.en ~350ms (CPU, int8, 4 threads).
    # If accuracy on outage matters: retain small.en (2.8s outage budget).
    # If outage latency matters: consider base.en or tiny.en (faster but less accurate).
    fallback_model: str = _env("FALLBACK_MODEL", "base.en")
    # Load the fallback into RAM at startup instead of on the first failover.
    preload_fallback: bool = _env_bool("PRELOAD_FALLBACK", True)
    cpu_threads: int = _env_int("CPU_THREADS", 4)

    # --- silence gate --------------------------------------------------- #
    # Segments quieter than this (dBFS RMS) never reach a backend: Whisper
    # hallucinates short phrases on near-silence ("you", "thank you").
    silence_threshold_dbfs: float = _env_float("SILENCE_THRESHOLD_DBFS", -55.0)

    # --- vad ----------------------------------------------------------- #
    vad_threshold: float = _env_float("VAD_THRESHOLD", 0.5)
    # Silence that ends an utterance (end-of-turn).
    min_silence_duration_ms: int = _env_int("MIN_SILENCE_DURATION_MS", 500)
    # Padding kept on both sides of a detected speech region.
    speech_pad_ms: int = _env_int("SPEECH_PAD_MS", 30)

    # --- streaming ----------------------------------------------------- #
    sample_rate: int = _env_int("SAMPLE_RATE", 16000)
    # Interim hypothesis cadence, measured in received speech.
    partial_interval_ms: int = _env_int("PARTIAL_INTERVAL_MS", 1000)
    # Shorter segments are VAD noise and are dropped.
    min_segment_seconds: float = _env_float("MIN_SEGMENT_SECONDS", 0.5)
    # Hard cap on utterance length (bounds memory and per-turn latency).
    max_segment_seconds: float = _env_float("MAX_SEGMENT_SECONDS", 20.0)

    # --- tts ----------------------------------------------------------- #
    tts_voice: str = _env("TTS_VOICE", "en-US-AndrewNeural")
    # Sentences fetched from edge-tts ahead of the consumer.
    tts_prefetch: int = _env_int("TTS_PREFETCH", 3)
    # Edge silence quieter than this (dBFS) is trimmed from each sentence.
    tts_silence_threshold_dbfs: float = _env_float("TTS_SILENCE_THRESHOLD_DBFS", -45.0)
    # Audio held back per sentence to allow trailing-silence removal.
    tts_trim_edge_ms: int = _env_int("TTS_TRIM_EDGE_MS", 100)
    # Digital silence inserted between trimmed sentences.
    tts_sentence_gap_ms: int = _env_int("TTS_SENTENCE_GAP_MS", 80)
    # New MP3 bytes buffered before re-decoding during streaming. Lower = faster
    # time-to-first-audio at the cost of a few more (small) re-decodes; 8192
    # was measured to hold the first ~1 s of MP3 before decoding anything.
    tts_decode_interval_bytes: int = _env_int("TTS_DECODE_INTERVAL_BYTES", 2048)
    # Transient edge-tts fetches fail (429 / network blip); a failed sentence is
    # otherwise dropped silently, which reads as "a word that never gets spoken".
    # Retry the fetch a few times before giving up on a sentence.
    tts_retries: int = _env_int("TTS_RETRIES", 2)
    tts_retry_backoff_ms: int = _env_int("TTS_RETRY_BACKOFF_MS", 250)
    # Speak every final transcript back to the client automatically.
    tts_auto_speak: bool = _env_bool("TTS_AUTO_SPEAK", False)
    # Synthesize and discard one short phrase at startup so the first speak
    # does not pay the edge-tts cold start.
    tts_warmup: bool = _env_bool("TTS_WARMUP", True)

    max_connections: int = _env_int("MAX_CONNECTIONS", 64)
    # Maximum size of a single WebSocket binary frame (bytes). Prevents DoS.
    max_frame_size: int = _env_int("MAX_FRAME_SIZE", 1048576)  # 1 MB default

    # --- admission control ---------------------------------------------- #
    # Shared secret required on ``/ws/transcribe``. Empty (the default) leaves
    # the endpoint open, which is fine only for a loopback/dev deployment.
    api_key: str = _env("API_KEY", _env("SERVICE_AUTH_TOKEN", "")).strip()

    # --- llm (brain) ------------------------------------------------------- #
    # NVIDIA NIM as the primary conversational brain.
    # Benchmarked: nvidia/nemotron-3-ultra-550b-a55b is the fastest working
    # model on integrate.api.nvidia.com/v1 (~1s per turn, clean output).
    # Other tested models (nemotron-3.5-lightning-30b-a3b, etc.) return
    # thinking-process text or fail with NotFoundError for this API key.
    llm_api_key: str = _env("LLM_API_KEY", _env("NVIDIA_API_KEY", "")).strip()
    llm_base_url: str = _env("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    llm_model: str = _env("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    llm_enabled: bool = _env_bool("LLM_ENABLED", True)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 128)
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.7)
    llm_system_prompt: str = _env(
        "LLM_SYSTEM_PROMPT",
        "You are Alvin, a helpful, concise AI voice assistant. "
        "Keep responses short and natural, suitable for text-to-speech. "
        "Do not use emojis or markdown. Answer in plain text only.",
    )
    # Timeout for the LLM API call (seconds). NVIDIA NIM needs more headroom.
    llm_timeout_sec: float = _env_float("LLM_TIMEOUT_SEC", 15.0)
    llm_warmup: bool = _env_bool("LLM_WARMUP", True)

    # --- cloudflare fallback (secondary LLM provider) ---------------------- #
    # Cloudflare Workers AI fallback when NVIDIA NIM fails or times out.
    # Tested free @cf/ models:
    # @cf/meta/llama-3.2-3b-instruct    ~880ms avg (selected as fallback)
    # @cf/meta/llama-3.3-70b-instruct-fp8-fast  ~1130ms avg (higher quality)
    # Non-@cf models (deepseek etc.) require payment on the Cloudflare dashboard.
    cf_api_key: str = _env("CF_API_KEY", "").strip()
    cf_account_id: str = _env("CF_ACCOUNT_ID", "").strip()
    cf_base_url: str = _env(
        "CF_BASE_URL", "https://api.cloudflare.com/client/v4/accounts"
    )
    cf_model: str = _env("CF_MODEL", "@cf/meta/llama-3.2-3b-instruct")
    cf_enabled: bool = _env_bool("CF_ENABLED", True)
    cf_timeout_sec: float = _env_float("CF_TIMEOUT_SEC", 10.0)

    # --- conversation memory ---------------------------------------------- #
    # Maximum messages in the per-connection history (including system prompt).
    max_llm_history: int = _env_int("MAX_LLM_HISTORY", 12)
    # Tokens reserved for the system prompt + history when trimming.
    llm_max_history_tokens: int = _env_int("LLM_MAX_HISTORY_TOKENS", 2000)

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
