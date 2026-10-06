"""Unit tests for the Settings configuration.

These tests use :func:`app.config.build_settings` instead of reloading
``app.config``. Reloading produced a *second* ``Settings`` singleton that the
rest of the app never saw, which made the suite order-dependent: patching the
fresh instance had no effect on ``app.tts``/``app.connection``, which kept
references to the original one.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import os
from app.config import ENV_VAR_NAMES, Settings, build_settings, settings


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every variable the service reads, so defaults are exercised."""
    for name in ENV_VAR_NAMES:
        for alias in name.split("|"):
            monkeypatch.delenv(alias, raising=False)



def _all_env_vars_default() -> dict[str, str]:
    """Every known env var mapped to the empty string (meaning: unset)."""
    return dict.fromkeys(ENV_VAR_NAMES, "")


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should load defaults when env vars are unset."""
    _clean_env(monkeypatch)
    # DEBUG: Check what's left in environment after cleanup
    remaining = [name for name in ENV_VAR_NAMES if any(os.environ.get(alias) is not None for alias in name.split("|"))]
    print(f"DEBUG: Number of remaining env vars: {len(remaining)}")
    if remaining:
        print(f"DEBUG: Env vars still present after cleanup: {remaining}")
        for name in remaining:
            for alias in name.split("|"):
                val = os.environ.get(alias)
                print(f"  {alias} = {repr(val)}")
    else:
        print("DEBUG: No env vars remaining after cleanup")

    settings = build_settings()

    # Check defaults
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level == "info"
    assert settings.whisper_language == "en"
    assert settings.groq_api_key == ""
    assert settings.groq_model == "whisper-large-v3-turbo"
    assert settings.groq_language == "en"
    assert settings.groq_fallback is True
    assert settings.groq_timeout_sec == 1.5
    assert settings.groq_timeout_base_sec == 1.0
    assert settings.groq_timeout_slope_sec == 0.12
    assert settings.groq_initial_prompt == (
        "Cagayan de Oro City, order IDs like 4829-B, "
        "email addresses like test@domain.org."
    )
    assert settings.groq_warmup is True
    assert settings.fallback_model == "base.en"
    assert settings.preload_fallback is True
    assert settings.cpu_threads == 4
    assert settings.silence_threshold_dbfs == -60.0
    assert settings.vad_threshold == 0.6
    assert settings.min_silence_duration_ms == 500
    assert settings.speech_pad_ms == 30
    assert settings.sample_rate == 16000
    assert settings.partial_interval_ms == 1000
    assert settings.min_segment_seconds == 0.5
    assert settings.max_segment_seconds == 20.0
    assert settings.tts_voice == "en-US-AndrewNeural"
    assert settings.tts_prefetch == 3
    assert settings.tts_silence_threshold_dbfs == -45.0
    assert settings.tts_trim_edge_ms == 100
    assert settings.tts_sentence_gap_ms == 80
    assert settings.tts_decode_interval_bytes == 2048
    assert settings.tts_retries == 2
    assert settings.tts_retry_backoff_ms == 250
    assert settings.tts_auto_speak is False
    assert settings.tts_warmup is True
    assert settings.max_connections == 64
    assert settings.max_frame_size == 1048576
    assert settings.intent_enabled is True
    assert settings.intent_speak_confirmation is False
    assert settings.intent_min_confidence == 0.5
    assert settings.api_key == ""
    assert settings.llm_api_key == ""
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.llm_enabled is True
    assert settings.llm_max_tokens == 128
    assert settings.llm_temperature == 0.7
    assert "You are Alvin" in settings.llm_system_prompt
    assert settings.llm_timeout_sec == 15.0
    assert settings.llm_warmup is True
    assert settings.cf_api_key == ""
    assert settings.cf_account_id == ""
    assert settings.cf_base_url == "https://api.cloudflare.com/client/v4/accounts"
    assert settings.cf_model == "@cf/meta/llama-3.2-3b-instruct"
    assert settings.cf_enabled is True
    assert settings.cf_timeout_sec == 10.0
    assert settings.max_llm_history == 12
    assert settings.llm_max_history_tokens == 2000
    assert settings.transcription_queue_maxsize == 100


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should read from environment variables."""
    _clean_env(monkeypatch)

    test_cases = [
        ("HOST", "127.0.0.1", "host"),
        ("PORT", "9000", "port", int),
        ("LOG_LEVEL", "debug", "log_level"),
        ("WHISPER_LANGUAGE", "fr", "whisper_language"),
        ("GROQ_API_KEY", "test-groq-key", "groq_api_key"),
        ("GROQ_MODEL", "whisper-large-v3", "groq_model"),
        ("GROQ_LANGUAGE", "fr", "groq_language"),
        ("GROQ_FALLBACK", "false", "groq_fallback", "bool"),
        ("GROQ_TIMEOUT_SEC", "2.5", "groq_timeout_sec", float),
        ("GROQ_TIMEOUT_BASE_SEC", "2.0", "groq_timeout_base_sec", float),
        ("GROQ_TIMEOUT_SLOPE_SEC", "0.2", "groq_timeout_slope_sec", float),
        ("GROQ_INITIAL_PROMPT", "test prompt", "groq_initial_prompt"),
        ("GROQ_WARMUP", "false", "groq_warmup", "bool"),
        ("FALLBACK_MODEL", "tiny.en", "fallback_model"),
        ("PRELOAD_FALLBACK", "false", "preload_fallback", "bool"),
        ("CPU_THREADS", "2", "cpu_threads", int),
        ("SILENCE_THRESHOLD_DBFS", "-60.0", "silence_threshold_dbfs", float),
        ("VAD_THRESHOLD", "0.3", "vad_threshold", float),
        ("MIN_SILENCE_DURATION_MS", "300", "min_silence_duration_ms", int),
        ("SPEECH_PAD_MS", "15", "speech_pad_ms", int),
        ("SAMPLE_RATE", "8000", "sample_rate", int),
        ("PARTIAL_INTERVAL_MS", "500", "partial_interval_ms", int),
        ("MIN_SEGMENT_SECONDS", "0.25", "min_segment_seconds", float),
        ("MAX_SEGMENT_SECONDS", "10.0", "max_segment_seconds", float),
        ("TTS_VOICE", "en-US-AriaNeural", "tts_voice"),
        ("TTS_PREFETCH", "5", "tts_prefetch", int),
        ("TTS_SILENCE_THRESHOLD_DBFS", "-40.0", "tts_silence_threshold_dbfs", float),
        ("TTS_TRIM_EDGE_MS", "50", "tts_trim_edge_ms", int),
        ("TTS_SENTENCE_GAP_MS", "40", "tts_sentence_gap_ms", int),
        ("TTS_DECODE_INTERVAL_BYTES", "4096", "tts_decode_interval_bytes", int),
        ("TTS_RETRIES", "5", "tts_retries", int),
        ("TTS_RETRY_BACKOFF_MS", "100", "tts_retry_backoff_ms", int),
        ("TTS_AUTO_SPEAK", "true", "tts_auto_speak", "bool"),
        ("TTS_WARMUP", "false", "tts_warmup", "bool"),
        ("MAX_CONNECTIONS", "32", "max_connections", int),
        ("MAX_FRAME_SIZE", "512000", "max_frame_size", int),
        ("INTENT_ENABLED", "false", "intent_enabled", "bool"),
        ("INTENT_SPEAK_CONFIRMATION", "true", "intent_speak_confirmation", "bool"),
        ("INTENT_MIN_CONFIDENCE", "0.8", "intent_min_confidence", float),
        ("API_KEY", "test-api-key", "api_key"),
        ("LLM_API_KEY", "test-llm-key", "llm_api_key"),
        ("LLM_BASE_URL", "https://test.example.com/v1", "llm_base_url"),
        ("LLM_MODEL", "test-model", "llm_model"),
        ("LLM_ENABLED", "false", "llm_enabled", "bool"),
        ("LLM_MAX_TOKENS", "256", "llm_max_tokens", int),
        ("LLM_TEMPERATURE", "0.0", "llm_temperature", float),
        ("LLM_SYSTEM_PROMPT", "Test system prompt", "llm_system_prompt"),
        ("LLM_TIMEOUT_SEC", "30.0", "llm_timeout_sec", float),
        ("LLM_WARMUP", "false", "llm_warmup", "bool"),
        ("CF_API_KEY", "test-cf-key", "cf_api_key"),
        ("CF_ACCOUNT_ID", "test-account-id", "cf_account_id"),
        ("CF_BASE_URL", "https://test.cf.com/v4", "cf_base_url"),
        ("CF_MODEL", "@cf/test/model", "cf_model"),
        ("CF_ENABLED", "true", "cf_enabled", "bool"),
        ("CF_TIMEOUT_SEC", "5.0", "cf_timeout_sec", float),
        ("MAX_LLM_HISTORY", "20", "max_llm_history", int),
        ("LLM_MAX_HISTORY_TOKENS", "3000", "llm_max_history_tokens", int),
        ("TRANSCRIPTION_QUEUE_MAXSIZE", "50", "transcription_queue_maxsize", int),
    ]

    for case in test_cases:
        monkeypatch.setenv(case[0], case[1])
    settings = build_settings()

    def convert(value: str, conv) -> object:
        if conv == "bool":
            return value.lower() in ("1", "true", "yes", "on")
        if conv is not None:
            return conv(value)
        return value

    for case in test_cases:
        var_name, var_value, attr_name = case[0], case[1], case[2]
        converter = case[3] if len(case) > 3 else None
        expected = convert(var_value, converter)
        actual = getattr(settings, attr_name)
        assert actual == expected, (
            f"Failed for {var_name} -> {attr_name}: "
            f"expected {expected!r}, got {actual!r}"
        )


def test_legacy_env_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy aliases still feed their modern counterparts."""
    _clean_env(monkeypatch)

    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "legacy-service-token")
    assert build_settings().api_key == "legacy-service-token"

    monkeypatch.setenv("NVIDIA_API_KEY", "legacy-nvidia-key")
    assert build_settings().llm_api_key == "legacy-nvidia-key"


def test_modern_env_var_wins_over_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``API_KEY`` takes precedence over the legacy ``SERVICE_AUTH_TOKEN``."""
    _clean_env(monkeypatch)

    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "legacy-service-token")
    monkeypatch.setenv("API_KEY", "modern-api-key")
    assert build_settings().api_key == "modern-api-key"


def test_settings_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test derived properties."""
    _clean_env(monkeypatch)

    # "en" is the default language, so it is reported as-is.
    assert build_settings().language == "en"

    monkeypatch.setenv("WHISPER_LANGUAGE", "fr")
    assert build_settings().language == "fr"

    # The "auto"/"none"/"null" sentinels mean "let the model decide".
    for sentinel in ("auto", "none", "null", "  AUTO  "):
        monkeypatch.setenv("WHISPER_LANGUAGE", sentinel)
        assert build_settings().language is None

    # Test auth_required
    monkeypatch.setenv("API_KEY", "test-key")
    assert build_settings().auth_required is True

    monkeypatch.setenv("API_KEY", "")
    assert build_settings().auth_required is False

    # Test max_segment_samples
    monkeypatch.setenv("MAX_SEGMENT_SECONDS", "10.0")
    monkeypatch.setenv("SAMPLE_RATE", "16000")
    assert build_settings().max_segment_samples == 160000

    # Test min_segment_samples
    monkeypatch.setenv("MIN_SEGMENT_SECONDS", "0.5")
    assert build_settings().min_segment_samples == 8000


def test_build_settings_overrides_beat_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyword overrides win over the environment and keep their type."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("PORT", "9000")

    built = build_settings(port=1234, tts_sentence_gap_ms=250)
    assert built.port == 1234
    assert isinstance(built.port, int)
    assert built.tts_sentence_gap_ms == 250


def test_build_settings_rejects_malformed_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric value for a numeric setting fails loudly."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("PORT", "not-a-number")

    with pytest.raises(ValueError, match="PORT"):
        build_settings()


def test_build_settings_returns_independent_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each call yields a separate object, so tests cannot cross-contaminate."""
    _clean_env(monkeypatch)

    first = build_settings(tts_sentence_gap_ms=10)
    second = build_settings(tts_sentence_gap_ms=20)

    assert first is not second
    assert first.tts_sentence_gap_ms == 10
    assert second.tts_sentence_gap_ms == 20
    # The process-wide singleton is untouched by either call.
    assert settings.tts_sentence_gap_ms != 10
    assert settings.tts_sentence_gap_ms != 20


def test_env_var_table_matches_settings_fields() -> None:
    """Every dataclass field must have an env source and vice versa."""
    from app.config import _FIELD_ENV_SOURCES

    field_names = {f.name for f in fields(Settings)}
    assert set(_FIELD_ENV_SOURCES) == field_names


def test_env_var_names_are_unique() -> None:
    """The documented env var list must not contain duplicates."""
    flat = [alias for name in ENV_VAR_NAMES for alias in name.split("|")]
    assert len(flat) == len(set(flat))


# --------------------------------------------------------------------------- #
# Assistant persona
# --------------------------------------------------------------------------- #
def test_default_prompt_states_name_and_creator() -> None:
    """The built-in persona must name Alvin and credit Alwin Casagan.

    This guards the *default*; a deployment may override LLM_SYSTEM_PROMPT.
    """
    from app.config import DEFAULT_LLM_SYSTEM_PROMPT as prompt

    assert "Alvin" in prompt
    assert "Alwin Casagan" in prompt
    assert "Dinagat Island" in prompt
    assert "web developer" in prompt.lower()


def test_default_prompt_keeps_the_voice_constraints() -> None:
    from app.config import DEFAULT_LLM_SYSTEM_PROMPT as prompt

    low = prompt.lower()
    # These matter for TTS output quality and were easy to lose when editing.
    assert "text-to-speech" in low
    assert "emoji" in low
    assert "markdown" in low
    assert "short" in low


def test_default_prompt_does_not_claim_a_body() -> None:
    """The model once said it lived in Dinagat; that is the creator's home."""
    from app.config import DEFAULT_LLM_SYSTEM_PROMPT as prompt

    low = prompt.lower()
    assert "do not have a body" in low or "no body" in low
