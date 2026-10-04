"""Unit tests for the STT pre-flight guards (no Whisper model, no network).

Covers the three guards that sit in front of every transcription:

* :func:`app.stt.audio_dbfs` and the silence short-circuit in
  ``STTService.transcribe_async``;
* :func:`app.stt.breaker_timeout`, the duration-scaled circuit breaker;
* ``GroqBackend._compress_audio``, the in-memory Opus encoder that replaced the
  pydub/ffmpeg path.
"""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
import soundfile as sf

from app.config import settings
from app.stt import (
    GroqBackend,
    STTService,
    TranscriptionResult,
    audio_dbfs,
    breaker_timeout,
)

SILENCE_GATE_BUDGET_SEC = 0.002  # the gate must stay far under the 2 ms target


def _silence(seconds: float) -> np.ndarray:
    """Digital silence as float32, the shape the connection layer produces."""
    return np.zeros(int(settings.sample_rate * seconds), dtype=np.float32)


def _tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """Full-scale-ish sine, comfortably above the gate."""
    t = np.linspace(0.0, seconds, int(settings.sample_rate * seconds), False)
    return (np.sin(2 * np.pi * 440.0 * t) * amplitude).astype(np.float32)


# --------------------------------------------------------------------------- #
# audio_dbfs
# --------------------------------------------------------------------------- #
def test_dbfs_of_digital_silence_is_negative_infinity():
    assert audio_dbfs(_silence(1.5)) == float("-inf")


def test_dbfs_of_empty_buffer_is_negative_infinity():
    assert audio_dbfs(np.empty(0, dtype=np.float32)) == float("-inf")


def test_dbfs_tracks_amplitude():
    # A sine of amplitude A has RMS A/sqrt(2), i.e. -3 dBFS below A.
    quiet = audio_dbfs(_tone(1.0, amplitude=0.1))
    loud = audio_dbfs(_tone(1.0, amplitude=0.8))
    assert loud > quiet
    assert abs(quiet - (20.0 * np.log10(0.1 / np.sqrt(2)))) < 0.1


def test_real_speech_is_above_the_gate(speech_array):
    audio, _sr = speech_array
    assert audio_dbfs(audio) > settings.silence_threshold_dbfs


# --------------------------------------------------------------------------- #
# silence gate
# --------------------------------------------------------------------------- #
def test_silence_returns_empty_without_touching_a_backend(monkeypatch):
    """The gate must short-circuit before any backend is even created."""
    service = STTService()

    async def _boom(*_args, **_kwargs):
        raise AssertionError("a backend was called for a silent buffer")

    monkeypatch.setattr(GroqBackend, "transcribe_async", _boom)
    monkeypatch.setattr(
        "app.stt.LocalWhisperBackend.transcribe_async", _boom, raising=False
    )

    assert asyncio.run(service.transcribe_async(_silence(2.0), "en")) == ("", 0.0)


def test_silence_gate_is_far_under_two_milliseconds():
    service = STTService()
    audio = _silence(2.0)

    async def _run() -> tuple[tuple[str, float], float]:
        start = time.perf_counter()
        result = await service.transcribe_async(audio, "en")
        return result, time.perf_counter() - start

    result, elapsed = asyncio.run(_run())

    assert result == ("", 0.0)
    assert elapsed < SILENCE_GATE_BUDGET_SEC, f"gate took {elapsed * 1000:.3f} ms"


def test_quiet_but_audible_audio_still_reaches_a_backend(monkeypatch):
    """A -30 dBFS buffer is quiet, not silent: it must not be dropped."""
    service = STTService()
    seen: list[int] = []

    async def _fake(_self, audio, _language):
        seen.append(audio.size)
        raise RuntimeError("stop here")  # falls back, proving the gate passed

    async def _local(_self, _audio, _language):
        return TranscriptionResult("ok", 0.5, 1.0, "fake-local")

    monkeypatch.setattr(GroqBackend, "transcribe_async", _fake)
    monkeypatch.setattr(
        "app.stt.LocalWhisperBackend.transcribe_async", _local, raising=False
    )

    quiet = _tone(1.0, amplitude=0.03)  # ~-33 dBFS
    text, _confidence = asyncio.run(service.transcribe_async(quiet, "en"))

    assert seen, "the quiet buffer was dropped by the gate"
    assert text == "ok"


# --------------------------------------------------------------------------- #
# breaker timeout
# --------------------------------------------------------------------------- #
def test_short_audio_uses_the_configured_floor():
    assert breaker_timeout(1.5) == pytest.approx(settings.groq_timeout_sec)


def test_long_audio_gets_a_scaled_budget():
    budget = breaker_timeout(12.4)
    expected = settings.groq_timeout_base_sec + 12.4 * settings.groq_timeout_slope_sec
    assert budget == pytest.approx(expected)
    assert budget > settings.groq_timeout_sec


def test_breaker_timeout_is_monotonic_and_never_below_the_floor():
    budgets = [breaker_timeout(seconds / 10) for seconds in range(1, 130)]
    assert budgets == sorted(budgets)
    assert min(budgets) >= settings.groq_timeout_sec


# --------------------------------------------------------------------------- #
# in-memory Opus encoder
# --------------------------------------------------------------------------- #
def test_opus_encoder_needs_no_ffmpeg_binary(monkeypatch):
    """The encoder must not shell out: a missing ffmpeg cannot break it."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    ogg = GroqBackend()._compress_audio(_tone(3.0))
    assert ogg[:4] == b"OggS"


def test_opus_encoder_round_trips_at_16k_mono(speech_array):
    audio, _sr = speech_array
    ogg = GroqBackend()._compress_audio(audio)

    decoded, rate = sf.read(io.BytesIO(ogg), dtype="int16")
    assert rate == settings.sample_rate
    assert decoded.ndim == 1
    # Opus is lossy, so allow a small duration drift on the decoded length.
    assert abs(decoded.size - audio.size) / audio.size < 0.02


def test_opus_encoder_shrinks_the_upload(speech_array):
    audio, _sr = speech_array
    ogg = GroqBackend()._compress_audio(audio)
    pcm_bytes = audio.size * 2  # 16-bit mono
    assert len(ogg) < pcm_bytes / 4


def test_opus_encoder_handles_stereo_input():
    stereo = np.stack([_tone(1.0), np.zeros(settings.sample_rate, dtype=np.float32)])
    ogg = GroqBackend()._compress_audio(stereo)
    decoded, rate = sf.read(io.BytesIO(ogg), dtype="int16")
    assert rate == settings.sample_rate
    assert decoded.ndim == 1
