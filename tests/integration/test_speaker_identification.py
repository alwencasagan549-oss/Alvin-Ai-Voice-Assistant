"""Optional end-to-end checks for the real SpeechBrain speaker model."""

from __future__ import annotations

import asyncio
import importlib.util
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HAS_SPEAKER_DEPENDENCIES = all(
    importlib.util.find_spec(package) is not None
    for package in ("torch", "torchaudio", "speechbrain")
)


@pytest.mark.skipif(
    not HAS_SPEAKER_DEPENDENCIES,
    reason="install requirements-speaker.txt to run the real model test",
)
@pytest.mark.asyncio
async def test_enabled_app_lifespan_loads_enrolled_voice_and_identifies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import speaker
    from app.config import settings
    from app.main import app, lifespan
    from app.speaker import SpeakerService

    with wave.open(str(ROOT / "tests" / "data" / "sample_speech.wav"), "rb") as wav:
        assert wav.getframerate() == settings.sample_rate
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        audio = audio.astype(np.float32) / 32768.0

    profile_path = tmp_path / "speaker_profiles.pkl"
    monkeypatch.setattr(settings, "speaker_enabled", True)
    monkeypatch.setattr(settings, "speaker_profiles_path", str(profile_path))
    monkeypatch.setattr(settings, "speaker_min_identify_sec", 1.0)
    monkeypatch.setattr(settings, "preload_fallback", False)
    monkeypatch.setattr(settings, "groq_fallback", False)
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "tts_warmup", False)
    monkeypatch.setattr(settings, "llm_enabled", False)
    monkeypatch.setattr(settings, "llm_api_key", "")

    enrollment_service = SpeakerService(persistence_path=profile_path)
    await enrollment_service.initialize()
    profile = await enrollment_service.enroll(audio, "Fixture Speaker")
    embedding = await asyncio.to_thread(
        enrollment_service._model.extract_embedding, audio
    )
    assert embedding.shape == (192,)

    speaker._speaker_service = None
    monkeypatch.setattr("app.main.stt_service.ensure_loaded", AsyncMock())
    monkeypatch.setattr("app.main.stt_service.shutdown", AsyncMock())
    monkeypatch.setattr("app.main.llm_service.shutdown", AsyncMock())

    try:
        async with lifespan(app):
            service = await speaker.get_speaker_service()
            result = await service.identify(audio)

            assert result.is_known
            assert result.speaker_id == profile.speaker_id
            assert result.name == "Fixture Speaker"
    finally:
        await speaker.shutdown_speaker_service()
