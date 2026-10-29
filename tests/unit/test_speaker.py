"""Unit tests for speaker identification."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import numpy as np

from app.speaker import (
    _SpeakerModel,
    SpeakerProfile,
    IdentificationResult,
    SpeakerService,
)

# _generate_speaker_id is a static method on SpeakerService


class TestSpeakerProfile:
    def test_new_profile(self):
        """Test creating a new speaker profile."""
        profile = SpeakerProfile(speaker_id="test_123", name="Test User")
        assert profile.speaker_id == "test_123"
        assert profile.name == "Test User"
        assert profile.embedding_count == 0
        assert profile.enrollment_duration_sec == 0.0

    def test_add_embedding(self):
        """Test adding embeddings to a profile."""
        profile = SpeakerProfile(speaker_id="test_123", name="Test User")
        embedding = np.random.randn(192).astype(np.float32)

        profile.add_embedding(embedding, 2.0)
        assert profile.embedding_count == 1
        assert profile.enrollment_duration_sec == 2.0

        profile.add_embedding(embedding, 3.0)
        assert profile.embedding_count == 2
        assert profile.enrollment_duration_sec == 5.0

    def test_average_embedding(self):
        """Test average embedding computation."""
        profile = SpeakerProfile(speaker_id="test_123", name="Test User")

        # Empty profile should return zeros
        avg = profile.average_embedding
        assert avg.shape == (192,)
        assert np.allclose(avg, 0.0)

        # Add embeddings
        e1 = np.ones(192, dtype=np.float32)
        e2 = np.ones(192, dtype=np.float32) * 2
        profile.add_embedding(e1, 1.0)
        profile.add_embedding(e2, 1.0)

        avg = profile.average_embedding
        assert np.allclose(avg, 1.5)

    def test_serialization(self):
        """Test profile serialization and deserialization."""
        profile = SpeakerProfile(speaker_id="test_123", name="Test User")
        embedding = np.random.randn(192).astype(np.float32)
        profile.add_embedding(embedding, 2.5)

        data = profile.to_dict()
        assert data["speaker_id"] == "test_123"
        assert data["name"] == "Test User"
        assert len(data["embeddings"]) == 1

        restored = SpeakerProfile.from_dict(data)
        assert restored.speaker_id == "test_123"
        assert restored.name == "Test User"
        assert restored.embedding_count == 1
        assert np.allclose(restored.embeddings[0], embedding)


class TestSpeakerService:
    @pytest.fixture
    def service(self, tmp_path):
        """Create a speaker service with temporary persistence."""
        return SpeakerService(persistence_path=tmp_path / "profiles.pkl")

    def test_generate_speaker_id(self):
        """Test speaker ID generation."""
        id1 = SpeakerService._generate_speaker_id("John Doe")
        id2 = SpeakerService._generate_speaker_id("John Doe")
        id3 = SpeakerService._generate_speaker_id("Jane Smith")

        assert id1 == id2  # Deterministic
        assert id1 != id3  # Different names produce different IDs
        assert id1.startswith("john_doe_")

    def test_enroll_and_identify(self, service):
        """Enrollment persists and identification uses the enrolled embedding."""
        embedding = np.zeros(192, dtype=np.float32)
        embedding[0] = 1.0
        service._model = SimpleNamespace(
            extract_embedding=lambda _audio: embedding.copy()
        )
        audio = np.zeros(32000, dtype=np.float32)

        async def enroll_and_identify():
            profile = await asyncio.wait_for(
                service.enroll(audio, "Alice"), timeout=0.05
            )
            result = await service.identify(audio)
            return profile, result

        profile, result = asyncio.run(enroll_and_identify())
        assert profile.name == "Alice"
        assert profile.embedding_count == 1
        assert result.is_known
        assert result.speaker_id == profile.speaker_id

    def test_list_profiles(self, service):
        """Test listing profiles."""
        profiles = service.list_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) == 0

    def test_delete_profile(self, service):
        """Test deleting a profile."""
        # Add a profile directly
        profile = SpeakerProfile(speaker_id="test_123", name="Test")
        service.profiles["test_123"] = profile

        import asyncio
        result = asyncio.run(service.delete_profile("test_123"))
        assert result is True
        assert "test_123" not in service.profiles

        result = asyncio.run(service.delete_profile("nonexistent"))
        assert result is False

    def test_update_threshold(self, service):
        """Test updating similarity threshold."""
        service.update_threshold(0.5)
        assert service.similarity_threshold == 0.5

        service.update_threshold(1.5)  # Should clamp to 1.0
        assert service.similarity_threshold == 1.0

        service.update_threshold(-0.5)  # Should clamp to 0.0
        assert service.similarity_threshold == 0.0

    def test_identify_returns_unknown_if_model_failed_to_initialize(self, service):
        profile = SpeakerProfile(speaker_id="alice", name="Alice")
        profile.add_embedding(np.ones(192, dtype=np.float32), 2.0)
        service.profiles[profile.speaker_id] = profile

        result = asyncio.run(service.identify(np.zeros(16000, dtype=np.float32)))

        assert result.is_known is False
        assert result.speaker_id is None
        assert result.name is None

    def test_identify_offloads_profile_comparison(self, service, monkeypatch):
        embedding = np.ones(192, dtype=np.float32)
        profile = SpeakerProfile(speaker_id="alice", name="Alice")
        profile.add_embedding(embedding, 2.0)
        service.profiles[profile.speaker_id] = profile
        service._model = SimpleNamespace(
            extract_embedding=lambda _audio: embedding.copy()
        )

        original_to_thread = asyncio.to_thread
        calls = []

        async def record_to_thread(function, *args, **kwargs):
            calls.append(function)
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", record_to_thread)
        result = asyncio.run(service.identify(np.zeros(16000, dtype=np.float32)))

        assert result.is_known
        assert len(calls) == 2

    def test_persistence_path_uses_configured_setting(self, monkeypatch, tmp_path):
        from app.config import settings

        configured_path = tmp_path / "custom_profiles.pkl"
        monkeypatch.setattr(settings, "speaker_profiles_path", str(configured_path))

        service = SpeakerService()

        assert service.persistence_path == configured_path


def test_extract_embedding_converts_audio_to_torch_tensor(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        @property
        def shape(self):
            return self.values.shape

        def float(self):
            self.values = self.values.astype(np.float32, copy=False)
            return self

        def squeeze(self, axis):
            return FakeTensor(np.squeeze(self.values, axis=axis))

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    @contextmanager
    def inference_mode():
        yield

    fake_torch = SimpleNamespace(
        float32=np.float32,
        from_numpy=FakeTensor,
        ones=lambda size, dtype=None: FakeTensor(np.ones(size, dtype=np.float32)),
        inference_mode=inference_mode,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeEncoder:
        def encode_batch(self, wavs, wav_lens):
            assert isinstance(wavs, FakeTensor)
            assert wavs.shape == (1, 10)
            assert wavs.values.flags.c_contiguous
            assert wav_lens.shape == (1,)
            return FakeTensor(np.ones((1, 192), dtype=np.float32))

    model = _SpeakerModel()
    model._loaded = True
    model._model = FakeEncoder()

    embedding = model.extract_embedding(np.arange(20, dtype=np.float32)[::2])

    assert embedding.shape == (192,)
    assert embedding.dtype == np.float32


def test_importing_app_main_does_not_require_speaker_packages():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockSpeakerPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".", 1)[0]
                if root in {"torch", "torchaudio", "speechbrain"}:
                    raise ModuleNotFoundError(f"blocked optional package: {fullname}")
                return None

        sys.meta_path.insert(0, BlockSpeakerPackages())
        import app.main
        """
    )
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
