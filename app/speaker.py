"""Speaker identification using voice embeddings.

This module provides speaker enrollment and identification capabilities using
ECAPA-TDNN embeddings from SpeechBrain. The embeddings are computed on-device
for privacy and low latency.

Architecture:
    Audio segment -> Voice Activity Detection -> Embedding extraction
                                      -> Cosine similarity search
                                      -> Speaker ID or "unknown"

The enrollment process:
    1. User speaks an enrollment phrase (or multiple phrases)
    2. System extracts embeddings from each phrase
    3. Embeddings are averaged to create a speaker profile
    4. Profile is stored in memory (and optionally persisted)

The identification process:
    1. Incoming audio segment is processed through VAD
    2. If speech detected, embedding is extracted
    3. Embedding is compared against enrolled speaker profiles
    4. Best match above threshold returns speaker ID; otherwise "unknown"
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import settings

log = logging.getLogger("alvin.speaker")

# Embedding configuration
_EMBEDDING_DIM = 192  # ECAPA-TDNN embedding dimension
_DEFAULT_SIMILARITY_THRESHOLD = 0.75
_MIN_ENROLLMENT_SECONDS = 2.0
_MAX_ENROLLMENT_SECONDS = 10.0
_SAMPLE_RATE = 16000

# Global model instance (loaded lazily)
_speaker_model: Optional["_SpeakerModel"] = None
_model_lock = asyncio.Lock()


@dataclass
class SpeakerProfile:
    """Enrolled speaker profile with averaged embeddings."""

    speaker_id: str
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    enrollment_duration_sec: float = 0.0

    def add_embedding(self, embedding: np.ndarray, duration_sec: float) -> None:
        """Add a new embedding to the profile and update the average."""
        self.embeddings.append(embedding.copy())
        self.enrollment_duration_sec += duration_sec
        self.updated_at = time.time()

    @property
    def average_embedding(self) -> np.ndarray:
        """Compute the average embedding vector."""
        if not self.embeddings:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        return np.mean(np.stack(self.embeddings), axis=0).astype(np.float32)

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "embeddings": [e.tolist() for e in self.embeddings],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enrollment_duration_sec": self.enrollment_duration_sec,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        """Deserialize from dictionary."""
        profile = cls(
            speaker_id=data["speaker_id"],
            name=data["name"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            enrollment_duration_sec=data.get("enrollment_duration_sec", 0.0),
        )
        profile.embeddings = [np.array(e, dtype=np.float32) for e in data.get("embeddings", [])]
        return profile


@dataclass
class IdentificationResult:
    """Result of a speaker identification attempt."""

    speaker_id: Optional[str]
    name: Optional[str]
    similarity: float
    is_known: bool
    latency_ms: float


class _SpeakerModel:
    """Wrapper for the ECAPA-TDNN speaker embedding model."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    async def ensure_loaded(self) -> None:
        """Load the model once, coalescing concurrent callers."""
        if self._loaded:
            return
        if self._model is not None:
            self._loaded = True
            return

        loop = asyncio.get_running_loop()
        try:
            # Import here to avoid loading torch at module import time
            from speechbrain.inference.speaker import EncoderClassifier

            log.info("Loading ECAPA-TDNN speaker embedding model...")
            self._model = await asyncio.to_thread(
                EncoderClassifier.from_hparams,
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")), "speechbrain", "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
            self._loaded = True
            log.info("Speaker embedding model loaded successfully")
        except Exception as exc:
            log.exception("Failed to load speaker embedding model: %s", exc)
            self._model = None
            self._loaded = False
            raise

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio segment.

        Args:
            audio: Mono audio at 16kHz, float32 in [-1, 1]

        Returns:
            192-dimensional embedding vector (float32)
        """
        # Import lazily so importing app.main remains torch-free when speaker
        # identification is disabled or its optional dependencies are absent.
        import torch  # noqa: PLC0415

        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        # Ensure correct shape: (1, samples) for batch processing
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # SpeechBrain expects CPU torch tensors, not NumPy arrays.
        wavs = torch.from_numpy(np.ascontiguousarray(audio)).float()
        wav_lens = torch.ones(wavs.shape[0], dtype=torch.float32)

        with torch.inference_mode():
            embedding = self._model.encode_batch(
                wavs,
                wav_lens=wav_lens,
            )

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)


async def get_speaker_model() -> _SpeakerModel:
    """Get the global speaker model instance, loading if necessary."""
    global _speaker_model
    async with _model_lock:
        if _speaker_model is None:
            _speaker_model = _SpeakerModel()
        await _speaker_model.ensure_loaded()
        return _speaker_model


class SpeakerService:
    """Speaker enrollment and identification service."""

    def __init__(self, persistence_path: Optional[Path] = None) -> None:
        self.profiles: dict[str, SpeakerProfile] = {}
        self.similarity_threshold = settings.speaker_similarity_threshold
        self.persistence_path = (
            Path(persistence_path)
            if persistence_path is not None
            else Path(settings.speaker_profiles_path)
        )
        self._lock = asyncio.Lock()
        self._model: Optional[_SpeakerModel] = None

    async def initialize(self) -> None:
        """Initialize the service: load model and persisted profiles."""
        self._model = await get_speaker_model()
        await self._load_profiles()
        log.info("Speaker service initialized with %d profiles", len(self.profiles))

    async def _load_profiles(self) -> None:
        """Load speaker profiles from disk."""
        if not self.persistence_path.exists():
            return
        try:
            async with self._lock:
                with open(self.persistence_path, "rb") as f:
                    data = pickle.load(f)
                for speaker_id, profile_data in data.items():
                    self.profiles[speaker_id] = SpeakerProfile.from_dict(profile_data)
                log.info("Loaded %d speaker profiles from %s", len(self.profiles), self.persistence_path)
        except Exception as exc:
            log.warning("Failed to load speaker profiles: %s", exc)

    async def _save_profiles(self) -> None:
        """Save speaker profiles to disk."""
        try:
            async with self._lock:
                data = {sid: p.to_dict() for sid, p in self.profiles.items()}
                with open(self.persistence_path, "wb") as f:
                    pickle.dump(data, f)
        except Exception as exc:
            log.warning("Failed to save speaker profiles: %s", exc)

    @staticmethod
    def _generate_speaker_id(name: str) -> str:
        """Generate a unique speaker ID from name."""
        import hashlib
        base = name.lower().replace(" ", "_")
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{base}_{hash_suffix}"

    async def enroll(
        self,
        audio: np.ndarray,
        name: str,
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> SpeakerProfile:
        """Enroll a new speaker or add samples to existing profile.

        Args:
            audio: Mono audio at 16kHz, float32 in [-1, 1]
            name: Human-readable name for the speaker
            speaker_id: Optional existing speaker ID to add to
            language: Optional language hint (unused, for future use)

        Returns:
            Updated or new SpeakerProfile

        Raises:
            ValueError: If audio is too short or too long
        """
        duration = audio.size / _SAMPLE_RATE
        if duration < _MIN_ENROLLMENT_SECONDS:
            raise ValueError(f"Enrollment audio too short: {duration:.1f}s (minimum {_MIN_ENROLLMENT_SECONDS}s)")
        if duration > _MAX_ENROLLMENT_SECONDS:
            raise ValueError(f"Enrollment audio too long: {duration:.1f}s (maximum {_MAX_ENROLLMENT_SECONDS}s)")

        # Extract embedding
        embedding = await asyncio.to_thread(self._model.extract_embedding, audio)

        async with self._lock:
            if speaker_id and speaker_id in self.profiles:
                profile = self.profiles[speaker_id]
                profile.add_embedding(embedding, duration)
                log.info("Added enrollment sample to %s (%s), total samples: %d, duration: %.1fs",
                         profile.name, speaker_id, profile.embedding_count, profile.enrollment_duration_sec)
            else:
                speaker_id = speaker_id or self._generate_speaker_id(name)
                profile = SpeakerProfile(speaker_id=speaker_id, name=name)
                profile.add_embedding(embedding, duration)
                self.profiles[speaker_id] = profile
                log.info("Enrolled new speaker: %s (%s), duration: %.1fs", name, speaker_id, duration)

        await self._save_profiles()
        return profile

    async def identify(self, audio: np.ndarray) -> IdentificationResult:
        """Identify speaker from audio segment.

        Args:
            audio: Mono audio at 16kHz, float32 in [-1, 1]

        Returns:
            IdentificationResult with speaker info and similarity score
        """
        start = time.perf_counter()

        if not self.profiles or self._model is None:
            return IdentificationResult(
                speaker_id=None,
                name=None,
                similarity=0.0,
                is_known=False,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Extract embedding from input audio
        embedding = await asyncio.to_thread(self._model.extract_embedding, audio)

        async with self._lock:
            profiles = [
                (speaker_id, profile.name, profile.average_embedding)
                for speaker_id, profile in self.profiles.items()
            ]

        def compare_profiles() -> tuple[str | None, str | None, float]:
            best_speaker_id = None
            best_name = None
            best_similarity = 0.0
            embedding_norm = np.linalg.norm(embedding)

            for speaker_id, name, avg_embedding in profiles:
                similarity = float(
                    np.dot(embedding, avg_embedding)
                    / (embedding_norm * np.linalg.norm(avg_embedding) + 1e-8)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker_id = speaker_id
                    best_name = name

            return best_speaker_id, best_name, best_similarity

        best_speaker_id, best_name, best_similarity = await asyncio.to_thread(
            compare_profiles
        )

        is_known = best_similarity >= self.similarity_threshold
        latency_ms = (time.perf_counter() - start) * 1000

        if is_known:
            log.debug("Speaker identified: %s (%.3f)", best_name, best_similarity)
        else:
            log.debug("Unknown speaker (best: %s, %.3f)", best_name, best_similarity)

        return IdentificationResult(
            speaker_id=best_speaker_id if is_known else None,
            name=best_name if is_known else None,
            similarity=best_similarity,
            is_known=is_known,
            latency_ms=latency_ms,
        )

    async def identify_streaming(
        self,
        audio_queue: asyncio.Queue,
        min_duration_sec: float = 1.0,
        max_duration_sec: float = 5.0,
    ) -> IdentificationResult:
        """Identify speaker from streaming audio queue.

        Accumulates audio from the queue until min_duration_sec is reached,
        then performs identification. Continues accumulating up to max_duration_sec
        for better accuracy if confidence is low.

        Args:
            audio_queue: Queue of audio chunks (np.ndarray)
            min_duration_sec: Minimum audio duration before identification
            max_duration_sec: Maximum audio duration to accumulate

        Returns:
            IdentificationResult
        """
        chunks: list[np.ndarray] = []
        total_samples = 0
        min_samples = int(min_duration_sec * _SAMPLE_RATE)
        max_samples = int(max_duration_sec * _SAMPLE_RATE)

        # Accumulate audio
        while total_samples < min_samples:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
                if chunk is None:
                    break
                if isinstance(chunk, np.ndarray):
                    chunks.append(chunk)
                    total_samples += chunk.size
                elif isinstance(chunk, bytes):
                    # Convert bytes to float32
                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    chunks.append(arr)
                    total_samples += arr.size
            except asyncio.TimeoutError:
                break

        if total_samples == 0:
            return IdentificationResult(
                speaker_id=None, name=None, similarity=0.0, is_known=False, latency_ms=0.0
            )

        # Combine chunks
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        # Trim to max duration
        if total_samples > max_samples:
            audio = audio[:max_samples]

        return await self.identify(audio)

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by ID."""
        return self.profiles.get(speaker_id)

    def list_profiles(self) -> list[SpeakerProfile]:
        """List all enrolled speaker profiles."""
        return list(self.profiles.values())

    async def delete_profile(self, speaker_id: str) -> bool:
        """Delete a speaker profile."""
        async with self._lock:
            if speaker_id not in self.profiles:
                return False
            del self.profiles[speaker_id]

        await self._save_profiles()
        return True

    async def update_threshold(self, threshold: float) -> None:
        """Update the similarity threshold for identification."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        log.info("Speaker similarity threshold updated to %.3f", self.similarity_threshold)


# Global service instance
_speaker_service: Optional[SpeakerService] = None
_service_lock = asyncio.Lock()


async def get_speaker_service() -> SpeakerService:
    """Get the global speaker service instance."""
    global _speaker_service
    async with _service_lock:
        if _speaker_service is None:
            _speaker_service = SpeakerService()
            await _speaker_service.initialize()
        return _speaker_service


async def shutdown_speaker_service() -> None:
    """Shutdown the speaker service and save profiles."""
    global _speaker_service
    async with _service_lock:
        if _speaker_service is not None:
            await _speaker_service._save_profiles()
            _speaker_service = None
