"""Streaming voice-activity-detection front-end.

Wraps ``silero_vad``'s :class:`VADIterator` to provide frame-by-frame
endpointing for real-time STT. A fresh, independent VAD model is created per
detector instance, so multiple WebSocket connections never share recurrent
LSTM state (silero's model is stateful across frames).

The ONNX runtime back-end is used (``load_silero_vad(onnx=True)``): it is
torch-free, releases the GIL during the kernels and avoids the TorchScript
dtype fragility seen with ``torch.jit.load`` on some builds.
"""

from __future__ import annotations

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from .config import settings

# silero VAD operates on fixed 32 ms frames (512 samples at 16 kHz).
FRAME_SAMPLES: int = 512

_SHARED_MODEL = None


class VADStreamDetector:
    """Stateful streaming VAD. Feed one 512-sample float32 frame at a time."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        min_silence_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
        sampling_rate: int | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate or settings.sample_rate
        threshold = settings.vad_threshold if threshold is None else threshold
        min_silence_duration_ms = (
            settings.min_silence_duration_ms
            if min_silence_duration_ms is None
            else min_silence_duration_ms
        )
        speech_pad_ms = (
            settings.speech_pad_ms if speech_pad_ms is None else speech_pad_ms
        )

        # Shared ONNX model across all connections; VADIterator isolates the recurrent state.
        global _SHARED_MODEL
        if _SHARED_MODEL is None:
            _SHARED_MODEL = load_silero_vad(onnx=True)
        self.vad = VADIterator(
            _SHARED_MODEL,
            threshold=threshold,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def reset(self) -> None:
        """Reset recurrent state (call at the start of a new utterance)."""
        self.vad.reset_states()

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """Process a single ``FRAME_SAMPLES`` frame.

        Returns ``{"start": int}`` when speech begins, ``{"end": int}`` when an
        utterance ends (after ``min_silence_duration_ms`` of silence), or
        ``None`` while the frame is uneventful. Sample indices are absolute
        within the detector's active lifetime.
        """
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(
                f"VAD frame must be {FRAME_SAMPLES} samples, got {len(frame)}"
            )
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        return self.vad(frame)
