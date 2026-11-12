"""Unit tests for the streaming VAD endpointing logic (no Whisper model)."""

from __future__ import annotations

import numpy as np

from app.vad import FRAME_SAMPLES, VADStreamDetector

SPEECH_RMS_FLOOR = 0.02  # real speech segments are far louder than silence


def _feed(audio: np.ndarray, detector: VADStreamDetector):
    """Feed ``audio`` 512-sample frames; collect VAD start/end events + coords."""
    events: list[tuple[int, dict]] = []
    feed = 0
    for off in range(0, len(audio), FRAME_SAMPLES):
        chunk = audio[off : off + FRAME_SAMPLES]
        if len(chunk) != FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        ev = detector.process_frame(chunk)
        feed += FRAME_SAMPLES
        if ev is not None:
            events.append((feed, ev))
    return events


def test_vad_loads_bundled_model():
    detector = VADStreamDetector()
    assert detector.process_frame(np.zeros(FRAME_SAMPLES, dtype=np.float32)) is None
    detector.reset()


def test_real_speech_produces_start_end_events(speech_array):
    audio, _sr = speech_array
    detector = VADStreamDetector()
    events = _feed(audio, detector)

    starts = [e for _, e in events if "start" in e]
    ends = [e for _, e in events if "end" in e]
    assert starts, "expected at least one speech start event"
    assert ends, "expected at least one speech end event"

    total = len(audio)
    for _, ev in events:
        coord = ev["start"] if "start" in ev else ev["end"]
        assert 0 <= coord <= total + FRAME_SAMPLES  # start/end may reference pad


def test_vad_coordinates_are_ordered_and_have_speech(speech_array):
    audio, _sr = speech_array
    detector = VADStreamDetector()
    events = _feed(audio, detector)
    pairs = []
    last_start = None
    for _, ev in events:
        if "start" in ev:
            last_start = ev["start"]
        elif "end" in ev and last_start is not None:
            pairs.append((last_start, ev["end"]))
            last_start = None

    assert pairs, "expected at least one complete start->end pair"
    for start, end in pairs:
        assert 0 <= start < end <= len(audio) + FRAME_SAMPLES
        seg = audio[start : max(end, 0)] if end <= len(audio) else audio[start:]
        assert seg.size > 0
        # Each finalized segment should carry audible energy.
        rms = float(np.sqrt((seg**2).mean()))
        assert rms > SPEECH_RMS_FLOOR, f"segment {start}:{end} too quiet (rms={rms})"
