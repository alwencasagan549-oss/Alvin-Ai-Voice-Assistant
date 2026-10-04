"""Unit tests for the TTS text pipeline, silence trimmer, and error events.

No network access: everything here is pure logic. The streaming integration
itself is covered by ``scripts/bench_tts.py`` and the WebSocket tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.tts import (
    EdgeSilenceTrimmer,
    TTSSynthesisError,
    chunk_sentences,
    clean_text_for_tts,
    dbfs_to_amplitude,
    silence_gap_bytes,
    synthesize_pcm,
)

SR = 16000


def pcm(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()


def tone(ms: int, amplitude: int = 8000) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000))
    return (amplitude * np.sin(2 * np.pi * 220 * t / SR)).astype(np.int16)


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


def cat(*blocks: np.ndarray) -> np.ndarray:
    return np.concatenate(blocks)


def test_clean_text_strips_markdown_urls_and_emoji():
    text = "## Heading\n**Bold** and *italic* see https://x.io \U0001f600 `code`"
    assert clean_text_for_tts(text) == "Heading Bold and italic see code"


def test_chunk_sentences_splits_on_terminators():
    chunks = chunk_sentences("One. Two! Three? Four without punctuation")
    assert chunks == ["One.", "Two!", "Three?", "Four without punctuation"]


def test_chunk_sentences_ignores_empty_input():
    assert chunk_sentences("") == []
    assert chunk_sentences("   \n ") == []


def test_dbfs_threshold_maps_to_int16_amplitude():
    assert dbfs_to_amplitude(0) == 32767
    assert dbfs_to_amplitude(-50) == pytest.approx(103, abs=1)
    assert dbfs_to_amplitude(-50) == dbfs_to_amplitude(-50)


def test_silence_gap_bytes_length():
    assert len(silence_gap_bytes(120)) == int(0.12 * SR) * 2


def test_trimmer_drops_leading_silence():
    trimmer = EdgeSilenceTrimmer(threshold=dbfs_to_amplitude(-50), hold=1600)
    out = trimmer.push(pcm(silence(500)))
    assert out == b""  # nothing but silence so far
    out = trimmer.push(pcm(tone(200)))
    assert out != b""
    assert trimmer.speech_started is True


def test_trimmer_drops_trailing_silence_on_flush():
    trimmer = EdgeSilenceTrimmer(threshold=dbfs_to_amplitude(-50), hold=1600)
    emitted = trimmer.push(pcm(cat(silence(100), tone(300))))
    trimmed = trimmer.push(pcm(cat(tone(100), silence(400))))
    tail = trimmer.flush()

    combined = np.frombuffer(emitted + trimmed + tail, dtype="<i2")
    assert combined.size > 0
    # The trailing 400 ms of silence must not survive.
    assert abs(int(combined[-1])) <= dbfs_to_amplitude(-50)
    assert int(np.abs(combined).max()) > dbfs_to_amplitude(-50)


def test_trimmer_keeps_speech_when_flushed():
    trimmer = EdgeSilenceTrimmer(threshold=dbfs_to_amplitude(-50), hold=800)
    trimmer.push(pcm(cat(silence(100), tone(100))))
    trimmer.push(pcm(tone(200)))  # held tail is speech, not silence
    tail = trimmer.flush()
    assert np.frombuffer(tail, dtype="<i2").size > 0
    assert int(np.abs(np.frombuffer(tail, dtype="<i2")).max()) > dbfs_to_amplitude(-50)


def test_trimmer_hold_zero_is_passthrough_after_onset():
    trimmer = EdgeSilenceTrimmer(threshold=dbfs_to_amplitude(-50), hold=0)
    assert trimmer.push(pcm(cat(silence(100), tone(100)))) != b""
    block = pcm(tone(100))
    assert trimmer.push(block) == block


def test_trimmer_discards_all_silent_sentence():
    trimmer = EdgeSilenceTrimmer(threshold=dbfs_to_amplitude(-50), hold=1600)
    trimmer.push(pcm(silence(2000)))
    assert trimmer.flush() == b""
    assert trimmer.speech_started is False


def test_synthesize_pcm_raises_on_error_event():
    import asyncio

    import app.tts as tts_module

    async def fake_stream(text, voice=None, prefetch=None):
        yield pcm(tone(50))
        yield {
            "type": "error",
            "stage": "tts",
            "message": "boom",
            "sentence_index": 1,
            "sentence": "second",
            "fatal": False,
        }

    async def run():
        frames = []
        with pytest.raises(TTSSynthesisError) as excinfo:
            async for frame in synthesize_pcm("hello"):
                frames.append(frame)
        assert frames  # audio before the failure is still delivered
        assert excinfo.value.event["sentence_index"] == 1

    original = tts_module.synthesize_stream
    tts_module.synthesize_stream = fake_stream
    try:
        asyncio.run(run())
    finally:
        tts_module.synthesize_stream = original


def _fake_pipeline(monkeypatch, chunks_per_sentence: int = 3):
    """Replace the network + decoder with deterministic fakes.

    Returns the number of PCM samples each emitted audio frame carries.
    """
    import asyncio

    import app.tts as tts_module
    from app.config import settings

    frame_samples = 1600  # 100 ms of audio per fake chunk

    class FakeAssembler:
        def __init__(self, threshold, hold, interval_bytes):
            self._left = chunks_per_sentence

        async def push(self, mp3_chunk):
            if self._left <= 0:
                return b""
            self._left -= 1
            return pcm(tone(100))

        async def finish(self):
            return pcm(tone(100))

    async def fake_fetch(sentence, voice, queue):
        for _ in range(chunks_per_sentence):
            queue.put_nowait(b"mp3")
        queue.put_nowait(tts_module._SENTENCE_DONE)

    monkeypatch.setattr(tts_module, "_PcmAssembler", FakeAssembler)
    monkeypatch.setattr(tts_module, "_fetch_sentence_mp3", fake_fetch)
    monkeypatch.setattr(settings, "tts_trim_edge_ms", 0)
    monkeypatch.setattr(settings, "tts_sentence_gap_ms", 120)
    return frame_samples, asyncio


def test_gap_inserted_once_per_sentence(monkeypatch):
    """Regression: the joining gap must not repeat per decoded chunk."""
    import asyncio

    frame_samples, _ = _fake_pipeline(monkeypatch, chunks_per_sentence=3)
    from app.config import settings
    from app.tts import synthesize_stream

    async def run():
        stream = [
            item async for item in synthesize_stream("One. Two. Three.", prefetch=1)
        ]
        assert all(isinstance(item, bytes) for item in stream)
        return stream

    stream = asyncio.run(run())
    gap_frames = int(settings.tts_sentence_gap_ms * SR / 1000)
    audio_frames = sum(
        len(item) // 2 for item in stream if len(item) // 2 != gap_frames
    )
    gaps = sum(1 for item in stream if len(item) // 2 == gap_frames)

    sentences = 3
    assert gaps == sentences - 1  # one gap between sentences, none before the first
    # 3 sentences x (3 streamed chunks + 1 final chunk) x 100 ms
    assert audio_frames == 3 * (3 + 1) * frame_samples


def test_prefetch_window_is_respected(monkeypatch):
    """Sentences are fetched concurrently up to the prefetch window."""
    frame_samples, asyncio = _fake_pipeline(monkeypatch, chunks_per_sentence=1)
    import app.tts as tts_module
    from app.tts import synthesize_stream

    live = 0
    peak = 0

    original_fetch = tts_module._fetch_sentence_mp3

    async def counting_fetch(sentence, voice, queue):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            await original_fetch(sentence, voice, queue)
        finally:
            live -= 1

    tts_module._fetch_sentence_mp3 = counting_fetch

    async def run():
        return [item async for item in synthesize_stream("A. B. C. D.", prefetch=2)]

    try:
        stream = asyncio.run(run())
    finally:
        tts_module._fetch_sentence_mp3 = original_fetch

    assert frame_samples  # sanity: fake chunks produced audio
    assert len(stream) > 0
    assert peak <= 2, f"prefetch window exceeded: {peak} concurrent fetches"


# --------------------------------------------------------------------------- #
# warm_up
# --------------------------------------------------------------------------- #
def test_warm_up_reports_success_when_audio_comes_back(monkeypatch):
    """A warm-up that returns audio is a success and yields nothing to callers."""
    import asyncio

    from app.tts import warm_up

    async def fake_stream(_text, _voice=None, _prefetch=None):
        yield pcm(tone(20))
        yield pcm(tone(20))

    monkeypatch.setattr("app.tts.synthesize_stream", fake_stream)
    assert asyncio.run(warm_up()) is True


def test_warm_up_swallows_error_events(monkeypatch):
    """A sentence failure must not raise; it only means a slower first turn."""
    import asyncio

    from app.tts import warm_up

    async def fake_stream(_text, _voice=None, _prefetch=None):
        yield {"type": "error", "stage": "tts", "message": "boom"}

    monkeypatch.setattr("app.tts.synthesize_stream", fake_stream)
    assert asyncio.run(warm_up()) is False


def test_warm_up_never_raises_on_transport_failure(monkeypatch):
    import asyncio

    from app.tts import warm_up

    async def fake_stream(_text, _voice=None, _prefetch=None):
        raise ConnectionError("edge-tts unreachable")
        yield b""  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("app.tts.synthesize_stream", fake_stream)
    assert asyncio.run(warm_up()) is False


def test_warmup_text_is_a_single_short_sentence():
    from app.tts import WARMUP_TEXT, chunk_sentences, clean_text_for_tts

    sentences = chunk_sentences(clean_text_for_tts(WARMUP_TEXT))
    assert len(sentences) == 1
    assert len(WARMUP_TEXT) <= 16


def _run_fetch(fake_comm, retries=2, backoff_ms=1):
    """Drive one ``_fetch_sentence_mp3`` with a monkeypatched edge_tts.Communicate."""
    import asyncio

    import app.tts as tts_module
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr(tts_module.edge_tts, "Communicate", fake_comm)
    queue: asyncio.Queue = asyncio.Queue()
    try:
        asyncio.run(
            tts_module._fetch_sentence_mp3(
                "hi", "voice", queue, retries=retries, backoff_ms=backoff_ms
            )
        )
    finally:
        mp.undo()
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return tts_module, items


def test_fetch_sentence_mp3_retries_transient_failure():
    state = {"attempts": 0}

    class _FakeComm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self):
            async def gen():
                state["attempts"] += 1
                if state["attempts"] == 1:
                    raise RuntimeError("network blip")
                yield {"type": "audio", "data": b"ok"}

            return gen()

    tts_module, items = _run_fetch(_FakeComm)
    assert state["attempts"] == 2  # first failed, retry succeeded
    assert b"ok" in items
    assert tts_module._SENTENCE_DONE in items
    assert not any(isinstance(item, Exception) for item in items)


def test_fetch_sentence_mp3_does_not_retry_after_partial_audio():
    state = {"attempts": 0}

    class _FakeComm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self):
            async def gen():
                state["attempts"] += 1
                # Emit one audio chunk, then fail: the sentence already started,
                # so a restart would double-speak it and must not happen.
                yield {"type": "audio", "data": b"part"}
                raise RuntimeError("mid-stream drop")

            return gen()

    tts_module, items = _run_fetch(_FakeComm)
    assert state["attempts"] == 1  # no retry once audio was emitted
    assert b"part" in items
    assert any(isinstance(item, Exception) for item in items)
    assert tts_module._SENTENCE_DONE in items


def test_fetch_sentence_mp3_gives_up_after_exhausting_retries():
    state = {"attempts": 0}

    class _FakeComm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self):
            async def gen():
                state["attempts"] += 1
                raise RuntimeError("still down")
                yield  # pragma: no cover - makes this an async generator

            return gen()

    tts_module, items = _run_fetch(_FakeComm)
    assert state["attempts"] == 3  # initial attempt + 2 retries
    assert sum(1 for item in items if isinstance(item, Exception)) == 1
    assert tts_module._SENTENCE_DONE in items
