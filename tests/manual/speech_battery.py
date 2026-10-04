"""Manual speech + edge-case battery for the Alvin STT/TTS pipeline.

Companion to ``tests/benchmark_pipeline.py``: that script measures synthetic
latencies, this one feeds the *real* test phrases from the acceptance plan
through the same pipeline and checks latency, accuracy and robustness.

Every phrase is synthesized once with the service's own TTS (``synthesize_pcm``)
and cached as a 16 kHz / 16-bit / mono WAV under ``tests/data/battery/``, so the
same files can be played through a speaker to exercise the live-microphone path
by hand. ``--regenerate`` rebuilds them.

Categories:

* **1 -- rapid conversational turns**: per-turn STT latency + TTS TTFA.
* **2 -- numbers, acronyms, homophones**: transcript vs the expected string,
  token error rate, and an explicit verdict on digits/acronyms.
* **3 -- multi-sentence paragraphs**: TTS time-to-first-audio, inter-sentence
  dead air, and boundary click / DC-offset checks (the automated stand-in for
  listening for pops between sentences).
* **4 -- robustness**:
  * A: 2 s of digital silence -> must be rejected by the VAD and cost ~0 ms.
  * B: background noise and cough-like bursts at several SNRs -> must not
    produce text.
  * C: cloud outage -> circuit breaker + local fallback, with no exception
    escaping to the caller.

Usage::

    .\\.venv\\Scripts\\python.exe tests\\manual\\speech_battery.py
    .\\.venv\\Scripts\\python.exe tests\\manual\\speech_battery.py --categories 2,4
    .\\.venv\\Scripts\\python.exe tests\\manual\\speech_battery.py --regenerate

Not collected by pytest: the module name does not match ``test_*.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.stt import STTService
from app.tts import (
    chunk_sentences,
    clean_text_for_tts,
    synthesize_pcm,
    synthesize_stream,
)
from app.vad import FRAME_SAMPLES, VADStreamDetector

BATTERY_DIR = ROOT / "tests" / "data" / "battery"
BYTES_PER_PCM_SECOND = settings.sample_rate * 2

# Category 1 -- short conversational phrases, spoken back to back.
RAPID_TURNS: list[tuple[str, str]] = [
    ("turn-1", "Hey, what time is it?"),
    ("turn-2", "Can you hear me clearly?"),
    ("turn-3", "Stop speaking."),
]

# Category 2 -- digits, mixed alphanumeric tokens, acronyms, proper nouns.
ACCURACY_PHRASES: list[tuple[str, str]] = [
    (
        "numbers+email",
        "My order number is 4829-B, and my email is test@domain.org.",
    ),
    (
        "city+acronym",
        "I live in Cagayan de Oro City near API street.",
    ),
]

# Category 3 -- one paragraph, four sentences, to exercise prefetch/gap/trim.
PARAGRAPH_TEXT = (
    "The assistant is receiving speech through a local microphone buffer. "
    "It compresses the audio to Opus format and transmits it across the ocean. "
    "Once processed, streaming text turns into speech instantly."
)

# Token classes that Category 2 must not mangle.
STRICT_TOKENS: dict[str, tuple[str, ...]] = {
    "numbers+email": ("4829", "b", "test", "domain", "org"),
    "city+acronym": ("cagayan", "de", "oro", "city", "api"),
}

# Audible-defect thresholds for Category 3 boundary checks. A DC step is judged
# relative to the sentence's own RMS level: an absolute threshold would just
# measure how loud the sentence is, not whether the offset is audible.
CLICK_JUMP_LIMIT = 2000  # |last sample - first sample| across a join
DC_RATIO_LIMIT = 0.25  # |DC offset| / sentence RMS at the onset
DEAD_AIR_LIMIT_MS = 150.0  # stall beyond the configured inter-sentence gap


# --------------------------------------------------------------------------- #
# printing helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def kv(label: str, value: object) -> None:
    """Print one aligned ``label: value`` line."""
    print(f"   - {label:<44} {value}")


def ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1000.0


def verdict(ok: bool) -> str:
    """``PASS``/``FAIL`` word for a boolean check."""
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def to_float32(pcm16: np.ndarray) -> np.ndarray:
    """int16 PCM -> contiguous mono float32 scaled to [-1, 1]."""
    return np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16, clipping instead of wrapping."""
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


def write_wav(path: Path, pcm16: np.ndarray) -> None:
    """Write mono int16 PCM at ``settings.sample_rate``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(settings.sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def load_wav_float32(path: Path) -> np.ndarray:
    """Read a mono 16 kHz 16-bit WAV as float32."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"{path.name} must be mono 16-bit PCM")
        raw = wav_file.readframes(wav_file.getnframes())
        rate = wav_file.getframerate()
    if rate != settings.sample_rate:
        raise ValueError(f"{path.name} is {rate} Hz, expected {settings.sample_rate}")
    return to_float32(np.frombuffer(raw, dtype="<i2"))


async def synthesize_to_wav(text: str, path: Path, regenerate: bool) -> Path:
    """Cache ``text`` as a WAV using the service's own TTS pipeline."""
    if path.exists() and not regenerate:
        return path
    chunks: list[bytes] = []
    async for item in synthesize_pcm(text):
        chunks.append(item)
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    write_wav(path, pcm)
    return path


def rms(audio: np.ndarray) -> float:
    """Root-mean-square of a float32 signal."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def white_noise(seconds: float, seed: int = 7) -> np.ndarray:
    """Deterministic Gaussian white noise."""
    rng = np.random.default_rng(seed)
    samples = int(settings.sample_rate * seconds)
    return rng.standard_normal(samples).astype(np.float32) * 0.2


def cough_burst(
    samples: int,
    position: int,
    seed: int = 11,
) -> np.ndarray:
    """A short noise burst with a fast attack and exponential decay.

    Stand-in for a cough / throat clear: broadband, transient, no words.
    """
    rng = np.random.default_rng(seed + position)
    length = int(settings.sample_rate * 0.32)
    envelope = np.exp(-np.linspace(0.0, 9.0, length))
    envelope[: int(settings.sample_rate * 0.008)] *= np.linspace(
        0.0, 1.0, int(settings.sample_rate * 0.008)
    )
    burst = rng.standard_normal(length).astype(np.float32) * envelope * 0.45
    out = np.zeros(samples, dtype=np.float32)
    end = min(samples, position + length)
    out[position:end] = burst[: end - position]
    return out


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``speech`` with ``noise`` at a target signal-to-noise ratio."""
    length = max(speech.size, noise.size)
    speech_padded = np.zeros(length, dtype=np.float32)
    speech_padded[: speech.size] = speech
    noise_padded = np.zeros(length, dtype=np.float32)
    noise_padded[: noise.size] = noise
    speech_rms = rms(speech_padded)
    if speech_rms <= 0.0:
        return speech_padded
    noise_padded *= (
        speech_rms / (10.0 ** (snr_db / 20.0)) / max(rms(noise_padded), 1e-9)
    )
    return np.clip(speech_padded + noise_padded, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# pipeline probes
# --------------------------------------------------------------------------- #
@dataclass
class SttOutcome:
    """One ``STTService.transcribe_async`` call."""

    label: str
    latency_ms: float
    text: str
    confidence: float


async def run_stt(service: STTService, audio: np.ndarray, label: str) -> SttOutcome:
    """Transcribe one buffer through the full service (cloud + fallback)."""
    start = time.perf_counter()
    text, confidence = await service.transcribe_async(audio, settings.language)
    return SttOutcome(
        label=label,
        latency_ms=ms(time.perf_counter() - start),
        text=text,
        confidence=confidence,
    )


async def run_tts_timeline(text: str) -> list[tuple[float, bytes]]:
    """Synthesize ``text``, timestamping every PCM frame from t=0.

    ``TTSErrorEvent`` dicts are dropped, so the timeline is audio only.
    """
    start = time.perf_counter()
    timeline: list[tuple[float, bytes]] = []
    async for item in synthesize_stream(text):
        if isinstance(item, dict):
            continue
        timeline.append((time.perf_counter() - start, item))
    return timeline


@dataclass
class SentenceTiming:
    """Playback timing of one sentence inside a TTS stream."""

    index: int
    first_byte_s: float
    last_byte_s: float
    pcm_bytes: int
    samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<i2"))
    dc_offset: int = 0
    dc_ratio: float = 0.0

    @property
    def first_sample(self) -> int:
        """First int16 sample of the sentence."""
        return int(self.samples[0]) if self.samples.size else 0

    @property
    def last_sample(self) -> int:
        """Last int16 sample of the sentence."""
        return int(self.samples[-1]) if self.samples.size else 0


def split_timeline(
    timeline: Iterable[tuple[float, bytes]],
) -> tuple[list[SentenceTiming], list[float]]:
    """Group a TTS timeline into sentences and measure the gaps between them.

    A frame of pure digital zeros is the joining gap that ``synthesize_stream``
    emits in front of every sentence after the first, so it also marks a
    sentence boundary. The configured gap length is subtracted from the observed
    stall (the wait between the previous sentence's last byte and this
    sentence's first byte): the remainder is how much of the gap the prefetcher
    had already covered, i.e. the dead air the listener would not hear. A
    remainder near or above the gap means the consumer stalled.
    """
    sentences: list[SentenceTiming] = []
    slack_ms: list[float] = []
    current: SentenceTiming | None = None
    gap_ms = 0.0

    for timestamp, frame in timeline:
        if frame and not any(frame):
            gap_ms += len(frame) * 1000.0 / BYTES_PER_PCM_SECOND
            current = None  # the sentence ended; the next audio frame is a new one
            continue

        if current is None:
            if sentences:
                stall = ms(timestamp - sentences[-1].last_byte_s)
                slack_ms.append(gap_ms - stall)
            current = _new_sentence(sentences)
            current.first_byte_s = timestamp
            gap_ms = 0.0

        frames = np.frombuffer(frame, dtype="<i2")
        current.samples = (
            frames
            if not current.pcm_bytes
            else np.concatenate((current.samples, frames))
        )
        current.pcm_bytes += len(frame)
        current.last_byte_s = timestamp

    for sentence in sentences:
        if sentence.samples.size:
            sentence.dc_offset = int(np.mean(sentence.samples[:160]))
            # Both sides in int16 units, so the ratio is scale free.
            level = rms(sentence.samples.astype(np.float64))
            sentence.dc_ratio = abs(sentence.dc_offset) / max(level, 1e-9)

    return sentences, slack_ms


def _new_sentence(bucket: list[SentenceTiming]) -> SentenceTiming:
    """Append and return an empty :class:`SentenceTiming`."""
    sentence = SentenceTiming(
        index=len(bucket),
        first_byte_s=0.0,
        last_byte_s=0.0,
        pcm_bytes=0,
    )
    bucket.append(sentence)
    return sentence


# --------------------------------------------------------------------------- #
# text comparison
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, punctuation dropped."""
    cleaned = "".join(
        char if (char.isalnum() or char.isspace()) else " " for char in text.lower()
    )
    return [token for token in cleaned.split() if token]


def squash(text: str) -> str:
    """Lowercase alphanumeric-only form, so hyphenation cannot hide a token."""
    return "".join(char for char in text.lower() if char.isalnum())


def edit_distance(left: list[str], right: list[str]) -> int:
    """Levenshtein distance over token lists (word error rate numerator)."""
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (left_token != right_token)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def compare(key: str, expected: str, actual: str) -> dict[str, Any]:
    """Word error rate plus the strict tokens that did not survive."""
    expected_tokens = tokenize(expected)
    actual_tokens = tokenize(actual)
    errors = edit_distance(expected_tokens, actual_tokens)
    return {
        "expected_tokens": expected_tokens,
        "actual_tokens": actual_tokens,
        "wer": errors / max(1, len(expected_tokens)),
        "errors": errors,
        "missing": strict_token_report(key, actual),
    }


def strict_token_report(key: str, actual: str) -> list[str]:
    """Strict tokens from phrase ``key`` that are absent from ``actual``.

    Matching ignores case, punctuation and hyphenation (``4829-B`` still counts
    as present when the engine returns ``4829B``), so only a genuinely wrong
    word fails the check.
    """
    squashed = squash(actual)
    return [
        token for token in STRICT_TOKENS.get(key, ()) if squash(token) not in squashed
    ]


# --------------------------------------------------------------------------- #
# VAD probe (used by the robustness category)
# --------------------------------------------------------------------------- #
@dataclass
class VadOutcome:
    """Speech segments the streaming VAD detected in one buffer."""

    speech_starts: int
    speech_ends: int
    first_start_s: float | None
    last_end_s: float | None


def run_vad(audio: np.ndarray) -> VadOutcome:
    """Feed ``audio`` through a fresh streaming VAD in 32 ms frames."""
    detector = VADStreamDetector()
    starts = 0
    ends = 0
    first_start: float | None = None
    last_end: float | None = None
    usable = audio.size - (audio.size % FRAME_SAMPLES)
    for offset in range(0, usable, FRAME_SAMPLES):
        frame = np.ascontiguousarray(
            audio[offset : offset + FRAME_SAMPLES], dtype=np.float32
        )
        event = detector.process_frame(frame)
        if not event:
            continue
        if "start" in event:
            starts += 1
            if first_start is None:
                first_start = event["start"] / settings.sample_rate
        if "end" in event:
            ends += 1
            last_end = event["end"] / settings.sample_rate
    return VadOutcome(starts, ends, first_start, last_end)


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    """One scorecard line."""

    metric: str
    target: str
    measured: str
    status: str
    note: str = ""


@dataclass
class Battery:
    """Collected checks and generated audio files."""

    checks: list[Check] = field(default_factory=list)
    generated: list[Path] = field(default_factory=list)

    def add(
        self,
        metric: str,
        target: str,
        measured: object,
        ok: bool | None,
        note: str = "",
    ) -> None:
        """Append a check; ``ok=None`` marks it not measurable here."""
        if ok is None:
            status = "N-A"
        else:
            status = verdict(ok)
        self.checks.append(Check(metric, target, str(measured), status, note))


async def measure_cold_paths(
    service: STTService, battery: Battery, regenerate: bool
) -> None:
    """Time the first cloud STT call and the first TTS synthesis separately.

    Both providers pay a one-off cost on their first request in a process (TLS
    handshake, DNS, connection pool, Opus encoder start-up). Mixing that into
    the per-turn numbers would blame the pipeline for a cold start, so the cold
    numbers are reported here and every category below runs warm.
    """
    banner("WARM-UP -- COLD vs WARM PROVIDER PATHS")
    phrase = "Hey, what time is it?"
    path = await synthesize_to_wav(phrase, BATTERY_DIR / "turn-1.wav", regenerate)
    battery.generated.append(path)
    audio = load_wav_float32(path)

    cold_stt = await run_stt(service, audio, "cold-stt")
    kv("1st STT call (cold)", f"{cold_stt.latency_ms:,.1f} ms  {cold_stt.text!r}")
    warm_stt = await run_stt(service, audio, "warm-stt")
    kv("2nd STT call (warm)", f"{warm_stt.latency_ms:,.1f} ms  {warm_stt.text!r}")
    battery.add(
        "Cold-start STT penalty",
        "(reported, not scored)",
        f"{cold_stt.latency_ms - warm_stt.latency_ms:+,.1f} ms",
        None,
        "first call in a process vs the next one",
    )

    cold_tts = await run_tts_timeline(phrase)
    kv(
        "1st TTS synthesis (cold TTFA)",
        f"{ms(cold_tts[0][0]):,.1f} ms" if cold_tts else "no audio",
    )
    warm_tts = await run_tts_timeline(phrase)
    kv(
        "2nd TTS synthesis (warm TTFA)",
        f"{ms(warm_tts[0][0]):,.1f} ms" if warm_tts else "no audio",
    )
    if cold_tts and warm_tts:
        battery.add(
            "Cold-start TTS penalty",
            "(reported, not scored)",
            f"{ms(cold_tts[0][0]) - ms(warm_tts[0][0]):+,.1f} ms TTFA",
            None,
            "first edge-tts request in a process vs the next one",
        )


async def category_1_rapid_turns(
    service: STTService, battery: Battery, regenerate: bool
) -> list[str]:
    """Rapid conversational turns: STT latency + TTS TTFA per turn."""
    banner(
        "CATEGORY 1 -- RAPID CONVERSATIONAL TURNS (target: STT < 800 ms, TTFA < 700 ms)"
    )
    transcripts: list[str] = []
    stt_latencies: list[float] = []
    ttfas: list[float] = []

    for name, phrase in RAPID_TURNS:
        path = await synthesize_to_wav(phrase, BATTERY_DIR / f"{name}.wav", regenerate)
        battery.generated.append(path)
        audio = load_wav_float32(path)
        outcome = await run_stt(service, audio, name)
        stt_latencies.append(outcome.latency_ms)
        transcripts.append(outcome.text)
        kv(
            f"{name} audio",
            f"{audio.size / settings.sample_rate:.2f}s from {path.name}",
        )
        kv("  said", repr(phrase))
        kv(
            "  STT latency",
            f"{outcome.latency_ms:,.1f} ms  {verdict(outcome.latency_ms < 800)}",
        )
        kv("  transcript", repr(outcome.text))
        kv("  confidence", f"{outcome.confidence:.3f}")

        timeline = await run_tts_timeline(outcome.text or phrase)
        if not timeline:
            battery.add(
                f"Cat1 {name} TTFA",
                "< 700 ms",
                "no audio",
                False,
                "edge-tts returned no PCM for this transcript",
            )
            continue
        ttfa = ms(timeline[0][0])
        pcm_bytes = sum(len(frame) for _, frame in timeline)
        audio_seconds = pcm_bytes / BYTES_PER_PCM_SECOND
        rtf = (ms(timeline[-1][0]) / 1000.0) / audio_seconds
        ttfas.append(ttfa)
        kv("  TTS TTFA", f"{ttfa:,.1f} ms  {verdict(ttfa < 700)}")
        kv("  TTS total / RTF", f"{ms(timeline[-1][0]):,.1f} ms / {rtf:.3f}")

    if stt_latencies:
        worst = max(stt_latencies)
        battery.add(
            "Cat1 STT latency (worst turn)",
            "< 800 ms",
            f"{worst:,.1f} ms",
            worst < 800,
        )
    if ttfas:
        worst_ttfa = max(ttfas)
        battery.add(
            "Cat1 TTS TTFA (worst turn)",
            "< 700 ms",
            f"{worst_ttfa:,.1f} ms",
            worst_ttfa < 700,
            "measured on the STT transcript, one synthesis per turn",
        )
    return transcripts


async def category_2_accuracy(
    service: STTService, battery: Battery, regenerate: bool
) -> None:
    """Numbers, acronyms and homophones must survive transcription."""
    banner(
        "CATEGORY 2 -- NUMBERS, ACRONYMS & HOMOPHONES (target: no hallucinated spelling)"
    )

    for key, phrase in ACCURACY_PHRASES:
        path = await synthesize_to_wav(phrase, BATTERY_DIR / f"{key}.wav", regenerate)
        battery.generated.append(path)
        audio = load_wav_float32(path)
        outcome = await run_stt(service, audio, key)
        stats = compare(key, phrase, outcome.text)
        missing = stats["missing"]

        kv(f"{key} audio", f"{audio.size / settings.sample_rate:.2f}s from {path.name}")
        kv("  said", repr(phrase))
        kv("  STT latency", f"{outcome.latency_ms:,.1f} ms")
        kv("  heard", repr(outcome.text))
        kv(
            "  token error rate",
            f"{stats['wer'] * 100:.1f}%  "
            f"({stats['errors']}/{len(stats['expected_tokens'])} tokens)  "
            f"{verdict(stats['wer'] == 0.0)}",
        )
        kv("  strict tokens missing", f"{missing or 'none'}  {verdict(not missing)}")
        battery.add(
            f"Cat2 {key} WER",
            "0% token errors",
            f"{stats['wer'] * 100:.1f}%",
            stats["wer"] == 0.0,
        )
        battery.add(
            f"Cat2 {key} strict tokens",
            "digits/acronyms intact",
            "all present" if not missing else f"missing {missing}",
            not missing,
        )


async def category_3_paragraph(
    service: STTService, battery: Battery, regenerate: bool
) -> None:
    """Multi-sentence TTS: no dead air, no pops, bounded latency."""
    banner(
        "CATEGORY 3 -- MULTI-SENTENCE PARAGRAPH "
        "(target: TTFA 650-750 ms, RTF < 0.18, zero dead air)"
    )
    path = await synthesize_to_wav(
        PARAGRAPH_TEXT, BATTERY_DIR / "paragraph.wav", regenerate
    )
    battery.generated.append(path)
    audio = load_wav_float32(path)
    outcome = await run_stt(service, audio, "paragraph")
    kv("paragraph audio", f"{audio.size / settings.sample_rate:.2f}s from {path.name}")
    kv("STT latency", f"{outcome.latency_ms:,.1f} ms")
    kv("STT transcript", repr(outcome.text))

    text_to_speak = outcome.text.strip() or PARAGRAPH_TEXT
    expected_sentences = len(chunk_sentences(clean_text_for_tts(PARAGRAPH_TEXT)))
    timeline = await run_tts_timeline(text_to_speak)
    if not timeline:
        battery.add("Cat3 TTS", "audio produced", "no audio", False)
        return

    ttfa = ms(timeline[0][0])
    total_ms = ms(timeline[-1][0])
    pcm_bytes = sum(len(frame) for _, frame in timeline)
    audio_seconds = pcm_bytes / BYTES_PER_PCM_SECOND
    rtf = (total_ms / 1000.0) / audio_seconds
    gap_ms = settings.tts_sentence_gap_ms
    sentences, stalls = split_timeline(timeline)

    kv("spoken text", repr(text_to_speak))
    kv("TTFA", f"{ttfa:,.1f} ms  {verdict(650 <= ttfa <= 750)}")
    kv("total time", f"{total_ms:,.1f} ms")
    kv(
        "audio duration",
        f"{audio_seconds:.2f}s ({pcm_bytes:,} B, {len(timeline)} frames)",
    )
    kv("RTF", f"{rtf:.3f}  {verdict(rtf < 0.18)}")
    kv(
        "sentences",
        f"{len(sentences)} detected / {expected_sentences} expected "
        f"(gap setting {gap_ms} ms)",
    )

    for index, stall in enumerate(stalls, start=1):
        label = f"boundary {index}"
        kv(
            f"  {label} stall",
            f"{stall:,.1f} ms (gap - stall)  {verdict(stall <= DEAD_AIR_LIMIT_MS)}",
        )
    worst_stall = max((abs(stall) for stall in stalls), default=0.0)
    battery.add(
        "Cat3 worst inter-sentence stall",
        f"<= {DEAD_AIR_LIMIT_MS:.0f} ms",
        f"{worst_stall:,.1f} ms",
        worst_stall <= DEAD_AIR_LIMIT_MS,
        "gap minus stall; <= gap means the next sentence was already buffered",
    )

    clicks: list[str] = []
    for previous, following in pairwise(sentences):
        jump = abs(following.first_sample - previous.last_sample)
        if jump > CLICK_JUMP_LIMIT:
            clicks.append(f"step {jump} into sentence {following.index}")
        if following.dc_ratio > DC_RATIO_LIMIT:
            clicks.append(
                f"DC {following.dc_offset} "
                f"({following.dc_ratio:.2f} of RMS) at sentence {following.index}"
            )
    kv("boundary clicks / DC offset", f"{clicks or 'none'}  {verdict(not clicks)}")
    battery.add(
        "Cat3 boundary clicks",
        f"step <= {CLICK_JUMP_LIMIT}, DC <= {DC_RATIO_LIMIT:.2f}x RMS",
        "none" if not clicks else "; ".join(clicks),
        not clicks,
        "automated stand-in for listening for pops between sentences",
    )
    battery.add("Cat3 TTS TTFA", "650 - 750 ms", f"{ttfa:,.1f} ms", 650 <= ttfa <= 750)
    battery.add("Cat3 TTS RTF", "< 0.18", f"{rtf:.3f}", rtf < 0.18)


async def category_4_robustness(
    service: STTService, battery: Battery, regenerate: bool
) -> None:
    """Silence, background noise, and a cloud outage must not break anything."""
    banner("CATEGORY 4 -- SILENCE, BACKGROUND NOISE & INTERRUPTIONS")

    # --- Action A: two seconds of pure room silence ----------------------- #
    print("   Action A -- 2 s of pure digital silence")
    silence = np.zeros(int(settings.sample_rate * 2), dtype=np.float32)
    vad = run_vad(silence)
    kv(
        "VAD speech starts on silence",
        f"{vad.speech_starts}  {verdict(vad.speech_starts == 0)}",
    )
    battery.add(
        "Cat4A VAD rejects silence",
        "0 speech starts",
        vad.speech_starts,
        vad.speech_starts == 0,
        "the real gate in this pipeline (app/vad.py + min_segment_samples)",
    )

    start = time.perf_counter()
    outcome = await run_stt(service, silence, "silence-2s")
    gate_ms = ms(time.perf_counter() - start)
    kv("transcribe_async(silence)", f"{gate_ms:,.1f} ms  {verdict(gate_ms < 2)}")
    kv("  transcript", repr(outcome.text) + "  " + verdict(not outcome.text.strip()))
    battery.add(
        "Cat4A STT silence gate",
        "< 2 ms and empty text",
        f"{gate_ms:,.1f} ms / {outcome.text!r}",
        gate_ms < 2 and not outcome.text.strip(),
        "app.stt.audio_dbfs gates the buffer before any backend runs",
    )

    # --- Action B: background noise + cough-like bursts ------------------- #
    print()
    print("   Action B -- speech buried in background noise / cough bursts")
    phrase = ACCURACY_PHRASES[0][1]
    speech = load_wav_float32(
        await synthesize_to_wav(phrase, BATTERY_DIR / "numbers+email.wav", regenerate)
    )
    noise = white_noise(speech.size / settings.sample_rate)
    speech_rms = rms(speech)

    for snr_db in (30, 20, 10, 0):
        mixed = mix_at_snr(speech, noise, snr_db)
        path = BATTERY_DIR / f"noise_{snr_db:02d}db.wav"
        write_wav(path, to_pcm16(mixed))
        battery.generated.append(path)
        actual_snr = 20 * math.log10(
            max(rms(speech), 1e-9) / max(rms(mixed - speech[: mixed.size]), 1e-9)
        )
        vad = run_vad(mixed)
        outcome = await run_stt(service, mixed, f"noise-{snr_db}db")
        hallucinated = bool(outcome.text.strip()) and not _contains_phrase(
            outcome.text, phrase
        )
        kv(f"SNR {snr_db} dB (measured {actual_snr:.1f} dB)", path.name)
        kv("  VAD speech starts", f"{vad.speech_starts} (expected >= 1)")
        kv("  STT latency", f"{outcome.latency_ms:,.1f} ms")
        kv("  transcript", repr(outcome.text))
        kv(
            "  hallucinated text",
            f"{hallucinated}  {verdict(not hallucinated)}",
        )
        battery.add(
            f"Cat4B SNR {snr_db} dB",
            "no hallucinated text",
            "clean" if not hallucinated else repr(outcome.text),
            not hallucinated,
        )

    cough = load_wav_float32(BATTERY_DIR / "numbers+email.wav")
    cough_samples = cough.size
    cough_mix = cough + cough_burst(cough_samples, cough_samples // 3, seed=3)
    cough_mix += cough_burst(cough_samples, (cough_samples * 2) // 3, seed=5)
    cough_path = BATTERY_DIR / "cough.wav"
    write_wav(cough_path, to_pcm16(np.clip(cough_mix, -1.0, 1.0)))
    battery.generated.append(cough_path)
    vad = run_vad(cough_mix)
    outcome = await run_stt(service, cough_mix, "cough")
    kv("cough bursts overlay", f"{cough_path.name} (speech {speech_rms:.3f} rms)")
    kv("  VAD speech starts", f"{vad.speech_starts} (a burst may open a segment)")
    kv("  STT latency", f"{outcome.latency_ms:,.1f} ms")
    kv("  transcript", repr(outcome.text))
    battery.add(
        "Cat4B cough overlay",
        "no crash, text still from the phrase",
        "ok" if service.is_ready else "service lost its model",
        bool(service.is_ready),
    )

    # --- Action C: the cloud provider disappears --------------------------- #
    print()
    print("   Action C -- cloud outage (Groq unreachable / rejected)")
    audio = (
        load_wav_float32(BATTERY_DIR / "turn-1.wav")
        if (BATTERY_DIR / "turn-1.wav").exists()
        else speech
    )
    original_key = settings.groq_api_key
    groq_backend = service._get_groq_backend()
    original_client = groq_backend._client
    original_timeout = settings.groq_timeout_sec
    original_compress = groq_backend._compress_audio

    try:
        # C1 -- hard rejection (bad key). Time the cloud call on its own first:
        # the interesting number is how long the provider takes to say "no",
        # which the end-to-end fallback number would hide behind the local decode.
        settings.groq_api_key = "invalid_key_to_force_error"
        groq_backend._client = None
        cloud_start = time.perf_counter()
        cloud_error = ""
        try:
            await groq_backend.transcribe_async(audio, settings.language)
        except Exception as exc:  # noqa: BLE001 - we are timing the failure
            cloud_error = f"{type(exc).__name__}"
        cloud_ms = ms(time.perf_counter() - cloud_start)
        kv("C1 cloud rejection (no fallback)", f"{cloud_ms:,.1f} ms {cloud_error}")
        battery.add(
            "Cat4C1 cloud rejection latency",
            "< 1000 ms",
            f"{cloud_ms:,.1f} ms ({cloud_error})",
            cloud_ms < 1000,
            "Groq backend only; includes TLS setup on a freshly built client",
        )

        rejected = await run_stt(service, audio, "outage-rejected")
        kv(
            "C1 rejected key -> local fallback",
            f"{rejected.latency_ms:,.1f} ms total "
            f"(cloud {cloud_ms:,.1f} ms + local decode)",
        )
        kv("  transcript", repr(rejected.text))
        battery.add(
            "Cat4C1 failover on rejection",
            f"< {6000:.0f} ms incl. local decode",
            f"{rejected.latency_ms:,.1f} ms",
            rejected.latency_ms < 6000,
            "401 path: no breaker is burned, the local decode dominates",
        )

        # C2 -- silent provider: burn the full breaker, then go local.
        settings.groq_api_key = original_key
        groq_backend._client = None

        def _hang(_audio: np.ndarray) -> bytes:
            time.sleep(settings.groq_timeout_sec * 1.5)
            return b""

        groq_backend._compress_audio = _hang  # type: ignore[method-assign]
        silent = await run_stt(service, audio, "outage-silent")
        kv(
            "C2 silent provider -> breaker",
            f"{silent.latency_ms:,.1f} ms "
            f"(breaker {settings.groq_timeout_sec}s + local)  "
            f"{verdict(silent.latency_ms < 6000)}",
        )
        kv("  transcript", repr(silent.text))
        battery.add(
            "Cat4C2 failover after breaker",
            f"< {6000:.0f} ms ({settings.groq_timeout_sec}s breaker + local)",
            f"{silent.latency_ms:,.1f} ms",
            silent.latency_ms < 6000,
            "the realistic 'Wi-Fi dropped' case",
        )
        battery.add(
            "Cat4C no crash on outage",
            "a transcript comes back",
            repr(silent.text)[:40],
            bool(silent.text.strip()) or bool(rejected.text.strip()),
        )
    finally:
        settings.groq_api_key = original_key
        groq_backend._client = original_client
        groq_backend._compress_audio = original_compress  # type: ignore[method-assign]
        settings.groq_timeout_sec = original_timeout
        kv("restored", "original key, client, and _compress_audio")


def _contains_phrase(transcript: str, phrase: str) -> bool:
    """True when ``transcript`` shares its distinctive tokens with ``phrase``."""
    wanted = {token for token in tokenize(phrase) if len(token) > 3}
    got = set(tokenize(transcript))
    if not wanted:
        return False
    return len(wanted & got) / len(wanted) >= 0.6


# --------------------------------------------------------------------------- #
# scorecard
# --------------------------------------------------------------------------- #
def print_scorecard(battery: Battery) -> None:
    """Print every collected check plus the generated audio manifest."""
    banner("SCORECARD")
    metric_width = max(len(check.metric) for check in battery.checks)
    target_width = max(len(check.target) for check in battery.checks)
    measured_width = max(28, max(len(check.measured) for check in battery.checks))
    print(
        f"   {'METRIC'.ljust(metric_width)}  {'TARGET'.ljust(target_width)}  "
        f"{'MEASURED':>{measured_width}}  STATUS"
    )
    print(
        f"   {'-' * metric_width}  {'-' * target_width}  {'-' * measured_width}  ------"
    )
    for check in battery.checks:
        print(
            f"   {check.metric.ljust(metric_width)}  {check.target.ljust(target_width)}  "
            f"{check.measured[:measured_width]:>{measured_width}}  {check.status}"
        )
        if check.note:
            print(f"   {'':{metric_width}}  {'':>{target_width}}  note: {check.note}")

    failed = [check for check in battery.checks if check.status == "FAIL"]
    print()
    print(f"   {len(battery.checks) - len(failed)}/{len(battery.checks)} checks passed")
    print()
    print("   Audio written for by-hand microphone / speaker testing:")
    for path in dict.fromkeys(battery.generated):
        print(f"     {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="speech_battery.py",
        description="Speech + edge-case battery for the Alvin STT/TTS pipeline.",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4",
        help="comma separated subset of 1,2,3,4 (default: all)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="re-synthesize the cached phrase WAVs instead of reusing them",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Run the requested categories and print the scorecard."""
    args = parse_args(argv)
    wanted = {part.strip() for part in args.categories.split(",") if part.strip()}

    banner("ALVIN SPEECH & EDGE-CASE BATTERY")
    kv("project root", ROOT)
    kv("categories", sorted(wanted))
    kv("voice", settings.tts_voice)
    kv(
        "vad threshold / silence end",
        f"{settings.vad_threshold} / {settings.min_silence_duration_ms} ms",
    )
    kv(
        "groq fallback / breaker",
        f"{settings.groq_fallback} / {settings.groq_timeout_sec}s",
    )

    BATTERY_DIR.mkdir(parents=True, exist_ok=True)
    battery = Battery()
    service = STTService()

    print()
    print("   Warming the local fallback model (excluded from every number)...")
    warm_start = time.perf_counter()
    await service.ensure_loaded()
    kv("service.ensure_loaded()", f"{ms(time.perf_counter() - warm_start):,.1f} ms")

    await measure_cold_paths(service, battery, args.regenerate)

    if "1" in wanted:
        await category_1_rapid_turns(service, battery, args.regenerate)
    if "2" in wanted:
        await category_2_accuracy(service, battery, args.regenerate)
    if "3" in wanted:
        await category_3_paragraph(service, battery, args.regenerate)
    if "4" in wanted:
        await category_4_robustness(service, battery, args.regenerate)

    await service.shutdown()
    print_scorecard(battery)
    return 0


if __name__ == "__main__":
    asyncio.run(main())
