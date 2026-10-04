"""End-to-end latency benchmark for the Alvin STT/TTS pipeline.

Measured here (real audio, real network calls wherever the environment allows):

* **STT model warm-up** -- ``STTService.ensure_loaded()`` pre-warms the local
  ``settings.fallback_model`` (``small.en``, int8). Timed on its own so it never
  contaminates any other number.
* **Silence path** -- 1.5 s of digital-zero audio through
  ``STTService.transcribe_async``.  Since the silence gate was added this is a
  microsecond short circuit that never reaches a backend (see
  ``app.stt.audio_dbfs``).  The **local-only** Whisper decode of the same
  buffer is timed separately, because that is the fallback's worst case.
* **Cloud-path prerequisite** -- the Opus payload is encoded in memory by
  ``soundfile`` (libsndfile ships inside the wheel), so no ``ffmpeg`` binary on
  PATH is required.  ``ffmpeg`` is reported for information only: its absence
  no longer blocks the cloud path.
* **Primary cloud path** -- the real Groq call with the real key.  When
  ``settings.groq_api_key`` is empty the cloud latency is reported as NOT
  MEASURED with the reason rather than being faked from a local-fallback
  number.
* **Forced failover** -- an invalid key plus a dropped cached ``AsyncGroq``
  client (so ``GroqBackend._get_client`` rebuilds it with the bad key), timing the
  whole ``transcribe_async`` call and identifying which ``except`` branch of
  ``STTService.transcribe_async`` was taken from the captured log records.
  A second probe silently delays the cloud attempt past the breaker
  (``_compress_audio`` is replaced) so the ``asyncio.TimeoutError`` branch is
  exercised too.
* **TTS** -- ``app.tts.synthesize_stream`` TTFA, total wall time, PCM bytes,
  playback duration and RTF for three sentences.  ``TTSErrorEvent`` **dicts** are
  collected separately instead of being counted as audio bytes.

Run it from anywhere (the project root is pushed onto ``sys.path`` here)::

    .\\.venv\\Scripts\\python.exe tests\\benchmark_pipeline.py
    .\\.venv\\Scripts\\python.exe tests\\benchmark_pipeline.py --skip-stt
    .\\.venv\\Scripts\\python.exe tests\\benchmark_pipeline.py --skip-stt --repeat 3

pytest does not collect this file: the name does not match ``test_*.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import shutil
import statistics
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported after the sys.path bootstrap above; ruff does not flag these as E402
# in this project, so no noqa is needed.
from app.config import settings
from app.stt import STTService
from app.stt import audio_dbfs as app_audio_dbfs
from app.tts import synthesize_stream

FIXTURE = ROOT / "tests" / "data" / "sample_speech.wav"
STT_LOGGER = "alvin.stt"
BYTES_PER_PCM_SECOND = settings.sample_rate * 2  # 16 kHz, 16-bit, mono -> 32000 B/s
REFERENCE_ITERATIONS = 200
INVALID_KEY = "invalid_key_to_force_error"

# The three benchmark sentences, verbatim from the original script.
TEST_SENTENCES: list[tuple[str, str]] = [
    ("Short", "Hello Alvin, how are you doing today?"),
    (
        "Medium",
        "The primary bottleneck in low latency voice pipelines is network upload time, not inference speed.",
    ),
    (
        "Paragraph",
        "Building a real-time voice assistant requires tight latency control. First, we compress incoming audio to Opus. Next, we run Groq STT with local fallbacks. Finally, streaming TTS decodes audio chunks on the fly to eliminate dead air.",
    ),
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def kv(label: str, value: object) -> None:
    """Print one aligned ``label: value`` line."""
    print(f"   - {label:<46} {value}")


def ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1000.0


def per_call_ms(fn: Callable[[], Any], iterations: int) -> list[float]:
    """Return the millisecond duration of each of ``iterations`` calls to ``fn``."""
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append(ms(time.perf_counter() - start))
    return timings


def to_float32(pcm16: np.ndarray) -> np.ndarray:
    """int16 PCM -> contiguous mono float32 scaled to [-1, 1].

    ``STTService.transcribe_async`` expects this shape, not raw int16 bytes.
    """
    return np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0, dtype=np.float32)


def rms_dbfs(audio: np.ndarray) -> float:
    """Full-scale dBFS of a float32 signal (``-inf`` for digital silence)."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def describe_audio(audio: np.ndarray) -> str:
    """Human-readable size/duration string for a float32 signal."""
    seconds = audio.size / settings.sample_rate
    return (
        f"{audio.size} float32 samples @ {settings.sample_rate} Hz "
        f"= {seconds:.2f}s (ndim={audio.ndim})"
    )


# --------------------------------------------------------------------------- #
# audio generators (the original script's helpers, kept verbatim in spirit)
# --------------------------------------------------------------------------- #
def generate_dummy_pcm(
    duration_sec: float,
    sample_rate: int = 16000,
    tone_freq: float = 440.0,
) -> np.ndarray:
    """Synthetic 440 Hz sine PCM, amplitude 16384 (original helper, as int16)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    audio_data = (np.sin(2 * np.pi * tone_freq * t) * 16384).astype(np.int16)
    return audio_data


def generate_silence_pcm(
    duration_sec: float = 1.0, sample_rate: int = 16000
) -> np.ndarray:
    """Pure digital silence PCM (original helper, as int16)."""
    return np.zeros(int(sample_rate * duration_sec), dtype=np.int16)


def load_speech_fixture() -> np.ndarray:
    """Load ``tests/data/sample_speech.wav`` as float32 in [-1, 1]."""
    with wave.open(str(FIXTURE), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    return to_float32(np.frombuffer(raw, dtype=np.int16))


# --------------------------------------------------------------------------- #
# log capture (used to prove which except branch the failover took)
# --------------------------------------------------------------------------- #
class _LogCapture(logging.Handler):
    """Collect records emitted by a logger while a sub-benchmark runs."""

    def __init__(self, logger_name: str = STT_LOGGER) -> None:
        super().__init__(level=logging.DEBUG)
        self._logger = logging.getLogger(logger_name)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def __enter__(self) -> Self:
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._logger.removeHandler(self)

    def find(self, needle: str) -> logging.LogRecord | None:
        """First record whose formatted message contains ``needle``."""
        for record in self.records:
            if needle in record.getMessage():
                return record
        return None

    def exception_detail(self) -> str:
        """``TypeName: message`` of the first logged exception, or ``""``."""
        for record in self.records:
            if record.exc_info is not None:
                exc = record.exc_info[1]
                if exc is not None:
                    return f"{type(exc).__name__}: {exc}"
        return ""


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass
class Preflight:
    """Environment facts that decide what is measurable at all."""

    python: str
    cpu_count: int | None
    ffmpeg: str | None
    key_present: bool
    key_length: int
    groq_model: str
    fallback_model: str
    groq_fallback: bool
    groq_timeout_sec: float
    min_segment_samples: int
    min_segment_seconds: float


@dataclass
class SttSample:
    """One timed ``transcribe_async`` style measurement."""

    label: str
    audio_source: str
    latency_ms: float
    text: str
    confidence: float
    backend_model: str


@dataclass
class CloudProbe:
    """One direct ``GroqBackend.transcribe_async`` attempt."""

    label: str
    audio_source: str
    measured: bool
    latency_ms: float | None
    model: str
    text: str
    error_type: str
    error_message: str
    logged_exception: str


@dataclass
class FailoverProbe:
    """One forced-failure ``STTService.transcribe_async`` measurement."""

    audio_source: str
    total_ms: float | None
    path: str
    groq_error_type: str
    groq_error_message: str
    text: str
    confidence: float
    backend_model: str


@dataclass
class SttReport:
    """Everything section 1 produced, for the scorecard."""

    warmup_ms: float | None = None
    is_ready: bool | None = None
    silence_service_ms: float | None = None
    silence_local_ms: float | None = None
    energy_ref_ms: float | None = None
    min_segment_ref_ms: float | None = None
    cloud_latency_ms: float | None = None
    cloud_reason: str = ""
    failover_total_ms: float | None = None
    failover_path: str = ""
    failover_note: str = ""
    local_ms: float | None = None
    fixture_text: str = ""
    fixture_seconds: float = 0.0
    samples: list[SttSample] = field(default_factory=list)


@dataclass
class TtsSample:
    """One ``synthesize_stream`` run."""

    ttfa_ms: float | None
    total_ms: float
    pcm_bytes: int
    frames: int
    max_frame_bytes: int
    errors: list[dict[str, Any]]


@dataclass
class TtsAggregate:
    """Min / median / max of a TTS metric across ``--repeat`` runs."""

    label: str
    text: str
    runs: int
    ttfa_min: float | None
    ttfa_median: float | None
    ttfa_max: float | None
    total_min: float
    total_median: float
    total_max: float
    pcm_bytes: int
    audio_seconds: float
    frames: int
    max_frame_bytes: int
    rtf_min: float | None
    rtf_median: float | None
    rtf_max: float | None
    errors: list[dict[str, Any]]

    @property
    def score_ttfa(self) -> float | None:
        """TTFA used by the scorecard (median, or the single run when repeat=1)."""
        return self.ttfa_median

    @property
    def score_rtf(self) -> float | None:
        """RTF used by the scorecard (median, or the single run when repeat=1)."""
        return self.rtf_median


@dataclass
class ScoreRow:
    """One scorecard line."""

    metric: str
    target: str
    measured: str
    status: str
    note: str = ""


# --------------------------------------------------------------------------- #
# Section 1 -- STT
# --------------------------------------------------------------------------- #
def collect_preflight() -> Preflight:
    """Snapshot the environment facts that gate what can be measured."""
    return Preflight(
        python=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        ffmpeg=shutil.which("ffmpeg"),
        key_present=bool(settings.groq_api_key),
        key_length=len(settings.groq_api_key),
        groq_model=settings.groq_model,
        fallback_model=settings.fallback_model,
        groq_fallback=settings.groq_fallback,
        groq_timeout_sec=settings.groq_timeout_sec,
        min_segment_samples=settings.min_segment_samples,
        min_segment_seconds=settings.min_segment_seconds,
    )


def print_preflight(info: Preflight) -> None:
    """Print the preflight block, including the cloud blockers."""
    print(f"   - {'python':<46} {info.python}")
    print(f"   - {'logical CPUs':<46} {info.cpu_count}")
    print(
        f"   - {'sample_rate':<46} {settings.sample_rate} Hz (PCM = {BYTES_PER_PCM_SECOND} B/s)"
    )
    print(f"   - {'groq_model (cloud)':<46} {info.groq_model}")
    print(f"   - {'fallback_model (local)':<46} {info.fallback_model}")
    print(f"   - {'groq_fallback':<46} {info.groq_fallback}")
    print(
        f"   - {'groq breaker':<46} floor {info.groq_timeout_sec} s"
        f" + {settings.groq_timeout_base_sec} s +"
        f" {settings.groq_timeout_slope_sec} s per audio second"
    )
    print(f"   - {'silence_threshold_dbfs':<46} {settings.silence_threshold_dbfs} dBFS")
    print(f"   - {'min_segment_samples':<46} {info.min_segment_samples}")
    print()
    print("   Cloud reachability preflight:")
    if info.key_present:
        print(
            f"     GROQ_API_KEY      : present in settings.groq_api_key "
            f"({info.key_length} chars, value never printed)"
        )
    else:
        print(
            '     GROQ_API_KEY      : ABSENT -- settings.groq_api_key == "" '
            "(length 0, value never printed)"
        )
        print(
            "                         => the real cloud call CANNOT be measured here."
            " Primary STT latency is reported as NOT MEASURED."
        )
    if info.ffmpeg:
        print(f"     ffmpeg on PATH     : YES ({info.ffmpeg}) -- informational only")
    else:
        print("     ffmpeg on PATH     : NO -- informational only. The Opus payload")
        print("                         is encoded in memory by soundfile/libsndfile,")
        print("                         so the cloud path no longer needs the binary.")


async def measure_cloud(
    service: STTService,
    audio: np.ndarray,
    language: str | None,
    label: str,
    audio_source: str,
) -> CloudProbe:
    """Attempt the real Groq call once and report exactly what happened.

    Bypasses ``STTService.transcribe_async`` on purpose so the raw Groq outcome
    (and its own latency) is visible instead of being hidden by the fallback.
    """
    probe = CloudProbe(
        label=label,
        audio_source=audio_source,
        measured=False,
        latency_ms=None,
        model=settings.groq_model,
        text="",
        error_type="",
        error_message="",
        logged_exception="",
    )
    backend = service._get_groq_backend()  # private cache: no public accessor exists
    start = time.perf_counter()
    with _LogCapture() as cap:
        try:
            result = await asyncio.wait_for(
                backend.transcribe_async(audio, language),
                timeout=settings.groq_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            probe.latency_ms = ms(time.perf_counter() - start)
            probe.error_type = type(exc).__name__
            probe.error_message = (
                f"asyncio.TimeoutError after {probe.latency_ms:.1f} ms "
                f"(circuit breaker = {settings.groq_timeout_sec} s)"
            )
        except Exception as exc:  # noqa: BLE001 - we are classifying, not handling
            probe.latency_ms = ms(time.perf_counter() - start)
            probe.error_type = type(exc).__name__
            probe.error_message = str(exc).replace("\n", " ")[:400]
        else:
            probe.measured = True
            probe.latency_ms = ms(time.perf_counter() - start)
            probe.model = result.model
            probe.text = result.text
    probe.logged_exception = cap.exception_detail()
    return probe


async def measure_local(
    service: STTService,
    audio: np.ndarray,
    language: str | None,
    label: str,
    audio_source: str,
) -> SttSample:
    """Time ``LocalWhisperBackend.transcribe_async`` on its own.

    Uses the private ``_get_local_backend()`` because ``STTService`` exposes no
    public handle for it. This is the honest "fallback STT (local)" number: a
    real local decode, not a failed-cloud timing.
    """
    backend = service._get_local_backend()
    start = time.perf_counter()
    result = await backend.transcribe_async(audio, language)
    return SttSample(
        label=label,
        audio_source=audio_source,
        latency_ms=ms(time.perf_counter() - start),
        text=result.text,
        confidence=result.confidence,
        backend_model=result.model,
    )


async def measure_failover(
    service: STTService,
    audio: np.ndarray,
    language: str | None,
    audio_source: str,
) -> FailoverProbe:
    """Force the cloud path to fail and time the whole failover.

    ``GroqBackend`` does not carry the API key -- it reads ``settings.groq_api_key``
    inside ``_get_client()`` and caches the ``AsyncGroq`` client on ``self._client``.
    So forcing failure means setting the key AND dropping that cached client, or
    the already-built client would keep working. The original key is restored in
    ``finally``.
    """
    probe = FailoverProbe(
        audio_source=audio_source,
        total_ms=None,
        path="unknown",
        groq_error_type="",
        groq_error_message="",
        text="",
        confidence=0.0,
        backend_model="",
    )
    backend = service._get_groq_backend()
    original_key = settings.groq_api_key
    original_client = backend._client
    try:
        settings.groq_api_key = INVALID_KEY
        backend._client = None  # force AsyncGroq to be rebuilt with the bad key
        start = time.perf_counter()
        with _LogCapture() as cap:
            text, confidence = await service.transcribe_async(audio, language)
        probe.total_ms = ms(time.perf_counter() - start)
        probe.text = text
        probe.confidence = confidence
        probe.logged_exception = cap.exception_detail()
        if cap.find("Groq STT timeout") is not None:
            probe.path = (
                f"asyncio.TimeoutError branch -- 'Groq STT timeout "
                f"({settings.groq_timeout_sec}s)' logged, so the full "
                f"{settings.groq_timeout_sec} s circuit breaker elapsed"
            )
        elif cap.find("Groq STT error") is not None:
            probe.path = (
                "bare `except Exception` branch -- 'Groq STT error' logged, so "
                "Groq raised before the timeout elapsed (FAILS FAST)"
            )
        else:
            probe.path = "unclassified: neither failover log line was emitted"
        probe.groq_error_message = probe.logged_exception
        probe.groq_error_type = (
            probe.logged_exception.split(":", 1)[0] if probe.logged_exception else ""
        )
    except Exception as exc:  # noqa: BLE001 - keep the report complete
        probe.groq_error_type = type(exc).__name__
        probe.groq_error_message = str(exc).replace("\n", " ")[:400]
        probe.path = "transcribe_async itself raised (no local transcript)"
    finally:
        settings.groq_api_key = original_key
        backend._client = original_client
    return probe


async def benchmark_stt(repeat: int = 1) -> SttReport:
    """Run section 1 and return everything the scorecard needs."""
    report = SttReport()
    banner("SECTION 1 -- STT (warm-up, silence, cloud, forced failover)")
    info = collect_preflight()
    print_preflight(info)

    silence_pcm = generate_silence_pcm(1.5)
    silence_audio = to_float32(silence_pcm)
    tone_pcm = generate_dummy_pcm(2.5)
    tone_audio = to_float32(tone_pcm)
    language = settings.language

    print()
    print("   Audio under test:")
    kv("silence (digital zero, 1.5s)", describe_audio(silence_audio))
    kv("tone (440 Hz sine @16384, 2.5s)", describe_audio(tone_audio))
    if FIXTURE.exists():
        speech_audio = load_speech_fixture()
        kv("speech fixture (real WAV)", describe_audio(speech_audio))
        kv("  -> file", str(FIXTURE.relative_to(ROOT)))
        report.fixture_seconds = speech_audio.size / settings.sample_rate
    else:
        speech_audio = tone_audio
        kv("speech fixture MISSING", f"{FIXTURE} not found, tone used instead")

    service = STTService()

    # --- model warm-up, timed on its own --------------------------------- #
    print()
    print("   [1] Model warm-up (excluded from every other number)")
    try:
        start = time.perf_counter()
        await service.ensure_loaded()
        report.warmup_ms = ms(time.perf_counter() - start)
        kv("await service.ensure_loaded()", f"{report.warmup_ms:,.1f} ms")
        kv("preload_fallback", settings.preload_fallback)
        report.is_ready = service.is_ready  # property, not a method
        kv("service.is_ready (property)", report.is_ready)
    except Exception as exc:  # noqa: BLE001
        print(f"   ! warm-up FAILED: {type(exc).__name__}: {exc}")
        report.warmup_ms = None

    # --- silence path ---------------------------------------------------- #
    print()
    print("   [2] Silence path -- 1.5s of digital-zero audio")
    print("       app/stt.py gates on RMS level (app.stt.audio_dbfs), so the service")
    print("       call below short-circuits before any backend. The local-only decode")
    print("       is timed separately as the fallback's worst case.")
    try:
        sample = await measure_local(
            service,
            silence_audio,
            language,
            "silence (local only)",
            "1.5s digital zero",
        )
        report.silence_local_ms = sample.latency_ms
        report.samples.append(sample)
        kv("local Whisper on pure silence", f"{sample.latency_ms:,.1f} ms")
        kv("  transcript", repr(sample.text))
        kv("  confidence", f"{sample.confidence:.3f}")
        kv("  backend model", sample.backend_model)
    except Exception as exc:  # noqa: BLE001
        print(f"   ! local silence probe FAILED: {type(exc).__name__}: {exc}")

    try:
        start = time.perf_counter()
        text, confidence = await service.transcribe_async(silence_audio, language)
        report.silence_service_ms = ms(time.perf_counter() - start)
        kv(
            "full service.transcribe_async(silence)",
            f"{report.silence_service_ms:,.3f} ms",
        )
        kv("  transcript / confidence", f"{text!r} / {confidence:.3f}")
        kv(
            "  gate cost",
            f"{report.silence_service_ms * 1000:,.1f} us "
            f"(threshold {settings.silence_threshold_dbfs} dBFS)",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   ! service silence probe FAILED: {type(exc).__name__}: {exc}")

    # --- gate cost breakdown --------------------------------------------- #
    print()
    print("   [3] Where the silence budget goes")

    def energy_check() -> float:
        return app_audio_dbfs(silence_audio)

    try:
        timings = per_call_ms(energy_check, REFERENCE_ITERATIONS)
        report.energy_ref_ms = statistics.median(timings)
        kv(
            f"(a) app.stt.audio_dbfs over 1.5s x {REFERENCE_ITERATIONS}",
            f"median {report.energy_ref_ms * 1000:.2f} us",
        )
        kv("    min / max", f"{min(timings) * 1000:.2f} / {max(timings) * 1000:.2f} us")
        kv("    dBFS of digital zero", f"{rms_dbfs(silence_audio)}")
        kv("    => this IS the gate in app/stt.py", "one RMS pass, no backend call")
    except Exception as exc:  # noqa: BLE001
        print(f"   ! energy reference FAILED: {type(exc).__name__}: {exc}")

    def min_segment_check() -> bool:
        return silence_audio.size < settings.min_segment_samples

    try:
        timings = per_call_ms(min_segment_check, REFERENCE_ITERATIONS)
        report.min_segment_ref_ms = statistics.median(timings)
        kv(
            f"(b) settings.min_segment_samples check x {REFERENCE_ITERATIONS}",
            f"median {report.min_segment_ref_ms * 1000:.3f} us",
        )
        kv("    min / max", f"{min(timings) * 1000:.3f} / {max(timings) * 1000:.3f} us")
        kv(
            "    threshold",
            f"{settings.min_segment_samples} samples "
            f"({settings.min_segment_seconds}s x {settings.sample_rate} Hz)",
        )
        kv(
            "    result on 24000-sample silence",
            f"{min_segment_check()} (False = not dropped)",
        )
        kv(
            "    => the upstream VAD gate in app/connection.py",
            "runs before the STT service",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   ! min-segment reference FAILED: {type(exc).__name__}: {exc}")

    # --- primary cloud path --------------------------------------------- #
    print()
    print("   [4] Primary cloud path (real Groq call, real key)")
    if not info.key_present:
        print(
            "       settings.groq_api_key is empty -> the real call cannot be measured."
        )
    probes: list[CloudProbe] = []
    cloud_repeats = max(1, repeat)
    for index in range(cloud_repeats):
        probes.append(
            await measure_cloud(
                service,
                speech_audio,
                language,
                "primary Groq (speech fixture)",
                f"real speech WAV ({report.fixture_seconds:.2f}s)",
            )
        )
        if cloud_repeats > 1:
            probe = probes[-1]
            state = (
                f"{probe.latency_ms:,.1f} ms"
                if probe.measured and probe.latency_ms is not None
                else "failed"
            )
            print(f"       run {index + 1}: {state}")

    probe = probes[0]
    measured = [p.latency_ms for p in probes if p.measured and p.latency_ms is not None]
    if measured:
        report.cloud_latency_ms = statistics.median(measured)
        report.cloud_reason = (
            f"measured via the real Groq API as {probes[0].model} "
            f"({len(measured)}/{cloud_repeats} runs ok)"
        )
        kv("Groq latency (median)", f"{report.cloud_latency_ms:,.1f} ms")
        if len(measured) > 1:
            kv(
                "  min / max",
                f"{min(measured):,.1f} / {max(measured):,.1f} ms",
            )
        kv("model", probes[0].model)
        kv("transcript", repr(probes[0].text))
    else:
        probe = probes[0]
        reasons: list[str] = []
        if not info.key_present:
            reasons.append("settings.groq_api_key == '' (GROQ_API_KEY not set in .env)")
        if not reasons:
            reasons.append(
                f"the Groq call raised {probe.error_type or 'an error'}: "
                f"{probe.error_message or 'no message captured'}"
            )
        report.cloud_reason = "cloud latency NOT MEASURED because " + "; ".join(reasons)
        print("       CLOUD LATENCY: NOT MEASURED. Reasons:")
        for reason in reasons:
            print(f"         * {reason}")
        kv("exception type (from the attempt)", probe.error_type or "n/a")
        kv("exception message", probe.error_message or "n/a")
        kv(
            "exception from the app log (log.exception)",
            probe.logged_exception or "n/a",
        )
        kv("attempt wall time (not a cloud latency)", f"{probe.latency_ms:,.1f} ms")
        print("       NOTE: the attempt wall time above is NOT a Groq number and is")
        print("             deliberately NOT used in the scorecard.")
        print("       No local-fallback number is being passed off as a cloud number.")

    # --- forced failover ------------------------------------------------- #
    print()
    print("   [5] Forced failover -- invalid key + dropped cached AsyncGroq client")
    failover = await measure_failover(
        service,
        speech_audio,
        language,
        f"real speech WAV ({report.fixture_seconds:.2f}s)",
    )
    report.failover_total_ms = failover.total_ms
    report.failover_path = failover.path
    if failover.total_ms is not None:
        kv("total transcribe_async wall time", f"{failover.total_ms:,.1f} ms")
        kv(
            "  transcript / confidence",
            f"{failover.text!r} / {failover.confidence:.3f}",
        )
    kv("exception path taken", failover.path)
    kv("Groq exception type", failover.groq_error_type or "n/a")
    kv("Groq exception message", (failover.groq_error_message or "n/a")[:300])

    # --- direct local fallback measurement ------------------------------ #
    print()
    print("   [6] Direct local-fallback measurement (LocalWhisperBackend only)")
    for label, audio, source in (
        (
            "local fallback (speech fixture)",
            speech_audio,
            f"real speech WAV ({report.fixture_seconds:.2f}s)",
        ),
        (
            "local fallback (tone)",
            tone_audio,
            "440 Hz sine @16384, 2.5s",
        ),
    ):
        try:
            sample = await measure_local(service, audio, language, label, source)
            report.samples.append(sample)
            if label.endswith("(speech fixture)"):
                report.local_ms = sample.latency_ms
                report.fixture_text = sample.text
            kv(f"{label} [{source}]", f"{sample.latency_ms:,.1f} ms")
            kv(
                "  transcript / confidence",
                f"{sample.text!r} / {sample.confidence:.3f}",
            )
            kv("  backend model", sample.backend_model)
        except Exception as exc:  # noqa: BLE001
            print(f"   ! {label} FAILED: {type(exc).__name__}: {exc}")

    if report.fixture_text:
        kv(
            "speech fixture transcript (verbatim)",
            f"{report.fixture_text!r} ({report.fixture_seconds:.2f}s of audio)",
        )

    if failover.total_ms is not None:
        if report.failover_path.startswith("asyncio.TimeoutError"):
            report.failover_note = (
                f"invalid key consumed the FULL {settings.groq_timeout_sec}s timeout"
            )
        else:
            report.failover_note = "failed FAST: no timeout was consumed"
        print()
        print(f"   Verdict: {report.failover_note}")
        print(
            "   Contrast: a real network partition (packets dropped, no RST) burns the"
        )
        print(
            f"   entire breaker (floor {settings.groq_timeout_sec}s, scaled up for "
            f"longer audio) before the local"
        )
        print(
            "   model starts, whereas an auth rejection returns a 401 without waiting."
        )
        print("   This run shows whichever happened here; a partition cannot be")
        print("   simulated without blocking that long on purpose.")

    try:
        await service.shutdown()
    except Exception as exc:  # noqa: BLE001
        print(f"   ! shutdown warning: {type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------- #
# Section 2 -- TTS
# --------------------------------------------------------------------------- #
async def run_tts_once(text: str) -> TtsSample:
    """One ``synthesize_stream`` pass, filtering ``TTSErrorEvent`` dicts out."""
    start = time.perf_counter()
    first_chunk: float | None = None
    pcm_bytes = 0
    frames = 0
    max_frame = 0
    errors: list[dict[str, Any]] = []
    async for item in synthesize_stream(text):
        if isinstance(item, dict):
            # A TTSErrorEvent is NOT audio: recording its length would fabricate
            # playback duration and RTF.
            errors.append(dict(item))
            continue
        now = time.perf_counter()
        if first_chunk is None:
            first_chunk = now
        pcm_bytes += len(item)
        frames += 1
        max_frame = max(max_frame, len(item))
    return TtsSample(
        ttfa_ms=ms(first_chunk - start) if first_chunk is not None else None,
        total_ms=ms(time.perf_counter() - start),
        pcm_bytes=pcm_bytes,
        frames=frames,
        max_frame_bytes=max_frame,
        errors=errors,
    )


async def benchmark_tts(repeat: int) -> list[TtsAggregate]:
    """Run section 2 for all three sentences, ``repeat`` times each."""
    banner(f"SECTION 2 -- TTS (synthesize_stream, {repeat} run(s) per sentence)")
    kv("voice", settings.tts_voice)
    kv("PCM byte rate", f"{BYTES_PER_PCM_SECOND} B/s ({settings.sample_rate} Hz x 2)")
    kv(
        "prefetch / gap / trim",
        f"{settings.tts_prefetch} / "
        f"{settings.tts_sentence_gap_ms} ms / {settings.tts_trim_edge_ms} ms",
    )

    aggregates: list[TtsAggregate] = []
    for label, text in TEST_SENTENCES:
        print()
        print(
            f"   [{label}] ({len(text)} chars, {len(synthesized_sentences(text))} sentence(s))"
        )
        print(f'      "{text}"')
        samples: list[TtsSample] = []
        for run_index in range(repeat):
            try:
                sample = await run_tts_once(text)
            except Exception as exc:  # noqa: BLE001
                print(f"      run {run_index + 1} FAILED: {type(exc).__name__}: {exc}")
                continue
            samples.append(sample)
            audio_s = sample.pcm_bytes / BYTES_PER_PCM_SECOND
            rtf = (sample.total_ms / 1000.0) / audio_s if audio_s > 0 else None
            ttfa_text = (
                f"{sample.ttfa_ms:,.2f} ms" if sample.ttfa_ms is not None else "n/a"
            )
            rtf_text = f"{rtf:.3f}" if rtf is not None else "n/a"
            print(
                f"      run {run_index + 1}: TTFA {ttfa_text} | total "
                f"{sample.total_ms:,.2f} ms | {sample.pcm_bytes:,} B | "
                f"{audio_s:.2f}s audio | RTF {rtf_text} | "
                f"{sample.frames} frame(s) | errors {len(sample.errors)}"
            )

        if not samples:
            print("      !! no successful run; this sentence is unmeasured")
            continue

        ttfas = [s.ttfa_ms for s in samples if s.ttfa_ms is not None]
        totals = [s.total_ms for s in samples]
        pcm_bytes = max(s.pcm_bytes for s in samples)
        audio_s = pcm_bytes / BYTES_PER_PCM_SECOND
        rtfs = [
            (s.total_ms / 1000.0) / (s.pcm_bytes / BYTES_PER_PCM_SECOND)
            for s in samples
            if s.pcm_bytes > 0
        ]
        all_errors: list[dict[str, Any]] = []
        for sample in samples:
            all_errors.extend(sample.errors)

        aggregate = TtsAggregate(
            label=label,
            text=text,
            runs=len(samples),
            ttfa_min=min(ttfas) if ttfas else None,
            ttfa_median=statistics.median(ttfas) if ttfas else None,
            ttfa_max=max(ttfas) if ttfas else None,
            total_min=min(totals),
            total_median=statistics.median(totals),
            total_max=max(totals),
            pcm_bytes=pcm_bytes,
            audio_seconds=audio_s,
            frames=max(s.frames for s in samples),
            max_frame_bytes=max(s.max_frame_bytes for s in samples),
            rtf_min=min(rtfs) if rtfs else None,
            rtf_median=statistics.median(rtfs) if rtfs else None,
            rtf_max=max(rtfs) if rtfs else None,
            errors=all_errors,
        )
        aggregates.append(aggregate)

        print(f"      -- {label} summary over {aggregate.runs} run(s) --")
        if aggregate.runs == 1:
            kv("TTFA", f"{aggregate.ttfa_min:,.2f} ms" if aggregate.ttfa_min else "n/a")
            kv("total", f"{aggregate.total_min:,.2f} ms")
            kv("RTF", f"{aggregate.rtf_min:.3f}" if aggregate.rtf_min else "n/a")
        else:
            if aggregate.ttfa_min is not None:
                kv(
                    "TTFA min / median / max",
                    f"{aggregate.ttfa_min:,.2f} / {aggregate.ttfa_median:,.2f} / "
                    f"{aggregate.ttfa_max:,.2f} ms",
                )
            kv(
                "total min / median / max",
                f"{aggregate.total_min:,.2f} / {aggregate.total_median:,.2f} / "
                f"{aggregate.total_max:,.2f} ms",
            )
            if aggregate.rtf_min is not None:
                kv(
                    "RTF min / median / max",
                    f"{aggregate.rtf_min:.3f} / {aggregate.rtf_median:.3f} / "
                    f"{aggregate.rtf_max:.3f}",
                )
        kv("PCM bytes (max run)", f"{aggregate.pcm_bytes:,}")
        kv("audio playback duration", f"{aggregate.audio_seconds:.2f} s")
        kv("PCM frames yielded (max run)", aggregate.frames)
        kv("max frame size", f"{aggregate.max_frame_bytes:,} B")
        kv("TTSErrorEvent dicts", len(aggregate.errors))
        for error in aggregate.errors:
            print(
                f"        error sentence_index="
                f"{error.get('sentence_index')} message={error.get('message')!r} "
                f"fatal={error.get('fatal')}"
            )
    return aggregates


def synthesized_sentences(text: str) -> list[str]:
    """Sentence split as ``app.tts`` will see it (for reporting only)."""
    try:
        from app.tts import chunk_sentences, clean_text_for_tts

        return chunk_sentences(clean_text_for_tts(text))
    except Exception:  # noqa: BLE001
        return [text]


# --------------------------------------------------------------------------- #
# Section 3 -- Scorecard
# --------------------------------------------------------------------------- #
def score(
    metric: str,
    target: str,
    value: float | None,
    low: float,
    high: float,
    unit: str,
    reason: str = "",
    decimals: int = 2,
) -> ScoreRow:
    """Build one scorecard row; ``None`` becomes N-A, never a made-up number."""
    if value is None:
        return ScoreRow(metric, target, "N/A", "N-A", reason)
    status = "PASS" if low <= value <= high else "FAIL"
    return ScoreRow(metric, target, f"{value:,.{decimals}f} {unit}".strip(), status)


def print_scorecard(stt: SttReport | None, tts: list[TtsAggregate]) -> None:
    """Print the target scorecard. Anything unmeasured is N-A with a reason."""
    banner("SECTION 3 -- SCORECARD (vs the target benchmark)")

    silence_value = stt.silence_service_ms if stt else None
    silence_reason = "STT section was skipped (--skip-stt)"
    silence_note = ""
    if stt is not None:
        if stt.silence_service_ms is None:
            silence_reason = "silence probe failed to complete"
        else:
            silence_note = (
                "app.stt.audio_dbfs gates the buffer, so nothing is uploaded and no "
                "backend runs; Whisper never gets the chance to answer silence with "
                "a hallucination"
            )

    rows: list[ScoreRow] = [
        score(
            "Silence gate latency",
            "< 2 ms",
            silence_value,
            0.0,
            2.0,
            "ms",
            silence_reason,
        ),
        score(
            "Primary STT (Groq)",
            "450 - 800 ms",
            stt.cloud_latency_ms if stt else None,
            450.0,
            800.0,
            "ms",
            stt.cloud_reason if stt else "STT section was skipped (--skip-stt)",
        ),
        score(
            "Fallback STT (local)",
            "< 2150 ms (1.8s cutoff + ~350ms local)",
            stt.local_ms if stt else None,
            0.0,
            2150.0,
            "ms",
            "STT section was skipped (--skip-stt)"
            if stt is None
            else "local measurement did not complete",
        ),
    ]

    ttfa_values = [a.score_ttfa for a in tts if a.score_ttfa is not None]
    rtf_values = [a.score_rtf for a in tts if a.score_rtf is not None]
    if not tts:
        ttfa_reason = "TTS section was skipped (--skip-tts)"
    elif not ttfa_values:
        ttfa_reason = "no audio bytes were produced, so TTFA is undefined"
    else:
        ttfa_reason = ""
    rows.append(
        score(
            "TTS TTFA",
            "650 - 750 ms",
            max(ttfa_values) if ttfa_values else None,
            650.0,
            750.0,
            "ms",
            ttfa_reason,
        )
    )
    if not tts:
        rtf_reason = "TTS section was skipped (--skip-tts)"
    elif not rtf_values:
        rtf_reason = "no audio bytes were produced, so RTF is undefined"
    else:
        rtf_reason = ""
    rows.append(
        score(
            "TTS RTF",
            "< 0.18",
            max(rtf_values) if rtf_values else None,
            0.0,
            0.18,
            "",
            rtf_reason,
            decimals=3,
        )
    )

    metric_width = max(len(r.metric) for r in rows)
    target_width = max(len(r.target) for r in rows)
    print(
        f"   {'METRIC'.ljust(metric_width)}  {'TARGET'.ljust(target_width)}  "
        f"{'MEASURED':>28}  STATUS"
    )
    print(f"   {'-' * metric_width}  {'-' * target_width}  {'-' * 28}  ------")
    for row in rows:
        print(
            f"   {row.metric.ljust(metric_width)}  {row.target.ljust(target_width)}  "
            f"{row.measured:>28}  {row.status}"
        )
        if row.status == "N-A" and row.note:
            print(f"   {'':{metric_width}}  {'':>{target_width}}  reason: {row.note}")

    print()
    print("   Notes:")
    if silence_note:
        print(f"     * {silence_note}")
        if stt and stt.energy_ref_ms is not None and stt.min_segment_ref_ms is not None:
            print(
                f"       The gate itself costs "
                f"{stt.energy_ref_ms * 1000:.2f} us for 1.5 s of audio; the "
                f"connection-layer min_segment_samples check "
                f"{stt.min_segment_ref_ms * 1000:.3f} us."
            )
    if stt and stt.cloud_reason:
        print(f"     * Primary STT: {stt.cloud_reason}")
    if stt and stt.failover_note:
        print(f"     * Forced failover: {stt.failover_note}")
    if ttfa_values:
        worst = max(ttfa_values)
        best = min(ttfa_values)
        print(
            f"     * TTS TTFA / RTF rows use the WORST sentence of the three "
            f"({worst:,.2f} ms) so a single slow sentence cannot hide; best was "
            f"{best:,.2f} ms."
        )
    print("     * No number above was invented; anything unmeasurable is marked N-A.")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="benchmark_pipeline.py",
        description="Benchmark the Alvin STT + TTS pipeline latencies.",
    )
    parser.add_argument("--skip-stt", action="store_true", help="skip section 1 (STT)")
    parser.add_argument("--skip-tts", action="store_true", help="skip section 2 (TTS)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per cloud STT probe and per TTS sentence; min/median/max are reported (default: 1)",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Run every requested section, always printing a complete report."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="   [log] %(levelname)-8s %(name)s: %(message)s"
    )

    banner("ALVIN STT + TTS PIPELINE BENCHMARK")
    kv("project root", ROOT)
    kv("cwd", Path.cwd())
    kv("argv", sys.argv[1:])
    kv(
        "options",
        f"skip_stt={args.skip_stt} skip_tts={args.skip_tts} repeat={args.repeat}",
    )

    stt_report: SttReport | None = None
    tts_report: list[TtsAggregate] = []

    if not args.skip_stt:
        try:
            stt_report = await benchmark_stt(max(1, args.repeat))
        except Exception as exc:  # noqa: BLE001 - the report must still print
            print(f"   ! SECTION 1 ABORTED: {type(exc).__name__}: {exc}")
    else:
        print("\n   (STT section skipped by --skip-stt)")

    if not args.skip_tts:
        try:
            tts_report = await benchmark_tts(max(1, args.repeat))
        except Exception as exc:  # noqa: BLE001 - the report must still print
            print(f"   ! SECTION 2 ABORTED: {type(exc).__name__}: {exc}")
    else:
        print("\n   (TTS section skipped by --skip-tts)")

    print_scorecard(stt_report, tts_report)

    print()
    print("=" * 78)
    print(" END OF BENCHMARK")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    asyncio.run(main())
