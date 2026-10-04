"""Benchmark and validate Alvin's TTS path (no server required).

Measures time-to-first-audio, total synthesis time, real-time factor, the
cost split between the edge-tts network call and local MP3 decoding, and
validates the emitted 16 kHz PCM.

Usage::

    python scripts\\bench_tts.py
    python scripts\\bench_tts.py --repeat 3 --text "Your custom sentence."

Matrix mode A/Bs the two deterministic TTS knobs (``TTS_TRIM_EDGE_MS`` and
``TTS_DECODE_INTERVAL_BYTES``) with N samples per cell, after one discarded
warm-up request per cell::

    python scripts\\bench_tts.py --matrix-test --samples 20

The TTS input text is fixed for every cell: it is transcribed once from a
pre-rendered speech fixture, so all cells synthesize byte-identical input and
only the engine parameters differ. Note that edge-tts network jitter dominates
the absolute TTFA numbers, so the meaningful comparison is the per-cell median
and spread, not any single sample.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import wave
from array import array
from pathlib import Path

import edge_tts
import miniaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.tts import (
    TTSSynthesisError,
    chunk_sentences,
    clean_text_for_tts,
    synthesize_pcm,
)

# Fixed ingest for matrix mode: a pre-rendered 3.4 s / 16 kHz / 16-bit mono WAV.
FIXTURE = ROOT / "tests" / "data" / "sample_speech.wav"

# Used when the fixture or the STT leg is unavailable, so the matrix still runs.
FALLBACK_TEXT = "Hey, what time is it?"

# (label, trim_edge_ms, decode_interval_bytes)
MATRIX_CELLS: list[tuple[str, int, int]] = [
    ("A baseline", 200, 8192),
    ("B low-decode-interval", 200, 2048),
    ("C zero-edge-trim", 0, 8192),
    ("D aggressive-combined", 0, 2048),
]

SHORT_TEXT = "Sure, setting your timer now."
PARAGRAPH_TEXT = (
    "Alvin here. I found three calendar events for tomorrow. "
    "The first is a design review at ten in the morning. "
    "The second is a one on one with your manager at one. "
    "The third is a dentist appointment at four thirty. "
    "Would you like me to add reminders for any of them?"
)
LONG_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the assistant streams "
    "audio chunk by chunk so that playback can begin before the entire "
    "response has finished rendering on the server"
)


def pcm_stats(pcm: bytes) -> dict:
    """Summarize a 16 kHz mono int16 buffer: level, clipping, silence, DC offset."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])

    if not samples:
        return {"frames": 0}

    peak = max(abs(s) for s in samples)
    mean = sum(samples) / len(samples)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    window = 320  # 20 ms
    silent = sum(
        1
        for i in range(0, len(samples), window)
        if max((abs(s) for s in samples[i : i + window]), default=0) < 100
    )
    windows = max(1, -(-len(samples) // window))

    return {
        "frames": len(samples),
        "seconds": round(len(samples) / 16000, 3),
        "peak_dbfs": round(20 * math_log10(peak / 32768), 1) if peak else None,
        "rms_dbfs": round(20 * math_log10(rms / 32768), 1) if rms else None,
        "clipped_samples": sum(1 for s in samples if abs(s) >= 32767),
        "dc_offset": round(mean, 1),
        "silence_pct": round(100 * silent / windows, 1),
    }


def math_log10(value: float) -> float:
    import math

    return math.log10(value) if value > 0 else -120.0


async def bench_stream(label: str, text: str, repeat: int) -> dict:
    """Time synthesize_stream end to end and validate the emitted PCM."""
    ttfa: list[float] = []
    totals: list[float] = []
    chunk_counts: list[int] = []
    pcm = b""

    for _ in range(repeat):
        started = time.perf_counter()
        first = None
        buf = b""
        count = 0
        async for chunk in synthesize_pcm(text):
            if first is None:
                first = time.perf_counter() - started
            buf += chunk
            count += 1
        elapsed = time.perf_counter() - started

        ttfa.append((first or elapsed) * 1000)
        totals.append(elapsed * 1000)
        chunk_counts.append(count)
        pcm = buf

    stats = pcm_stats(pcm)
    audio_seconds = stats["seconds"] or 0.001
    mean_total = statistics.mean(totals)

    return {
        "label": label,
        "sentences": len(chunk_sentences(clean_text_for_tts(text))),
        "chunks": chunk_counts[-1],
        "ttfa_ms": round(statistics.mean(ttfa), 1),
        "total_ms": round(mean_total, 1),
        "rtf": round(mean_total / 1000 / audio_seconds, 3),
        "audio_s": stats["seconds"],
        **stats,
    }


async def bench_split(text: str) -> dict:
    """Split synthesis time into edge-tts network time vs local MP3 decode time."""
    sentences = chunk_sentences(clean_text_for_tts(text))
    net_ms = 0.0
    decode_ms = 0.0
    mp3_bytes = 0

    for sentence in sentences:
        communicate = edge_tts.Communicate(sentence, settings.tts_voice)
        buffer = bytearray()
        started = time.perf_counter()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.extend(chunk["data"])
        net_ms += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        miniaudio.decode(
            bytes(buffer),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000,
        )
        decode_ms += (time.perf_counter() - started) * 1000
        mp3_bytes += len(buffer)

    return {
        "sentences": len(sentences),
        "mp3_kb": round(mp3_bytes / 1024, 1),
        "network_ms": round(net_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "decode_share_pct": round(100 * decode_ms / (net_ms + decode_ms), 1),
    }


async def bench_concurrency(text: str, jobs: int) -> dict:
    """Check whether independent sentences can be synthesized in parallel."""
    sentences = chunk_sentences(clean_text_for_tts(text)) or [text]

    started = time.perf_counter()
    await asyncio.gather(*(_drain(sentence) for sentence in sentences[:jobs]))
    parallel_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    for sentence in sentences[:jobs]:
        await _drain(sentence)
    sequential_ms = (time.perf_counter() - started) * 1000

    return {
        "jobs": min(jobs, len(sentences)),
        "parallel_ms": round(parallel_ms, 1),
        "sequential_ms": round(sequential_ms, 1),
        "speedup": round(sequential_ms / parallel_ms, 2) if parallel_ms else None,
    }


async def _drain(sentence: str) -> int:
    total = 0
    async for chunk in synthesize_pcm(sentence):
        total += len(chunk)
    return total


# --------------------------------------------------------------------------- #
# matrix mode
# --------------------------------------------------------------------------- #
def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile over ``samples`` (pct in 0-100)."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (pct / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(samples: list[float]) -> dict[str, float]:
    """Mean / median / p90 / p95 / stddev / min / max for one cell."""
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def load_fixture_float32() -> object:
    """Read the fixed ingest fixture as float32 in [-1, 1]."""
    import numpy as np

    with wave.open(str(FIXTURE), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
        rate = wav_file.getframerate()
    if rate != settings.sample_rate:
        raise ValueError(
            f"{FIXTURE.name} is {rate} Hz, expected {settings.sample_rate}"
        )
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def fixed_text() -> tuple[str, str]:
    """Transcribe the fixed fixture once so every matrix cell speaks the same text.

    Returns ``(text, note)``. Falls back to a constant when STT is unavailable
    (no cloud key, no network) so the TTS matrix is never blocked by the STT leg.
    """
    try:
        from app.stt import STTService
    except ImportError as exc:  # pragma: no cover - import guard
        return FALLBACK_TEXT, f"STT unavailable ({exc})"

    if not FIXTURE.exists():
        return FALLBACK_TEXT, f"fixture missing: {FIXTURE}"

    audio = load_fixture_float32()
    started = time.perf_counter()
    try:
        service = STTService()
        text, _confidence = await service.transcribe_async(audio, settings.language)
        stt_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001 - the matrix only needs the text
        return FALLBACK_TEXT, f"STT leg failed ({type(exc).__name__}: {exc})"
    finally:
        try:
            await service.shutdown()
        except Exception:  # noqa: BLE001, S110 - shutdown is best effort
            pass

    text = text.strip() or FALLBACK_TEXT
    return text, (
        f"fixed ingest: {FIXTURE.relative_to(ROOT)} "
        f"({audio.size / settings.sample_rate:.2f}s), STT once in {stt_ms:.0f} ms"
    )


async def measure_cell(text: str, samples: int) -> tuple[list[float], list[int]]:
    """One discarded warm-up request, then ``samples`` timed TTFA measurements."""
    warm_started = time.perf_counter()
    async for _chunk in synthesize_pcm(text):
        pass
    warm_ms = (time.perf_counter() - warm_started) * 1000

    ttfa_ms: list[float] = []
    totals_ms: list[float] = []
    sizes: list[int] = []
    failures = 0
    for _index in range(samples):
        outcome = await time_one(text)
        if outcome is None:
            failures += 1
            continue
        first, total, elapsed = outcome
        ttfa_ms.append(first)
        totals_ms.append(elapsed)
        sizes.append(total)
    print(
        f"   warm-up {warm_ms:,.0f} ms discarded; "
        f"{samples - failures}/{samples} trials ok, "
        f"{sum(totals_ms) / 1000:.1f}s of synthesis"
    )
    return ttfa_ms, sizes


async def time_one(text: str) -> tuple[float, int, float] | None:
    """One timed synthesis: (ttfa_ms, audio_bytes, total_ms), or None on failure.

    edge-tts is a public endpoint, so a trial can fail with a DNS/TCP error or a
    5xx. That is recorded as a failed sample rather than aborting the whole
    matrix -- and the failure count is reported, because a flaky provider is
    itself a latency finding.
    """
    started = time.perf_counter()
    first = None
    total = 0
    try:
        async for chunk in synthesize_pcm(text):
            if first is None:
                first = time.perf_counter() - started
            total += len(chunk)
    except TTSSynthesisError as exc:
        print(f"      ! trial failed: {str(exc)[:90]}")
        return None
    elapsed = time.perf_counter() - started
    return (first or elapsed) * 1000, total, elapsed * 1000


async def measure_interleaved(
    text: str, samples: int
) -> dict[str, tuple[list[float], list[int]]]:
    """Round-robin one sample per cell per round (paired design).

    Running all 20 samples of a cell back to back confounds the engine
    parameters with network drift: a slow minute inflates whichever cell
    happened to run then. Interleaving puts every cell in every time window, so
    a per-round paired difference cancels slow drift.
    """
    for _label, trim, interval in MATRIX_CELLS:
        settings.tts_trim_edge_ms = trim
        settings.tts_decode_interval_bytes = interval
        async for _chunk in synthesize_pcm(text):  # discarded warm-up per cell
            pass
    print(f"   one warm-up per cell discarded; {samples} interleaved rounds")

    collected: dict[str, tuple[list[float], list[int]]] = {
        label: ([], []) for label, _trim, _interval in MATRIX_CELLS
    }
    failures = 0
    for round_index in range(samples):
        for label, trim, interval in MATRIX_CELLS:
            settings.tts_trim_edge_ms = trim
            settings.tts_decode_interval_bytes = interval
            outcome = await time_one(text)
            if outcome is None:
                failures += 1
                continue
            first, total, _elapsed = outcome
            ttfa_ms, sizes = collected[label]
            ttfa_ms.append(first)
            sizes.append(total)
        if (round_index + 1) % 5 == 0:
            print(f"      ... {round_index + 1}/{samples} rounds")
    print(
        f"   {failures} failed trial(s) excluded "
        f"({samples * len(MATRIX_CELLS) - failures} of "
        f"{samples * len(MATRIX_CELLS)} ok)"
    )
    return collected


async def run_matrix(samples: int, interleave: bool = False) -> int:
    """Run the 4-cell A/B matrix and print per-cell TTFA statistics."""
    print("=" * 78)
    print(" MATRIX TEST -- TTS_TRIM_EDGE_MS x TTS_DECODE_INTERVAL_BYTES")
    print("=" * 78)
    print(f"voice        : {settings.tts_voice}")
    print(f"edge-tts     : {edge_tts.__version__}")
    print(f"samples      : {samples} per cell (plus 1 discarded warm-up each)")
    print(f"design       : {'interleaved (paired)' if interleave else 'blocked'}")
    print(f"prefetch/gap : {settings.tts_prefetch} / {settings.tts_sentence_gap_ms} ms")

    text, note = await fixed_text()
    sentences = chunk_sentences(clean_text_for_tts(text))
    print(f"input        : {len(sentences)} sentence(s), {len(text)} chars")
    print(f"               {text!r}")
    print(f"ingest       : {note}")

    results: list[tuple[str, int, int, dict[str, float], list[int], list[float]]] = []
    original = (settings.tts_trim_edge_ms, settings.tts_decode_interval_bytes)
    try:
        if interleave:
            collected = await measure_interleaved(text, samples)
            for label, trim, interval in MATRIX_CELLS:
                ttfa_ms, sizes = collected[label]
                results.append(
                    (label, trim, interval, summarize(ttfa_ms), sizes, ttfa_ms)
                )
        else:
            for label, trim, interval in MATRIX_CELLS:
                settings.tts_trim_edge_ms = trim
                settings.tts_decode_interval_bytes = interval
                print()
                print(f"   [{label}] trim={trim}ms decode_interval={interval}B")
                ttfa_ms, sizes = await measure_cell(text, samples)
                stats = summarize(ttfa_ms)
                results.append((label, trim, interval, stats, sizes, ttfa_ms))
                print(
                    f"      mean {stats['mean']:,.1f}  median {stats['median']:,.1f}  "
                    f"p90 {stats['p90']:,.1f}  p95 {stats['p95']:,.1f}  "
                    f"sd {stats['stdev']:,.1f} ms"
                )
    finally:
        settings.tts_trim_edge_ms, settings.tts_decode_interval_bytes = original

    print()
    print("=" * 78)
    print(" RESULTS -- TTFA (ms), lower is better")
    print("=" * 78)
    header = (
        f"{'cell':<24}{'trim':>6}{'interval':>10}{'n':>4}{'mean':>10}{'median':>10}"
        f"{'p90':>10}{'p95':>10}{'sd':>9}{'audio B':>11}"
    )
    print(header)
    print("-" * len(header))
    for label, trim, interval, stats, sizes, _raw in results:
        audio_b = int(statistics.mean(sizes)) if sizes else 0
        print(
            f"{label:<24}{trim:>5}ms{interval:>10}{stats['n']:>4}"
            f"{stats['mean']:>10,.1f}{stats['median']:>10,.1f}"
            f"{stats['p90']:>10,.1f}{stats['p95']:>10,.1f}"
            f"{stats['stdev']:>9,.1f}{audio_b:>11,}"
        )

    if interleave:
        print()
        print(" PAIRED DELTA vs A (same round, so slow periods cancel)")
        print("   mean +/- 95% CI of the per-round difference; a CI excluding 0 = real")
        base_raw = results[0][5] if results else []
        for label, _trim, _interval, _stats, _sizes, raw in results[1:]:
            pairs = min(len(base_raw), len(raw))
            diffs = [raw[i] - base_raw[i] for i in range(pairs)]
            mean_diff = statistics.mean(diffs)
            sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            sem_diff = sd_diff / (len(diffs) ** 0.5) if diffs else 0.0
            low = mean_diff - 1.96 * sem_diff
            high = mean_diff + 1.96 * sem_diff
            significant = "REAL" if low * high > 0 else "not significant"
            print(
                f"   {label:<24}{mean_diff:>+8,.1f} ms "
                f"[{low:>+7,.1f}, {high:>+7,.1f}]  n={pairs}  -> {significant}"
            )
    else:
        print()
        print(" DELTA vs A (baseline), by median TTFA")
        baseline = results[0][3]["median"] if results else 0.0
        for label, _trim, _interval, stats, _sizes, _raw in results[1:]:
            delta = stats["median"] - baseline
            sem = stats["stdev"] / (stats["n"] ** 0.5)
            verdict = (
                "within noise"
                if abs(delta) < 2 * sem
                else "improvement"
                if delta < 0
                else "regression"
            )
            print(f"   {label:<24}{delta:>+8,.1f} ms   (sem {sem:,.1f} ms -> {verdict})")

    if results:
        trimmed = results[0][4]
        untrimmed = next(
            (sizes for _l, trim, _i, _s, sizes, _raw in results if trim == 0),
            None,
        )
        if untrimmed:
            extra = statistics.mean(untrimmed) - statistics.mean(trimmed)
            print()
            print(
                f" AUDIO COST of trim=0: {extra:+,.0f} bytes "
                f"({100 * extra / statistics.mean(trimmed):+.1f}%) -- the tail hold "
                "that"
            )
            print(" drops end-of-sentence silence does not exist at hold=0, so that")
            print(" silence is encoded and streamed instead of discarded.")

    print()
    print(" NOTE: edge-tts is a public endpoint, so absolute TTFA carries network")
    print(" jitter of several hundred ms. Judge a cell by its median and spread, and")
    print(
        " by the paired delta -- not by any single sample. An apparent change smaller"
    )
    print(" than one standard deviation is not evidence of an engine effect.")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--text", default=None)
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument("--trim-edge-ms", type=int, default=None)
    parser.add_argument("--decode-interval-bytes", type=int, default=None)
    parser.add_argument("--gap-ms", type=int, default=None)
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="TTFA samples per cell in matrix mode (default: 20)",
    )
    parser.add_argument(
        "--matrix-test",
        action="store_true",
        help="run the trim x decode-interval A/B matrix instead of the default cases",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="with --matrix-test, interleave cells round-robin (paired design) so "
        "network drift cancels instead of biasing one cell",
    )
    args = parser.parse_args()

    if args.matrix_test:
        return await run_matrix(max(1, args.samples), args.interleave)

    for attr, value in (
        ("tts_prefetch", args.prefetch),
        ("tts_trim_edge_ms", args.trim_edge_ms),
        ("tts_decode_interval_bytes", args.decode_interval_bytes),
        ("tts_sentence_gap_ms", args.gap_ms),
    ):
        if value is not None:
            setattr(settings, attr, value)

    print(f"voice        : {settings.tts_voice}")
    print(f"edge-tts     : {edge_tts.__version__}")
    print(f"repeat       : {args.repeat}")
    print(
        f"pipeline     : prefetch={settings.tts_prefetch} "
        f"trim_edge={settings.tts_trim_edge_ms}ms "
        f"gap={settings.tts_sentence_gap_ms}ms "
        f"decode_interval={settings.tts_decode_interval_bytes}B\n"
    )

    cases = [
        ("short sentence", SHORT_TEXT),
        ("long sentence", LONG_SENTENCE),
        ("paragraph", args.text or PARAGRAPH_TEXT),
    ]

    print("=" * 78)
    print(
        f"{'case':<16}{'ttfa':>9}{'total':>10}{'rtf':>7}{'audio':>8}{'chunks':>8}{'peak':>8}"
    )
    print("-" * 78)
    rows = []
    for label, text in cases:
        row = await bench_stream(label, text, args.repeat)
        rows.append(row)
        peak = row.get("peak_dbfs")
        print(
            f"{row['label']:<16}{row['ttfa_ms']:>8.0f}ms{row['total_ms']:>9.0f}ms"
            f"{row['rtf']:>7.2f}{row['audio_s']:>7.2f}s{row['chunks']:>8}"
            f"{str(peak) + 'dB':>8}"
        )
    print("=" * 78)

    print("\nPCM validation")
    for row in rows:
        print(
            f"  {row['label']:<16} frames={row['frames']:<8} "
            f"silence={row['silence_pct']}%  clipped={row['clipped_samples']:<5} "
            f"dc={row['dc_offset']:<7} rms={row.get('rms_dbfs')}dBFS"
        )

    split = await bench_split(args.text or PARAGRAPH_TEXT)
    print(
        f"\nCost split ({split['sentences']} sentences, {split['mp3_kb']} kB mp3)\n"
        f"  edge-tts network : {split['network_ms']:.0f} ms\n"
        f"  miniaudio decode : {split['decode_ms']:.1f} ms "
        f"({split['decode_share_pct']}% of total)"
    )

    concurrency = await bench_concurrency(PARAGRAPH_TEXT, 4)
    print(
        f"\nParallelism ({concurrency['jobs']} independent sentences)\n"
        f"  sequential : {concurrency['sequential_ms']:.0f} ms\n"
        f"  parallel   : {concurrency['parallel_ms']:.0f} ms\n"
        f"  speedup    : {concurrency['speedup']}x"
    )


if __name__ == "__main__":
    asyncio.run(main())
