#!/usr/bin/env python3
"""CLI tool for speaker enrollment and management.

Usage:
    python scripts/speaker_enroll.py --name "John" --audio enrollment.wav
    python scripts/speaker_enroll.py --list
    python scripts/speaker_enroll.py --delete "john_abc123"
    python scripts/speaker_enroll.py --identify --audio test.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.speaker import SpeakerService, get_speaker_service, shutdown_speaker_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("speaker_enroll")


def load_audio(path: Path) -> np.ndarray:
    """Load audio file and convert to mono float32 at 16kHz."""
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != settings.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)

    # Normalize to float32 [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

    return audio


async def enroll_speaker(args: argparse.Namespace) -> int:
    """Enroll a new speaker or add to existing profile."""
    audio = load_audio(args.audio)
    duration = audio.size / settings.sample_rate

    if duration < 2.0:
        log.error("Audio too short: %.1fs (minimum 2s)", duration)
        return 1
    if duration > 10.0:
        log.warning("Audio longer than 10s, truncating")
        audio = audio[:int(10.0 * settings.sample_rate)]

    try:
        service = await get_speaker_service()
        profile = await service.enroll(audio, args.name, speaker_id=args.speaker_id)
        log.info("Enrolled: %s (ID: %s, samples: %d, duration: %.1fs)",
                 profile.name, profile.speaker_id, profile.embedding_count, profile.enrollment_duration_sec)
        return 0
    except Exception as exc:
        log.exception("Enrollment failed: %s", exc)
        return 1
    finally:
        await shutdown_speaker_service()


async def list_speakers(args: argparse.Namespace) -> int:
    """List all enrolled speakers."""
    try:
        service = await get_speaker_service()
        profiles = service.list_profiles()

        if not profiles:
            log.info("No speakers enrolled")
            return 0

        print(f"{'Speaker ID':<30} {'Name':<20} {'Samples':<8} {'Duration':<10}")
        print("-" * 70)
        for p in profiles:
            print(f"{p.speaker_id:<30} {p.name:<20} {p.embedding_count:<8} {p.enrollment_duration_sec:.1f}s")
        return 0
    except Exception as exc:
        log.exception("Failed to list speakers: %s", exc)
        return 1
    finally:
        await shutdown_speaker_service()


async def delete_speaker(args: argparse.Namespace) -> int:
    """Delete a speaker profile."""
    try:
        service = await get_speaker_service()
        result = await service.delete_profile(args.speaker_id)
        if result:
            log.info("Deleted speaker: %s", args.speaker_id)
        else:
            log.error("Speaker not found: %s", args.speaker_id)
            return 1
        return 0
    except Exception as exc:
        log.exception("Delete failed: %s", exc)
        return 1
    finally:
        await shutdown_speaker_service()


async def identify_speaker(args: argparse.Namespace) -> int:
    """Identify speaker from audio file."""
    audio = load_audio(args.audio)

    try:
        service = await get_speaker_service()
        result = await service.identify(audio)

        if result.is_known:
            log.info("Identified: %s (ID: %s, similarity: %.3f, latency: %.1fms)",
                     result.name, result.speaker_id, result.similarity, result.latency_ms)
        else:
            log.info("Unknown speaker (best match: %s, similarity: %.3f, latency: %.1fms)",
                     result.name or "N/A", result.similarity, result.latency_ms)
        return 0
    except Exception as exc:
        log.exception("Identification failed: %s", exc)
        return 1
    finally:
        await shutdown_speaker_service()


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker enrollment and management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("--name", required=True, help="Speaker name")
    enroll_parser.add_argument("--audio", required=True, type=Path, help="Audio file path")
    enroll_parser.add_argument("--speaker-id", help="Existing speaker ID to add to")
    enroll_parser.set_defaults(func=enroll_speaker)

    # List
    list_parser = subparsers.add_parser("list", help="List enrolled speakers")
    list_parser.set_defaults(func=list_speakers)

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete a speaker")
    delete_parser.add_argument("speaker_id", help="Speaker ID to delete")
    delete_parser.set_defaults(func=delete_speaker)

    # Identify
    identify_parser = subparsers.add_parser("identify", help="Identify speaker from audio")
    identify_parser.add_argument("--audio", required=True, type=Path, help="Audio file path")
    identify_parser.set_defaults(func=identify_speaker)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())