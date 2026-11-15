"""Bounded thread pools for CPU-bound audio work.

The pipeline offloads blocking work off the event loop: Opus encoding
(:mod:`app.stt`), MP3 decode and Piper synthesis (:mod:`app.tts`), and speaker
embeddings (:mod:`app.speaker`). The default ``asyncio.to_thread`` executor is
the loop's shared ``ThreadPoolExecutor`` sized ``min(32, cpu+4)``, which is
unbounded from the application's point of view: a burst of concurrent
connections could saturate it and starve unrelated ``to_thread`` callers,
indirectly stalling the event loop.

A single shared pool would let throughput work starve latency-critical work:
Piper synthesis (the edge-tts fallback) can occupy a worker for hundreds of ms
to seconds, and ``run_in_executor`` cannot cancel a running thread, so repeated
barge-ins during an edge-tts outage could pin every worker and delay Opus
encoding -- which sits on the critical path of every utterance.

So there are two pools, split by criticality:

* :func:`stt_pool` -- small, reserved for the latency-critical Opus encode.
* :func:`media_pool` -- larger, for MP3 decode, Piper synthesis, and speaker
  embeddings (throughput work that can tolerate queueing).

Both are created lazily on first use and torn down by :func:`shutdown_pools`
(called from the app lifespan).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config import settings

# Prefixes for worker thread names, so a stuck worker is identifiable in a dump.
STT_POOL_PREFIX = "alvin-stt"
MEDIA_POOL_PREFIX = "alvin-media"

# Never allow a zero/negative pool: at least one worker must exist.
_MIN_WORKERS = 1

# Upper clamp so a misconfigured AUDIO_WORKER_THREADS cannot spawn an unbounded
# number of OS threads.
_MAX_WORKERS = 32

_stt_pool: ThreadPoolExecutor | None = None
_media_pool: ThreadPoolExecutor | None = None


def _clamp_workers(configured: int) -> int:
    return max(_MIN_WORKERS, min(_MAX_WORKERS, configured))


def stt_max_workers() -> int:
    """Worker count for the latency-critical STT pool.

    Roughly a quarter of the configured total (at least 2), so the Opus
    encode -- which sits on every utterance's critical path -- keeps its own
    headroom even while the media pool is saturated by Piper synthesis or
    speaker embeddings.
    """
    total = _clamp_workers(settings.audio_worker_threads)
    return max(2, min(8, total // 4))


def media_max_workers() -> int:
    """Worker count for the throughput (TTS/speaker) pool."""
    total = _clamp_workers(settings.audio_worker_threads)
    return max(_MIN_WORKERS, total - stt_max_workers())


def get_stt_pool() -> ThreadPoolExecutor:
    """Return the process-wide STT pool, creating it on first use.

    Safe to call from coroutines: the event loop is single-threaded, so the
    check-then-create is serialized without a lock.
    """
    global _stt_pool
    if _stt_pool is None:
        _stt_pool = ThreadPoolExecutor(
            max_workers=stt_max_workers(),
            thread_name_prefix=STT_POOL_PREFIX,
        )
    return _stt_pool


def get_media_pool() -> ThreadPoolExecutor:
    """Return the process-wide media pool, creating it on first use."""
    global _media_pool
    if _media_pool is None:
        _media_pool = ThreadPoolExecutor(
            max_workers=media_max_workers(),
            thread_name_prefix=MEDIA_POOL_PREFIX,
        )
    return _media_pool


async def run_in_stt_pool(fn, *args):
    """Run the latency-critical Opus encode on the dedicated STT pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_stt_pool(), fn, *args)


async def run_in_media_pool(fn, *args):
    """Run TTS/speaker throughput work on the media pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_media_pool(), fn, *args)


def shutdown_pools() -> None:
    """Tear down both pools if they were created. Safe to call repeatedly."""
    global _stt_pool, _media_pool
    for pool in (_stt_pool, _media_pool):
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    _stt_pool = None
    _media_pool = None
