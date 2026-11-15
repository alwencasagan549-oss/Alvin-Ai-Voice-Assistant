"""Shared low-latency TTS playback engine for the manual mic client.

Extracted from tests/manual/stream_mic.py so the barge-in/flush algorithm
has a single canonical implementation (see scripts/speaker_service.py for the
service-side variant).
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

PLAYBACK_BLOCK_SEC = 0.02


class AudioPlayer:
    """Single-owner, thread-safe TTS playback to one ``OutputStream``.

    Exactly one output stream is created for the lifetime of this object and
    it is never recreated per turn.  That is the fix for the PortAudio
    ``-9988 / Stream closed or not active`` failures and device lockouts that
    happen when a second ``OutputStream`` is opened while the previous one is
    still aborting or closing.

    * ``flush()`` (barge-in) only aborts/restarts the shared stream, it never
      closes it, so the next turn reuses the same device handle.
    * Writes are split into ``PLAYBACK_BLOCK_SEC`` blocks and a generation
      counter lets an in-flight write notice a flush and drop the remainder
      instead of continuing to talk over the user.
    * All stream mutations are serialized with ``self._lock``; the blocking
      ``stream.write()`` runs outside the lock so a flush never waits behind a
      full sentence.
    """

    def __init__(
        self,
        samplerate: int,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self._samplerate = samplerate
        self._channels = channels
        self._dtype = dtype
        self._lock = threading.Lock()
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._generation = 0
        self._closed = False
        self._started = False
        self._writing = False
        self._thread = threading.Thread(
            target=self._run, name="alvin-tts-playback", daemon=True
        )

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def is_playing(self) -> bool:
        """True while queued PCM is still being pushed to the device.

        The mic gate uses this so capture stays muted until the speaker has
        actually gone quiet, not merely until the server finished *sending*
        (``tts_end``): the local buffer can lag the network by a few hundred
        milliseconds, and unmuting there would let the assistant hear the tail
        of its own sentence.
        """
        if not self._thread.is_alive():
            return False
        with self._lock:
            return self._writing or not self._queue.empty()

    def submit(self, data: bytes) -> None:
        """Queue PCM bytes for playback (non-blocking, any thread)."""
        if data:
            self._queue.put(data)

    def flush(self) -> None:
        """Drop queued audio and reset the stream so the next write is silent.

        Safe to call from any thread.  The shared stream is aborted (not
        closed), then restarted, so the device handle is preserved.
        """
        with self._lock:
            self._generation += 1
            stream = self._stream
        self._drain()
        if stream is None:
            return
        try:
            stream.abort()
        except Exception as exc:  # noqa: BLE001
            print(f"[playback] abort failed: {exc}", file=sys.stderr, flush=True)
        time.sleep(0.1)
        try:
            stream.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[playback] restart failed: {exc}", file=sys.stderr, flush=True)

    def close(self) -> None:
        """Stop playback and close the device exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            stream = self._stream
        if stream is not None:
            # Unblock a write that is in progress so the worker can exit.
            with contextlib.suppress(Exception):
                stream.abort()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        try:
            stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=self._channels,
                dtype=self._dtype,
                blocksize=0,
                latency="low",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[playback] cannot open output device: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        with self._lock:
            if self._closed:
                with contextlib.suppress(Exception):
                    stream.close()
                return
            self._stream = stream

        try:
            stream.start()
            while True:
                chunk = self._queue.get()
                if chunk is None:
                    break
                self._write(stream, chunk)
        finally:
            with self._lock:
                if self._stream is stream:
                    self._stream = None
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

    def _write(self, stream: sd.OutputStream, data: bytes) -> None:
        samples = np.frombuffer(data, dtype="<i2")
        block = max(1, int(self._samplerate * PLAYBACK_BLOCK_SEC))
        with self._lock:
            self._writing = True
        try:
            self._write_blocks(stream, samples, block)
        finally:
            with self._lock:
                self._writing = False

    def _write_blocks(self, stream: sd.OutputStream, samples, block: int) -> None:
        for offset in range(0, samples.size, block):
            chunk = samples[offset : offset + block]
            with self._lock:
                generation = self._generation
                if self._closed:
                    return
            try:
                stream.write(chunk)
            except Exception as exc:  # noqa: BLE001
                # ``sounddevice.PortAudioError`` is not an ``OSError``, so catch
                # broadly: an abort (barge-in) or a closed device both surface
                # here.  A superseded write is expected; anything else is a real
                # device error worth one restart attempt.
                with self._lock:
                    superseded = self._generation != generation
                if superseded:
                    # Barge-in flushed mid-write; drop the rest of this chunk.
                    return
                print(
                    f"[playback] write failed, restarting: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    stream.abort()
                    stream.start()
                    stream.write(chunk)
                except Exception as retry_exc:  # noqa: BLE001
                    print(
                        f"[playback] write failed after restart: {retry_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
            with self._lock:
                if self._generation != generation:
                    return
