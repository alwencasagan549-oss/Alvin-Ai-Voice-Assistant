"""YouTube search helper for the ``play_youtube`` voice command.

Uses yt-dlp (no API key, actively maintained) for the search itself and caches
results for 5 minutes. A hit gives us a ready-to-play URL; a miss falls back
to the YouTube search-results page so the client still has something useful.

``yt-dlp`` is imported lazily so a fresh install without it still starts the
server; searches then transparently fall through to the fallback URL.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

log = logging.getLogger("alvin.youtube")

_CACHE_TTL = 300.0        # seconds
_CACHE_MAX_ENTRIES = 200
_SEARCH_TIMEOUT = 15.0

_CACHE: dict[str, tuple[float, YouTubeMatch | None]] = {}


@dataclass
class YouTubeMatch:
    video_id: str
    title: str
    url: str


def _normalize_query(query: str) -> str:
    return " ".join((query or "").lower().split())


async def _ytdlp_search(query: str) -> YouTubeMatch | None:
    """Search YouTube for the best match. Returns ``None`` on any failure."""
    try:
        import yt_dlp  # lazy — optional dependency
    except ImportError:
        log.warning(
            "yt-dlp is not installed; play_youtube will only open the search page. "
            "Install with: pip install yt-dlp"
        )
        return None

    def _run() -> YouTubeMatch | None:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,       # search-results page, not full extract
            "skip_download": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "default_search": "ytsearch1",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)

        # Failure-mode visibility: log what came back when it is unusable.
        if not info:
            log.warning("yt-dlp returned no info for %r", query)
            return None
        if not info.get("entries"):
            log.warning(
                "yt-dlp returned empty entries for %r (type=%r)",
                query, info.get("_type"),
            )
            return None

        entry = info["entries"][0]

        # Primary path: search results carry an explicit `id`.
        vid = entry.get("id")

        # Fallback: extract the video ID from the URL when `id` is absent
        # (some yt-dlp versions ship search entries without it).
        if not vid:
            url = entry.get("url") or ""
            m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
            if m:
                vid = m.group(1)

        if not vid:
            log.warning(
                "yt-dlp entry has no usable video id: %r",
                {k: entry.get(k) for k in ("id", "url", "title", "_type")},
            )
            return None

        return YouTubeMatch(
            video_id=vid,
            title=entry.get("title", "") or "",
            url=f"https://www.youtube.com/watch?v={vid}",
        )

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run), timeout=_SEARCH_TIMEOUT
        )
    except TimeoutError:
        log.warning("YouTube search timed out after %.1fs for %r", _SEARCH_TIMEOUT, query)
        return None
    except Exception as exc:  # noqa: BLE001 — search is best-effort
        log.warning("YouTube search failed for %r: %s", query, exc)
        return None


def _evict_if_needed(now: float) -> None:
    """TTL first, then hard cap by oldest timestamp."""
    cutoff = now - _CACHE_TTL
    for key in [k for k, (ts, _) in _CACHE.items() if ts <= cutoff]:
        _CACHE.pop(key, None)
    if len(_CACHE) > _CACHE_MAX_ENTRIES:
        overflow = len(_CACHE) - _CACHE_MAX_ENTRIES
        oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:overflow]
        for key, _ in oldest:
            _CACHE.pop(key, None)


async def search_youtube(query: str) -> YouTubeMatch | None:
    """Search YouTube. Returns the best match or ``None``. Never raises."""
    key = _normalize_query(query)
    if not key:
        return None

    hit = _CACHE.get(key)
    now = time.time()
    if hit and now - hit[0] < _CACHE_TTL:
        log.debug("youtube cache hit for %r", key)
        return hit[1]

    match = await _ytdlp_search(query)
    # Only cache positive results. A transient failure (network blip,
    # yt-dlp rate limit) should not lock the query out for 5 minutes.
    if match is not None:
        _CACHE[key] = (now, match)
        _evict_if_needed(now)
    return match


def fallback_search_url(query: str) -> str:
    """Search-results URL used when the video ID could not be resolved."""
    return f"https://www.youtube.com/results?search_query={quote(query, safe='')}"


# --------------------------------------------------------------------------- #
# Ad-free playback: resolve a direct googlevideo stream URL.
#
# YouTube's website injects ads into its player; the direct media URL yt-dlp
# extracts has none. A local player (VLC/mpv/ffplay) can open it, so the
# assistant can start a song with no ad break at all. Direct URLs are valid
# for a few hours, hence the cache TTL. Everything is best-effort: any failure
# returns ``None`` and the caller falls back to the normal YouTube page.
# --------------------------------------------------------------------------- #
_STREAM_CACHE: dict[tuple[str, bool], tuple[float, str]] = {}
_STREAM_CACHE_MAX = 100
_STREAM_TTL = 4 * 3600.0   # googlevideo URLs expire after roughly six hours
_STREAM_TIMEOUT = 25.0


def _resolve_stream_sync(video_id: str, audio_only: bool) -> str | None:
    """Blocking yt-dlp extraction of the direct media URL."""
    try:
        import yt_dlp  # lazy — optional dependency
    except ImportError:
        log.warning("yt-dlp is not installed; ad-free playback is unavailable")
        return None

    fmt = (
        "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"
        if audio_only
        else "best[ext=mp4]/best"
    )
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "format": fmt,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}", download=False
        )
    if not info:
        return None

    # Single-format selections expose ``url``; merged video+audio selections
    # expose the parts in ``requested_formats``.
    url = info.get("url")
    if not url:
        for part in info.get("requested_formats") or []:
            if part.get("url") and (
                not audio_only or part.get("acodec") not in (None, "none")
            ):
                url = part["url"]
                break
    return url or None


async def resolve_stream_url(video_id: str, audio_only: bool = True) -> str | None:
    """Return a direct, ad-free media URL for ``video_id``. Never raises."""
    if not video_id:
        return None

    key = (video_id, bool(audio_only))
    now = time.time()
    hit = _STREAM_CACHE.get(key)
    if hit and now - hit[0] < _STREAM_TTL:
        log.debug("stream cache hit for %s", video_id)
        return hit[1]

    try:
        url = await asyncio.wait_for(
            asyncio.to_thread(_resolve_stream_sync, video_id, audio_only),
            timeout=_STREAM_TIMEOUT,
        )
    except TimeoutError:
        log.warning("stream resolution timed out for %s", video_id)
        return None
    except Exception as exc:  # noqa: BLE001 — playback is best-effort
        log.warning("stream resolution failed for %s: %s", video_id, exc)
        return None

    if url:
        if len(_STREAM_CACHE) >= _STREAM_CACHE_MAX:
            oldest = min(_STREAM_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _STREAM_CACHE.pop(oldest, None)
        _STREAM_CACHE[key] = (now, url)
    return url
