"""Web search for the Alvin voice assistant.

Tavily (primary) + DuckDuckGo (fallback), 5-minute in-memory cache, formatted
for LLM tool calls.

Fixes vs. the previous version:

* The Tavily synthesized ``answer`` is now included in the output. It was paid
  for (``include_answer=True``) and then thrown away, so the LLM only saw raw
  snippets and often answered worse than Tavily's own summary.
* ``TavilyClient`` and ``DDGS`` are cached across calls. Recreating them per
  search paid a TLS handshake inside an 8s budget, which is why searches
  "randomly" timed out.
* ``topic`` is normalized once before the API call. ``"News"`` used to fail the
  ``topic == "news"`` check and reach Tavily as an invalid enum.
* Empty Tavily results (``[]``) now fall through to DDG. Only ``None`` (an
  exception) triggered the fallback before, so a successful-but-empty search
  was cached as "No results found." for 5 minutes.
* ``.get(key, "")`` is replaced with ``.get(key) or ""`` so explicit JSON
  ``null`` values do not crash ``_format_results``.
* The DDG import tries both package names (``ddgs`` and ``duckduckgo_search``).
* News results include the published date when Tavily provides one.
* Cache is bounded by both TTL and a hard entry cap.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .config import settings

log = logging.getLogger("alvin.web_search")

_CACHE_TTL = 300.0        # seconds
_CACHE_MAX_ENTRIES = 500  # hard cap
_SEARCH_TIMEOUT = 8.0     # per-provider budget, seconds
_DEFAULT_MAX_RESULTS = 3
_SNIPPET_LEN = 400


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    domain: str
    published: str | None = None


# ── client caches ─────────────────────────────────────────────────────────
# Rebuilding these per call was the root cause of intermittent timeouts:
# each fresh TavilyClient opens a new httpx pool and pays a TLS handshake
# inside an 8 s budget.
_tavily_client = None
_tavily_key_used: str | None = None
_tavily_import_failed = False

_ddg_cls = None
_ddg_import_failed = False


def _get_tavily_client():
    global _tavily_client, _tavily_key_used, _tavily_import_failed
    if _tavily_import_failed:
        return None
    api_key = (settings.tavily_api_key or "").strip()
    if not api_key:
        return None
    if _tavily_client is not None and _tavily_key_used == api_key:
        return _tavily_client
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        _tavily_import_failed = True  # log once, not on every search
        log.error("tavily-python is not installed: %s", exc)
        return None
    _tavily_client = TavilyClient(api_key=api_key)
    _tavily_key_used = api_key
    return _tavily_client


def _get_ddg_cls():
    global _ddg_cls, _ddg_import_failed
    if _ddg_cls is not None or _ddg_import_failed:
        return _ddg_cls
    for module_name in ("ddgs", "duckduckgo_search"):
        try:
            module = __import__(module_name, fromlist=["DDGS"])
            _ddg_cls = getattr(module, "DDGS")
            return _ddg_cls
        except ImportError:
            continue
    _ddg_import_failed = True
    log.error(
        "Neither 'ddgs' nor 'duckduckgo-search' is installed; "
        "web search will only work when Tavily succeeds."
    )
    return None


def _extract_domain(url: str | None) -> str:
    if not url:
        return "unknown"
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"


def _normalize_topic(topic: str | None) -> str:
    """Clamp to the only two values Tavily / DDG accept."""
    return "news" if (topic or "").strip().lower() == "news" else "general"


def _format_results(
    results: list[SearchResult],
    answer: str | None = None,
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> str:
    if not results and not answer:
        return "No results found."

    lines: list[str] = []
    # Tavily's synthesized answer is the highest-signal line for the LLM.
    if answer:
        lines.append(f"Summary: {answer}")

    for r in results[:max_results]:
        prefix = f"[{r.domain}]"
        if r.published:
            prefix = f"{prefix} ({r.published})"
        title = r.title or "(untitled)"
        snippet = (r.content or "")[:_SNIPPET_LEN]
        lines.append(f"{prefix} {title}: {snippet}")
    return "\n".join(lines)


async def _tavily_search(
    query: str, topic: str, max_results: int
) -> tuple[list[SearchResult], str | None]:
    """Return ``(results, answer)``. Raises on hard failure."""
    client = _get_tavily_client()
    if client is None:
        return [], None  # not configured — caller falls back to DDG

    topic = _normalize_topic(topic)  # bug #4
    kwargs: dict = {
        "query": query,
        "search_depth": "basic",
        "include_answer": "basic",        # bug #1: now consumed below
        "max_results": max_results,
        "topic": topic,
    }
    if topic == "news":
        kwargs["time_range"] = "week"

    loop = asyncio.get_running_loop()
    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: client.search(**kwargs)),
        timeout=_SEARCH_TIMEOUT,
    )

    results: list[SearchResult] = []
    for item in response.get("results") or []:
        url = item.get("url") or ""          # bug #6
        results.append(SearchResult(
            title=item.get("title") or "",
            url=url,
            content=item.get("content") or "",
            domain=_extract_domain(url),
            published=item.get("published_date") or None,  # bug #9
        ))
    return results, response.get("answer") or None   # bug #1


async def _duckduckgo_search(
    query: str, topic: str, max_results: int
) -> list[SearchResult]:
    """Return DDG results. Raises on hard failure."""
    DDGS = _get_ddg_cls()          # bug #7
    if DDGS is None:
        return []

    topic = _normalize_topic(topic)  # bug #4

    def _search():
        with DDGS() as ddgs:
            raw = ddgs.news(query, max_results=max_results) \
                if topic == "news" \
                else ddgs.text(query, max_results=max_results)
            return list(raw or [])

    loop = asyncio.get_running_loop()
    raw_results = await asyncio.wait_for(
        loop.run_in_executor(None, _search),
        timeout=_SEARCH_TIMEOUT,
    )

    results: list[SearchResult] = []
    for item in raw_results:
        url = item.get("href") or item.get("url") or ""   # bug #6
        results.append(SearchResult(
            title=item.get("title") or "",
            url=url,
            content=item.get("body") or item.get("snippet") or "",
            domain=_extract_domain(url),
            published=item.get("date") or None,           # bug #9
        ))
    return results


# ── cache ─────────────────────────────────────────────────────────────────
_SEARCH_CACHE: dict[tuple[str, str, int], tuple[float, str]] = {}


def _cache_key(query: str, topic: str, max_results: int) -> tuple[str, str, int]:
    return (
        " ".join((query or "").lower().split()),
        _normalize_topic(topic),
        max_results,
    )


def _evict_if_needed(now: float) -> None:
    """TTL first, then hard cap by oldest timestamp (bugs #10, #12)."""
    cutoff = now - _CACHE_TTL
    for key in [k for k, (ts, _) in _SEARCH_CACHE.items() if ts <= cutoff]:
        _SEARCH_CACHE.pop(key, None)

    if len(_SEARCH_CACHE) > _CACHE_MAX_ENTRIES:
        overflow = len(_SEARCH_CACHE) - _CACHE_MAX_ENTRIES
        oldest = sorted(_SEARCH_CACHE.items(), key=lambda kv: kv[1][0])[:overflow]
        for key, _ in oldest:
            _SEARCH_CACHE.pop(key, None)


async def web_search(
    query: str,
    topic: str = "general",
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> str:
    """Search the web and return formatted plain text for the LLM.

    Never raises; on total failure returns a ``SEARCH_FAILED`` marker.
    """
    key = _cache_key(query, topic, max_results)
    now = time.time()

    cached = _SEARCH_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        log.debug("cache hit for %r", key)
        return cached[1]

    results: list[SearchResult] = []
    answer: str | None = None

    try:
        results, answer = await _tavily_search(query, topic, max_results)
    except Exception as exc:  # noqa: BLE001 - fall through to DDG
        log.warning("Tavily failed for %r: %s", query, exc)

    # Bug #5: fall back on "no results OR no answer", not just on exception.
    if not results and not answer:
        try:
            results = await _duckduckgo_search(query, topic, max_results)
        except Exception as exc:  # noqa: BLE001
            log.warning("DuckDuckGo failed for %r: %s", query, exc)
            results = []

    if not results and not answer:
        log.error("both providers returned nothing for %r", query)
        return "SEARCH_FAILED: web search is unavailable right now."

    formatted = _format_results(results, answer=answer, max_results=max_results)
    _SEARCH_CACHE[key] = (now, formatted)
    _evict_if_needed(now)
    return formatted


# ── tool schema (unchanged, but made explicit) ────────────────────────────
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use 'topic=news' for current "
            "events, recent news, live scores, prices, rankings, or anything that "
            "changes frequently. Use 'topic=general' for stable knowledge, how-to "
            "guides, or explanations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and concise.",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Search topic: 'general' or 'news'.",
                    "default": "general",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}