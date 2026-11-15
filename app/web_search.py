"""Web search for the Alvin voice assistant.

Gemini CLI (primary) + DuckDuckGo (fallback), 5-minute in-memory cache,
formatted for LLM tool calls.

The primary path shells out to the free Gemini CLI
(`npm install -g @google/gemini-cli`), the same integration the
``gemini-bridge`` MCP server exposes: the query is piped to the CLI on stdin,
the CLI uses its built-in Google Search grounding tool, and the final answer
is read from stdout. CLI noise (retry logs, tool warnings) arrives on stderr
and is discarded. Authentication is whatever the CLI is configured with —
either a logged-in personal Google account or the ``GEMINI_API_KEY`` it picks
up from the project ``.env``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("alvin.web_search")

_CACHE_TTL = 300.0        # seconds
_CACHE_MAX_ENTRIES = 500  # hard cap
_SEARCH_TIMEOUT = 5.0     # DuckDuckGo budget, seconds
_DEFAULT_MAX_RESULTS = 3
_SNIPPET_LEN = 400

# The CLI pays node startup + CLI init on every call (~2-5 s) and the model
# may run several search rounds. Keep it short so DDG gets a turn quickly.
_DEFAULT_CLI_TIMEOUT = 8.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    domain: str
    published: str | None = None


# ── Gemini CLI ────────────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+")


def _cli_timeout() -> float:
    raw = (os.getenv("GEMINI_CLI_TIMEOUT_SEC") or "").strip()
    if not raw:
        return _DEFAULT_CLI_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_CLI_TIMEOUT
    return value if value > 0 else _DEFAULT_CLI_TIMEOUT


def _gemini_cli_path() -> str | None:
    """Locate the ``gemini`` executable; ``None`` means not installed."""
    return shutil.which("gemini")


def _build_search_prompt(query: str, topic: str) -> str:
    topic_note = (
        "Focus on the latest news and recent developments."
        if topic == "news"
        else "Prefer authoritative sources."
    )
    return (
        "Use the Google Search tool to research the query below, then answer "
        f"from what you find. {topic_note}\n"
        f"Query: {query}\n\n"
        "Respond with a concise plain-prose answer (no markdown), then a "
        "'Sources:' heading followed by the source URLs, one per line."
    )


def _split_answer_and_sources(
    text: str, max_results: int
) -> tuple[str, list[SearchResult]]:
    """Split CLI output into ``(answer, results)``.

    Everything before the ``Sources:`` heading is the answer; each URL line
    after it becomes a SearchResult. If no heading is present the whole text
    is the answer and no results are returned.
    """
    marker = re.compile(r"^sources?:\s*$", re.IGNORECASE | re.MULTILINE)
    match = marker.search(text)
    if not match:
        return text.strip(), []

    answer = text[: match.start()].strip()
    results: list[SearchResult] = []
    for line in text[match.end() :].splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line:
            continue
        url_match = _URL_RE.search(line)
        if not url_match:
            continue
        url = url_match.group(0).rstrip(").,;")
        label = (line[: url_match.start()] or "").strip(" -:•*")
        results.append(SearchResult(
            title=label or _extract_domain(url),
            url=url,
            content="",
            domain=_extract_domain(url),
        ))
        if len(results) >= max_results:
            break
    return answer, results


async def _gemini_search(
    query: str, topic: str, max_results: int
) -> tuple[list[SearchResult], str | None]:
    """Run the query through the Gemini CLI with Google Search grounding.

    Returns ``(results, answer)``. Returns empty on any failure so the caller
    falls back to DuckDuckGo. Raises only on the asyncio timeout path, which
    the caller also treats as a fallback signal.
    """
    cli = _gemini_cli_path()
    if cli is None:
        log.warning(
            "Gemini CLI not found; install with: npm install -g @google/gemini-cli"
        )
        return [], None

    topic = _normalize_topic(topic)
    cmd = [cli]
    model = (os.getenv("GEMINI_MODEL") or "").strip()
    if model:
        cmd += ["-m", model]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=_PROJECT_ROOT,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(
                _build_search_prompt(query, topic).encode("utf-8")
            ),
            timeout=_cli_timeout(),
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise

    if process.returncode != 0:
        log.warning(
            "Gemini CLI exited with code %s for %r", process.returncode, query
        )
        return [], None

    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return [], None

    answer, results = _split_answer_and_sources(text, max_results)
    return results, answer or None


# ── Tavily primary ───────────────────────────────────────────────────────

_TAVILY_URL = "https://api.tavily.com/search"


async def _tavily_search(
    query: str, topic: str, max_results: int, api_key: str
) -> tuple[list[SearchResult], str | None]:
    """Run a Tavily search. Returns ``(results, answer)``."""
    import aiohttp

    params = {
        "query": query,
        "topic": topic,
        "max_results": max_results,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _TAVILY_URL,
                json=params,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=8.0),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning("Tavily HTTP %d: %s", resp.status, text[:200])
                    return [], None
                data = await resp.json()
    except Exception as exc:
        log.warning("Tavily search failed: %s", exc)
        return [], None

    results: list[SearchResult] = []
    for item in data.get("results", []):
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            content=item.get("content", "") or item.get("snippet", ""),
            domain=_extract_domain(item.get("url")),
            published=item.get("published_date") or None,
        ))

    answer = data.get("answer")
    return results, answer


# ── DuckDuckGo fallback ───────────────────────────────────────────────────
_ddg_cls = None
_ddg_import_failed = False


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
        "web search will only work when the Gemini CLI succeeds."
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
    """Clamp to the only two values the providers accept."""
    return "news" if (topic or "").strip().lower() == "news" else "general"


def _format_results(
    results: list[SearchResult],
    answer: str | None = None,
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> str:
    if not results and not answer:
        return "No results found."

    lines: list[str] = []
    # The Gemini answer (already grounded in live search) is the
    # highest-signal line for the LLM.
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


async def _duckduckgo_search(
    query: str, topic: str, max_results: int
) -> list[SearchResult]:
    """Return DDG results. Raises on hard failure."""
    DDGS = _get_ddg_cls()
    if DDGS is None:
        return []

    topic = _normalize_topic(topic)

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
        url = item.get("href") or item.get("url") or ""
        results.append(SearchResult(
            title=item.get("title") or "",
            url=url,
            content=item.get("body") or item.get("snippet") or "",
            domain=_extract_domain(url),
            published=item.get("date") or None,
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
    """TTL first, then hard cap by oldest timestamp."""
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

    Tavily is primary, DuckDuckGo is fallback.
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

    # Run Tavily and DuckDuckGo concurrently; take whichever returns
    # usable results first. This avoids waiting for Tavily if DDG is faster.
    tavily_task: asyncio.Task | None = None
    ddg_task: asyncio.Task | None = None

    tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("GEMINI_API_KEY")
    if tavily_key:
        tavily_task = asyncio.create_task(_tavily_search(query, topic, max_results, tavily_key))
    ddg_task = asyncio.create_task(_duckduckgo_search(query, topic, max_results))

    pending = {t for t in (tavily_task, ddg_task) if t is not None}
    if pending:
        done, _ = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                result = await task
            except Exception:
                continue
            if isinstance(result, tuple):
                r, a = result
            else:
                r, a = result, None
            if r or a:
                results, answer = r, a
                break

    # If the first finisher came up empty, wait for the other provider too.
    if not results and not answer:
        for task in pending:
            if task.done():
                continue
            try:
                result = await task
            except Exception:
                continue
            if isinstance(result, tuple):
                r, a = result
            else:
                r, a = result, None
            if r or a:
                results, answer = r, a
                break

    # Cancel any still-running task so we don't leak subprocesses.
    for task in pending:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Fall back on "no results OR no answer", not just on exception.
    if not results and not answer:
        log.error("both providers returned nothing for %r", query)
        return "SEARCH_FAILED: web search is unavailable right now."

    formatted = _format_results(results, answer=answer, max_results=max_results)
    _SEARCH_CACHE[key] = (now, formatted)
    _evict_if_needed(now)
    return formatted


# ── tool schema ───────────────────────────────────────────────────────────
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
