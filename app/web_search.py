"""Web search module for the Alvin voice assistant.

Provides a `web_search` function with Tavily as primary and DuckDuckGo as fallback,
5-minute in-memory cache, and formatted output suitable for LLM tool calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config import settings

log = logging.getLogger("alvin.web_search")


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    content: str
    domain: str


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def _format_results(results: list[SearchResult], max_results: int = 3) -> str:
    """Format search results as plain text for LLM consumption."""
    if not results:
        return "No results found."
    
    lines = []
    for i, r in enumerate(results[:max_results], 1):
        snippet = r.content[:400]
        lines.append(f"[{r.domain}] {r.title}: {snippet}")
    return "\n".join(lines)


async def _tavily_search(query: str, topic: str, max_results: int) -> list[SearchResult] | None:
    """Search using Tavily API. Returns None on failure."""
    api_key = getattr(settings, "tavily_api_key", "") or ""
    if not api_key:
        log.debug("Tavily API key not configured")
        return None
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        search_depth = "basic"
        include_answer = True
        time_range = "week" if topic == "news" else None
        
        # Run in thread pool since TavilyClient is synchronous
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    max_results=max_results,
                    topic=topic,
                    time_range=time_range,
                ),
            ),
            timeout=8.0,
        )
        
        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                domain=_extract_domain(item.get("url", "")),
            ))
        return results
        
    except Exception as exc:
        log.warning("Tavily search failed: %s", exc)
        return None


async def _duckduckgo_search(query: str, topic: str, max_results: int) -> list[SearchResult] | None:
    """Search using DuckDuckGo via ddgs. Returns None on failure."""
    try:
        from ddgs import DDGS
        
        # Run in thread pool since DDGS is synchronous
        loop = asyncio.get_running_loop()
        
        def _search():
            with DDGS() as ddgs:
                if topic == "news":
                    return list(ddgs.news(query, max_results=max_results))
                return list(ddgs.text(query, max_results=max_results))
        
        raw_results = await asyncio.wait_for(
            loop.run_in_executor(None, _search),
            timeout=8.0,
        )
        
        results = []
        for item in raw_results:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", item.get("url", "")),
                content=item.get("body", item.get("snippet", "")),
                domain=_extract_domain(item.get("href", item.get("url", ""))),
            ))
        return results
        
    except Exception as exc:
        log.warning("DuckDuckGo search failed: %s", exc)
        return None


# 5-minute cache (300 seconds)
_SEARCH_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_CACHE_TTL = 300  # seconds


def _cache_key(query: str, topic: str) -> tuple[str, str]:
    return (query.lower().strip(), topic.lower().strip())


async def web_search(query: str, topic: str = "general", max_results: int = 3) -> str:
    """Search the web and return formatted plain text results.
    
    Args:
        query: The search query.
        topic: "general" or "news".
        max_results: Maximum number of results to return (default 3).
    
    Returns:
        Formatted plain text with summary and up to 3 results.
        Never raises; on total failure returns an error message.
    """
    key = _cache_key(query, topic)
    now = time.time()
    
    # Check cache
    if key in _SEARCH_CACHE:
        cached_time, cached_result = _SEARCH_CACHE[key]
        if now - cached_time < _CACHE_TTL:
            log.debug("Cache hit for query: %s", query)
            return cached_result
    
    # Try Tavily first
    results = await _tavily_search(query, topic, max_results)
    
    # Fallback to DuckDuckGo
    if results is None:
        log.info("Tavily failed or unavailable, falling back to DuckDuckGo")
        results = await _duckduckgo_search(query, topic, max_results)
    
    # Total failure
    if results is None:
        error_msg = "SEARCH_FAILED: web search is unavailable right now."
        log.error("Both Tavily and DuckDuckGo failed for query: %s", query)
        return error_msg
    
    formatted = _format_results(results, max_results)
    
    # Cache the result
    _SEARCH_CACHE[key] = (now, formatted)
    
    # Clean old cache entries (simple LRU-ish cleanup)
    if len(_SEARCH_CACHE) > 100:
        cutoff = now - _CACHE_TTL
        for k in list(_SEARCH_CACHE.keys()):
            if _SEARCH_CACHE[k][0] <= cutoff:
                del _SEARCH_CACHE[k]
    
    return formatted


# Tool schema for LLM function calling
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use 'topic=news' for current events, recent news, live scores, prices, rankings, or anything that changes frequently. Use 'topic=general' for stable knowledge, how-to guides, or explanations.",
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
                    "description": "Search topic: 'general' for most queries, 'news' for current events, recent news.",
                    "default": "general",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}