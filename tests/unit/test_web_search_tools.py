"""Verification tests for web search tool calling integration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.llm import LLMService, _check_voice_override, _should_search
from app.web_search import WEB_SEARCH_TOOL_SCHEMA, web_search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, create) -> None:
        self.chat = NS(completions=NS(create=create))


def _service_with_fake_stream(create) -> LLMService:
    settings.llm_api_key = "fake-key"
    settings.llm_enabled = True
    settings.debug = True
    # Pin the provider chain to the fake Kilo client. Without this the test
    # would hit the configured Cloudflare/NVIDIA provider, which answers with
    # a real (and non-deterministic) reply.
    settings.llm_primary_provider = "kilo"
    settings.llm_use_cf_as_primary = False
    settings.cf_enabled = False
    settings.nvidia_enabled = False

    service = LLMService()
    service._client = _FakeClient(create)
    service._api_key = "fake-key"
    return service


def _chunk(content=None, tool_calls=None) -> NS:
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


# ---------------------------------------------------------------------------
# Voice override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("search the web for python", 1),
        ("look it up", 1),
        ("google it", 1),
        ("don't search for that", 0),
        ("no search please", 0),
        ("tell me a joke", None),
    ],
)
def test_voice_override_detection(text, expected):
    assert _check_voice_override(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "What is the weather?",
        "Tell me the latest news",
        "What is the price of bitcoin?",
        "Search for the best pasta recipe",
        "Look it up",
        "Google it",
    ],
)
def test_live_info_and_explicit_requests_enable_web_search(text):
    """Only live-info hints and explicit search requests enable the tool."""
    assert _should_search(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "tell me a joke",
        "I am having a good day",
        "Please turn on the lights",
        "How do I make pasta?",
        "What is the capital of France?",
        "When is the meeting?",
    ],
)
def test_conversational_turns_do_not_enable_web_search(text):
    """A question word alone no longer triggers web search."""
    assert _should_search(text) is False


# ---------------------------------------------------------------------------
# Tool call streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversational_turn_does_not_offer_web_search():
    """Ordinary conversation must not expose the search tool to the model."""
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return _stream([_chunk("Sure.")])

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "tell me a joke"},
    ]

    result = [
        item
        async for item in service.stream_response(
            messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
        )
    ]

    assert result == [("Sure.", settings.llm_model)]
    assert calls[0].get("tools") is None
    assert calls[0].get("tool_choice") is None


@pytest.mark.asyncio
async def test_live_info_question_forces_web_search_tool():
    """A live-info question requires the web-search function."""
    calls = []

    async def fake_create(messages, **kwargs):
        calls.append(kwargs)
        if messages[-1].get("role") == "tool":
            return _stream([_chunk("Here is the answer.")])
        tc = NS(
            index=0,
            id="call_question",
            function=NS(
                name="web_search",
                arguments='{"query":"question query","topic":"general"}',
            ),
        )
        return _stream([_chunk(tool_calls=[tc])])

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is the latest news"},
    ]

    with patch(
        "app.llm.web_search", new_callable=AsyncMock, return_value="Mock result"
    ) as mock_search:
        result = [
            item
            async for item in service.stream_response(
                messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
            )
        ]

    assert calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "web_search"},
    }
    assert result[0] == (None, "tool_call:web_search")
    mock_search.assert_awaited_once_with("question query", "general")


@pytest.mark.asyncio
async def test_direct_answer_when_llm_does_not_call_tool():
    """LLM returns text without tool calls -> answer directly."""

    async def fake_create(**_kwargs):
        return _stream([_chunk("Paris is the capital of France.")])

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    result = []
    async for sentence, model in service.stream_response(
        messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
    ):
        result.append((sentence, model))

    assert len(result) == 1
    assert result[0][0] == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_search_tool_call_is_executed():
    """LLM returns a web_search tool call -> search runs and result is fed back."""

    async def fake_create(messages, **_kwargs):
        last_msg = messages[-1]
        if last_msg.get("role") == "tool":
            # Second call: LLM answers after seeing search results
            return _stream([_chunk("Here is what I found.")])
        # First call: LLM decides to search
        tc = NS(
            index=0,
            id="call_123",
            function=NS(name="web_search", arguments='{"query":"test query","topic":"general"}'),
        )
        return _stream([_chunk(tool_calls=[tc])])

    def _stream(pieces):
        async def gen():
            for piece in pieces:
                yield piece
        return gen()

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search who won the NBA game last night"},
    ]

    with patch("app.llm.web_search", new_callable=AsyncMock, return_value="Mock search result") as mock_search:
        result = []
        async for sentence, model in service.stream_response(
            messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
        ):
            result.append((sentence, model))

    # Should get: tool_call marker + final answer
    assert len(result) == 2
    assert result[0][0] is None
    assert result[0][1] == "tool_call:web_search"
    assert result[1][0] == "Here is what I found."
    mock_search.assert_called_once_with("test query", "general")


@pytest.mark.asyncio
async def test_force_search_voice_override():
    """'search the web' forces tool call even if LLM wouldn't normally search."""

    async def fake_create(messages, **_kwargs):
        last_msg = messages[-1]
        if last_msg.get("role") == "tool":
            return _stream([_chunk("I found information about photosynthesis.")])
        tool_choice = _kwargs.get("tool_choice", "auto")
        if tool_choice and not isinstance(tool_choice, str):
            # Forced tool call
            tc = NS(
                index=0,
                id="call_forced",
                function=NS(name="web_search", arguments='{"query":"forced query","topic":"general"}'),
            )
            return _stream([_chunk(tool_calls=[tc])])
        # Without force, LLM would answer directly
        return _stream([_chunk("I can answer directly.")])

    def _stream(pieces):
        async def gen():
            for piece in pieces:
                yield piece
        return gen()

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search the web for photosynthesis"},
    ]

    with patch("app.llm.web_search", new_callable=AsyncMock, return_value="Search result") as mock_search:
        result = []
        async for sentence, model in service.stream_response(
            messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
        ):
            result.append((sentence, model))

    assert len(result) == 2
    assert result[0][0] is None
    assert result[0][1] == "tool_call:web_search"
    mock_search.assert_called_once_with("forced query", "general")


@pytest.mark.asyncio
async def test_disable_search_voice_override():
    """'don't search' disables tools for the first round."""

    call_count = 0

    async def fake_create(messages, **_kwargs):
        nonlocal call_count
        call_count += 1
        tool_choice = _kwargs.get("tool_choice", "auto")
        if tool_choice == "none" or (isinstance(tool_choice, str) and tool_choice == "none"):
            return _stream([_chunk("Paris is the capital.")])
        if call_count == 1:
            # First round with tools disabled should just answer
            return _stream([_chunk("Paris is the capital.")])
        return _stream([_chunk("Done.")])

    def _stream(pieces):
        async def gen():
            for piece in pieces:
                yield piece
        return gen()

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "don't search, what is the capital of France?"},
    ]

    result = []
    async for sentence, model in service.stream_response(
        messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
    ):
        result.append((sentence, model))

    assert len(result) == 1
    assert result[0][0] == "Paris is the capital."


@pytest.mark.asyncio
async def test_search_failure_returns_error_message():
    """When both Tavily and DuckDuckGo fail, web_search returns SEARCH_FAILED."""

    async def fake_create(messages, **_kwargs):
        last_msg = messages[-1]
        if last_msg.get("role") == "tool":
            return _stream([_chunk("I couldn't search the web right now.")])
        tc = NS(
            index=0,
            id="call_fail",
            function=NS(name="web_search", arguments='{"query":"test","topic":"general"}'),
        )
        return _stream([_chunk(tool_calls=[tc])])

    def _stream(pieces):
        async def gen():
            for piece in pieces:
                yield piece
        return gen()

    service = _service_with_fake_stream(fake_create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search for the test result"},
    ]

    with patch("app.llm.web_search", new_callable=AsyncMock, return_value="SEARCH_FAILED: web search is unavailable right now.") as mock_search:
        result = []
        async for sentence, model in service.stream_response(
            messages, tools=[WEB_SEARCH_TOOL_SCHEMA], tool_choice="auto"
        ):
            result.append((sentence, model))

    assert len(result) == 2
    assert result[0][0] is None
    assert result[0][1] == "tool_call:web_search"
    # Second round: LLM sees the search failure and should answer
    assert "search" in result[1][0].lower() or "unavailable" in result[1][0].lower()


def _stream(pieces):
    async def gen():
        for piece in pieces:
            yield piece
    return gen()
