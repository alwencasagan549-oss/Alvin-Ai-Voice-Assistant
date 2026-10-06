"""Unit tests for the LLM text pipeline and streaming robustness.

No network: the streaming helpers are exercised with synthetic chunks.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from types import SimpleNamespace as NS

import pytest

from app.config import settings
from app.llm import (
    LLMService,
    _chunk_delta,
    build_messages,
    chunk_sentences,
    sanitize_for_voice,
)


def _chunk(text: str | None) -> NS:
    """A streamed chunk shaped like the OpenAI SDK's."""
    return NS(choices=[NS(delta=NS(content=text))])


# --------------------------------------------------------------------------- #
# streamed chunk parsing (regression)
# --------------------------------------------------------------------------- #


def test_chunk_delta_reads_content():
    assert _chunk_delta(_chunk("Hello")) == "Hello"


def test_chunk_delta_tolerates_chunk_with_no_choices():
    """A keep-alive / usage-only frame must not raise IndexError.

    Regression: indexing ``chunk.choices[0]`` unguarded crashed the whole turn
    with ``IndexError`` whenever the gateway emitted an empty chunk, which
    surfaced as intermittent ``llm_error`` events on the fastest models.
    """
    assert _chunk_delta(NS(choices=[])) == ""
    assert _chunk_delta(NS(choices=None)) == ""


def test_chunk_delta_tolerates_missing_and_none_fields():
    assert _chunk_delta(NS(choices=[NS(delta=NS(content=None))])) == ""
    assert _chunk_delta(NS(choices=[NS()])) == ""
    assert _chunk_delta(object()) == ""


# --------------------------------------------------------------------------- #
# sanitising
# --------------------------------------------------------------------------- #


def test_sanitize_strips_thinking_blocks():
    # Built programmatically: heredocs/shells have been known to mangle "</"
    # inside inline test literals.
    open_tag, close_tag = "<" + "thinking>", "<" + "/thinking>"
    assert (
        sanitize_for_voice(open_tag + "reasoning" + close_tag + "Hello there.")
        == "Hello there."
    )
    assert (
        sanitize_for_voice(open_tag + "reasoning" + close_tag + " Hello there.")
        == "Hello there."
    )
    assert sanitize_for_voice(open_tag + "multi\nline" + close_tag + "Hi.") == "Hi."
    assert (
        sanitize_for_voice("Hello " + open_tag + "x" + close_tag + "there.")
        == "Hello there."
    )


def test_sanitize_strips_markdown_and_collapses_whitespace():
    assert sanitize_for_voice("**Bold** and `code`") == "Bold and code"
    assert sanitize_for_voice("  spaced   out  ") == "spaced out"


def test_sanitize_empty_input():
    assert sanitize_for_voice("") == ""


# --------------------------------------------------------------------------- #
# sentence chunking
# --------------------------------------------------------------------------- #


def test_chunk_sentences_splits_on_terminators():
    assert chunk_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunk_sentences_keeps_trailing_fragment():
    assert chunk_sentences("Done. And more") == ["Done.", "And more"]


def test_chunk_sentences_empty():
    assert chunk_sentences("") == []
    assert chunk_sentences("   ") == []


# --------------------------------------------------------------------------- #
# message assembly
# --------------------------------------------------------------------------- #


def test_build_messages_pins_the_system_prompt():
    messages = build_messages([], "hello", max_history=12)
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_build_messages_trims_to_cap_keeping_system():
    history = [{"role": "system", "content": "sys"}]
    history += [{"role": "user", "content": str(i)} for i in range(30)]
    messages = build_messages(history, "newest", max_history=12)

    assert len(messages) == 12
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "newest"


def test_build_messages_is_self_trimming_on_append():
    """ConversationHistory enforces the cap without a separate trim step."""
    from app.llm import ConversationHistory

    history = ConversationHistory([{"role": "system", "content": "sys"}], max_history=4)
    for i in range(20):
        history.append({"role": "user", "content": str(i)})

    assert len(history) == 4
    assert history[0]["role"] == "system"


@pytest.mark.parametrize("cap", [2, 3, 8])
def test_conversation_history_never_drops_the_system_prompt(cap):
    from app.llm import ConversationHistory

    history = ConversationHistory(
        [{"role": "system", "content": "sys"}], max_history=cap
    )
    for i in range(50):
        history.append({"role": "user", "content": str(i)})
        assert history[0]["role"] == "system"
        assert len(history) <= cap


# --------------------------------------------------------------------------- #
# GeneratorExit safety (regression)
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Async-iterable stand-in for the OpenAI streaming response."""

    def __init__(self, pieces: list[str], *, stall: bool = False) -> None:
        self._pieces = pieces
        self._stall = stall

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for piece in self._pieces:
            yield NS(choices=[NS(delta=NS(content=piece))])
        if self._stall:
            # Keep the generator open so the consumer can close it mid-stream.
            await asyncio.sleep(3600)


async def _drain_then_close(pieces: list[str]) -> list[tuple[str, str]]:
    """Consume one sentence, then close the generator while it is suspended.

    This is the exact shape of a barge-in: the outer generator is parked on its
    own ``yield`` when the consumer closes it, so GeneratorExit is raised there
    and any ``yield`` in a ``finally`` blows up.
    """

    async def fake_create(**_kwargs):
        return _FakeStream(pieces)

    service = _service_with_fake_stream(fake_create)

    collected: list[tuple[str, str]] = []
    agen = service._stream_primary([{"role": "user", "content": "hi"}], None, None)
    async with contextlib.aclosing(agen):
        async for item in agen:
            collected.append(item)
            break  # leave a trailing partial sentence buffered, then close
    return collected


class _FakeClient:
    def __init__(self, create) -> None:
        self.chat = NS(completions=NS(create=create))


def _service_with_fake_stream(create) -> LLMService:
    """An LLMService whose primary client returns a canned stream.

    ``_get_client`` reads the API key from ``settings`` and rebuilds the real
    client whenever ``_api_key`` doesn't match the configured key, so the test
    must prime BOTH ``settings.llm_api_key`` (so the key is non-empty and
    matches) and ``service._api_key`` before the client is first requested.
    """
    from app.config import settings

    # Prime the runtime-read settings singleton with a fake key so ``_get_client``
    # accepts the fake client we install below instead of returning None (the
    # suite's conftest hard-wires these values for a hermetic run). The primed
    # state is kept so the service can use the fake client during the test.
    settings.llm_api_key = "fake-key"
    settings.llm_enabled = True

    service = LLMService()
    service._client = _FakeClient(create)
    service._api_key = "fake-key"

    # This helper is test-only and the suite runs serially; no caller relies on
    # the original values after it returns.
    return service


def _source_of(func) -> str:
    """Dedented source of a function, for structural assertions."""
    import inspect
    import textwrap

    return textwrap.dedent(inspect.getsource(func))


@pytest.mark.parametrize("method_name", ["_stream_primary", "_stream_cloudflare"])
def test_streaming_generators_never_yield_inside_finally(method_name):
    """No ``yield`` may appear inside a ``finally`` of these generators.

    This is the structural rule that prevents "RuntimeError: async generator
    ignored GeneratorExit". When a generator is closed while suspended on a
    ``yield`` (barge-in, ``stop_speak``, disconnect), CPython throws
    GeneratorExit at that suspension point; a generator that then yields again
    from a ``finally`` makes the *caller's* ``aclose`` raise that RuntimeError.
    A local ``except RuntimeError`` inside the generator cannot catch it, because
    the error is raised in the caller. The trailing fragment is therefore emitted
    from ``else`` (normal-completion only).
    """
    func = getattr(LLMService, method_name)
    src = _source_of(func)

    # Find each finally: block and assert it contains no yield.
    for match in re.finditer(r"^([ ]*)finally:[ ]*$", src, re.MULTILINE):
        indent = match.group(1)
        start = match.end()
        # Consume the block: lines that are blank or deeper-indented.
        block_lines = []
        for line in src[start:].splitlines(keepends=True):
            stripped = line.strip()
            if stripped and not line.startswith(indent + " "):
                break
            block_lines.append(line)
        block = "".join(block_lines)
        assert "yield" not in block, (
            f"{method_name}: yield inside finally triggers GeneratorExit errors\n"
            f"{block}"
        )


@pytest.mark.asyncio
async def test_normal_completion_still_emits_trailing_fragment():
    """A stream that ends without punctuation must still speak its tail."""

    async def fake_create(**_kwargs):
        return _FakeStream(["All done, no terminator"])

    service = _service_with_fake_stream(fake_create)

    got = [
        item
        async for item in service._stream_primary(
            [{"role": "user", "content": "hi"}], None, None
        )
    ]
    assert got == [("All done, no terminator", settings.llm_model)]
