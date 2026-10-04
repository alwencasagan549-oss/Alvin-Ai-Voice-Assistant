"""LLM service — the conversational brain of the Alvin voice assistant.

Routes every finalized utterance through an OpenAI-compatible LLM. The default
backend is NVIDIA NIM (``nvidia/nemotron-3-ultra-550b-a55b`` on
``https://integrate.api.nvidia.com/v1``), benchmarked as the fastest working
model (~1s per turn with clean output). Cloudflare Workers AI provides a
transparent fallback (``@cf/meta/llama-3.2-3b-instruct``).

The service supports two modes:

* **Blocking** (:meth:`generate_response`) — awaits the full completion, used by
  the simple echo path.
* **Streaming** (:meth:`stream_response`) — yields completed sentences as they
  arrive, so TTS can begin speaking the first sentence before the LLM finishes
  writing the rest (lower Time-To-First-Audio).

Pipeline per turn::

    STT (transcript) -> LLM (streamed sentences) -> TTS (spoken PCM)

Thinking blocks (``<thinking>...</thinking>``), Markdown formatting, and inline
code are stripped from every sentence before it reaches the TTS layer.

Fallback strategy: if the primary LLM (Kilo AI) errors or times out, the
service falls back to Cloudflare Workers AI (``cloudflare/{cf_model}``). When
both fail, a single fallback string (the raw transcript) is emitted so silence
is never the answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import settings

log = logging.getLogger("alvin.llm")


# Type alias matching what ``AsyncOpenAI.chat.completions.create`` expects.
# Using the openai param union (instead of bare ``dict``) keeps the message list
# type-checked end-to-end and silences the ``reportArgumentType`` diagnostics
# that plain ``list[dict]`` tripped at every ``create`` call site.
MessageParam = ChatCompletionMessageParam

# --- text cleaning ------------------------------------------------------- #
# Remove thinking blocks (including content): <thinking>...</thinking>
_THINK_BLOCK = re.compile(r"<thinking\b[^>]*>.*?</thinking>", re.IGNORECASE | re.DOTALL)
# Markdown artifacts that degrade speech quality.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"\*(.+?)\*")
_MD_INLINE_CODE = re.compile(r"`(.+?)`")
_MD_HEADING = re.compile(r"^#+ ", re.MULTILINE)
_MD_LIST = re.compile(r"^\s*[-*] ", re.MULTILINE)
# Sentence boundary for streaming chunking.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sanitize_for_voice(text: str) -> str:
    """Strip thinking blocks, Markdown, and code artifacts from LLM output.

    Removes ``<thinking>...</thinking>`` tags, bold/italic/code formatting,
    heading markers, and list bullets so edge-tts reads clean prose without
    audible artifacts like "asterisk" or "markdown".
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_sentences(text: str) -> list[str]:
    """Split text into completed sentence chunks."""
    if not text.strip():
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def build_messages(
    history: list[MessageParam],
    user_text: str,
    max_history: int | None = None,
) -> list[MessageParam]:
    """Build a messages list with system prompt pinned first and truncated history.

    Args:
        history: Existing conversation turns (system message expected at index 0).
        user_text: The new user utterance to append.
        max_history: Maximum messages including system prompt. ``None`` keeps all.

    Returns:
        A new list for ``AsyncOpenAI.chat.completions.create``.
    """
    system_prompt = settings.llm_system_prompt
    if not history or history[0].get("role") != "system":
        sys_msg: MessageParam = {"role": "system", "content": system_prompt}
        history = [sys_msg] + history

    user_msg: MessageParam = {"role": "user", "content": user_text}
    history = history + [user_msg]

    if max_history is not None and len(history) > max_history:
        history = [history[0]] + history[-(max_history - 1) :]

    return history


# --- result dataclass ---------------------------------------------------- #
@dataclass
class LLMResult:
    """One LLM outcome."""

    text: str
    latency_ms: float
    model: str


# --- service ------------------------------------------------------------- #
class LLMService:
    """Conversational brain with streaming, multi-provider fallback.

    Configuration:

    * ``LLM_API_KEY`` / ``NVIDIA_API_KEY`` — NVIDIA NIM API token.
    * ``LLM_MODEL`` — primary model id, default ``nvidia/nemotron-3-ultra-550b-a55b``.
    * ``LLM_BASE_URL`` — NIM endpoint base URL.
    * ``LLM_MAX_TOKENS``, ``LLM_TEMPERATURE`` — decoding knobs.
    * ``LLM_SYSTEM_PROMPT`` — persona / style steer.
    * ``LLM_TIMEOUT_SEC`` — per-call timeout for the primary provider.
    * ``LLM_WARMUP`` — prime the TLS pool at startup.
    * ``CF_API_KEY`` / ``CF_ACCOUNT_ID`` / ``CF_MODEL`` — Cloudflare fallback.
    * ``CF_ENABLED`` — toggle the Cloudflare fallback path.
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._cf_client: AsyncOpenAI | None = None
        self._api_key: str | None = None
        self._cf_api_key: str | None = None

    # --- client management ----------------------------------------------- #
    def _resolve_api_key(self) -> str | None:
        key = settings.llm_api_key
        if not key:
            key = settings.groq_api_key or None
        return key

    def _get_client(self) -> AsyncOpenAI | None:
        api_key = self._resolve_api_key()
        if not api_key:
            return None
        if api_key != self._api_key or self._client is None:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.llm_base_url,
            )
            self._api_key = api_key
        return self._client

    def _get_cf_client(self) -> AsyncOpenAI | None:
        if not settings.cf_enabled:
            return None
        key = settings.cf_api_key
        if not key or not settings.cf_account_id:
            return None
        base_url = f"{settings.cf_base_url}/{settings.cf_account_id}/ai/v1"
        if key != self._cf_api_key or self._cf_client is None:
            self._cf_client = AsyncOpenAI(
                api_key=key,
                base_url=base_url,
            )
            self._cf_api_key = key
        return self._cf_client

    @property
    def enabled(self) -> bool:
        """True when the NVIDIA NIM LLM path is configured."""
        return settings.llm_enabled and bool(self._resolve_api_key())

    @property
    def cf_enabled(self) -> bool:
        """True when the Cloudflare fallback is configured."""
        return (
            settings.cf_enabled
            and bool(settings.cf_api_key)
            and bool(settings.cf_account_id)
        )

    @property
    def model(self) -> str:
        return settings.llm_model

    # --- warm-up --------------------------------------------------------- #
    async def warm_up(self) -> bool:
        """Send a tiny request to prime the TLS/HTTP pool at startup.

        Uses the streaming path (``stream_response``) — the same code path
        used in production — since some gateway backends return empty
        content on non-streaming completions while streaming works fine.

        Retries once if the primary model yields nothing (cold-start on the
        gateway), then falls back to Cloudflare if available.

        Returns ``True`` when the LLM answered; failures are logged and never
        raised so startup is never blocked by a slow/unreachable gateway.
        """
        if not self.enabled:
            log.info("LLM warm-up skipped (disabled or no API key)")
            return False

        messages = build_messages([], "Hello")
        start = time.perf_counter()

        for attempt in (1, 2):
            try:
                count = 0
                async for _sentence, _model in self.stream_response(
                    messages, max_tokens=16
                ):
                    count += 1
                if count:
                    elapsed = (time.perf_counter() - start) * 1000
                    log.info(
                        "LLM warm in %.0f ms (%d sentence(s), attempt=%d)",
                        elapsed,
                        count,
                        attempt,
                    )
                    return True
                if attempt == 1:
                    log.debug("LLM warm-up yielded no content, retrying...")
            except Exception:  # noqa: BLE001 — best-effort warm-up
                if attempt == 1:
                    log.debug("LLM warm-up attempt 1 failed, retrying...")
                else:
                    elapsed = (time.perf_counter() - start) * 1000
                    log.warning(
                        "LLM warm-up failed after %.0f ms; LLM path remains enabled",
                        elapsed,
                    )
                    return False

        elapsed = (time.perf_counter() - start) * 1000
        log.warning("LLM warm-up returned no content after %.0f ms", elapsed)
        return False

    # --- blocking response ----------------------------------------------- #
    async def generate_response(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        messages: list[MessageParam] | None = None,
    ) -> str:
        """Generate a conversational reply to the user's utterance (blocking).

        Args:
            text: The transcript from STT (user's spoken words).
            max_tokens: Override for response length.
            temperature: Override for sampling temperature.
            messages: Optional pre-built conversation history. If provided,
                ``text`` is ignored for message construction.

        Returns:
            The LLM's text response, cleaned of Markdown/thinking.

        Raises:
            RuntimeError: when the LLM is disabled or returns no content.
            asyncio.TimeoutError: when the call exceeds ``LLM_TIMEOUT_SEC``.
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError("LLM service is not configured (no API key)")

        if messages is None:
            messages = [
                {"role": "system", "content": settings.llm_system_prompt},
                {"role": "user", "content": text},
            ]

        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature
                    if temperature is not None
                    else settings.llm_temperature,
                ),
                timeout=settings.llm_timeout_sec,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            if isinstance(exc, asyncio.TimeoutError):
                log.warning(
                    "LLM primary timed out after %.0f ms, trying Cloudflare",
                    (time.perf_counter() - start) * 1000,
                )
            else:
                log.warning(
                    "LLM primary failed (%s), trying Cloudflare: %s",
                    type(exc).__name__,
                    exc,
                )
            # Fall back to Cloudflare if configured.
            if self.cf_enabled:
                return await self._generate_cf_response(
                    messages, max_tokens, temperature
                )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = msg.content or ""
            if not content:
                # ``reasoning_content`` is a non-standard extension set by some
                # providers (e.g. reasoning models); it is absent from the openai
                # type stubs, so read it safely with getattr instead of hasattr.
                content = getattr(msg, "reasoning_content", "") or ""

        if not content:
            log.warning(
                "LLM returned empty response (model=%s, latency=%.0fms)",
                settings.llm_model,
                latency_ms,
            )
            raise RuntimeError(f"LLM ({settings.llm_model}) returned an empty response")

        cleaned = sanitize_for_voice(content)
        log.debug(
            "LLM [%s] latency=%.0fms tokens=%s",
            settings.llm_model,
            latency_ms,
            getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        )
        return cleaned

    async def _generate_cf_response(
        self,
        messages: list[MessageParam],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a response using the Cloudflare Workers AI fallback."""
        client = self._get_cf_client()
        if client is None:
            raise RuntimeError("Cloudflare LLM fallback is not configured")

        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.cf_model,
                    messages=messages,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature
                    if temperature is not None
                    else settings.llm_temperature,
                ),
                timeout=settings.cf_timeout_sec,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            log.warning("Cloudflare LLM timed out after %.0f ms", elapsed)
            raise
        except Exception as exc:
            log.error("Cloudflare LLM failed: %s", exc)
            raise

        elapsed = (time.perf_counter() - start) * 1000
        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = msg.content or ""

        if not content:
            raise RuntimeError(
                f"Cloudflare LLM ({settings.cf_model}) returned an empty response"
            )

        cleaned = sanitize_for_voice(content)
        log.info(
            "Cloudflare LLM [%s] latency=%.0fms (fallback succeeded)",
            settings.cf_model,
            elapsed,
        )
        return cleaned

    # --- streaming ------------------------------------------------------- #
    async def stream_response(
        self,
        messages: list[MessageParam],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        fallback_text: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Stream LLM tokens, yielding completed sentences as they arrive.

        Tries the primary provider (Kilo AI) first; on failure or empty output,
        falls back to Cloudflare Workers AI. If both fail and ``fallback_text``
        is provided, yields that as a single fallback sentence so the user
        always hears *something*.

        Args:
            messages: Full conversation history (system first).
            max_tokens: Override for response length.
            temperature: Override for sampling temperature.
            fallback_text: Text to yield if all LLM calls fail.

        Yields:
            ``(sentence, model_name)`` tuples where ``model_name`` is the id of
            the provider that actually generated the sentence (primary or cf).
        """
        primary_count = 0
        try:
            async for sentence, model in self._stream_primary(
                messages, max_tokens, temperature
            ):
                primary_count += 1
                yield sentence, model
        except Exception as exc:  # noqa: BLE001 — try fallback
            log.warning(
                "LLM primary streaming failed (%s: %s), trying Cloudflare",
                type(exc).__name__,
                exc,
            )

        # If primary failed or produced no content, fall back to Cloudflare.
        if primary_count == 0:
            if self.cf_enabled:
                log.info("LLM primary yielded no content; falling back to Cloudflare")
                cf_count = 0
                try:
                    async for sentence, model in self._stream_cloudflare(
                        messages, max_tokens, temperature
                    ):
                        cf_count += 1
                        yield sentence, model
                except Exception as exc2:  # noqa: BLE001 — CF also failed
                    log.error("Cloudflare streaming fallback also failed: %s", exc2)
                    cf_count = 0

            if cf_count == 0 and fallback_text:
                cleaned = sanitize_for_voice(fallback_text)
                if cleaned:
                    yield cleaned, settings.llm_model

    async def _stream_primary(
        self,
        messages: list[MessageParam],
        max_tokens: int | None,
        temperature: float | None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Stream sentences from the primary LLM provider (Kilo AI)."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("LLM service is not configured (no API key)")

        start = time.perf_counter()
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else settings.llm_temperature,
                stream=True,
            ),
            timeout=settings.llm_timeout_sec,
        )
        log.debug("LLM stream started (model=%s)", settings.llm_model)

        model_name = settings.llm_model
        buffer = ""
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                buffer += delta

                # Yield completed sentences as soon as a delimiter is seen.
                if _SENTENCE_END.search(buffer):
                    for i in range(len(buffer) - 1, -1, -1):
                        if buffer[i] in ".!?":
                            sentence = buffer[: i + 1].strip()
                            buffer = buffer[i + 1 :]
                            cleaned = sanitize_for_voice(sentence)
                            if cleaned:
                                yield cleaned, model_name
                            break
        except asyncio.TimeoutError:
            log.warning(
                "LLM stream timed out after %.0f ms",
                (time.perf_counter() - start) * 1000,
            )
            raise
        finally:
            remainder = buffer.strip()
            if remainder:
                cleaned = sanitize_for_voice(remainder)
                if cleaned:
                    # When the task is cancelled during stream_response,
                    # the async generator is being closed (GeneratorExit).
                    # Yielding in a finally block under those conditions
                    # raises "RuntimeError: async generator ignored
                    # GeneratorExit", so guard against that.
                    try:
                        yield cleaned, model_name
                    except RuntimeError:
                        pass
            log.debug(
                "LLM stream closed (latency=%.0fms)",
                (time.perf_counter() - start) * 1000,
            )

    async def _stream_cloudflare(
        self,
        messages: list[MessageParam],
        max_tokens: int | None,
        temperature: float | None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Stream sentences from the Cloudflare Workers AI fallback."""
        client = self._get_cf_client()
        if client is None:
            raise RuntimeError("Cloudflare LLM fallback is not configured")

        start = time.perf_counter()
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.cf_model,
                messages=messages,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else settings.llm_temperature,
                stream=True,
            ),
            timeout=settings.cf_timeout_sec,
        )
        log.debug("Cloudflare stream started (model=%s)", settings.cf_model)

        model_name = settings.cf_model
        buffer = ""
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                buffer += delta
                if _SENTENCE_END.search(buffer):
                    for i in range(len(buffer) - 1, -1, -1):
                        if buffer[i] in ".!?":
                            sentence = buffer[: i + 1].strip()
                            buffer = buffer[i + 1 :]
                            cleaned = sanitize_for_voice(sentence)
                            if cleaned:
                                yield cleaned, model_name
                            break
        finally:
            remainder = buffer.strip()
            if remainder:
                cleaned = sanitize_for_voice(remainder)
                if cleaned:
                    try:
                        yield cleaned, model_name
                    except RuntimeError:
                        pass
            log.debug(
                "Cloudflare stream closed (latency=%.0fms)",
                (time.perf_counter() - start) * 1000,
            )

    # --- cleanup --------------------------------------------------------- #
    async def shutdown(self) -> None:
        """Close all pooled HTTP clients."""
        for client in (self._client, self._cf_client):
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:  # noqa: BLE001 — cleanup must never raise
                    log.warning("error closing LLM client: %s", exc)
        self._client = None
        self._cf_client = None
        self._api_key = None
        self._cf_api_key = None


llm_service = LLMService()
