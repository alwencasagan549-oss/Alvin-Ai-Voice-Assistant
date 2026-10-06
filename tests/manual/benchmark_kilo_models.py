"""Benchmark all available Kilo AI gateway models for latency.

Usage:
    set LLM_API_KEY=your-token-here
    python tests/manual/benchmark_kilo_models.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from app.config import settings

MODELS_TO_TEST = [
    "stepfun/step-3.7-flash:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "dots-studio/dots-3-note-preview:free",
    "poolside/laguna-s-2.1:free",
    "apodex/apodex-1.1-mini:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "thinkingmachines/inkling-small:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "openrouter/free",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "google/gemma-2-9b-it",
    "qwen/qwen3-8b",
    "qwen/qwen3.8-27b",
]

PROMPT = "Say exactly: latency test"
MAX_TOKENS = 32


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm_api_key or "",
        base_url=str(settings.llm_base_url),
    )


async def _stream_chat(model: str) -> tuple[float, str, str | None, str | None]:
    client = _client()
    start = time.perf_counter()
    text_parts: list[str] = []
    model_name = model
    err: str | None = None
    try:
        async for chunk in await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            text_parts.append(delta)
            if chunk.choices[0].finish_reason:
                pass
        elapsed = time.perf_counter() - start
        return elapsed, "".join(text_parts).strip(), model_name, None
    except Exception as exc:
        elapsed = time.perf_counter() - start
        err = f"{type(exc).__name__}: {exc}"
        return elapsed, "", model_name, err


async def main() -> int:
    if not (settings.llm_api_key or "").strip():
        print("Set LLM_API_KEY first.", flush=True)
        return 1

    print(f"Base URL : {settings.llm_base_url}")
    print(f"Models   : {len(MODELS_TO_TEST)}")
    print(f"Prompt   : {PROMPT!r}")
    print(flush=True)

    results: list[tuple[str, float, str, str | None]] = []
    for model in MODELS_TO_TEST:
        elapsed, text, name, err = await _stream_chat(model)
        status = text if text else "FAILED"
        print(f"[{elapsed:6.2f}s] {model}: {status[:80]}", flush=True)
        if err:
            print(f"         ERROR: {err[:160]}", flush=True)
        elif not text:
            print(f"         ERROR: empty response", flush=True)
        results.append((model, elapsed, text, err))

    results.sort(key=lambda item: item[1] if item[2] else 9999)
    print(flush=True)
    print("=== Ranked by latency ===")
    for rank, (model, elapsed, text, _err) in enumerate(results, 1):
        ok = "OK" if text else "FAIL"
        print(f"{rank}. [{elapsed:6.2f}s] {ok} {model}", flush=True)

    fastest = next((r for r in results if r[2]), None)
    if fastest:
        print(
            f"\nFastest working model: {fastest[0]} ({fastest[1]:.2f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
