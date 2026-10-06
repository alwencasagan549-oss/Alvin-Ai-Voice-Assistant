"""Benchmark LLM models for the Alvin voice assistant.

Tests multiple LLM providers/models to find the fastest with good quality.

Usage:
    python tests/tests/benchmark_llm_models.py

Environment variables required:
- LLM_API_KEY or NVIDIA_API_KEY (for NVIDIA NIM primary)
- GROQ_API_KEY (for Groq models including openai/gpt-oss-20b)
- CF_API_KEY + CF_ACCOUNT_ID (for Cloudflare fallback)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

# Shared test queries that exercise the Alvin persona
TEST_QUERIES = [
    "What is your name and who created you?",
    "What is Dinagat Island?",
    "Tell me the time in 5 words.",
    "How are you today?",
]

# Models to benchmark, as (provider_name, base_url, api_key_env, model_id)
# A None api_key skips that provider.
PROVIDERS = [
    (
        "NVIDIA NIM",
        "https://integrate.api.nvidia.com/v1",
        "LLM_API_KEY",
        "nvidia/nemotron-3-super-120b-a12b",
    ),
    (
        "Groq openai/gpt-oss-20b",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-20b",
    ),
    (
        "Cloudflare @cf/meta/llama-3.2-3b",
        "https://api.cloudflare.com/client/v4/accounts",
        "CF_API_KEY",
        "@cf/meta/llama-3.2-3b-instruct",
    ),
]

# Minimal system prompt matching Alvin's persona
SYSTEM_PROMPT = (
    "You are Alvin, a concise AI voice assistant. Your name is Alvin. "
    "You were created by Alwin Casagan. Keep responses short. "
    "Do not use emojis, markdown, or lists."
)


@dataclass
class ModelResult:
    name: str
    model: str
    latencies: list[float] = field(default_factory=list)
    first_token_times: list[float] = field(default_factory=list)
    errors: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def avg_ttfb(self) -> float:
        return (
            sum(self.first_token_times) / len(self.first_token_times)
            if self.first_token_times
            else 0.0
        )

    @property
    def success_rate(self) -> float:
        total = len(self.latencies) + self.errors
        return (len(self.latencies) / total * 100) if total > 0 else 0.0


async def benchmark_provider(
    name: str,
    base_url: str,
    api_key_env: str,
    model: str,
    account_id: str | None = None,
) -> ModelResult:
    result = ModelResult(name=name, model=model)
    api_key = os.getenv(api_key_env)

    if not api_key:
        print(f"  {name}: SKIPPED (no {api_key_env} set)")
        return result

    full_url = base_url
    if account_id:
        full_url = f"{base_url}/{account_id}/ai/v1"

    client = AsyncOpenAI(api_key=api_key, base_url=full_url)

    for i, query in enumerate(TEST_QUERIES):
        start = time.perf_counter()
        ttfb = None
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=128,
                    temperature=0.7,
                    stream=True,
                ),
                timeout=15.0,
            )

            full_response = ""
            async for chunk in stream:
                # Tolerate chunks with no choices (keep-alives, usage-only frames)
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                content = delta.content or ""
                if content:
                    if ttfb is None:
                        ttfb = (time.perf_counter() - start) * 1000
                    full_response += content

            if not full_response.strip():
                result.errors += 1
                continue

            total = (time.perf_counter() - start) * 1000
            result.latencies.append(total)
            result.first_token_times.append(ttfb or 0.0)
            result.samples.append(full_response[:80])

        except Exception as exc:  # noqa: BLE001 — benchmark should not crash
            result.errors += 1
            elapsed = (time.perf_counter() - start) * 1000
            print(
                f"  {name} query {i + 1}: ERROR {type(exc).__name__}: {exc} ({elapsed:.0f}ms)"
            )

    return result


async def main():
    print("=" * 70)
    print(
        f"{'Model Provider':<30} {'Avg TTFB':>10} {'Avg Latency':>12} {'Success':>10}"
    )
    print("=" * 70)

    results: list[ModelResult] = []
    for name, base_url, api_key_env, model in PROVIDERS:
        account_id = os.getenv("CF_ACCOUNT_ID") if "CF_API_KEY" == api_key_env else None
        result = await benchmark_provider(
            name, base_url, api_key_env, model, account_id
        )
        results.append(result)
        if result.latencies:
            print(
                f"{name:<30} {result.avg_ttfb:>8.0f}ms {result.avg_latency:>10.0f}ms "
                f"{result.success_rate:>8.0f}%"
            )
        else:
            print(f"{name:<30} {'N/A':>10} {'N/A':>12} {'N/A':>10}")

    print("=" * 70)

    for r in results:
        if r.samples:
            print(f"\n{r.name} ({r.model}):")
            for i, (q, s) in enumerate(zip(TEST_QUERIES, r.samples)):
                print(f"  Q{i + 1}: {q}")
                print(f"  A{i + 1}: {s}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
