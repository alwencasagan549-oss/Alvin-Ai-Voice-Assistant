"""Benchmark kilo.ai free models to compare their speed and response quality.

Tests each model with a conversational prompt representative of a voice
assistant interaction, with proper error handling for empty responses
and multiple iterations for stable latency measurements.
"""

import asyncio
import sys
import time

from openai import AsyncOpenAI

from app.config import settings

BASE_URL = settings.llm_base_url
API_KEY = settings.llm_api_key

MODELS = [
    "apodex/apodex-1.1-mini:free",
    "cohere/north-mini-code:free",
    "dots-studio/dots-3-note-preview:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "kilo-auto/free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "thinkingmachines/inkling-small:free",
    "stepfun/step-3.7-flash:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
]

TEST_PROMPT = (
    "You are Alvin, a helpful AI voice assistant. "
    "The user said: 'What is the weather like today?' Reply with a short, "
    "natural-sounding response suitable for text-to-speech."
)


async def benchmark_model(client: AsyncOpenAI, model: str, iterations: int = 2) -> dict:
    """Benchmark a single model across multiple iterations."""
    latencies = []
    response_texts = []
    tokens_used = 0
    errors = []

    for i in range(iterations):
        try:
            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=80,
            )
            end = time.perf_counter()

            latencies.append((end - start) * 1000)
            if response.choices and response.choices[0].message.content:
                response_texts.append(response.choices[0].message.content.strip())
            else:
                response_texts.append("")

            if response.usage:
                tokens_used = max(tokens_used, response.usage.total_tokens)

        except Exception as e:  # noqa: BLE001 — benchmark catches all API errors
            errors.append(str(e)[:200])

    success = len(response_texts) > 0 and response_texts[-1] != ""
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "model": model,
        "latency_ms": round(avg_latency, 2) if avg_latency else None,
        "all_latencies": [round(l, 2) for l in latencies],
        "tokens": tokens_used,
        "response": response_texts[-1][:200] if response_texts else "",
        "errors": errors,
        "success": success,
    }


async def main():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Testing {len(MODELS)} kilo.ai free models...")
    print("Iterations per model: 2")
    print(f"Test prompt: {TEST_PROMPT[:80]}...")
    print("-" * 80)

    results = []
    for model in MODELS:
        print(f"Testing {model}...", end=" ", flush=True)
        result = await benchmark_model(client, model)
        results.append(result)

        if result["success"]:
            print(
                f"OK  {result['latency_ms']}ms avg ({result['all_latencies']})  tokens={result['tokens']}"
            )
        else:
            err = result["errors"][-1] if result["errors"] else "empty response"
            print(f"FAIL  {err[:60]}")

    print("\n" + "=" * 80)
    print("RESULTS (sorted by latency)")
    print("=" * 80)

    successful = [r for r in results if r["success"] and r["latency_ms"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        successful.sort(key=lambda x: x["latency_ms"])
        print(f"\nSuccessful models ({len(successful)}):")
        for i, r in enumerate(successful, 1):
            print(f"  {i}. {r['model']}")
            print(f"     Latency: {r['latency_ms']}ms avg  ({r['all_latencies']})")
            print(f"     Tokens:  {r['tokens']}")
            print(f"     Response: {r['response'][:150]}")

    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for r in failed:
            err = r["errors"][-1] if r["errors"] else "empty response"
            print(f"  - {r['model']}: {err[:120]}")

    # Summary: recommend the fastest model with a non-empty, useful response
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    # Filter out models that returned empty responses
    usable = [r for r in successful if r["response"]]
    if usable:
        fastest = usable[0]
        print(f"\nFastest usable model: {fastest['model']}")
        print(f"  Latency: {fastest['latency_ms']}ms avg")
        print(f"  Response: {fastest['response'][:150]}")
    else:
        print("\nNo usable models found. Check errors above.")


async def test_single(model: str, prompt: str):
    """Quick single-model test for ad-hoc use."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    elapsed = (time.perf_counter() - start) * 1000
    text = (
        response.choices[0].message.content
        if response.choices and response.choices[0].message.content
        else ""
    )
    print(f"Model: {model}")
    print(f"Latency: {elapsed:.0f}ms")
    print(f"Response: {text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else TEST_PROMPT
        asyncio.run(test_single(model, prompt))
    else:
        asyncio.run(main())
