"""Quick Kilo AI gateway token test.
Usage:
    set LLM_API_KEY=your-token-here
    python tests/manual/test_kilo_token.py
"""

from __future__ import annotations

import os
import sys

from app.config import settings
from app.llm import LLMService


def main() -> int:
    token = os.environ.get("LLM_API_KEY") or getattr(settings, "llm_api_key", "")
    if not token:
        print("Set LLM_API_KEY first.", file=sys.stderr)
        return 1

    settings.llm_api_key = token
    settings.llm_model = "nvidia/nemotron-3.5-lightning:free"
    settings.llm_base_url = "https://api.kilo.ai/api/gateway"
    settings.llm_enabled = True

    service = LLMService()
    print("Resolved base URL:", settings.llm_base_url)
    print("Model:", settings.llm_model)

    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Say 'Kilo token is valid' only, nothing else."},
    ]

    try:
        texts = []
        async def run():
            async for sentence, model in service.stream_response(messages, fallback_text=""):
                texts.append(sentence)

        import asyncio
        asyncio.run(run())
        joined = " ".join(texts).strip()
        print("Response:", repr(joined))
        return 0 if joined else 1
    except Exception as exc:
        print("Request failed:", type(exc).__name__, exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
