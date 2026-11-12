"""Manual launcher: run the Alvin server with HuggingFace forced offline.

Use when the Whisper model is already cached and the machine has no network.

Usage:
    python tests/manual/start_server_offline.py
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
