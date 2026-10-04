# AGENTS.md — Alvin Voice Assistant

A low-latency, real-time WebSocket voice assistant for Alvin AI. It ingests raw
16 kHz / 16-bit / mono PCM audio over WebSockets and runs the full voice pipeline:

**STT (Whisper) → LLM (Kilo AI) → TTS (edge-tts)**

```

## Stack

- Python 3.12
- [FastAPI](https://fastapi.tiangolo.com) + Starlette WebSockets + uvicorn
- [faster-whisper](https://github.com/SYStran/faster-whisper) (ctranslate2) STT engine
- [silero-vad](https://github.com/snakers4/silero-vad) ONNX runtime VAD (streaming endpointing)
- OpenAI-compatible client for the Kilo AI gateway LLM brain
- [edge-tts](https://github.com/rany2/edge-tts) for TTS, [miniaudio](https://github.com/irliam/miniaudio) for MP3 decode

## Layout

```
app/
  config.py      Settings loaded from environment/.env
  vad.py         VADStreamDetector — streaming 32 ms frame VAD wrapper
  stt.py         STTService — shared Whisper model + serialized transcription
   llm.py         LLMService — Kilo AI gateway brain (nvidia/nemotron-3-super-120b-a12b:free)
  tts.py         Edge TTS — sentence-chunked 16 kHz PCM streaming
  connection.py  Per-connection state machine (VAD feed, segments, JSON events)
  main.py        FastAPI app: GET /health, WS /ws/transcribe
run.py          Server entrypoint (uvicorn)
requirements.txt
.env / .env.example   Runtime config (override per deployment)
tests/
  conftest.py   Shared fixtures: uvicorn subprocess (`server_url`), speech WAV
  unit/         Pure-logic tests, no Whisper
  integration/  End-to-end tests against a live server subprocess
  manual/       Run-by-hand client scripts (excluded from collection)
  data/         sample_speech.wav fixture
scripts/        Benchmark/measurement helpers
                `tts_voice_samples.py` — list voices + render sample MP3s
```

## Protocols

### WebSocket `ws://<host>:<port>/ws/transcribe`

**Input** — binary frames of little-endian int16 PCM (16 kHz, mono). Frames of
any size are accepted and buffered. Text frames carry JSON control messages:

- `{"type": "config", "language": "en"}` — override language for the session
  (`"auto"`/`""` → auto-detect). `"voice"` overrides the TTS voice.
- `{"type": "stop"}` — flush the current utterance as a final event, then close.
- `{"type": "ping"}` → `{"type": "pong"}`.
- `{"type": "speak", "text": "hello", "voice": "en-US-AndrewNeural"}` —
  synthesize and stream PCM back as binary frames. Emits `tts_end` when done and
  `tts_error` per failed sentence.
- `{"type": "stop_speak"}` — cancel in-flight synthesis; the socket stays open.

**Output** — JSON objects on every partial/final:

```json
{"transcript": "hello alvin how are you", "is_partial": true, "is_final": false, "confidence": 0.94}
```

- Interim hypothesis: `is_partial: true, is_final: false`
- End-of-turn (silence endpoint, `MIN_SILENCE_DURATION_MS=500`): `is_partial: false, is_final: true`
- `llm_enabled` field on final events indicates the LLM brain will process the utterance.
- When the LLM is enabled, a final transcript is followed by:
  - `{"type": "llm_response", "text": "...", "model": "..."}` — LLM reply
  - `{"type": "llm_error", "error": "..."}` — LLM failure (TTS falls back to raw transcript)

### Admission control

Both guards run **before** `accept()`, so a rejected client gets a denied
handshake (close code `1008`) instead of an open socket that closes immediately.

- `API_KEY` — when set, every client must present it as `Authorization: Bearer
  <key>` or `X-API-Key: <key>`; `?token=<key>` also works for browsers that
  cannot set WebSocket headers, but query strings land in access/proxy logs.
  `SERVICE_AUTH_TOKEN` is a legacy alias. Empty (the default) leaves the
  endpoint open — fine for loopback, not for a shared host. Compared in
  constant time (`app/connection.py:verify_token`).
- `MAX_CONNECTIONS` (default 64) — live sockets are tracked by
  `app/connection.py:ConnectionManager`; over the cap the handshake is refused
  with `1008`. `/health` reports `auth_required`, `active_connections`, and
  `max_connections`.

## Commands

```bash
# install
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
# run
python run.py                         # or: uvicorn app.main:app
# health
curl http://localhost:8000/health
# lint
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
# test (uses tiny.en for speed)
.venv\Scripts\pytest -q
# manual clients (need a running server)
python tests/manual/stream_wav.py                  # stream the fixture, print events
python tests/manual/run_with_server.py              # boot a throwaway tiny.en server + stream
python tests/manual/stream_mic.py                   # live mic (needs `pip install sounddevice`)
# voice catalogue + previews
python scripts\tts_voice_samples.py                 # every voice -> voice_samples/
python scripts\tts_voice_samples.py --lang en       # subset
# TTS latency / audio validation
python scripts\bench_tts.py                        # TTFA, RTF, silence, cost split
# pipeline latency + accuracy scorecard (no server needed)
python tests\benchmark_pipeline.py                 # STT cloud/local/silence + TTS TTFA/RTF
python tests\manual\speech_battery.py              # phrases, numbers/acronyms, noise, outage
```

## TTS

`app.tts` streams edge-tts output as 16 kHz / 16-bit / mono PCM, sentence by
sentence. The voice comes from `TTS_VOICE` (default `en-US-AndrewNeural`);
`synthesize_stream(text, voice=...)` overrides it per call. MP3 is decoded with
`miniaudio`, so no ffmpeg binary is required at runtime.

Pipeline: sentences are fetched by producer tasks (`TTS_PREFETCH` ahead of the
consumer), decoded progressively as MP3 arrives, run through `EdgeSilenceTrimmer`
(dropping silence below `TTS_SILENCE_THRESHOLD_DBFS` at both edges), and joined
with `TTS_SENTENCE_GAP_MS` of digital silence. `synthesize_stream` yields `bytes`
audio plus `{"type": "error", ...}` events instead of dropping failed sentences;
`synthesize_pcm` raises `TTSSynthesisError` on those. Measure with
`python scripts\bench_tts.py` (A/B against the old behaviour via `--prefetch 1
--trim-edge-ms 0 --decode-interval-bytes 999999999 --gap-ms 0`).

## STT

`app.stt` routes every segment through four guards, in order:

1. **Silence gate** — `audio_dbfs()` computes the buffer's RMS level; anything
   under `SILENCE_THRESHOLD_DBFS` (default `-55`) returns `("", 0.0)` in ~25 µs.
   This is what stops Whisper answering silence with hallucinations ("you",
   "thank you", "Subtitles by...").
2. **Cloud primary** — Groq `whisper-large-v3-turbo`. The audio is encoded as
   Ogg/Opus **in memory** by `soundfile` (libsndfile ships inside the wheel), so
   no ffmpeg binary is needed; the previous pydub path raised `FileNotFoundError`
   on machines without one and silently forced every turn onto the local model.
   `GROQ_INITIAL_PROMPT` steers proper nouns and punctuation.
3. **Circuit breaker** — `breaker_timeout()` gives the cloud
   `max(GROQ_TIMEOUT_SEC, GROQ_TIMEOUT_BASE_SEC + GROQ_TIMEOUT_SLOPE_SEC *
   duration)`, because encode + upload grows with the utterance and a flat
   cutoff strands long audio.
4. **Local fallback** — `FALLBACK_MODEL` (`small.en`, int8) on timeout or any
   Groq error, so the caller always gets a transcript.

The cloud connection pool is warmed during the lifespan
(`STTService.warm_up_cloud()`, disable with `GROQ_WARMUP=false`) so the first
turn does not pay TCP + TLS.

Measured fallback speed (int8, 4 CPU threads, 3.4 s of speech): `base.en`
~690 ms, `distil-small.en` ~1.99 s, `small.en` ~2.2 s. `distil-small.en` is not
worth the switch — it is barely faster than `small.en` at similar accuracy;
`base.en` is 3× faster but mangles proper nouns and email addresses.

## LLM

`app.llm` provides the conversational brain of the voice assistant. After STT
produces a final transcript, the LLM generates a reply that is then spoken via
TTS. Without an `LLM_API_KEY`, the LLM path is disabled and `TTS_AUTO_SPEAK`
(if on) echoes the raw transcript instead.

### Architecture

The pipeline per turn is:

```
STT (transcript) → LLM (streamed sentences) → TTS (per-sentence PCM)
```

**Token streaming & sentence-chunked TTS:** The LLM response is streamed via
`stream=True`, with tokens buffered into complete sentences (delimited by
`.`, `!`, `?`, or newlines). Each completed sentence is immediately fed to
`tts.synthesize_stream()` for speech output, so the first sentence reaches the
speaker before the LLM finishes writing the full reply — reducing
Time-To-First-Audio from ~1.5–3s to near-instant for short replies.

**Conversation memory:** Each WebSocket connection maintains its own
`history` list (`self.history` in `Connection`). The system prompt is pinned
at index 0; turns are appended after each LLM response and trimmed to
`MAX_LLM_HISTORY` messages (default 12). The `build_messages()` helper ensures
correct ordering and truncation.

**Interruption & barge-in:** When the user starts speaking again (VAD
"start" event), `_cancel_ongoing_turn()` aborts both the in-progress LLM
stream task and any in-flight TTS. This also applies to `stop_speak`,
`speak`, and connection disconnects.

**Thinking tag stripping & speech normalization:** The `sanitize_for_voice()`
function strips `<thinking>...</thinking>` blocks, Markdown formatting
(bold, italic, code, headings, lists) so edge-tts reads clean prose without
artifacts like "asterisk" or "markdown".

**Multi-provider fallback:** If the primary provider (Kilo AI) errors or times
out, the service falls back to Cloudflare Workers AI (`CF_MODEL`). If both
fail, the raw STT transcript is spoken as a last resort.

### Configuration

The LLM connects to the Kilo AI gateway (`LLM_BASE_URL`, default
`https://api.kilo.ai/api/gateway`) using an OpenAI-compatible client. The model
defaults to `nvidia/nemotron-3-super-120b-a12b:free`, identified by
`benchmark_kilo_models.py` as the fastest consistently-working free model
(~2.0 s per turn). `qwen/qwen3.8-27b:free` was the previous default but is now
intermittent (429 / empty responses).

Key knobs:

- `LLM_API_KEY` — Kilo AI gateway API token (required to enable the LLM path).
  Falls back to `GROQ_API_KEY` if unset.
- `LLM_MODEL` — model id on the gateway (default `nvidia/nemotron-3-super-120b-a12b:free`).
- `LLM_MAX_TOKENS` — cap on response length (default 128).
- `LLM_TEMPERATURE` — sampling temperature (default 0.7).
- `LLM_SYSTEM_PROMPT` — system prompt steering the assistant's persona and
  style. Keep responses short and TTS-friendly.
- `LLM_TIMEOUT_SEC` — per-call timeout before falling back to the raw transcript
  (default 5.0).
- `LLM_WARMUP` — if true, sends a tiny request at startup to prime the TLS/HTTP
  pool (default true).
- `MAX_LLM_HISTORY` — max conversation turns per connection, including system
  prompt (default 12).
- `LLM_MAX_HISTORY_TOKENS` — token budget for history trimming (default 2000).

### Cloudflare AI Fallback

Cloudflare Workers AI provides a secondary LLM provider for when Kilo AI is
rate-limited or unavailable. Several free `@cf/` models are accessible via the
standard Cloudflare API key:

| Model | Avg latency | Status |
|-------|-------------|--------|
| `@cf/meta/llama-3.2-3b-instruct` | ~880 ms | Fastest (selected as fallback) |
| `@cf/meta/llama-3.2-1b-instruct` | ~1375 ms | 1B parameter |
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | ~1130 ms | 70B, higher quality |
| `@cf/openai/gpt-oss-120b` | ~1510 ms | 120B parameter |
| `@cf/meta/llama-4-scout-17b-16e-instruct` | ~1643 ms | Llama 4 |

Tested models that are deprecated or not available:

| Model | Issue |
|-------|-------|
| `@cf/meta/llama-3.1-8b-instruct` | Deprecated (410), deprecated on 2026-05-30 |
| `@cf/qwen/qwen3.8-27b` | Returns empty content streams |
| Non-`@cf/` models (`deepseek/deepseek-chat`, `openai/gpt-4o-mini`, `cohere/command-r`) | Require adding payment to the Cloudflare account |

Configuration:

- `CF_API_KEY` — Cloudflare API token.
- `CF_ACCOUNT_ID` — Cloudflare account ID (32-char hex string).
- `CF_MODEL` — model id (default `@cf/meta/llama-3.2-3b-instruct`).
- `CF_ENABLED` — toggle the Cloudflare fallback (default true).
- `CF_BASE_URL` — Cloudflare API base URL.
- `CF_TIMEOUT_SEC` — per-call timeout for Cloudflare (default 10.0).

Note: The free `@cf/` models are available without payment on this account;
non-`@cf/` models require billing setup. See `benchmark_kilo_models.py` to
re-run the Kilo AI benchmark.

## Notes

- The VAD model is bundled with `silero-vad`; the Whisper model is downloaded
  to `~/.cache/huggingface` on first run.
- A single shared Whisper model is used across connections; transcriptions are
  serialized (thread-safe) via an `asyncio.Lock`. Tune `MAX_SEGMENT_SECONDS` to
  bound CPU cost per turn.
- Nothing at runtime needs an ffmpeg binary: TTS decodes MP3 with `miniaudio`
  and STT encodes Opus with `soundfile`.
