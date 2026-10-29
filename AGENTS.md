# AGENTS.md — Alvin Voice Assistant

A low-latency, real-time WebSocket voice assistant for Alvin AI. It ingests raw
16 kHz / 16-bit / mono PCM audio over WebSockets and runs the full voice pipeline:

**STT (Groq cloud primary + local faster-whisper fallback) → LLM (Kilo AI gateway primary / Cloudflare fallback + web search tool calling) → TTS (edge-tts)**

## Stack

- Python 3.12
- [FastAPI](https://fastapi.tiangolo.com) + Starlette WebSockets + uvicorn
- [faster-whisper](https://github.com/SYStran/faster-whisper) (ctranslate2) STT engine (local fallback)
- [silero-vad](https://github.com/snakers4/silero-vad) ONNX runtime VAD (streaming endpointing)
- Groq Python client for cloud STT (`whisper-large-v3-turbo`), Opus encoded in-memory via `soundfile`
- OpenAI-compatible client for the Kilo AI gateway LLM brain
- Cloudflare Workers AI as alternate / primary LLM provider (`LLM_USE_CF_AS_PRIMARY=true`)
- [Tavily](https://tavily.com) web search tool calling (primary) + [DuckDuckGo](https://duckduckgo.com) key-free fallback
- [edge-tts](https://github.com/rany2/edge-tts) for TTS, [miniaudio](https://github.com/irliam/miniaudio) for MP3 decode
- [python-dotenv](https://github.com/theskumar/python-dotenv) for `.env` loading
- [pynput](https://github.com/moses-palmer/pynput) for global PTT hotkey (Shift+Z)
- Optional speaker identification uses SpeechBrain ECAPA-TDNN; install its
  dependencies with `pip install -r requirements-speaker.txt`.

## Layout

```
app/
  config.py      Settings loaded from environment/.env (dataclass, pydantic-free)
  vad.py         VADStreamDetector — streaming 32 ms frame VAD wrapper
  stt.py         STTService — Groq cloud primary + local faster-whisper fallback
  llm.py         LLMService — Kilo AI gateway brain + Cloudflare fallback, web search tool calling
  tts.py         Edge TTS — sentence-chunked 16 kHz PCM streaming
  speaker.py     Optional ECAPA-TDNN speaker enrollment and identification
  connection.py  Per-connection state machine (VAD feed, segments, JSON events, barge-in, PTT, echo)
  intents.py     Deterministic device-command registry (~20 intents: stop/mute/volume/lights/media/query_time/wake)
  itn.py         Inverse text normalization (spoken → written form)
  echo.py        Echo / self-trigger suppression
  main.py        FastAPI app: GET /health, WS /ws/transcribe
run.py          Server entrypoint (uvicorn). --mic also opens the live-mic client
requirements.txt
.env / .env.example   Runtime config (override per deployment)
tests/
  conftest.py   Shared fixtures: forces tiny.en, GROQ_FALLBACK=false, collects offline
  unit/         Pure-logic tests, no Whisper / no network
  integration/  End-to-end tests against a live server subprocess
  manual/       Run-by-hand client scripts (excluded from collection)
  data/         sample_speech.wav fixture
scripts/        Benchmark + speaker service + voice-catalogue helpers
                bench_tts.py    — TTS benchmark matrix tool
                speaker_service.py — standalone PCM playback service with barge-in
                tts_voice_samples.py — list voices + render sample MP3s
                measure_stt_tts_latency.py — manual STT→TT latency benchmark
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
- `{"type": "talk", "active": true|false}` — push-to-talk gating. When active,
  audio is captured; otherwise audio is discarded (mic idle).

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
- Deterministic commands are emitted as `{"type": "command", "intent": "...", "slots": {...}}` events and never reach the LLM.
- TTS audio is sent as binary frames; `tts_end` / `tts_error` are JSON.

### Admission control

Both guards run **before** `accept()`, so a rejected client gets a denied
handshake (HTTP 403 from Starlette) instead of an open socket. Browser
WebSocket APIs do not expose the internal policy close code for a failed handshake.

- `API_KEY` — when set, every client must present it as `Authorization: Bearer
  <key>` or `X-API-Key: <key>`; `?token=<key>` also works for browsers that
  cannot set WebSocket headers, but query strings land in access/proxy logs.
  `SERVICE_AUTH_TOKEN` is a legacy alias. Empty (the default) leaves the
  endpoint open — fine for loopback, not for a shared host. Compared in
  constant time (`app/connection.py:verify_token`).
- `MAX_CONNECTIONS` (default 64) — live sockets are tracked by
  `app/connection.py:ConnectionManager`; over the cap the handshake is refused
  before acceptance. `/health` reports `auth_required`, `active_connections`, and
  `max_connections`.
- `MAX_FRAME_SIZE` (default 1048576 = 1 MB) — oversized frames close with `1009`.
- `TRANSCRIPTION_QUEUE_MAXSIZE` (default 100) — caps the per-connection queue.

## Deterministic intent routing

`app.intents` matches fixed device commands on the ITN-normalized final transcript
with zero LLM involvement. Recognized commands are relayed as `command` events
and never enter the LLM conversation history.

Supported intents:

| Intent | Examples | Slots |
|--------|----------|-------|
| `stop` | "stop", "stop it", "stop the music", "shut up" | — |
| `pause` | "pause", "pause the music" | — |
| `resume` | "resume", "resume playing" | — |
| `mute_mic` | "mute", "mute the mic", "alvin mute" | — (action: `disable_stt`) |
| `unmute_mic` | "unmute", "unmute the mic", "alvin unmute" | — (action: `enable_stt`) |
| `mute` | "mute the speaker", "mute the music" | — |
| `unmute` | "unmute the speaker", "unmute the music" | — |
| `set_volume` | "set the volume to 50", "volume to forty", "volume 60 percent", "set volume to max" | `volume` (int) |
| `volume_up` | "volume up", "louder", "turn it up", "make it louder" | — |
| `volume_down` | "volume down", "quieter", "turn it down", "make it quieter" | — |
| `lights_on` | "turn on the lights", "lights on", "switch on the lights" | — |
| `lights_off` | "turn off the lights", "lights off", "please turn off the lights" | — |
| `next` | "next track", "skip to the next", "next" | — |
| `previous` | "previous song", "previous", "go back" | — |
| `play` | "play the music", "play some music", "play" | — |
| `query_time` | "what time is it", "what's the time", "tell me the time" | — |
| `wake` | "hey alvin", "alvin?", "ok alvin" | — |

Rules:

- The transcript is normalized with `itn.normalize_transcript` first, so "set the
  volume to fifty" matches the same parser as "set the volume to 50".
- The *whole* utterance must be the command (optionally prefixed with "please" and
  the "Alvin" wake word). "Stop the music and play jazz" stays an LLM turn.
- Sentence punctuation STT engines append to short commands ("Mute.", "Alvin, mute.")
  is ignored for matching.
- The STT confidence floor (`INTENT_MIN_CONFIDENCE`) guards against
  mis-transcriptions executing hardware.
- Wake word: when `WAKE_WORD_REQUIRED` is on, commands only execute if armed by a
  configured wake word. A wake word alone ("hey alvin") returns a `wake` match
  with confirmation "Yes?" instead of reaching the LLM.
- Mic-control intents (`mute_mic` / `unmute_mic`) always speak their hardcoded
  confirmation (`MUTE_RESPONSE_TEXT` / `UNMUTE_RESPONSE_TEXT`), never gated on
  `INTENT_SPEAK_CONFIRMATION`.

## Web search tool calling

`app.llm` supports web search via tool calling. When `TAVILY_API_KEY` is set,
Tavily is the primary search engine; otherwise DuckDuckGo is used as a key-free
fallback. Results are cached for 5 minutes. Up to 3 search rounds per LLM turn.

## Speaker identification

Optional, **off by default**. Requires `pip install -r requirements-speaker.txt`
(speechbrain + torch + torchaudio, ~2 GB) and `SPEAKER_ENABLED=true`.

- Embeddings: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), 192-dim, CPU.
- Enrolled via `python scripts/speaker_enroll.py` (CLI). Profiles persisted to
  `SPEAKER_PROFILES_PATH` (default `speaker_profiles.pkl`) — keep this file out
  of version control (pickle is not safe to load from untrusted sources).
- Matching: cosine similarity vs. `SPEAKER_SIMILARITY_THRESHOLD` (default 0.75).
- Audio is buffered only while VAD reports active speech (`seg_start is not None`),
  capped at `SPEAKER_MAX_IDENTIFY_SEC` (default 5 s); shorter than
  `SPEAKER_MIN_IDENTIFY_SEC` (default 1 s) is discarded without an embedding call.
- Runs as a background task spawned from `_handle_final_utterance`. The first
  turn is never personalized; turns 2+ use `current_speaker_name` in the
  system prompt.
- Emits `{"type": "speaker_change", "speaker_id", "name", "similarity"}` on the
  WebSocket when a *different* known speaker is identified, or when the speaker
  context is cleared after 2 consecutive unknown turns.

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
`python scripts\bench_tts.py`.

Key knobs:

- `TTS_PREFETCH` (default 4) — sentences fetched ahead of the consumer.
- `TTS_SILENCE_THRESHOLD_DBFS` (default -45.0) — edge silence quieter than this is trimmed.
- `TTS_TRIM_LEADING_MS` (default 10) — audio held back at the START of a sentence to allow leading-silence removal.
- `TTS_TRIM_TRAILING_MS` (default 60) — audio held back at the END to allow trailing-silence removal.
- `TTS_SENTENCE_GAP_MS` (default 40) — digital silence between trimmed sentences.
- `TTS_DECODE_INTERVAL_BYTES` (default 8192) — new MP3 bytes buffered before re-decoding.
- `TTS_RETRIES` (default 2) — transient fetch retries before giving up on a sentence (3 total attempts).
- `TTS_RETRY_BACKOFF_MS` (default 150) — backoff between retries.
- `TTS_TIMEOUT_SEC` (default 6.0) — per-sentence edge-tts fetch timeout.
- `TTS_AUTO_SPEAK` (default false) — speak every final transcript automatically.
- `TTS_WARMUP` (default true) — synthesize one short phrase at startup.

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
4. **Local fallback** — `FALLBACK_MODEL` (`base.en`, int8) on timeout or any
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

**Multi-provider fallback:** If the primary provider errors or times out,
the service falls back to Cloudflare Workers AI (`CF_MODEL`). If both
fail, the raw STT transcript is spoken as a last resort.

### Configuration

The LLM connects to the Kilo AI gateway (`LLM_BASE_URL`, default
`https://api.kilo.ai/api/gateway`) using an OpenAI-compatible client. The model
defaults to `nvidia/nemotron-3.5-lightning:free`. `LLM_USE_CF_AS_PRIMARY`
(default `true`) makes Cloudflare Workers AI the primary provider instead,
with Kilo AI as the fallback.

Key knobs:

- `LLM_API_KEY` — Kilo AI gateway API token (required to enable the LLM path).
  Falls back to `NVIDIA_API_KEY` if unset.
- `LLM_BASE_URL` — Kilo AI gateway endpoint (default `https://api.kilo.ai/api/gateway`).
- `LLM_MODEL` — model id on the gateway (default `nvidia/nemotron-3.5-lightning:free`).
- `LLM_ENABLED` — master toggle for the LLM path (default true).
- `LLM_MAX_TOKENS` — cap on response length (default 300).
- `LLM_TEMPERATURE` — sampling temperature (default 0.7).
- `LLM_SYSTEM_PROMPT` — system prompt steering the assistant's persona and
  style. Keep responses short and TTS-friendly.
- `LLM_TIMEOUT_SEC` — per-call timeout before falling back to the raw transcript
  (default 15.0).
- `LLM_WARMUP` — if true, sends a tiny request at startup to prime the TLS/HTTP
  pool (default true).
- `LLM_USE_CF_AS_PRIMARY` — when true, Cloudflare Workers AI is tried first
  (default true).
- `MAX_LLM_HISTORY` — max conversation turns per connection, including system
  prompt (default 12).
- `LLM_MAX_HISTORY_TOKENS` — token budget for history trimming (default 2000).

### Cloudflare Workers AI

Cloudflare Workers AI provides an alternate LLM provider. Several free `@cf/`
models are accessible via the standard Cloudflare API key:

| Model | Avg latency | Status |
|-------|-------------|--------|
| `@cf/meta/llama-3.2-3b-instruct` | ~880 ms | Fastest (selected as fallback) |
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | ~1130 ms | 70B, higher quality |
| `@cf/meta/llama-3.2-1b-instruct` | ~1375 ms | 1B parameter |
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
- `CF_BASE_URL` — Cloudflare API base URL
  (default `https://api.cloudflare.com/client/v4/accounts`).
- `CF_MODEL` — model id (default `@cf/meta/llama-3.2-3b-instruct`).
- `CF_ENABLED` — toggle the Cloudflare provider (default true).
- `CF_TIMEOUT_SEC` — per-call timeout for Cloudflare (default 10.0).

Note: The free `@cf/` models are available without payment on this account;
non-`@cf/` models require billing setup.

## Echo / self-trigger suppression

`app.echo` prevents the assistant from hearing its own playback and replying to
itself in a loop:

- `EchoGuard` tracks `begin_speech()` / `end_speech()` per turn, with a cooldown
  (`ECHO_COOLDOWN_MS`, default 600 ms) after playback ends.
- `ECHO_MODE` (`filter` / `gate` / `off`, default `filter`):
  - `filter` — compare incoming transcripts against recent spoken history using
    `similarity()`; drop anything above `ECHO_SIMILARITY_THRESHOLD` (default 0.75).
  - `gate` — drop all mic audio while speaking (barge-in still works).
  - `off` — disable suppression entirely.
- `ECHO_HISTORY_SEC` (default 30.0) — seconds of spoken history retained.

## Push-to-talk (PTT)

`run.py --ptt-mode` supports two modes:

- `global` (default) — hold **Shift+Z** to speak (requires admin on Windows,
  uses `pynput`). Audio is discarded unless the key is held.
- `console` — press **Enter** to toggle listening (no admin required).

Control frame `{"type": "talk", "active": true|false}` sets the gating state on
the connection. Knobs:

- `PUSH_TO_TALK` (default false) — wake-word gating state.
- `PUSH_TO_TALK_TIMEOUT_MS` (default 30000) — force listening off after this long.
- `PUSH_TO_TALK_ECHO_GATE` (default true) — run the half-duplex playback gate
  during a hold.

## Scripts

- `scripts/bench_tts.py` — TTS benchmark matrix tool (TTFA, RTF, silence trim).
  A/B the two deterministic TTS knobs (`TTS_TRIM_LEADING_MS` and
  `TTS_DECODE_INTERVAL_BYTES`) with `--matrix-test --samples 20`.
- `scripts/speaker_service.py` — standalone PCM playback service that connects
  to the Alvin WebSocket server and plays audio with low-latency pre-buffering
  and barge-in support.
- `scripts/tts_voice_samples.py` — list every available edge-tts voice and render
  a playable sound sample to `voice_samples/`.
- `scripts/measure_stt_tts_latency.py` — manual benchmark: pipe the speech
  fixture through live STT, then TTS; measures end-to-end latency.

## Commands

```bash
# install
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
# run (server + mic client)
python run.py --mic
# run server only
python run.py
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
# standalone speaker service
python scripts\speaker_service.py
```

## Notes

- The VAD model is bundled with `silero-vad`; the Whisper model is downloaded
  to `~/.cache/huggingface` on first run.
- A single shared Whisper model is used across connections; transcriptions are
  serialized (thread-safe) via an `asyncio.Lock`. Tune `MAX_SEGMENT_SECONDS` to
  bound CPU cost per turn.
- Nothing at runtime needs an ffmpeg binary: TTS decodes MP3 with `miniaudio`
  and STT encodes Opus with `soundfile`.
- The service reads configuration from `.env` via python-dotenv; see
  `.env.example` for the full commented list of all environment variables.
