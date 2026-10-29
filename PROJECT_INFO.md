# Alvin AI Voice Assistant — Project Information

A low-latency, real-time voice assistant. Speak into your microphone and Alvin transcribes you, thinks with a large language model, and talks back — all over a live WebSocket, with interruption (barge-in) support.

```
You speak ──► VAD (silero) ──► STT (Groq cloud + local faster-whisper fallback)
            │                                      │
            ▼                                      ▼
      LLM (Kilo AI / Cloudflare) ◄──────────────┘
            │
            ▼
      TTS (edge-tts) ──► Alvin speaks
```

Sentences are spoken as soon as the LLM finishes writing them, so short replies come back fast.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Web framework | FastAPI + Starlette WebSockets + uvicorn |
| VAD | silero-vad (ONNX runtime, streaming 32 ms frames) |
| STT (cloud) | Groq `whisper-large-v3-turbo` (Opus encoded in-memory via `soundfile`, no ffmpeg) |
| STT (local fallback) | faster-whisper (ctranslate2), `base.en` int8 |
| LLM (primary) | Kilo AI gateway (`https://api.kilo.ai/api/gateway`), OpenAI-compatible client, model `nvidia/nemotron-3.5-lightning:free` |
| LLM (fallback) | Cloudflare Workers AI (`@cf/meta/llama-3.2-3b-instruct`) |
| TTS | edge-tts (Microsoft online), MP3 decoded with `miniaudio`, sentence-chunked 16 kHz PCM streaming |
| Web search | Tavily (primary) + DuckDuckGo (key-free fallback) |
| Environment | `python-dotenv` for `.env` loading |
| PTT hotkey | `pynput` (global Shift+Z) |
| Audio I/O | `sounddevice` |
| WebSocket client lib | `websockets` |

---

## Project Layout

```
app/
  config.py       Settings loaded from environment/.env (dataclass, pydantic-free)
  vad.py          VADStreamDetector — streaming 32 ms frame VAD wrapper
  stt.py          STTService — Groq cloud primary + local faster-whisper fallback
  llm.py          LLMService — Kilo AI gateway brain + Cloudflare fallback, web search tool calling
  tts.py          edge-tts, sentence-chunked 16 kHz PCM streaming (with retry)
  connection.py   Per-connection state machine (VAD feed, JSON events, barge-in, PTT, echo guard)
  intents.py      Deterministic device-command registry (~20 intents)
  itn.py          Inverse text normalization (spoken → written form)
  echo.py         Echo / self-trigger suppression
  web_search.py   Web search tool calling (Tavily + DuckDuckGo)
  main.py         FastAPI app: GET /health, WS /ws/transcribe
  __init__.py
run.py            Server entrypoint (uvicorn); --mic also opens the live-mic client
.env              Runtime configuration (created from .env.example, not committed)
.env.example      Sample configuration (committed)
requirements.txt  Python dependencies
scripts/
  bench_tts.py                  — TTS benchmark matrix tool (TTFA, RTF, silence trim)
  speaker_service.py            — standalone PCM playback service with barge-in
  tts_voice_samples.py          — list voices + render sample MP3s to voice_samples/
  measure_stt_tts_latency.py    — manual STT→TTS latency benchmark
tests/
  conftest.py                   — shared fixtures (forces tiny.en, GROQ_FALLBACK=false, offline)
  unit/                         — pure-logic tests, no Whisper / no network
  integration/                  — end-to-end tests against a live server subprocess
  manual/                       — run-by-hand client scripts (excluded from collection)
  data/                         — sample_speech.wav fixture
docs/
  PRESS_TO_TALK_PRD.md          — PTT product requirements
.github/
  workflows/
    python-publish.yml          — GitHub Actions for PyPI publishing
.gitignore
```

---

## Configuration (.env)

All runtime settings come from `.env`. Below is the complete current state as on disk.

### Server

| Variable | Value | Description |
|----------|-------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |

### VAD / Audio

| Variable | Value | Description |
|----------|-------|-------------|
| `WHISPER_LANGUAGE` | `en` | Whisper language hint |
| `VAD_THRESHOLD` | `0.5` | Speech-detection sensitivity |
| `MIN_SILENCE_DURATION_MS` | `500` | Silence needed to end a turn |
| `SPEECH_PAD_MS` | `30` | Padding around speech segments |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `PARTIAL_INTERVAL_MS` | `1000` | Partial transcript interval |
| `MIN_SEGMENT_SECONDS` | `0.5` | Minimum segment length |
| `MAX_SEGMENT_SECONDS` | `20` | Maximum segment length |

### Admission Control

| Variable | Value | Description |
|----------|-------|-------------|
| `MAX_CONNECTIONS` | `64` | Max live WebSocket connections |
| `MAX_FRAME_SIZE` | `1048576` (1 MB) | Max size of a single WebSocket binary frame |
| `TRANSCRIPTION_QUEUE_MAXSIZE` | `100` | Per-connection transcription queue cap |

### Silence Gate

| Variable | Value | Description |
|----------|-------|-------------|
| `SILENCE_THRESHOLD_DBFS` | `-55.0` | Buffers quieter than this (dBFS RMS) never reach a backend |

### STT (Speech-to-Text)

| Variable | Value | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | *(set)* | Groq API key from https://console.groq.com/keys |
| `GROQ_MODEL` | `whisper-large-v3-turbo` | Groq Whisper model |
| `GROQ_LANGUAGE` | `en` | Groq language |
| `GROQ_FALLBACK` | `true` | Enable local fallback on Groq failure |
| `GROQ_TIMEOUT_SEC` | `1.5` | Cloud STT breaker floor (seconds) |
| `GROQ_TIMEOUT_BASE_SEC` | `1.0` | Base timeout for duration-based breaker |
| `GROQ_TIMEOUT_SLOPE_SEC` | `0.12` | Slope per audio second for breaker |
| `GROQ_INITIAL_PROMPT` | `Cagayan de Oro City, order IDs like 4829-B, email addresses like test@domain.org.` | Vocabulary/formatting steer |
| `GROQ_WARMUP` | `true` | Prime TLS connection pool at startup |

### Local STT Fallback

| Variable | Value | Description |
|----------|-------|-------------|
| `FALLBACK_MODEL` | `base.en` | Local faster-whisper model |
| `PRELOAD_FALLBACK` | `true` | Preload fallback model at startup |
| `CPU_THREADS` | `4` | CPU threads for local Whisper |

### TTS (Text-to-Speech)

| Variable | Value | Description |
|----------|-------|-------------|
| `TTS_VOICE` | `en-US-AndrewNeural` | edge-tts voice |
| `TTS_PREFETCH` | `4` | Sentences fetched ahead of consumer |
| `TTS_SILENCE_THRESHOLD_DBFS` | `-45.0` | Edge silence trimmed below this (dBFS) |
| `TTS_TRIM_LEADING_MS` | `10` | Audio held back at the START of a sentence for leading-silence removal |
| `TTS_TRIM_TRAILING_MS` | `60` | Audio held back at the END for trailing-silence removal |
| `TTS_SENTENCE_GAP_MS` | `40` | Digital silence between trimmed sentences |
| `TTS_DECODE_INTERVAL_BYTES` | `8192` | MP3 bytes buffered before re-decoding |
| `TTS_RETRIES` | `3` | Transient fetch retries |
| `TTS_RETRY_BACKOFF_MS` | `150` | Backoff between retries (ms) |
| `TTS_TIMEOUT_SEC` | `6.0` | Per-sentence edge-tts fetch timeout |
| `TTS_AUTO_SPEAK` | `false` | Speak every final transcript automatically |
| `TTS_WARMUP` | `true` | Synthesize one short phrase at startup |

### Admission / Auth

| Variable | Value | Description |
|----------|-------|-------------|
| `API_KEY` | *(empty)* | Shared secret for WebSocket; empty = open |
| `SERVICE_AUTH_TOKEN` | *(empty)* | Legacy alias for `API_KEY` |

### LLM Brain (Kilo AI Gateway)

| Variable | Value | Description |
|----------|-------|-------------|
| `LLM_API_KEY` | *(set)* | Kilo AI gateway API token |
| `LLM_BASE_URL` | `https://api.kilo.ai/api/gateway` | Kilo AI gateway endpoint |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning:free` | LLM model on the gateway |
| `LLM_ENABLED` | `true` | Master toggle for LLM path |
| `LLM_USE_CF_AS_PRIMARY` | `true` | Cloudflare as primary, Kilo as fallback |
| `LLM_MAX_TOKENS` | `300` | Reply length cap |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_SYSTEM_PROMPT` | `You are Alvin, a helpful, concise AI voice assistant created by Alwin T Casagan, a Full-Stack Developer and System Architect with 3+ years of experience who has built 20+ projects across a 10+ tech stack. Alwin is available for work and can be found on GitHub at alwincasagan549-oss. Keep responses short and natural, suitable for text-to-speech. Do not use emojis, markdown, or lists. Answer in plain text only.` | System prompt steering persona and style |
| `LLM_TIMEOUT_SEC` | `15.0` | Per-call LLM timeout |
| `LLM_WARMUP` | `true` | Prime TLS/HTTP pool at startup |

### Cloudflare Workers AI (Fallback / Primary)

| Variable | Value | Description |
|----------|-------|-------------|
| `CF_API_KEY` | *(set)* | Cloudflare API token |
| `CF_ACCOUNT_ID` | `(set — 32 hex chars)` | Cloudflare account ID |
| `CF_BASE_URL` | `https://api.cloudflare.com/client/v4/accounts` | Cloudflare API base URL |
| `CF_MODEL` | `@cf/meta/llama-3.2-3b-instruct` | Cloudflare model ID |
| `CF_ENABLED` | `true` | Toggle Cloudflare provider |
| `CF_TIMEOUT_SEC` | `10.0` | Per-call timeout for Cloudflare |

### Conversation Memory

| Variable | Value | Description |
|----------|-------|-------------|
| `MAX_LLM_HISTORY` | `12` | Max messages per connection (incl. system prompt) |
| `LLM_MAX_HISTORY_TOKENS` | `2000` | Token budget for history trimming |

### Intent Routing (Deterministic Device Commands)

| Variable | Value | Description |
|----------|-------|-------------|
| `INTENT_ENABLED` | `true` | Deterministic command matching |
| `INTENT_SPEAK_CONFIRMATION` | `false` | Speak confirmation after matched command |
| `INTENT_MIN_CONFIDENCE` | `0.5` | STT confidence floor for command execution |

### Wake Word

| Variable | Value | Description |
|----------|-------|-------------|
| `WAKE_WORDS` | `alvin` | Words that arm the command router |
| `WAKE_WORD_REQUIRED` | `false` | Require wake word before command executes |
| `WAKE_WORD_WINDOW_SEC` | `0.0` | Seconds of armed state after wake word |
| `WAKE_SPEAK_ACK` | `true` | Speak "Yes?" when wake word said alone |

### Push-to-Talk (PTT)

| Variable | Value | Description |
|----------|-------|-------------|
| `PUSH_TO_TALK` | `false` | Wake-word gating state |
| `PUSH_TO_TALK_TIMEOUT_MS` | `30000` | Force listening off after this long |
| `PUSH_TO_TALK_ECHO_GATE` | `true` | Run half-duplex playback gate during hold |

### Echo / Self-Trigger Suppression

| Variable | Value | Description |
|----------|-------|-------------|
| `ECHO_MODE` | `filter` | `filter` / `gate` / `off` |
| `ECHO_SIMILARITY_THRESHOLD` | `0.75` | Self-echo detection similarity threshold |
| `ECHO_HISTORY_SEC` | `30.0` | Seconds of spoken history retained |
| `ECHO_COOLDOWN_MS` | `600` | Cooldown after playback ends (ms) |

### Web Search

| Variable | Value | Description |
|----------|-------|-------------|
| `TAVILY_API_KEY` | *(set)* | Tavily API key from https://tavily.com |
| `DEBUG` | `false` | Debug flag for logging search decisions |

---

## Speaker Identification

Optional, **off by default**. Requires `pip install -r requirements-speaker.txt`
(speechbrain + torch + torchaudio, ~2 GB) and `SPEAKER_ENABLED=true`.

| Variable | Value | Description |
|----------|-------|-------------|
| `SPEAKER_ENABLED` | `false` | Enable speaker identification |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for speaker matching (0–1) |
| `SPEAKER_MIN_IDENTIFY_SEC` | `1.0` | Minimum audio duration for identification (seconds) |
| `SPEAKER_MAX_IDENTIFY_SEC` | `5.0` | Maximum audio duration for identification (seconds) |
| `SPEAKER_PROFILES_PATH` | `speaker_profiles.pkl` | Path to persist speaker profiles |

- **Embeddings:** ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), 192-dim, CPU.
- **Enrollment:** `python scripts/speaker_enroll.py` — profiles persisted to `SPEAKER_PROFILES_PATH` (pickle is not safe to load from untrusted sources; keep it out of version control).
- **Matching:** cosine similarity vs. `SPEAKER_SIMILARITY_THRESHOLD` (default 0.75).
- **Buffering:** audio is captured only during active speech (`seg_start is not None`), capped at `SPEAKER_MAX_IDENTIFY_SEC`; shorter than `SPEAKER_MIN_IDENTIFY_SEC` is discarded.
- **Personalization:** identification runs as a background task spawned from `_handle_final_utterance`. The first turn is never personalized; turns 2+ use `current_speaker_name` in the system prompt.
- **WebSocket events:** emits `{"type": "speaker_change", "speaker_id", "name", "similarity"}` when a *different* known speaker is identified, or when the speaker context is cleared after 2 consecutive unknown turns.
- `/health` reports speaker `enabled`, `min_identify_sec`, and `max_identify_sec` (the similarity threshold is intentionally not exposed).

---

## WebSocket Protocol (`ws://<host>:<port>/ws/transcribe`)

### Input — Binary Frames

Little-endian int16 PCM audio at 16 kHz, mono. Frames of any size are accepted and buffered.

### Input — Text Frames (JSON Control Messages)

| Message | Effect |
|---------|--------|
| `{"type": "config", "language": "en"}` | Override language (`"auto"`/`""` → auto-detect). `"voice"` overrides TTS voice. |
| `{"type": "stop"}` | Flush current utterance as final event, then close. |
| `{"type": "ping"}` | → `{"type": "pong"}` |
| `{"type": "speak", "text": "...", "voice": "..."}` | Synthesize and stream PCM back as binary frames. Emits `tts_end` / `tts_error`. |
| `{"type": "stop_speak"}` | Cancel in-flight synthesis; socket stays open. |
| `{"type": "talk", "active": true\|false}` | Push-to-talk gating. When active, audio is captured; otherwise discarded. |

### Output — JSON Events

| Event | Meaning |
|-------|---------|
| `{"transcript": "...", "is_partial": true, "is_final": false, "confidence": 0.8}` | Interim hypothesis |
| `{"transcript": "...", "is_partial": false, "is_final": true, "confidence": 0.8, "llm_enabled": true}` | End-of-turn final transcript |
| `{"type": "llm_response", "text": "...", "model": "..."}` | LLM reply |
| `{"type": "llm_error", "error": "..."}` | LLM failure (TTS falls back to raw transcript) |
| `{"type": "command", "intent": "...", "slots": {...}}` | Deterministic command (never reaches LLM) |
| `{"type": "tts_end"}` | TTS finished |
| `{"type": "tts_error", ...}` | TTS failure for a sentence |
| `{"type": "ptt_state", "active": true\|false}` | PTT state change |

---

## Deterministic Intent Routing

`app.intents` matches fixed device commands on the ITN-normalized final transcript with zero LLM involvement. Recognized commands are relayed as `command` events and never enter LLM conversation history.

**Supported intents:** `stop`, `pause`, `resume`, `mute_mic`, `unmute_mic`, `mute`, `unmute`, `set_volume`, `volume_up`, `volume_down`, `lights_on`, `lights_off`, `next`, `previous`, `play`, `query_time`, `wake`

---

## Admission Control (runs before `accept()`)

| Guard | Default | Behavior |
|-------|---------|----------|
| `API_KEY` | empty | When set, clients must present it as `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| `MAX_CONNECTIONS` | `64` | Over cap → handshake refused with close code `1008` |
| `MAX_FRAME_SIZE` | `1048576` | Oversized frames close with `1009` |
| `TRANSCRIPTION_QUEUE_MAXSIZE` | `100` | Caps per-connection transcription queue |

`/health` reports `auth_required`, `active_connections`, and `max_connections`.

---

## Echo / Self-Trigger Suppression

`app.echo` prevents Alvin from hearing its own playback and replying to itself:

- `ECHO_MODE=filter` — compare incoming transcripts against recent spoken history; drop anything above `ECHO_SIMILARITY_THRESHOLD=0.75`
- `ECHO_MODE=gate` — drop all mic audio while speaking
- `ECHO_MODE=off` — disable suppression
- `ECHO_COOLDOWN_MS=600` — cooldown after playback ends
- `ECHO_HISTORY_SEC=30.0` — seconds of spoken history retained

---

## Push-to-Talk (PTT)

`run.py --ptt-mode` supports two modes:

| Mode | How it works | Admin required |
|------|-------------|----------------|
| `global` | Hold **Shift+Z** to speak | Yes (on Windows, uses `pynput`) |
| `console` | Press **Enter** to toggle listening | No |

Control frame `{"type": "talk", "active": true\|false}` sets gating state on the connection.

---

## How to Run

From `F:\Alvin-Ai-Voice-Assistant\`:

```powershell
# Server + live mic client (one command, does everything):
.venv\Scripts\python.exe run.py --mic

# With console PTT (press Enter to toggle mic, no admin):
.venv\Scripts\python.exe run.py --mic --ptt-mode console

# With global PTT (hold Shift+Z to speak, needs admin on Windows):
.venv\Scripts\python.exe run.py --mic --ptt-mode global

# Server only (no mic):
.venv\Scripts\python.exe run.py

# Health check:
curl http://localhost:8000/health
```

**Useful flags:**
- `--voice en-US-AndrewNeural` — pick a different TTS voice
- `--vad-threshold 0.6` — less false barge-in in noisy rooms
- `--language en` — force STT language

Press `Ctrl+C` to stop.

---

## Running Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Integration tests spin up a throwaway server and stream a bundled sample WAV, so they need no microphone.

---

## Requirements

- **Python 3.12** (with "Add python.exe to PATH" on Windows)
- **Microphone and speakers**
- **API keys** (all free to start): Groq (STT), Kilo AI (LLM), optional Cloudflare (LLM fallback), optional Tavily (web search)
- **No ffmpeg needed** — MP3 decoding is done in-process by `miniaudio`; Opus encoding by `soundfile`
- **First run** downloads the local Whisper model into `~/.cache/huggingface`

---

## Creator

Alvin was created by **Alwin T. Casagan**, Full-Stack Developer & System Architect, available for work, GitHub: [alwincasagan549-oss](https://github.com/alwincasagan549-oss).
