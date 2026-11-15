# AGENTS.md — Alvin Voice Assistant

A low-latency, real-time WebSocket voice assistant for Alvin AI. It ingests raw
16 kHz / 16-bit / mono PCM audio over WebSockets and runs the full voice pipeline:

**STT (Cloudflare Workers AI primary + Groq fallback) → LLM (Cloudflare Workers AI primary / Kilo AI gateway fallback + web search tool calling) → TTS (edge-tts)**

## Stack

- Python 3.12
- [FastAPI](https://fastapi.tiangolo.com) + Starlette WebSockets + uvicorn
- [silero-vad](https://github.com/snakers4/silero-vad) ONNX runtime VAD (streaming endpointing)
- Cloudflare Workers AI STT (`@cf/openai/whisper-large-v3-turbo`) — primary cloud STT
- Groq Python client for cloud STT fallback (`whisper-large-v3-turbo`), Opus encoded in-memory via `soundfile`
- Cloudflare Workers AI as primary LLM provider (`LLM_USE_CF_AS_PRIMARY=true`, default true)
- Kilo AI gateway (OpenAI-compatible) as LLM fallback
- [Tavily](https://tavily.com) web search tool calling (primary) + [DuckDuckGo](https://duckduckgo.com) key-free fallback
- [edge-tts](https://github.com/rany2/edge-tts) for TTS, [miniaudio](https://github.com/irliam/miniaudio) for MP3 decode
- [python-dotenv](https://github.com/theskumar/python-dotenv) for `.env` loading
- [pynput](https://github.com/moses-palmer/pynput) for global PTT hotkey (Shift+Z)
- Optional speaker identification uses SpeechBrain ECAPA-TDNN; install its
  dependencies with `pip install -r requirements-speaker.txt`.
- Test tooling: pytest, pytest-asyncio, ruff (in `requirements-dev.txt`)

## Layout

```
app/
  config.py      Settings loaded from environment/.env (dataclass, pydantic-free)
  vad.py         VADStreamDetector — streaming 32 ms frame VAD wrapper
  stt.py         STTService — Cloudflare Workers AI primary + Groq fallback
  llm.py         LLMService — Cloudflare Workers AI primary + Kilo AI gateway fallback, web search tool calling
  tts.py         Edge TTS — sentence-chunked 16 kHz PCM streaming
  speaker.py     Optional ECAPA-TDNN speaker enrollment and identification
  connection.py  Per-connection state machine (VAD feed, segments, JSON events, barge-in, PTT, echo)
  intents.py     Deterministic device-command registry (~20 intents: stop/mute/volume/lights/media/query_time/wake)
  itn.py         Inverse text normalization (spoken → written form)
  echo.py        Echo / self-trigger suppression
  briefing.py    Wake-up briefing: greeting, weather, tasks, email, optional LLM polish
  weather.py     Current conditions (Open-Meteo geocoding + MET Norway, no API key)
  email_source.py Read-only Gmail IMAP header fetch
  tasks.py       Google Tasks (read-only OAuth)
  main.py        FastAPI app: GET /health, WS /ws/transcribe
run.py          Server entrypoint (uvicorn). --mic also opens the live-mic client
alvin_hud.py    Desktop HUD frontend (PySide6): main window, live transcript,
                embedded system-settings dashboard (F2), camera preview (F3).
                `--hidden` starts it off-screen with the mic armed; it appears
                when "Alvin, wake up" is heard (used by start.bat).
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
                google_tasks_auth.py — one-time OAuth setup for Google Tasks
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
- The wake-up briefing is sent as `{"type": "briefing", "text": "...", "greeting": "...", "weather": {...}, "tasks": [...], "emails": [...], "errors": [...]}` before its TTS audio.
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
- `TRANSCRIPTION_QUEUE_MAXSIZE` (default 200) — caps the per-connection queue.

## Deterministic intent routing

`app.intents` matches fixed device commands on the ITN-normalized final transcript
with zero LLM involvement. Recognized commands are relayed as `command` events
and never enter the LLM conversation history.

Supported intents:

| Intent | Examples | Slots |
|--------|----------|-------|
| `stop` | "stop", "stop it", "stop the music", "shut up" | — (music stops also terminate the HUD's local player; the browser is only touched when playback fell back to it) |
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
| `open_system` | "open your system", "open system settings", "open your settings", "settings", "control panel" | — |
| `wake` | "hey alvin", "alvin?", "ok alvin" | — |
| `wake_up` | any sentence containing "wake up" ("wake up", "alvin wake up", "wake up now") | — (reveals the HUD fullscreen; starts one only for a non-HUD client) |
| `show_home` | "go back to homepage", "go home", "show homepage", "focus the app" | — (closes HUD pop-ups and brings the main window to the front) |
| `shutdown_laptop` | "turn off my laptop", "shut down the computer", "power off my pc" | — (action: powers off the local machine via the OS shutdown command) |
| `shutdown` | any utterance containing "goodbye" ("alvin goodbye", "I think it is time to say goodbye") | — (action: graceful server stop; never shuts down the OS) |

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

## Wake-up briefing

`app.briefing` builds the spoken briefing that follows the `wake_up` intent
("Alvin, wake up"). The text is assembled from templates outside the LLM, so it
works with small/free models and keeps a stable order; an optional LLM pass can
rewrite it. Each source is fetched independently, so a slow or failed source
adds an error line and the rest of the briefing is still spoken.

Order: time-aware greeting ("Good morning/afternoon/evening, sir" + date and
12-hour time) → current weather → open tasks → latest email → closing question.
The session receives a structured `briefing` event before its TTS audio:

```json
{"type": "briefing", "text": "...", "greeting": "...", "weather": {...}, "tasks": [...], "emails": [...], "errors": [...]}
```

The briefing is spawned as a background task from the `wake_up` handler
(`settings.briefing_enabled` and `settings.briefing_on_wake_up`), so gathering
does not block the audio consumer. It is cancelled if the turn is interrupted
or the connection closes. Weather and email responses are cached
(`BRIEFING_WEATHER_CACHE_SEC` 600 s, `BRIEFING_EMAIL_CACHE_SEC` 120 s) and
prefetched at startup when `BRIEFING_WARMUP=true`.

Sources:

- **Weather** (`app.weather`) — Open-Meteo geocoding resolves
  `BRIEFING_WEATHER_LOCATION`, MET Norway Locationforecast returns the current
  temperature and conditions. No API key or signup; CC BY 4.0 data.
- **Google Tasks** (`app.tasks`) — read-only `tasks.readonly` OAuth via
  `BRIEFING_TASKS_TOKEN_PATH` (create it with
  `python scripts/google_tasks_auth.py`). Returns the first
  `BRIEFING_TASKS_COUNT` open tasks from `BRIEFING_TASKS_LIST`.
- **Gmail** (`app.email_source`) — read-only IMAP with a Google app password
  (`BRIEFING_EMAIL_IMAP_USER` / `BRIEFING_EMAIL_IMAP_PASSWORD`). Fetches only
  `From`/`Subject`/`Date` with `BODY.PEEK`, never the body and never marks mail
  read. One-time codes, verification, and password-reset subjects are skipped
  when `BRIEFING_EMAIL_SKIP_SENSITIVE=true`.

Key knobs: `BRIEFING_ENABLED` (default true), `BRIEFING_ON_WAKE_UP` (default
true), `BRIEFING_WARMUP` (default true), `BRIEFING_LLM_POLISH` (default false),
`BRIEFING_LLM_MAX_TOKENS` (default 220), `BRIEFING_LLM_TIMEOUT_SEC` (default
10 s). If polish fails, the templated text is spoken.

## Web search tool calling

`app.llm` supports web search via tool calling. When `GEMINI_API_KEY` is set,
Google Gemini's built-in Google Search grounding is the primary search engine;
otherwise DuckDuckGo is used as a key-free fallback. Results are cached for 5 minutes. Up to 3 search rounds per LLM turn.

The automatic search gate recognizes the question words **what, how, when,
who, where, and if** anywhere in the utterance and requires the `web_search`
tool for those turns. A turn without a question word stays conversational and
is not offered the tool. Explicit search/no-search phrases, clear live-info
hints, and the small-talk/math exclusions still take precedence. When the model
actually calls `web_search`, the connection speaks the fixed acknowledgement
**“Let me check that on the web, sir.”** before returning the searched answer.

## YouTube playback

`play_youtube` resolves the song with yt-dlp on the server (cached 5 min). When
the client is the HUD and `YOUTUBE_ADFREE=true` (default), the server also
resolves a direct googlevideo stream URL (cached ~4 h); the HUD plays it in
VLC/mpv/ffplay, which carries no YouTube ads. With `YOUTUBE_AUDIO_ONLY=true`
(the default) VLC runs with `--intf dummy --no-video`, so audio plays in the
background and no player window opens; `--no-one-instance` keeps it from
attaching to a VLC window the user already has open. Set it to false to show
the video in a player window. If no local player is installed or stream
resolution fails, the normal watch page opens in the browser. `stop` (including
"stop the music") terminates the local player.

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
- `TTS_TRIM_LEADING_MS` (default 20) — audio held back at the START of a sentence to allow leading-silence removal.
- `TTS_TRIM_TRAILING_MS` (default 80) — audio held back at the END to allow trailing-silence removal.
- `TTS_SENTENCE_GAP_MS` (default 30) — digital silence between trimmed sentences.
- `TTS_DECODE_INTERVAL_BYTES` (default 4096) — new MP3 bytes buffered before re-decoding.
- `TTS_RETRIES` (default 5) — transient fetch retries before giving up on a sentence (6 total attempts).
- `TTS_RETRY_BACKOFF_MS` (default 200) — backoff between retries.
- `TTS_TIMEOUT_SEC` (default 20.0) — per-sentence edge-tts fetch timeout.
- `TTS_AUTO_SPEAK` (default false) — speak every final transcript automatically.
- `TTS_WARMUP` (default true) — synthesize one short phrase at startup.

## STT

`app.stt` routes every segment through a **fallback chain** with guards:

### Fallback order (configurable via `CF_STT_USE_AS_PRIMARY`)

| Setting | Chain |
|---------|-------|
| `CF_STT_USE_AS_PRIMARY=true` (default) | Cloudflare STT → Groq |
| `CF_STT_USE_AS_PRIMARY=false` | Groq → Cloudflare STT |
| `CF_STT_ENABLED=false` | Groq only |

**No local faster-whisper fallback exists** — only cloud backends are used.
The service is always ready (`is_ready` returns `True`) since there is no local model to load.

### Guards (applied in order for each backend)

1. **Silence gate** — `audio_dbfs()` computes the buffer's RMS level; anything
   under `SILENCE_THRESHOLD_DBFS` (default `-45 dBFS`) returns empty transcript
   immediately in ~25 µs. This stops Whisper answering silence with hallucinations
   ("you", "thank you", "Subtitles by...").

2. **In-memory Opus encoding** — Both backends encode mono audio as Ogg/Opus
   entirely in memory via `soundfile` (libsndfile ships inside the wheel).
   No ffmpeg binary is needed.

3. **Hallucination filter** — For segments near the silence gate
   (within `_HALLUCINATION_GUARD_MARGIN_DB = 6 dB` above threshold), transcripts
   matching known Whisper silence artifacts ("you", "thank you", "subscribe",
   "subtitles by", etc.) are dropped. Loud segments saying "thank you" are
   preserved — only near-silence artifacts are filtered.

4. **Duration-scaled circuit breaker** — `breaker_timeout()` gives each cloud
   attempt a budget of `max(GROQ_TIMEOUT_SEC, GROQ_TIMEOUT_BASE_SEC + duration * GROQ_TIMEOUT_SLOPE_SEC)`.
   Defaults: floor 5.0 s, base 2.0 s, slope 0.2 s/second of audio.
   Cloudflare STT uses fixed `CF_STT_TIMEOUT_SEC` (default 15 s).

5. **Fallback to next backend** — On timeout or any error, the next backend in
   the chain is tried. If all backends fail, empty transcript is returned.

### Warm-up

Both Cloudflare STT and Grok connections are warmed at startup
(`STTService.warm_up_cloud()`, disable with `GROQ_WARMUP=false` /
`CF_STT_WARMUP=false`) so the first turn does not pay TCP + TLS handshake.

### Key settings

- `SILENCE_THRESHOLD_DBFS` (default -45.0) — silence gate threshold
- `GROQ_API_KEY` — enables Groq backend
- `GROQ_MODEL` (default `whisper-large-v3-turbo`)
- `GROQ_TIMEOUT_SEC` (default 5.0) — floor timeout
- `GROQ_TIMEOUT_BASE_SEC` (default 2.0) — base for scaled timeout
- `GROQ_TIMEOUT_SLOPE_SEC` (default 0.2) — per-second slope
- `GROQ_INITIAL_PROMPT` — vocabulary steer for Whisper
- `CF_STT_ENABLED` (default true) — enable Cloudflare STT
- `CF_STT_MODEL` (default `@cf/openai/whisper-large-v3-turbo`)
- `CF_STT_TIMEOUT_SEC` (default 15.0)
- `CF_STT_USE_AS_PRIMARY` (default true) — Cloudflare STT first
- `CF_API_KEY` / `CF_ACCOUNT_ID` — required for Cloudflare STT

## LLM

`app.llm` provides the conversational brain of the voice assistant. After STT
produces a final transcript, the LLM generates a reply that is then spoken via
TTS. Without an `LLM_API_KEY` (Kilo AI) or `CF_API_KEY` (Cloudflare), the LLM
path is disabled and `TTS_AUTO_SPEAK` (if on) echoes the raw transcript instead.

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
the service falls back to the other provider. If both fail, the raw STT
transcript is spoken as a last resort.

### Configuration

The primary LLM provider is Cloudflare Workers AI (`LLM_USE_CF_AS_PRIMARY=true`, default).
Kilo AI gateway (`LLM_BASE_URL`, default `https://api.kilo.ai/api/gateway`) is the fallback.

Key knobs:

- `LLM_API_KEY` — Kilo AI gateway API token (fallback provider). Falls back to `NVIDIA_API_KEY` if unset.
- `LLM_BASE_URL` — Kilo AI gateway endpoint (default `https://api.kilo.ai/api/gateway`).
- `LLM_MODEL` — model id on the gateway (default `nvidia/nemotron-3.5-lightning:free`).
- `LLM_ENABLED` — master toggle for the LLM path (default true).
- `LLM_MAX_TOKENS` — cap on response length (default 200).
- `LLM_TEMPERATURE` — sampling temperature (default 0.7).
- `LLM_SYSTEM_PROMPT` — system prompt steering the assistant's persona and style. Keep responses short and TTS-friendly.
- `LLM_TIMEOUT_SEC` — per-call timeout before falling back (default 8.0).
- `LLM_WARMUP` — if true, sends a tiny request at startup to prime the TLS/HTTP pool (default true).
- `LLM_USE_CF_AS_PRIMARY` — when true, Cloudflare Workers AI is tried first (default true).
- `MAX_LLM_HISTORY` — max conversation turns per connection, including system prompt (default 12).
- `LLM_MAX_HISTORY_TOKENS` — token budget for history trimming (default 2000).

### Cloudflare Workers AI (primary LLM)

Free `@cf/` models accessible via standard Cloudflare API key:

| Model | Avg latency | Status |
|-------|-------------|--------|
| `@cf/meta/llama-3.2-3b-instruct` | ~880 ms | Fastest (used as primary) |
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
- `CF_BASE_URL` — Cloudflare API base URL (default `https://api.cloudflare.com/client/v4/accounts`).
- `CF_MODEL` — model id (default `@cf/meta/llama-3.2-3b-instruct`).
- `CF_ENABLED` — toggle the Cloudflare provider (default true).
- `CF_TIMEOUT_SEC` — per-call timeout for Cloudflare (default 6.0).

Note: The free `@cf/` models are available without payment on this account; non-`@cf/` models require billing setup.

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

- `console` (default) — press **Enter** to toggle listening (no admin required).
- `global` — hold **Shift+Z** to speak (requires admin on Windows,
  uses `pynput`). Audio is discarded unless the key is held.

Control frame `{"type": "talk", "active": true|false}` sets the gating state on
the connection. Knobs:

- `PUSH_TO_TALK` (default false) — wake-word gating state.
- `PUSH_TO_TALK_TIMEOUT_MS` (default 30000) — force listening off after this long.
- `PUSH_TO_TALK_ECHO_GATE` (default true) — run the half-duplex playback gate during a hold.

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
# one-click launcher: backend + hidden HUD (appears on "Alvin, wake up")
start.bat
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

- The VAD model is bundled with `silero-vad`; no Whisper model is downloaded locally (cloud-only STT).
- Transcriptions are serialized per backend via `asyncio.Lock` to respect API rate limits.
- Nothing at runtime needs an ffmpeg binary: TTS decodes MP3 with `miniaudio` and STT encodes Opus with `soundfile`.
- The service reads configuration from `.env` via python-dotenv; see `.env.example` for the full commented list of all environment variables.
- `VAD_THRESHOLD` (default 0.65) controls Silero VAD speech-detection sensitivity; raise to 0.7+ in noisy rooms to reduce false barge-in.
- `MIN_SILENCE_DURATION_MS` (default 500) controls end-of-turn detection; lower for faster response.
- `MAX_SEGMENT_SECONDS` (default 15.0) caps utterance length to bound memory and per-turn latency.