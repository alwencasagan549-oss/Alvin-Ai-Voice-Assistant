# Alvin AI Voice Assistant

**Current release: v1.1.2** — adds the wake-up briefing.

A low-latency, real-time voice assistant with a J.A.R.V.I.S.-style holographic
desktop UI. Speak into your microphone and Alvin transcribes you, thinks with a
large language model, and talks back — all over a live WebSocket, with
interruption (barge-in) support.

```
 You speak ──► VAD ──► STT (Cloudflare / Groq + local fallback) ──► LLM (Cloudflare / Kilo / NVIDIA) ──► TTS ──► Alvin speaks
              (silero)      (whisper-large-v3-turbo / faster-whisper)      (sentence-streamed)          (edge-tts → Piper, streamed)
```

The whole pipeline runs in Python over a single WebSocket. Sentences are spoken
as soon as the LLM finishes writing them, so short replies come back fast.

Alvin also includes:

- A PySide6 holographic HUD (`alvin_hud.py`) that can start hidden and appear
  when you say **"Alvin, wake up"**.
- A **wake-up briefing**: a time-aware greeting, current weather, open tasks,
  and the latest email, spoken as Alvin appears.
- A React dashboard (`alvin-ai-dashboard/`) for status, settings, and vision.
- Deterministic voice intents for media, microphone, volume, lights, apps,
  camera, YouTube, shutdown, and more.
- Ad-free YouTube playback through `yt-dlp` and a local player, with browser
  fallback.
- Web search tool calling, echo suppression, push-to-talk, and optional speaker
  identification.

---

## What you need

- **Python 3.12** — [download](https://www.python.org/downloads/). On Windows,
  tick **"Add python.exe to PATH"** during install.
- **A microphone and speakers.**
- **At least one LLM provider key**:
  - **Cloudflare Workers AI** — default primary (free tier)
  - **Kilo AI** — fallback provider
  - *(optional)* **NVIDIA NIM** — extra fallback
- **Speech-to-text keys are optional**: Groq is the default cloud provider,
  Cloudflare Workers AI can be primary instead, and local `faster-whisper`
  remains the offline fallback.
- **Node.js + npm** only if you want the React dashboard.
- **VLC, mpv, or ffplay** only if you want local ad-free YouTube playback;
  without one, videos open in the browser.

See [Getting the keys](#getting-the-keys) for the full list.

---

## Quick start (Windows)

> This guide assumes Windows with **PowerShell** or **Git Bash**. The commands
> work in both.

### 1. Install Python 3.12

From [python.org](https://www.python.org/downloads/). Verify:

```bash
python --version     # should print Python 3.12.x
```

### 2. Clone the repository

```bash
git clone https://github.com/alwencasagan549-oss/Alvin-Ai-Voice-Assistant.git
cd Alvin-Ai-Voice-Assistant
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls FastAPI, uvicorn, faster-whisper, silero-vad, edge-tts, miniaudio,
Groq, OpenAI, `sounddevice`, `websockets`, `PySide6`, `ddgs`, and `yt-dlp`.
No **ffmpeg** install is needed — MP3 decoding is done in-process by
`miniaudio` and Opus encoding by `soundfile`.

Speaker identification is optional and off by default. Its extras are large
(~2 GB) and are not part of `requirements.txt`. To enable it:

```bash
pip install speechbrain torch torchaudio
# then set SPEAKER_ENABLED=true in .env
```

### 4. Configure the environment

Copy the sample config and fill in your keys:

```bash
# Windows (PowerShell):
copy .env.example .env
# Windows (Git Bash) / macOS / Linux:
cp .env.example .env
```

Open `.env` and set at least one LLM provider:

```ini
# Cloudflare Workers AI (default primary)
CF_API_KEY=...
CF_ACCOUNT_ID=...

# and/or Kilo AI
LLM_API_KEY=...
```

`GROQ_API_KEY` is optional but recommended as the default speech-to-text
provider:

```ini
GROQ_API_KEY=gsk_....            # from https://console.groq.com/keys
```

Everything else has a working default. See [Getting the keys](#getting-the-keys)
and [Configuration](#configuration) for the full list.

> **Never commit `.env`** — it is in `.gitignore` for exactly that reason.

### 5. Start Alvin and talk to it

The normal Windows launch is:

```bat
start.bat
```

`start.bat` starts the backend and launches the **holographic HUD hidden**, so
nothing pops up on launch. The microphone stays live in the background. Say
**"Alvin, wake up"** to reveal the HUD.

The backend listens on `http://localhost:8000`; the HUD connects to
`ws://127.0.0.1:8000/ws/transcribe`.

Other launcher options:

```bash
# Visible HUD
python run-all.py --hud

# Hidden HUD (same behavior as start.bat)
python run-all.py --hud --hidden

# Backend only
python run-all.py

# Backend + terminal mic client, no HUD
python run.py --mic
```

Useful `run.py` flags:

```bash
python run.py --mic --voice en-US-AndrewNeural   # pick a different voice
python run.py --mic --vad-threshold 0.6          # less false barge-in
python run.py --mic --language en                # force STT language
python run.py --mic --ptt-mode global            # hold Shift+Z to speak
python run.py --mic --ptt-mode console           # press Enter to speak (no admin)
```

Push-to-talk (PTT): with `--ptt-mode global` you hold **Shift+Z** while
speaking; with `--ptt-mode console` you press **Enter** to toggle. While PTT is
active, audio is captured and processed; otherwise the mic is idle.

Wait for `Server ready.`, then start speaking. Say something, pause for about
half a second, and Alvin will answer out loud. Talk over Alvin at any time to
interrupt it. Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop.

<details>
<summary>Run the server and a client separately</summary>

```bash
# Terminal 1 — the server:
python run.py

# Terminal 2 — a live-microphone client (same machine):
python tests/manual/stream_mic.py
```

You can also point a client at a remote Alvin server:

```bash
python tests/manual/stream_mic.py --url ws://HOST:8000/ws/transcribe
```

</details>

### 6. Optional: run the React dashboard

The dashboard is **not** started automatically by `run-all.py`. Start it
manually:

```bash
cd alvin-ai-dashboard
npm install
npm run dev
```

It is served at `http://localhost:5173` with a Vite proxy to the backend.

You can also say **"Alvin, show dashboard"** — the backend starts the dev
server on demand (requires `npm`) and opens it in your browser.

### 7. Optional: verify with the test suite

```bash
pip install pytest pytest-asyncio
python -m pytest -q
```

Integration tests spin up a throwaway server and stream a bundled sample WAV,
so they need no microphone.

---

## Getting the keys

| Provider | Purpose | Where to get it | Env var |
|----------|---------|-----------------|---------|
| **Cloudflare Workers AI** | LLM (default primary) + optional STT | https://dash.cloudflare.com/ → API Tokens | `CF_API_KEY` + `CF_ACCOUNT_ID` |
| **Kilo AI** | LLM fallback | Kilo AI dashboard | `LLM_API_KEY` |
| **Groq** | Default speech-to-text | https://console.groq.com/keys | `GROQ_API_KEY` |
| **NVIDIA NIM** *(optional)* | Extra LLM provider | https://build.nvidia.com | `NVIDIA_API_KEY`, `NVIDIA_ENABLED=true` |
| **Tavily** *(optional)* | Web search tool | https://tavily.com | `TAVILY_API_KEY` |
| **DuckDuckGo** | Key-free web search fallback | None needed | — |
| **edge-tts** | Text-to-speech | None needed — online Microsoft voice endpoint | — |
| **Piper** | Local TTS fallback | None needed after model download | — |
| **yt-dlp** | YouTube search + stream resolution | None needed | — |

- **LLM providers** are tried in order. `LLM_PRIMARY_PROVIDER=cf` is the default,
  and the fallback chain is Cloudflare → NVIDIA → Kilo. At least one provider
  must be configured. If none is, the LLM is disabled: Alvin still transcribes
  you and, when `TTS_AUTO_SPEAK=true`, speaks the raw transcript back.
- **Speech-to-text** uses Groq as primary by default. Set
  `CF_STT_USE_AS_PRIMARY=true` to make Cloudflare Workers AI primary instead.
  Local `faster-whisper` is the offline fallback.
- **Web search** runs Tavily (when a key is configured) and DuckDuckGo
  concurrently, keeps whichever returns usable results, and caches results for
  5 minutes. Alvin searches when you explicitly ask ("search for…", "look it
  up", "google it") or when the utterance contains a live-info hint such as
  **latest, news, weather, or price**. Small talk and math questions do not
  trigger a search.
- **Cloudflare as LLM primary** is on by default. Set
  `LLM_PRIMARY_PROVIDER=kilo` to make Kilo AI primary.

---

## Voice commands

Commands are matched deterministically after inverse text normalization, so
they do not cost an LLM round-trip. `WAKE_WORD_REQUIRED=false` by default, so
you can say the command with or without "Alvin".

| Say | Does |
|-----|------|
| "Alvin, wake up" | Reveals the hidden HUD fullscreen and speaks the briefing |
| "stop" / "pause" / "resume" | Controls playback |
| "mute" / "unmute the mic" | Toggles microphone transcription |
| "volume up" / "set volume to 50" | Volume control |
| "turn on the lights" / "turn off the lights" | Lights intent |
| "play <song> on YouTube" | Resolves and plays a stream |
| "next" / "previous" | YouTube track control |
| "show dashboard" | Starts and opens the React dashboard |
| "open your system" / "open VS Code" / "open apps" | Opens the matching window |
| "open camera" / "close camera" / "what do you see" | Camera and vision |
| "what time is it" | Speaks the local time |
| "go back to homepage" | Returns the HUD to its main window |
| "goodbye" | Shuts down the assistant server |
| "turn off my laptop" | Starts the OS shutdown countdown |

The full registry lives in `app/intents.py`. The shutdown and laptop-shutdown
response texts are configurable in `.env`; other confirmations are constants in
that module.

---

## Wake-up briefing

Saying **"Alvin, wake up"** reveals the HUD and speaks a briefing in this shape:

> Good afternoon, sir. Today is Sunday, November 15, 2:35 PM. The weather in
> Cagayan de Oro is 31 degrees Celsius, partly cloudy. You have 2 open tasks:
> submit the quarterly report; call the bank. Your latest email is from Alice,
> about Project update. What's on your mind today, sir?

1. **Greeting** — "Good morning / afternoon / evening, sir", the date, and the
   time on a 12-hour clock.
2. **Weather** — current temperature and conditions from MET Norway.
3. **Open tasks** — the first few items in Google Tasks. If there are none,
   Alvin says "You don't have any available tasks right now, sir."
4. **Latest email** — sender and subject of the newest unread Gmail message. If
   there is none, Alvin says "You don't have any new email right now, sir."

All three sources are on by default. Every source is independent: if one fails
or is slow, Alvin says so and still speaks the rest. Weather and email are
cached (10 minutes and 2 minutes) and prefetched at startup, which keeps the
first wake-up fast.

### Weather (Open-Meteo + MET Norway)

No API key, signup, or billing. Set `BRIEFING_WEATHER_LOCATION` to your city;
Open-Meteo's geocoding endpoint resolves it to coordinates and MET Norway's
Locationforecast API returns the current temperature and conditions. Weather
data from MET Norway is used under CC BY 4.0.

### Gmail (read-only)

1. Turn on 2-Step Verification, then create an **App Password** at
   <https://myaccount.google.com/apppasswords>.
2. In `.env`:

   ```
   BRIEFING_EMAIL_ENABLED=true
   BRIEFING_EMAIL_IMAP_USER=you@gmail.com
   BRIEFING_EMAIL_IMAP_PASSWORD=the16characterapppassword
   ```

Alvin opens the mailbox read-only, fetches only `From`/`Subject`/`Date` with
`BODY.PEEK` (never the body), and never marks anything as read. One-time
codes, verification, and password-reset subjects are skipped by default.

### Google Tasks (OAuth)

1. In the Google Cloud console, enable the **Google Tasks API** and configure
   the OAuth consent screen (add your account as a test user).
2. Create an OAuth client of type **Desktop app** and save the JSON as
   `credentials/client_secret.json`.
3. Run:

   ```
   python scripts/google_tasks_auth.py
   ```

4. In `.env`, set `BRIEFING_TASKS_ENABLED=true`.

`credentials/` is gitignored; the token grants read-only `tasks.readonly`
access and is never used to modify your list.

### Polishing with the LLM

The briefing is composed from templates so it works without an LLM. Set
`BRIEFING_LLM_POLISH=true` to have the configured LLM rewrite the same facts
into one natural paragraph. If the model fails, Alvin falls back to the
template automatically.

---

## Configuration

All runtime settings come from `.env` (see `.env.example` for the full,
commented list). The ones you'll most often touch:

| Variable | Default | What it does |
|----------|---------|--------------|
| `HOST` | `127.0.0.1` | Server bind address. Set an API key before binding to a shared network. |
| `PORT` | `8000` | Server port |
| `TTS_VOICE` | `en-US-AndrewNeural` | Which voice Alvin speaks with |
| `VAD_THRESHOLD` | `0.65` | Speech-detection sensitivity. Raise it in noisy rooms / to reduce false barge-in |
| `SILENCE_THRESHOLD_DBFS` | `-45.0` | Buffers quieter than this never reach an STT backend |
| `GROQ_MODEL` | `whisper-large-v3-turbo` | Groq speech-to-text model |
| `GROQ_TIMEOUT_SEC` | `5.0` | Cloud STT breaker floor (seconds) |
| `CF_STT_ENABLED` | `true` | Enable Cloudflare Workers AI as an STT backend |
| `CF_STT_USE_AS_PRIMARY` | `false` | Make Cloudflare the primary STT provider |
| `LLM_PRIMARY_PROVIDER` | `cf` | `cf`, `nvidia`, or `kilo` |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning:free` | LLM model on the Kilo AI gateway |
| `LLM_BASE_URL` | `https://api.kilo.ai/api/gateway` | Kilo AI gateway endpoint |
| `LLM_MAX_TOKENS` | `200` | Reply length cap |
| `LLM_TIMEOUT_SEC` | `8.0` | Per-call LLM timeout |
| `TTS_PREFETCH` | `4` | Sentences fetched ahead of the consumer |
| `TTS_SENTENCE_GAP_MS` | `30` | Digital silence between trimmed sentences |
| `TTS_TIMEOUT_SEC` | `20.0` | Per-sentence edge-tts fetch timeout |
| `API_KEY` | *(empty)* | When set, clients must present it. Empty is suitable only for loopback use. |
| `INTENT_ENABLED` | `true` | Deterministic device-command routing |
| `WAKE_WORDS` | `alvin` | Words that arm the command router |
| `WAKE_WORD_REQUIRED` | `false` | Require the wake word before a command executes |
| `WAKE_WORD_WINDOW_SEC` | `0.0` | Seconds the command router stays armed after a wake word |
| `WAKE_SPEAK_ACK` | `true` | Speak "Yes?" when only the wake word is said |
| `ECHO_MODE` | `filter` | `"gate"` = drop mic audio during playback, `"filter"` = drop echoes, `"off"` = disable |
| `TAVILY_API_KEY` | *(empty)* | Tavily web search key; `GEMINI_API_KEY` is accepted as a fallback alias. Unset → DuckDuckGo only |
| `SPEAKER_ENABLED` | `false` | Enable speaker identification using ECAPA-TDNN embeddings |
| `YOUTUBE_ADFREE` | `true` | Resolve a direct stream for ad-free playback |
| `YOUTUBE_AUDIO_ONLY` | `true` | Audio-only playback (less bandwidth, faster start) |
| `BRIEFING_ENABLED` | `true` | Speak a time-aware briefing on "wake up" |
| `BRIEFING_WEATHER_LOCATION` | `Cagayan de Oro` | City for the current-weather line (Open-Meteo geocoding + MET Norway forecast, no key) |
| `BRIEFING_TASKS_ENABLED` | `true` | Read open Google Tasks |
| `BRIEFING_EMAIL_ENABLED` | `true` | Read the latest Gmail headers over IMAP |
| `BRIEFING_LLM_POLISH` | `false` | Rewrite the templated briefing with the LLM |
| `SHUTDOWN_DELAY_SEC` | `3.0` | Delay after the spoken goodbye before the server stops |
| `LAPTOP_SHUTDOWN_DELAY_SEC` | `10` | Windows shutdown countdown |

Full details, the WebSocket protocol, and the STT/LLM/TTS internals live in
[AGENTS.md](AGENTS.md).

### Where does the response time come from?

A turn is four serial stages — VAD end-of-turn wait (~0.5 s), STT (~1 s), LLM
(1–2 s, the biggest chunk), and TTS first audio (~1 s warm). `TTS_WARMUP`,
`GROQ_WARMUP`, `CF_STT_WARMUP`, and `LLM_WARMUP` (all on by default) remove the
one-time cold-start cost from your first real turn.

---

## Project layout

```
app/
  config.py       Settings loaded from the environment / .env
  vad.py          Streaming VAD wrapper (silero, 32 ms frames)
  stt.py          STTService — Groq / Cloudflare primary + local faster-whisper fallback
  llm.py          LLMService — Cloudflare / NVIDIA / Kilo failover, web search tool calling
  tts.py          edge-tts, sentence-chunked 16 kHz PCM streaming, Piper fallback
  speaker.py      Speaker identification — ECAPA-TDNN embeddings (enrollment + real-time ID)
  connection.py   Per-connection state machine (VAD feed, JSON events, barge-in, PTT, echo guard)
  intents.py      Deterministic device-command registry and wake-word router
  itn.py          Inverse text normalization (spoken → written form)
  echo.py         Echo / self-trigger suppression
  web_search.py   Tavily + DuckDuckGo search with a 5-minute cache
  youtube.py      yt-dlp search and stream resolution
  briefing.py     Wake-up briefing: greeting, weather, tasks, email, optional LLM polish
  weather.py      Current conditions (Open-Meteo geocoding + MET Norway, no API key)
  email_source.py Read-only Gmail IMAP header fetch
  tasks.py        Google Tasks (read-only OAuth)
  browser.py      Windows browser-tab automation for YouTube stop
  audio_pool.py   Bounded worker pools for Opus encode, MP3 decode, Piper, embeddings
  main.py         FastAPI app: GET /health, WS /ws/transcribe
alvin_hud.py      PySide6 holographic HUD (hidden-start wake-up flow)
run.py            Server entrypoint (uvicorn); --mic also opens the live-mic client
run-all.py        Combined launcher: backend + optional HUD / mic
start.bat         Windows one-click launcher (HUD hidden until "Alvin, wake up")
alvin-ai-dashboard/  React + Vite dashboard
tests/            unit / integration / manual-client suites
scripts/          Benchmarks, speaker CLI, voice catalogue, shared audio playback
.env.example      Sample configuration (copy to .env)
```

Scripts:

- `scripts/bench_tts.py` — TTS benchmark matrix tool (TTFA, RTF, silence trim)
- `scripts/speaker_service.py` — standalone PCM playback service with barge-in
- `scripts/speaker_enroll.py` — speaker enrollment CLI (enroll, list, delete, identify)
- `scripts/tts_voice_samples.py` — list voices + render sample MP3s to `voice_samples/`
- `scripts/measure_stt_tts_latency.py` — manual STT→TT latency benchmark
- `scripts/audio_playback.py` — shared low-latency playback engine
- `scripts/google_tasks_auth.py` — one-time OAuth setup for the Google Tasks
  briefing source (writes `credentials/google_tasks_token.json`)

---

## Notes & troubleshooting

- **First run downloads a Whisper model** (the local fallback) into
  `~/.cache/huggingface`. That one-time download happens before the first
  fallback transcription.
- **The HUD does not appear at launch.** That is intentional with `start.bat`:
  it starts hidden. Say **"Alvin, wake up"** or run `python run-all.py --hud` to
  show it immediately.
- **Muting the mic:** saying "mute" (or "unmute the mic") sets
  `mute_mic`/`unmute_mic` on the server. In `mute_mic` mode the client keeps the
  microphone stream open but the server stops sending its audio to the STT
  backend, instead running a local-only, cloud-free keyword pass over the
  incoming frames so "unmute" / "alvin unmute" still works without paying for a
  full transcription. There is no client-side keyword spotter; the detection
  lives on the server. Say "unmute" to resume transcription.
- **No audio out?** Check that your default playback device is correct and that
  nothing else has the device exclusive. The mic client prints `[event]` lines
  for every server message, which is the quickest way to see what's happening.
- **Dashboard does not load.** It is a separate Vite dev server. Run
  `npm install` and `npm run dev` in `alvin-ai-dashboard/`, or say
  "Alvin, show dashboard" and make sure `npm` is on `PATH`.
- **YouTube opens in a browser instead of playing locally.** Install VLC, mpv,
  or ffplay so `yt-dlp` can resolve and play the stream; otherwise Alvin falls
  back to the watch page.
- **Server not reachable from another machine?** Set `HOST=0.0.0.0`, configure
  `API_KEY`, check your firewall, and use the right `ws://` host:port.
- **Regenerate the voice previews** (not committed, ~14 MB):
  `python scripts/tts_voice_samples.py` → writes to `voice_samples/`.

---

## Roadmap

The **wake-up briefing** shipped in v1.1.2. v1.1.2 is the current release; this
README describes the current working tree.
