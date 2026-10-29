# Alvin AI Voice Assistant

A low-latency, real-time voice assistant. Speak into your microphone and Alvin
transcribes you, thinks with a large language model, and talks back — all
over a live WebSocket, with interruption (barge-in) support.

```
 You speak ──► VAD ──► STT (Groq cloud + local fallback) ──► LLM (Kilo AI / Cloudflare) ──► TTS ──► Alvin speaks
              (silero)            (whisper-large-v3-turbo / faster-whisper)   (sentence-streamed)   (edge-tts, streamed)
```

The whole pipeline runs in Python over a single WebSocket. Sentences are spoken
as soon as the LLM finishes writing them, so short replies come back fast.

---

## What you need

- **Python 3.12** — [download](https://www.python.org/downloads/). On Windows,
  tick **"Add python.exe to PATH"** during install.
- **A microphone and speakers.**
- **API keys** (all free to start — see [Getting the keys](#getting-the-keys)):
  - **Groq** — primary speech-to-text
  - **Kilo AI** — conversational LLM brain (get your key from the Kilo AI dashboard)
  - *(optional)* **Cloudflare Workers AI** — alternate LLM provider; becomes
    primary when `LLM_USE_CF_AS_PRIMARY=true`
  - *(optional)* **Tavily** — web search tool calling; DuckDuckGo is the key-free fallback
- Speech-to-text and text-to-speech also fall back to local / key-less paths,
  so the assistant still works even if one cloud provider is down.

---

## Run it on a new laptop (step by step)

> This guide assumes Windows with **PowerShell** or **Git Bash**. Adjust the
> command separator for your shell (`&&` works in both for these commands).

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
Groq, OpenAI, `sounddevice`, and `websockets`. No **ffmpeg** install is needed —
MP3 decoding is done in-process by `miniaudio` and Opus encoding by `soundfile`.

Speaker identification is optional and off by default. To enable it, install its
separate dependencies with `pip install -r requirements-speaker.txt`, then set
`SPEAKER_ENABLED=true` in `.env`.

### 4. Configure the environment

Copy the sample config and fill in your keys:

```bash
# Windows (PowerShell):
copy .env.example .env
# Windows (Git Bash) / macOS / Linux:
cp .env.example .env
```

Open `.env` and set at least the keys you plan to use:

```ini
GROQ_API_KEY=gsk_....            # from https://console.groq.com/keys
LLM_API_KEY=...                  # from the Kilo AI dashboard
```

Everything else has a working default. See [Getting the keys](#getting-the-keys)
and [Configuration](#configuration) for the full list.

> **Never commit `.env`** — it is in `.gitignore` for exactly that reason.

### 5. Start Alvin and talk to it

One command boots the server **and** opens the live microphone client:

```bash
python run.py --mic
```

#### Run the backend + dashboard together (one command)

To start **both** the FastAPI backend and the React dashboard UI together:

```bash
# Windows (PowerShell):
python run-all.py

# macOS / Linux:
./start-all.sh   # or: python3 run-all.py
```

- Backend starts on `http://localhost:8000` and is waited on until `/health` responds.
- Dashboard starts on `http://localhost:5173` (served by Vite with a proxy to the backend).
- Press `Ctrl+C` to stop both cleanly.

You can also start them separately if you prefer:

```bash
# Terminal 1 — backend:
python run.py

# Terminal 2 — dashboard:
cd alvin-ai-dashboard
npm install
npm run dev
```

Wait for `Server ready.`, then just start speaking. Say something, pause for
about half a second, and Alvin will answer out loud. Talk over Alvin at any
time to interrupt it. Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop.

Useful flags:

```bash
python run.py --mic --voice en-US-AndrewNeural   # pick a different voice
python run.py --mic --vad-threshold 0.6          # less false barge-in
python run.py --mic --language en                # force STT language
python run.py --mic --ptt-mode global            # hold Shift+Z to speak
python run.py --mic --ptt-mode console           # press Enter to speak (no admin)
```

Push-to-talk (PTT): with `--ptt-mode global` you hold **Shift+Z** while speaking;
with `--ptt-mode console` you press **Enter** to toggle. While PTT is active,
audio is captured and processed; otherwise the mic is idle.

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

### 6. (Optional) Verify with the test suite

```bash
python -m pytest -q
```

Integration tests spin up a throwaway server and stream a bundled sample WAV,
so they need no microphone.

---

## Getting the keys

| Provider | Purpose | Where to get it | Env var |
|----------|---------|-----------------|---------|
| **Groq** | Primary speech-to-text | https://console.groq.com/keys | `GROQ_API_KEY` |
| **Kilo AI** | Conversational LLM brain | Kilo AI dashboard | `LLM_API_KEY` (alias `NVIDIA_API_KEY`) |
| **Cloudflare Workers AI** *(optional)* | Alternate LLM provider; primary when `LLM_USE_CF_AS_PRIMARY=true` | https://dash.cloudflare.com/ → API Tokens | `CF_API_KEY` + `CF_ACCOUNT_ID` |
| **Tavily** *(optional)* | Web search tool calling | https://tavily.com | `TAVILY_API_KEY` |
| **edge-tts** | Text-to-speech | None needed — uses Microsoft's online voice endpoint | — |
| **DuckDuckGo** *(optional)* | Key-free web search fallback | None needed | — |

- **If you don't set `LLM_API_KEY`**, the LLM is disabled: Alvin will still
  transcribe you, and (if `TTS_AUTO_SPEAK=true`) simply speak the raw transcript
  back instead of a generated reply.
- **If you don't set `GROQ_API_KEY`**, every transcription falls back to the
  local `faster-whisper` model (slower, but works offline).
- **Web search** is enabled when `TAVILY_API_KEY` is set; otherwise DuckDuckGo
  is used as a key-free fallback.
- **Cloudflare-as-primary** is on by default (`LLM_USE_CF_AS_PRIMARY=true`),
  so Cloudflare is tried first and Kilo AI is the fallback. Set it to `false`
  to make Kilo AI primary.

---

## Configuration

All runtime settings come from `.env` (see `.env.example` for the full,
commented list). The ones you'll most often touch:

| Variable | Default | What it does |
|----------|---------|--------------|
| `HOST` | `127.0.0.1` | Server bind address. Set an API key before binding to a shared network. |
| `PORT` | `8000` | Server port |
| `TTS_VOICE` | `en-US-AndrewNeural` | Which voice Alvin speaks with |
| `VAD_THRESHOLD` | `0.5` | Speech-detection sensitivity. Raise to `0.6+` in noisy rooms / to reduce false barge-in |
| `SILENCE_THRESHOLD_DBFS` | `-55.0` | Buffers quieter than this never reach an STT backend |
| `GROQ_TIMEOUT_SEC` | `1.5` | Cloud STT breaker floor (seconds) |
| `FALLBACK_MODEL` | `base.en` | Local faster-whisper model used when Groq times out / errors |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning:free` | LLM model on the Kilo AI gateway |
| `LLM_BASE_URL` | `https://api.kilo.ai/api/gateway` | Kilo AI gateway endpoint |
| `LLM_USE_CF_AS_PRIMARY` | `true` | When true, Cloudflare Workers AI is the primary LLM provider |
| `LLM_MAX_TOKENS` | `300` | Reply length cap |
| `LLM_TIMEOUT_SEC` | `15.0` | Per-call LLM timeout |
| `TTS_PREFETCH` | `3` | Sentences fetched ahead of the consumer |
| `TTS_SENTENCE_GAP_MS` | `80` | Digital silence between trimmed sentences |
| `TTS_TIMEOUT_SEC` | `4.0` | Per-sentence edge-tts fetch timeout |
| `API_KEY` | *(empty)* | When set, clients must present it. Empty is suitable only for loopback use. |
| `INTENT_ENABLED` | `true` | Deterministic device-command routing (stop/mute/volume/lights/...) |
| `WAKE_WORDS` | `alvin` | Words that arm the command router |
| `WAKE_WORD_REQUIRED` | `false` | Require the wake word before a command executes |
| `PUSH_TO_TALK` | `false` | Wake-word gating state; set by `{"type": "talk", "active": ...}` |
| `ECHO_MODE` | `filter` | `"gate"` = drop mic audio during playback, `"filter"` = drop echoes, `"off"` = disable |
| `TAVILY_API_KEY` | *(empty)* | Tavily web search key; unset → DuckDuckGo fallback |
| `SPEAKER_ENABLED` | `false` | Enable speaker identification using ECAPA-TDNN embeddings |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for speaker matching (0-1) |
| `SPEAKER_MIN_IDENTIFY_SEC` | `1.0` | Minimum audio duration for identification (seconds) |
| `SPEAKER_MAX_IDENTIFY_SEC` | `5.0` | Maximum audio duration for identification (seconds) |
| `SPEAKER_PROFILES_PATH` | `speaker_profiles.pkl` | Path to persist speaker profiles |

Full details, the WebSocket protocol, and the STT/LLM/TTS internals live in
[AGENTS.md](AGENTS.md).

### Where does the response time come from?

A turn is four serial stages — VAD end-of-turn wait (~0.5 s), STT (~1 s), LLM
(1–2 s, the biggest chunk), and TTS first audio (~1 s warm). `TTS_WARMUP`,
`GROQ_WARMUP`, and `LLM_WARMUP` (all on by default) remove the one-time
cold-start cost from your first real turn.

---

## Project layout

```
app/
  config.py       Settings loaded from the environment / .env
  vad.py          Streaming VAD wrapper (silero, 32 ms frames)
  stt.py          STTService — Groq cloud primary + local faster-whisper fallback
  llm.py          LLMService — Kilo AI gateway primary + Cloudflare fallback, web search tool calling
  tts.py          edge-tts, sentence-chunked 16 kHz PCM streaming (with retry)
  speaker.py      Speaker identification — ECAPA-TDNN embeddings (enrollment + real-time ID)
  connection.py   Per-connection state machine (VAD feed, JSON events, barge-in, PTT, echo guard)
  intents.py      Deterministic device-command registry (~20 intents)
  itn.py          Inverse text normalization (spoken → written form)
  echo.py         Echo / self-trigger suppression
  main.py         FastAPI app: GET /health, WS /ws/transcribe
run.py            Server entrypoint (uvicorn); --mic also opens the live-mic client
tests/            unit / integration / manual-client suites
scripts/          Benchmark + speaker service + voice-catalogue helpers
.env.example      Sample configuration (copy to .env)
```

Scripts:

- `scripts/bench_tts.py` — TTS benchmark matrix tool (TTFA, RTF, silence trim)
- `scripts/speaker_service.py` — standalone PCM playback service with barge-in
- `scripts/speaker_enroll.py` — speaker enrollment CLI (enroll, list, delete, identify)
- `scripts/tts_voice_samples.py` — list voices + render sample MP3s to `voice_samples/`
- `scripts/measure_stt_tts_latency.py` — manual STT→TT latency benchmark

---

## Notes & troubleshooting

- **First run downloads a Whisper model** (the local fallback) into
  `~/.cache/huggingface`. That one-time download happens before the first
  fallback transcription.
- **Barge-in:** talking over Alvin interrupts it. The mic client mutes the
  uplink while Alvin is speaking so Alvin can't respond to its own voice;
  echo-cancellation is not used, so on-device barge-in detection is paused
  during playback (re-enable live barge-in by adding AEC if you need it).
- **No audio out?** Check that your default playback device is correct and that
  nothing else has the device exclusive. The mic client prints `[event]` lines
  for every server message, which is the quickest way to see what's happening.
- **Server not reachable from another machine?** Set `HOST=0.0.0.0`, configure
  `API_KEY`, check your firewall, and use the right `ws://` host:port.
- **Regenerate the voice previews** (not committed, ~14 MB):
  `python scripts/tts_voice_samples.py` → writes to `voice_samples/`.
