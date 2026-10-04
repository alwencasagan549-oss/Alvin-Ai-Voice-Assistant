# Alvin AI Voice Assistant

A low-latency, real-time voice assistant. Speak into your microphone and Alvin
transcribes you, thinks with a large language model, and talks back — all
over a live WebSocket, with interruption (barge-in) support.

```
 You speak ──► VAD ──► STT (Whisper) ──► LLM ──► TTS ──► Alvin speaks
              (silero)  (Groq cloud        (NVIDIA     (edge-tts,
                              + local        NIM, w/     sentence-
                              fallback)      CF fallback) streamed)
```

The whole pipeline runs in Python over a single WebSocket. Sentences are spoken
as soon as the LLM finishes writing them, so short replies come back fast.

---

## What you need

- **Python 3.12** — [download](https://www.python.org/downloads/). On Windows,
  tick **"Add python.exe to PATH"** during install.
- **A microphone and speakers.**
- **API keys** (all free to start — see [Getting the keys](#getting-the-keys)):
  - **Groq** — speech-to-text
  - **NVIDIA NIM** — the LLM "brain" (conversations)
  - *(optional)* **Cloudflare** — LLM fallback provider
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

### 4. Configure the environment

Copy the sample config and fill in your keys:

```bash
# Windows (PowerShell):
copy .env.example .env
# Windows (Git Bash) / macOS / Linux:
cp .env.example .env
```

Open `.env` and set at least the two required keys:

```ini
GROQ_API_KEY=gsk_....            # from https://console.groq.com/keys
LLM_API_KEY=nvapi-....           # from https://build.nvidia.com
```

Everything else has a working default. See [Getting the keys](#getting-the-keys)
and [Configuration](#configuration) for the full list.

> **Never commit `.env`** — it is in `.gitignore` for exactly that reason.

### 5. Start Alvin and talk to it

One command boots the server **and** opens the live microphone client:

```bash
python run.py --mic
```

Wait for `Server ready.`, then just start speaking. Say something, pause for
about half a second, and Alvin will answer out loud. Talk over Alvin at any
time to interrupt it. Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop.

Useful flags:

```bash
python run.py --mic --voice en-US-AndrewNeural   # pick a different voice
python run.py --mic --vad-threshold 0.6          # less false barge-in
python run.py --mic --language en                # force STT language
```

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
| **NVIDIA NIM** | The conversational LLM brain | https://build.nvidia.com | `LLM_API_KEY` (alias `NVIDIA_API_KEY`) |
| **Cloudflare** *(optional)* | LLM fallback provider | https://dash.cloudflare.com/ → API Tokens | `CF_API_KEY` + `CF_ACCOUNT_ID` |
| **edge-tts** | Text-to-speech | None needed — uses Microsoft's online voice endpoint | — |

- **If you don't set `LLM_API_KEY`**, the LLM is disabled: Alvin will still
  transcribe you, and (if `TTS_AUTO_SPEAK=true`) simply speak the raw transcript
  back instead of a generated reply.
- **If you don't set `GROQ_API_KEY`**, every transcription falls back to the
  local `faster-whisper` model (slower, but works offline).
- **Cloudflare fallback** is optional; leave `CF_ENABLED=false` until you add
  your keys.

---

## Configuration

All runtime settings come from `.env` (see `.env.example` for the full,
commented list). The ones you'll most often touch:

| Variable | Default | What it does |
|----------|---------|--------------|
| `PORT` | `8000` | Server port |
| `TTS_VOICE` | `en-US-AndrewNeural` | Which voice Alvin speaks with |
| `MIN_SILENCE_DURATION_MS` | `500` | How long you must be quiet to end a turn. Lower = snappier, but risks cutting you off mid-sentence |
| `VAD_THRESHOLD` | `0.5` | Speech-detection sensitivity. Raise to `0.6+` in noisy rooms / to reduce false barge-in |
| `LLM_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | Which LLM powers Alvin |
| `LLM_MAX_TOKENS` | `128` | Reply length cap (keeps answers short & TTS-friendly) |
| `LLM_SYSTEM_PROMPT` | *(short assistant)* | Alvin's persona / style |
| `API_KEY` | *(empty)* | When set, clients must present it. Empty = open (fine for a laptop, not for a shared host) |

Full details, the WebSocket protocol, and the STT/LLM/TTS internals live in
[AGENTS.md](AGENTS.md).

### Where does the response time come from?

A turn is four serial stages — VAD end-of-turn wait (~0.5 s), STT (~1 s), LLM
(1–2 s, the biggest chunk), and TTS first audio (~1 s warm). The LLM is a
reasoning model, so it thinks before it speaks. `TTS_WARMUP`, `GROQ_WARMUP`, and
`LLM_WARMUP` (all on by default) remove the one-time cold-start cost from your
first real turn.

---

## Project layout

```
app/
  config.py       Settings loaded from the environment / .env
  vad.py          Streaming VAD wrapper (silero, 32 ms frames)
  stt.py          STTService — Groq cloud primary + local faster-whisper fallback
  llm.py          LLMService — NVIDIA NIM primary + Cloudflare fallback, sentence-streamed
  tts.py          edge-tts, sentence-chunked 16 kHz PCM streaming (with retry)
  connection.py   Per-connection state machine (VAD feed, JSON events, barge-in)
  main.py         FastAPI app: GET /health, WS /ws/transcribe
run.py            Server entrypoint (uvicorn); --mic also opens the live-mic client
tests/            unit / integration / manual-client suites
scripts/          Benchmark + voice-catalogue helpers
.env.example      Sample configuration (copy to .env)
```

---

## Notes & troubleshooting

- **First run downloads a Whisper model** (the local fallback) into
  `~/.cache/huggingface`. That one-time download happens before the first
  fallback transcription.
- **Barge-in:** talking over Alvin interrupts it. The mic client mutes the
  uplink while Alvin is speaking so Alvin can't respond to its own voice;
  echo-cancellation is not used, so on-device barge-in detection is paused
  during playback (re-enable live barge-in by adding AEC if you need it).
- **Regenerate the voice previews** (not committed, ~14 MB):
  `python scripts/tts_voice_samples.py` → writes to `voice_samples/`.
- **No audio out?** Check that your default playback device is correct and that
  nothing else has the device exclusive. The mic client prints `[event]` lines
  for every server message, which is the quickest way to see what's happening.
- **Server not reachable from another machine?** Check `HOST` (default
  `0.0.0.0`), your firewall, and that you use the right `ws://` host:port.
