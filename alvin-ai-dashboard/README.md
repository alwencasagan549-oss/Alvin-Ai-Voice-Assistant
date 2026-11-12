# ALVIN AI Dashboard

A React + Vite dashboard styled after the ALVIN AI / J.A.R.V.I.S.-inspired Figma design. It connects to the existing FastAPI voice assistant described in `PROJECT_INFO.md` through `GET /health` and `WS /ws/transcribe`.

## Requirements

- Node.js 20 or newer
- Python 3.12 and the existing ALVIN backend
- Chrome or Edge recommended for microphone and AudioContext support

## Start the dashboard

Open a terminal in this folder:

```powershell
npm install
npm run dev
```

Open the local URL Vite prints, usually `http://localhost:5173`.

## Start the backend

In another terminal, go to the existing ALVIN backend folder and launch it the way you normally do. According to the current project notes, from `F:\Alvin-Ai-Voice-Assistant\`:

```powershell
.venv\Scripts\python.exe run.py
```

The backend listens on port `8000`. The Vite development server proxies:

- `/api/health` → `http://127.0.0.1:8000/health`
- `/ws/transcribe` → `ws://127.0.0.1:8000/ws/transcribe`

Confirm the backend itself is reachable by opening `http://localhost:8000/health` before troubleshooting the frontend.

## Configure a separately hosted backend

Copy `.env.example` to `.env.local` and set the URLs to addresses reachable from the user's browser:

```dotenv
VITE_API_BASE_URL=https://your-api.example.com
VITE_WS_URL=wss://your-api.example.com/ws/transcribe
```

Restart Vite after changing the environment values. Use HTTPS + WSS when accessing the UI over HTTPS; browsers normally block insecure microphone capture outside localhost.

## What is wired up

- Polls `/health` every 15 seconds and shows reachable/offline state.
- Connects to `/ws/transcribe` and sends `{"type":"config","language":"en","voice_enabled":<bool>}` on open.
- Sends microphone audio as little-endian mono PCM16 resampled to 16 kHz.
- Handles partial and final transcript messages, LLM response events, command events, TTS events, and the client heartbeat (`{"type":"ping"}` → `{"type":"pong"}`).
- The backend streams an LLM reply one sentence per `llm_response` event; the dashboard folds the sentences of a turn into a single assistant bubble.
- The backend speaks the reply itself and streams the audio back as binary frames — the client does **not** send a `speak` control frame per reply. The "Voice replies" toggle sends `{"type":"config","voice_enabled":false|true}` instead.
- Supports Shift+Z push-to-talk messages and the server's talk control frame. For real gated PTT behavior, also enable `PUSH_TO_TALK=true` in the backend configuration.
- Plays binary 16 kHz PCM audio frames received from the WebSocket.

## Honest status values

The current `/health` contract reports auth requirement, active connections, max connections, and the intent registry (including the registered command names, wake words, confidence floor, and the echo-guard settings). It does not report CPU or memory usage, so the dashboard shows an **Echo Guard** metric from the health payload instead of inventing host numbers. Provider rows describe project configuration; they are not a claim that every cloud provider has passed a live health check.

## Responsive behaviour

- The sidebar is a full column above 1080 px, an icon rail between 700 and 1080 px, and a horizontally scrollable strip below 700 px.
- The Voice Replies toggle stays in the hero at every width, so it is never hidden behind a collapsed panel.
- The Intent Router lists every intent the server registers (17 on the current backend) and scrolls inside its own panel so the row does not stretch the page.
- The conversation panel keeps the full transcript in a scrollable list; new messages scroll that list only and never move the page.

## Notes about browser WebSocket authentication

The documented backend can require `Authorization` or `X-API-Key` headers. The browser's built-in `WebSocket` API cannot add arbitrary request headers. The local project notes say `API_KEY` is empty by default, which should work locally. If you enable authentication for the browser UI, implement an authenticated session/cookie or a backend-supported short-lived token mechanism; avoid placing long-lived API keys in frontend source or query strings.

## Production checklist

- Serve the built `dist/` output from a web server or frontend host.
- Configure HTTPS/WSS and the backend's allowed origins/authentication for that deployment.
- Do not commit `.env.local` or API keys.
- The backend speaks LLM replies itself, so leave `TTS_AUTO_SPEAK` enabled and do not add a client-side `speak` request — that would produce duplicate audio.
- Use a server-side health endpoint if you want real CPU/memory charts; browser code should not pretend to see host metrics.

## Build

```powershell
npm run build
npm run preview
```
