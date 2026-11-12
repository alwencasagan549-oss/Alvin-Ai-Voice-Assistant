# Press-to-Talk (PTT) Feature PRD

## Problem
The voice assistant listens continuously by default. In a shared room or noisy environment this causes false triggers. Users need an explicit "push to talk" mode where the mic is off until they deliberately enable it.

## Solution
Implement press-to-talk using a Shift+Z hotkey. When PTT is active the server processes mic audio; when inactive it discards it. TTS playback auto-disables PTT. Pressing the hotkey during playback cancels TTS and re-enables the mic (barge-in).

## Protocol
Clients control PTT via a new WebSocket text control message:

```json
{"type": "talk", "active": true}
{"type": "talk", "active": false}
```

Server authoritative state: `Connection.ptt_active` (bool).

## Behavior
| Event | Client action | Server action |
|-------|--------------|---------------|
| Shift+Z press | Send `talk active=true` | Enable PTT, reset timeout watchdog, cancel ongoing turn |
| Shift+Z release | Send `talk active=false` | Disable PTT, cancel timeout watchdog |
| TTS starts | Mute local mic | Disable PTT (safety net) |
| TTS ends | Unmute mic if PTT active | No action |
| PTT held > timeout | None | Auto-disable PTT, log warning |
| WebSocket disconnect | Cleanup listener | Cancel timeout task |

## Config
- `PUSH_TO_TALK` (default `false`) — master switch
- `PUSH_TO_TALK_TIMEOUT_MS` (default `30000`) — safety timeout
- `PUSH_TO_TALK_ECHO_GATE` (default `true`) — reserved for future echo gate during PTT

## Client implementation
- `tests/manual/stream_mic.py` adds a `pynput` global hotkey listener
- Pressing Shift+Z sends `talk active=true` and flushes local TTS playback
- Releasing Shift+Z sends `talk active=false`
- Mic frames are only sent to the server when `ptt_active` is true and `tts_playing` is false

## Server implementation
- `app/connection.py` adds PTT gating in `_on_audio` and state handling in `_on_text`
- `_start_tts` disables PTT so the mic is off while the assistant speaks
- Timeout watchdog auto-disables PTT after the configured interval

## Compatibility
- No breaking changes to existing WebSocket protocol
- Existing clients that do not send `talk` messages continue to work (PTT off by default)
- Health endpoint already exposes `push_to_talk` and `push_to_talk_timeout_ms`
