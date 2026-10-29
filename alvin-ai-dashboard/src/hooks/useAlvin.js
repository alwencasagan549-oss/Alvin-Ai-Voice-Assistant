import { useCallback, useEffect, useRef, useState } from 'react';

// --- URL / env normalization ------------------------------------------------
const DEFAULT_API_BASE = '/api';
const configuredApi   = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const configuredWs    = (import.meta.env.VITE_WS_URL       || '').trim();
const configuredToken = (import.meta.env.VITE_API_KEY     || '').trim();

const API_BASE  = configuredApi || DEFAULT_API_BASE;
const HEALTH_URL = `${API_BASE}/health`;
const MAX_MESSAGES = 80;

const HEARTBEAT_INTERVAL_MS    = 25_000;
const RECONNECT_BASE_DELAY_MS  = 1_000;
const RECONNECT_MAX_DELAY_MS   = 30_000;
const RECONNECT_MAX_ATTEMPTS   = 6;
// A streamed reply arrives one sentence per event. If no further sentence
// shows up for this long the turn is considered finished, so a reply is not
// left open waiting for a `tts_end` that may never arrive.
const LLM_TURN_IDLE_MS        = 3_000;

function normalizeWsUrl(raw) {
  if (!raw) return '';
  const trimmed = String(raw).trim();
  if (trimmed.startsWith('ws://') || trimmed.startsWith('wss://')) return trimmed;
  if (trimmed.startsWith('https://')) return `wss://${trimmed.slice(8)}`;
  if (trimmed.startsWith('http://'))  return `ws://${trimmed.slice(7)}`;
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${trimmed}`;
}

function deriveWsUrl() {
  if (configuredWs) return normalizeWsUrl(configuredWs);
  if (configuredApi) {
    const apiRoot = configuredApi.replace(/\/api$/, '');
    return normalizeWsUrl(`${apiRoot}/ws/transcribe`);
  }
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${window.location.host}/ws/transcribe`;
}

function withToken(url) {
  if (!configuredToken) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}token=${encodeURIComponent(configuredToken)}`;
}

const WS_URL = withToken(deriveWsUrl());

function closeMessage(event) {
  if (!event) return 'WebSocket disconnected';
  switch (event.code) {
    case 1000: return 'WebSocket closed normally';
    case 1001: return 'WebSocket closed (going away)';
    case 1006: return 'WebSocket connection or handshake failed. Check backend availability, API key, and connection limits.';
    case 1008: return 'WebSocket rejected (unauthorized or server at capacity)';
    case 1009: return 'WebSocket closed: frame too large';
    case 1011: return 'WebSocket closed: server error';
    default:   return event.reason || `WebSocket closed (code ${event.code})`;
  }
}

function downsampleTo16k(input, inputRate) {
  const targetRate = 16000;
  if (inputRate === targetRate) return input;
  const ratio = inputRate / targetRate;
  const outLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < output.length) {
    const nextOffsetBuffer = Math.min(input.length, Math.round((offsetResult + 1) * ratio));
    let total = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer; i += 1) {
      total += input[i];
      count += 1;
    }
    output[offsetResult] = count ? total / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return output;
}

function floatToPcm16(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

function normalizeError(error) {
  if (!error) return 'Unknown connection error.';
  if (typeof error === 'string') return error;
  return error.message || String(error);
}

export default function useAlvin() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [serverStatus, setServerStatus]         = useState('checking');
  const [serverHealth, setServerHealth]         = useState(null);
  const serverHealthRef                         = useRef(null);
  const [lastError, setLastError]               = useState('');
  const [transcript, setTranscript]             = useState('');
  const [isMicActive, setIsMicActive]           = useState(false);
  const [isPushToTalk, setIsPushToTalk]         = useState(false);
  const [isSpeaking, setIsSpeaking]             = useState(false);
  const [voiceReplies, setVoiceRepliesState]    = useState(true);
  const [lastIntent, setLastIntent]             = useState(null);
  const [currentSpeaker, setCurrentSpeaker]     = useState(null); // { speaker_id, name, similarity } | null
  const [lastEvent, setLastEvent]               = useState('Waiting for a connection');
  const [modelName, setModelName]               = useState('Awaiting backend');
  const [pingLatency, setPingLatency]           = useState(null);
  const pingsInFlightRef                        = useRef([]);
  const [messages, setMessages]             = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Hello. I am ALVIN. Connect to the voice server, then start listening to begin a live conversation.',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      iso: new Date().toISOString(),
      local: true,
    },
  ]);

  const [lastYoutube, setLastYoutube] = useState(null);
  const lastYoutubeRef = useRef(null);
  const wsRef                = useRef(null);
  const mediaStreamRef       = useRef(null);
  const audioContextRef      = useRef(null);
  const processorRef         = useRef(null);
  const sourceRef            = useRef(null);
  const playbackContextRef   = useRef(null);
  const playbackNextTimeRef  = useRef(0);
  const playbackSourcesRef   = useRef(new Set());
  const mountedRef           = useRef(true);
  const micActiveRef         = useRef(false);
  const pttActiveRef         = useRef(false);
  const isSpeakingRef        = useRef(false);
  const voiceRepliesRef      = useRef(true);
  const messageCounterRef    = useRef(0);
  // Id of the most recently created message, and the id of the assistant
  // message the current LLM turn is streaming into.
  const lastMessageIdRef     = useRef('welcome');
  const llmTurnRef           = useRef(null);
  const llmTurnTimerRef      = useRef(null);

  // NEW: reconnect + heartbeat state
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef    = useRef(null);
  const userClosedRef        = useRef(false);
  const connectRef           = useRef(null);

  // --- playback / audio -----------------------------------------------------
  const stopPlayback = useCallback(() => {
    for (const source of playbackSourcesRef.current) {
      try { source.stop(); } catch { /* already stopped */ }
    }
    playbackSourcesRef.current.clear();
    playbackNextTimeRef.current = 0;
    isSpeakingRef.current = false;
    setIsSpeaking(false);
  }, []);

  const unlockAudio = useCallback(() => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!playbackContextRef.current || playbackContextRef.current.state === 'closed') {
        playbackContextRef.current = new AudioCtx();
      }
      if (playbackContextRef.current.state === 'suspended') {
        playbackContextRef.current.resume().catch(() => {});
      }
    } catch { /* AudioContext will be retried during playback */ }
  }, []);

  const setVoiceReplies = useCallback((value) => {
    const next = typeof value === 'function' ? value(voiceRepliesRef.current) : Boolean(value);
    voiceRepliesRef.current = next;
    setVoiceRepliesState(next);
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify({ type: 'config', voice_enabled: next }));
        setLastEvent(next ? 'Voice replies enabled' : 'Voice replies disabled');
      } catch { /* socket may be closing */ }
    }
    if (!next) {
      stopPlayback();
      setLastEvent('Voice replies disabled');
    }
    return next;
  }, [stopPlayback]);

  const addMessage = useCallback((role, text, extra = {}) => {
    const cleaned = String(text || '').trim();
    if (!cleaned) return null;
    messageCounterRef.current += 1;
    const stamp = new Date();
    const id = `${Date.now()}-${messageCounterRef.current}`;
    const entry = {
      id,
      role,
      text: cleaned,
      time: stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      iso: stamp.toISOString(),
      ...extra,
    };
    lastMessageIdRef.current = id;
    setMessages((current) => [...current, entry].slice(-MAX_MESSAGES));
    return id;
  }, []);

  // Extend an existing message. Used to fold the sentence-by-sentence LLM
  // stream into a single assistant bubble.
  const appendToMessage = useCallback((id, text) => {
    const addition = String(text || '').trim();
    if (!id || !addition) return;
    setMessages((current) => current.map((entry) => {
      if (entry.id !== id) return entry;
      const stamp = new Date();
      return {
        ...entry,
        text: entry.text ? `${entry.text} ${addition}` : addition,
        time: stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        iso: stamp.toISOString(),
      };
    }));
  }, []);

  const closeLlmTurn = useCallback(() => {
    llmTurnRef.current = null;
    if (llmTurnTimerRef.current) {
      window.clearTimeout(llmTurnTimerRef.current);
      llmTurnTimerRef.current = null;
    }
  }, []);

  const keepLlmTurnAlive = useCallback(() => {
    if (llmTurnTimerRef.current) window.clearTimeout(llmTurnTimerRef.current);
    llmTurnTimerRef.current = window.setTimeout(() => {
      llmTurnTimerRef.current = null;
      llmTurnRef.current = null;
    }, LLM_TURN_IDLE_MS);
  }, []);

  const clearError = useCallback(() => setLastError(''), []);

  const fetchHealth = useCallback(async () => {
    try {
      const response = await fetch(HEALTH_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Health endpoint returned HTTP ${response.status}`);
      const data = await response.json();
      if (!mountedRef.current) return;
      serverHealthRef.current = data;
      setServerHealth(data);
      setServerStatus('online');
    } catch {
      if (!mountedRef.current) return;
      serverHealthRef.current = null;
      setServerStatus('offline');
      setServerHealth(null);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchHealth();
    const interval = window.setInterval(fetchHealth, 15000);
    return () => {
      mountedRef.current = false;
      window.clearInterval(interval);
    };
  }, [fetchHealth]);

  const playPcm = useCallback(async (buffer) => {
    try {
      if (!voiceRepliesRef.current) return;
      if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 2) return;
      const usableBytes = buffer.byteLength - (buffer.byteLength % 2);
      if (usableBytes < 2) return;
      const view = new Int16Array(buffer, 0, usableBytes / 2);
      if (view.length === 0) return;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      let context = playbackContextRef.current;
      if (!context || context.state === 'closed') {
        context = new AudioCtx();
        playbackContextRef.current = context;
      }
      if (context.state === 'suspended') {
        try { await context.resume(); } catch { /* restricted until gesture */ }
      }
      // Voice can be disabled while resume() is pending. Do not queue audio
      // after stopPlayback() has already run for that toggle.
      if (!voiceRepliesRef.current) return;

      const floats = new Float32Array(view.length);
      for (let i = 0; i < view.length; i += 1) {
        floats[i] = view[i] / (view[i] < 0 ? 32768 : 32767);
      }

      const audioBuffer = context.createBuffer(1, floats.length, 16000);
      audioBuffer.getChannelData(0).set(floats);

      const source = context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(context.destination);

      const now = context.currentTime;
      if (playbackSourcesRef.current.size === 0 || playbackNextTimeRef.current < now) {
        playbackNextTimeRef.current = now + 0.02;
      }
      const startAt = Math.max(now + 0.015, playbackNextTimeRef.current);
      playbackNextTimeRef.current = startAt + audioBuffer.duration;
      playbackSourcesRef.current.add(source);

      source.onended = () => {
        playbackSourcesRef.current.delete(source);
        if (playbackSourcesRef.current.size === 0 && mountedRef.current) {
          isSpeakingRef.current = false;
          setIsSpeaking(false);
        }
      };

      isSpeakingRef.current = true;
      setIsSpeaking(true);
      source.start(startAt);
    } catch (error) {
      setLastError(`Audio playback failed: ${normalizeError(error)}`);
    }
  }, []);

  // ---- mic helpers (moved up so they can be in openSocket deps) -------------
  const stopMicrophone = useCallback(() => {
    const activeSocket = wsRef.current;
    if (pttActiveRef.current && activeSocket?.readyState === WebSocket.OPEN) {
      try { activeSocket.send(JSON.stringify({ type: 'talk', active: false })); } catch { /* closing */ }
    }
    micActiveRef.current = false;
    pttActiveRef.current = false;
    setIsMicActive(false);
    setIsPushToTalk(false);
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      try { processorRef.current.disconnect(); } catch { /* already disconnected */ }
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try { sourceRef.current.disconnect(); } catch { /* already disconnected */ }
      sourceRef.current = null;
    }
    if (mediaStreamRef.current) {
      for (const track of mediaStreamRef.current.getTracks()) track.stop();
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  }, []);

  // ---- open the socket (used by both manual connect and auto-reconnect) -----
  const openSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setLastError('');
    setConnectionStatus('connecting');
    setLastEvent('Opening WebSocket connection…');
    setPingLatency(null);
    pingsInFlightRef.current = [];

    let socket;
    try {
      socket = new WebSocket(WS_URL);
      socket.binaryType = 'arraybuffer';
      wsRef.current = socket;
    } catch (error) {
      setConnectionStatus('error');
      setLastError(normalizeError(error));
      return;
    }

    socket.onopen = () => {
      if (wsRef.current !== socket) return;
      reconnectAttemptsRef.current = 0;
      setConnectionStatus('connected');
      setLastEvent('Connected to /ws/transcribe');
      setLastError('');
      try {
        socket.send(JSON.stringify({
          type: 'config',
          language: 'en',
          voice_enabled: voiceRepliesRef.current,
        }));
        pingsInFlightRef.current.push(performance.now());
        socket.send(JSON.stringify({ type: 'ping' }));
      } catch { /* closing */ }
    };

    socket.onmessage = async (event) => {
      if (wsRef.current !== socket) return;
      if (event.data instanceof ArrayBuffer) {
        await playPcm(event.data);
        return;
      }
      if (event.data instanceof Blob) {
        const buffer = await event.data.arrayBuffer();
        if (wsRef.current !== socket) return;
        await playPcm(buffer);
        return;
      }
      let data;
      try { data = JSON.parse(event.data); } catch {
        setLastEvent('Received an unrecognized text frame');
        return;
      }

      if (typeof data.transcript === 'string') {
        if (data.is_partial) {
          if (isSpeakingRef.current) stopPlayback();
          setTranscript(data.transcript);
          setLastEvent('Receiving speech transcription');
        } else if (data.is_final) {
          setTranscript('');
          closeLlmTurn();
          addMessage('user', data.transcript, { confidence: data.confidence });
          setLastEvent('Final transcript received');
        }
      }

      if (data.type === 'llm_response') {
        const addition = String(data.text || '').trim();
        if (addition) {
          // The backend streams one sentence per event. Fold every sentence of
          // a turn into a single assistant bubble.
          if (llmTurnRef.current && llmTurnRef.current === lastMessageIdRef.current) {
            appendToMessage(llmTurnRef.current, addition);
          } else {
            llmTurnRef.current = addMessage('assistant', addition, { model: data.model });
          }
          keepLlmTurnAlive();
        }
        setModelName(data.model || 'Model unspecified');
        setLastEvent('Assistant response received');
      } else if (data.type === 'llm_error') {
        closeLlmTurn();
        const message = data.error || 'LLM request failed';
        setLastError(message);
        addMessage('system', `LLM error: ${message}`);
        setLastEvent('LLM error');
      } else if (data.type === 'command') {
        setLastIntent({ intent: data.intent || 'unknown', slots: data.slots || {}, time: new Date() });
        addMessage('system', `Command matched: ${data.intent || 'unknown'}`, { slots: data.slots || {} });
        setLastEvent(`Intent matched: ${data.intent || 'unknown'}`);
        if (data.action === 'disable_stt') onMicMuteRef.current?.();
        else if (data.action === 'enable_stt') onMicUnmuteRef.current?.();
        if (data.intent === 'play_youtube') {
          handleYoutubePlayRef.current?.(data.slots || {});
        } else if (data.intent === 'stop' || data.intent === 'pause') {
          stopYoutubeRef.current?.();
        } else if (data.intent === 'resume' || data.intent === 'play') {
          const last = lastYoutubeRef.current;
          if (last && last.playing) {
            // Already playing — acknowledge without trying to resume.
            setLastEvent('YouTube is already playing');
            addMessage('system', 'YouTube is already playing.');
          } else if (last && !last.videoId) {
            addMessage('system', 'YouTube search failed — click "Open ↗" to search manually, or say "play <song> on YouTube".');
            setLastEvent('YouTube search was unresolved');
          } else if (last && last.videoId) {
            resumeYoutubeRef.current?.();
          } else {
            addMessage('system', 'Nothing is paused — say "play <song> on YouTube" to start.');
            setLastEvent('No YouTube playback to resume');
          }
        }
      } else if (data.type === 'tts_stop') {
        stopPlayback();
        setLastEvent('Voice output stopped');
      } else if (data.type === 'tts_end') {
        closeLlmTurn();
        setIsSpeaking(playbackSourcesRef.current.size > 0);
        isSpeakingRef.current = playbackSourcesRef.current.size > 0;
        setLastEvent('Voice output completed');
      } else if (data.type === 'tts_error') {
        setLastError(data.error || 'Text-to-speech failed');
        setLastEvent('Voice output error');
      } else if (data.type === 'ptt_state') {
        setIsPushToTalk(Boolean(data.active));
      } else if (data.type === 'pong') {
        const sentAt = pingsInFlightRef.current.shift();
        if (sentAt) {
          setPingLatency(Math.round(performance.now() - sentAt));
        }
        setLastEvent('Server heartbeat received');
      } else if (data.type === 'speaker_change') {
        if (data.speaker_id) {
          setCurrentSpeaker({ speaker_id: data.speaker_id, name: data.name, similarity: data.similarity });
          setLastEvent(`Speaker identified: ${data.name}`);
          addMessage('system', `Speaker identified: ${data.name}`);
        } else {
          setCurrentSpeaker(null);
          setLastEvent('Speaker context cleared (unknown voice)');
        }
      }
    };

    socket.onerror = () => {
      if (wsRef.current !== socket) return;
      setConnectionStatus('error');
      setLastError((current) => current ||
        'WebSocket connection failed. Check that the ALVIN backend is reachable and any API key is correct.');
      setLastEvent('Connection error');
    };

    socket.onclose = (event) => {
      if (wsRef.current !== socket) return;
      if (micActiveRef.current) stopMicrophone();
      wsRef.current = null;
      if (!mountedRef.current) return;

      setConnectionStatus('disconnected');
      setLastEvent(closeMessage(event));

      // No reconnect on: user-initiated close, graceful close, policy violation.
      if (userClosedRef.current || event.code === 1000 || event.code === 1008) {
        if (event.code === 1008) {
          setLastError('Server rejected the connection (unauthorized or at capacity). Check VITE_API_KEY and server limits.');
        }
        return;
      }

      if (event.code === 1006) {
        const health = serverHealthRef.current;
        if (health?.auth_required && !configuredToken) {
          setLastError('This server requires authentication. Set VITE_API_KEY in the dashboard environment.');
          return;
        }
        if (
          Number.isFinite(health?.active_connections) &&
          Number.isFinite(health?.max_connections) &&
          health.active_connections >= health.max_connections
        ) {
          setLastError('The server is at its connection limit. Disconnect another client and retry.');
          return;
        }
      }

      const attempts = reconnectAttemptsRef.current;
      if (attempts >= RECONNECT_MAX_ATTEMPTS) {
        setLastError('Connection did not recover after six attempts. Check backend availability, API key, and connection limits. Click Connect to retry.');
        return;
      }
      const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempts, RECONNECT_MAX_DELAY_MS);
      reconnectAttemptsRef.current = attempts + 1;
      setLastEvent(`Reconnecting in ${Math.round(delay / 1000)}s (attempt ${attempts + 1}/${RECONNECT_MAX_ATTEMPTS})`);
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        if (!userClosedRef.current) connectRef.current?.({ automaticRetry: true });
      }, delay);
    };
  }, [addMessage, appendToMessage, closeLlmTurn, keepLlmTurnAlive, playPcm, stopMicrophone, stopPlayback, unlockAudio]);

  // ---- user-facing connect (resets the retry counter) -----------------------
  const connect = useCallback((options = {}) => {
    unlockAudio();
    userClosedRef.current = false;
    if (options?.automaticRetry !== true) reconnectAttemptsRef.current = 0;
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    openSocket();
  }, [openSocket, unlockAudio]);

  useEffect(() => { connectRef.current = connect; }, [connect]);

  // ---- client heartbeat keeps proxies from idling the socket ----------------
  useEffect(() => {
    const interval = window.setInterval(() => {
      const socket = wsRef.current;
      if (socket?.readyState === WebSocket.OPEN) {
        try {
          pingsInFlightRef.current.push(performance.now());
          socket.send(JSON.stringify({ type: 'ping' }));
        } catch { /* ignore */ }
      }
    }, HEARTBEAT_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, []);

  // ---- disconnect (explicit user action) -----------------------------------
  const disconnect = useCallback(() => {
    userClosedRef.current = true;
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectAttemptsRef.current = 0;
    stopMicrophone();
    stopPlayback();
    const socket = wsRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: 'stop' })); } catch { /* ignored */ }
    }
    if (socket) socket.close(1000, 'Disconnected by user');
    wsRef.current = null;
    setConnectionStatus('disconnected');
    setLastEvent('Disconnected by user');
    setPingLatency(null);
    pingsInFlightRef.current = [];
  }, [stopMicrophone, stopPlayback]);

  // ---- microphone -----------------------------------------------------------
  const startMicrophone = useCallback(async () => {
    if (micActiveRef.current) return;
    unlockAudio();
    if (isSpeakingRef.current) stopPlayback();
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setLastError('Connect to the ALVIN server before starting the microphone.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setLastError('Microphone capture requires HTTPS or localhost in a supported browser.');
      return;
    }
    try {
      setLastError('');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: false,
      });
      const context = new AudioContext();
      if (context.state === 'suspended') await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        const socket = wsRef.current;
        if (!micActiveRef.current || !socket || socket.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const resampled = downsampleTo16k(input, context.sampleRate);
        const pcm = floatToPcm16(resampled);
        if (pcm.length) socket.send(pcm.buffer);
        event.outputBuffer.getChannelData(0).fill(0);
      };
      source.connect(processor);
      processor.connect(context.destination);
      mediaStreamRef.current = stream;
      audioContextRef.current = context;
      sourceRef.current = source;
      processorRef.current = processor;
      micActiveRef.current = true;
      setIsMicActive(true);
      setLastEvent('Microphone streaming at 16 kHz mono PCM');
      addMessage('system', 'Microphone started. Speak naturally; VAD will detect the end of your turn.');
      // Send a reset frame so the server clears any leftover per-session state
      // (muted flag, PTT flag, wake-word window, echo cooldown, audio buffer).
      const socket = wsRef.current;
      if (socket?.readyState === WebSocket.OPEN) {
        try { socket.send(JSON.stringify({ type: 'reset' })); } catch { /* closing */ }
      }
    } catch (error) {
      setLastError(`Could not start microphone: ${normalizeError(error)}`);
      setLastEvent('Microphone permission or device error');
    }
  }, [addMessage, stopPlayback, unlockAudio]);

  const toggleMicrophone = useCallback(() => {
    if (micActiveRef.current) {
      stopMicrophone();
      setLastEvent('Microphone stopped');
      addMessage('system', 'Microphone stopped. The WebSocket connection remains open.');
    } else {
      startMicrophone();
    }
  }, [addMessage, startMicrophone, stopMicrophone]);

  const sendPtt = useCallback((active) => {
    if (!micActiveRef.current) return;
    if (pttActiveRef.current === active) return;
    if (active) {
      unlockAudio();
      if (isSpeakingRef.current) stopPlayback();
    }
    pttActiveRef.current = active;
    setIsPushToTalk(active);
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'talk', active }));
      setLastEvent(active ? 'Push-to-talk active' : 'Push-to-talk released');
    }
  }, [stopPlayback, unlockAudio]);

  // Send a command to the YouTube IFrame Player API via postMessage.
  // The iframe must have enablejsapi=1 in its src for this to work.
  const ytCommand = useCallback((func, args = '') => {
    const iframe = document.querySelector('iframe[title="YouTube Player"]');
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage(
        JSON.stringify({ event: 'command', func, args }),
        'https://www.youtube.com'
      );
    }
  }, []);

  const handleYoutubePlayRef = useRef(null);
  const handleYoutubePlay = useCallback((slots) => {
    const videoId = slots.video_id;
    const query = (slots.query || '').trim();

    // Fallback to search URL if yt-dlp failed to resolve a video ID
    const url = videoId
      ? `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`
      : (slots.url || `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`);

    if (videoId) {
      setLastYoutube({ url, videoId, title: slots.title || '', query, playing: true });
      // Kick the player in case the same video is already loaded but paused —
      // the iframe src is unchanged, so autoplay=1 won't re-fire on prop
      // re-render. For a genuinely new video these postMessages land on the
      // unloading iframe and are silently dropped; autoplay=1 starts the new one.
      ytCommand('unMute');
      ytCommand('playVideo');
      // Retry once after 200ms — the IFrame API may not be ready yet.
      // Guard on the ref: if a STOP arrived in the interim, the timeout
      // fires harmlessly instead of resurrecting a paused video.
      window.setTimeout(() => {
        if (!lastYoutubeRef.current?.playing) return;
        ytCommand('unMute');
        ytCommand('playVideo');
      }, 200);
      setLastEvent(slots.title ? `Playing "${slots.title}"` : `Playing: ${query}`);
      addMessage('system', slots.title ? `▶ ${slots.title}` : `▶ YouTube: ${query}`);
    } else {
      setLastError(`Could not resolve video ID for "${query}". Click "Open ↗" to search manually.`);
      setLastYoutube({ url, videoId: null, title: '', query, playing: false });
    }
  }, [addMessage, ytCommand]);
  useEffect(() => { handleYoutubePlayRef.current = handleYoutubePlay; }, [handleYoutubePlay]);

  const stopYoutube = useCallback(() => {
    setLastYoutube((prev) => (prev ? { ...prev, playing: false } : prev));
    setLastEvent('YouTube playback stopped');
    ytCommand('pauseVideo');
    // Also interrupt any TTS that's playing so the user gets immediate silence.
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: 'stop_speak' })); } catch { /* ignored */ }
    }
    stopPlayback();
  }, [ytCommand, stopPlayback]);

  const resumeYoutube = useCallback(() => {
    setLastYoutube((prev) => (prev && prev.videoId ? { ...prev, playing: true } : prev));
    setLastEvent('YouTube playback resumed');
    ytCommand('unMute');
    ytCommand('playVideo');
    // Retry once after 200ms — the IFrame API may drop the first postMessage
    // if the iframe just loaded or was refreshed.  Guard on the ref so a
    // STOP that arrived in the interim doesn't get undone here.
    window.setTimeout(() => {
      if (!lastYoutubeRef.current?.playing) return;
      ytCommand('unMute');
      ytCommand('playVideo');
    }, 200);
  }, [ytCommand]);

  const unmuteYoutube = useCallback(() => {
    // Fire twice 200ms apart; the YouTube IFrame API sometimes drops the
    // first postMessage before its JS bridge attaches.
    ytCommand('unMute');
    window.setTimeout(() => ytCommand('unMute'), 200);
    setLastEvent('YouTube unmuted');
  }, [ytCommand]);

  const stopYoutubeRef = useRef(null);
  const resumeYoutubeRef = useRef(null);
  useEffect(() => { stopYoutubeRef.current = stopYoutube; }, [stopYoutube]);
  useEffect(() => { resumeYoutubeRef.current = resumeYoutube; }, [resumeYoutube]);
  useEffect(() => { lastYoutubeRef.current = lastYoutube; }, [lastYoutube]);

  const interruptSpeech = useCallback(() => {
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: 'stop_speak' })); } catch { /* ignored */ }
    }
    stopPlayback();
    setLastEvent('Speech interruption requested');
  }, [stopPlayback]);

  const sendPing = useCallback(() => {
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      pingsInFlightRef.current.push(performance.now());
      socket.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  // ---- mute / unmute handlers (referenced by the message handler via refs) ---
  const onMicMute = useCallback(() => {
    if (micActiveRef.current) {
      setLastEvent('Microphone muted; listening locally for an unmute command');
    }
  }, []);

  const onMicUnmute = useCallback(() => {
    if (micActiveRef.current && processorRef.current && audioContextRef.current) {
      processorRef.current.onaudioprocess = (event) => {
        const socket = wsRef.current;
        if (!micActiveRef.current || !socket || socket.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const resampled = downsampleTo16k(input, audioContextRef.current.sampleRate);
        const pcm = floatToPcm16(resampled);
        if (pcm.length) socket.send(pcm.buffer);
        event.outputBuffer.getChannelData(0).fill(0);
      };
      setLastEvent('Microphone unmuted by command');
    }
  }, []);

  // The message handler is inside `openSocket`, which is created before these
  // two callbacks exist. Route through refs so the handler always hits the
  // current implementation.
  const onMicMuteRef = useRef(null);
  const onMicUnmuteRef = useRef(null);
  useEffect(() => { onMicMuteRef.current = onMicMute; }, [onMicMute]);
  useEffect(() => { onMicUnmuteRef.current = onMicUnmute; }, [onMicUnmute]);

  // ---- Shift+Z push-to-talk -------------------------------------------------
  useEffect(() => {
    const keyDown = (event) => {
      if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') return;
      if (event.code === 'KeyZ' && event.shiftKey && !event.repeat) {
        event.preventDefault();
        sendPtt(true);
      }
    };
    const keyUp = (event) => {
      if (event.code === 'KeyZ') sendPtt(false);
      if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') sendPtt(false);
    };
    const release = () => sendPtt(false);
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', release);
    return () => {
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', release);
    };
  }, [sendPtt]);

  // ---- cleanup on unmount ---------------------------------------------------
  useEffect(() => () => {
    userClosedRef.current = true;
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (llmTurnTimerRef.current) {
      window.clearTimeout(llmTurnTimerRef.current);
      llmTurnTimerRef.current = null;
    }
    stopMicrophone();
    stopPlayback();
    if (wsRef.current) wsRef.current.close();
    if (playbackContextRef.current) playbackContextRef.current.close().catch(() => {});
  }, [stopMicrophone, stopPlayback]);

  const clearConversation = useCallback(() => {
    closeLlmTurn();
    lastMessageIdRef.current = null;
    setMessages([]);
  }, [closeLlmTurn]);

  return {
    apiBase: API_BASE,
    wsUrl: WS_URL,
    connectionStatus,
    serverStatus,
    serverHealth,
    currentSpeaker,
    lastError,
    clearError,
    transcript,
    isMicActive,
    isPushToTalk,
    isSpeaking,
    voiceReplies,
    setVoiceReplies,
    sendPtt,
    lastIntent,
    lastEvent,
    modelName,
    messages,
    connect,
    disconnect,
    fetchHealth,
    toggleMicrophone,
    interruptSpeech,
    sendPing,
    pingLatency,
    clearConversation,
    handleYoutubePlay,
    stopYoutube,
    resumeYoutube,
    unmuteYoutube,
    lastYoutube,
  };
}
