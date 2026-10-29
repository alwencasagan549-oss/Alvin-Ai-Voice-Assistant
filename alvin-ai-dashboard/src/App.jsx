import { useEffect, useRef, useState } from 'react';
import useAlvin from './hooks/useAlvin.js';
import Visualizer from './components/Visualizer.jsx';
import { scrollToElement, stickyChromeHeight } from './lib/scroll.js';

const NAV_ITEMS = [
  { label: 'Dashboard', icon: 'home', target: 'dashboard-top' },
  { label: 'Voice Console', icon: 'audio', target: 'voice-console' },
  { label: 'Transcripts', icon: 'transcript', target: 'conversation' },
  { label: 'Intent Router', icon: 'command', target: 'intent-router' },
  { label: 'System Health', icon: 'activity', target: 'system-diagnostics' },
  { label: 'Settings', icon: 'settings', target: 'dashboard-settings' },
];

// Descriptions for every deterministic intent the server can register.
// The router itself is driven by /health, so this table only supplies the
// human-readable label and the icon for a given intent name.
const INTENT_DETAILS = {
  wake: ['Wake phrase · "alvin"', 'pulse'],
  stop: ['Stop playback or the current turn', 'stop'],
  pause: ['Pause the active media', 'stop'],
  resume: ['Resume paused media', 'arrow'],
  mute_mic: ['Microphone gating · local keyword pass', 'mic'],
  unmute_mic: ['Restore microphone capture', 'mic'],
  mute: ['Mute the speaker or the music', 'stop'],
  unmute: ['Unmute the speaker or the music', 'audio'],
  set_volume: ['Absolute volume level', 'bolt'],
  volume_up: ['Relative volume increase', 'bolt'],
  volume_down: ['Relative volume decrease', 'bolt'],
  lights_on: ['Smart-home lighting', 'bolt'],
  lights_off: ['Smart-home lighting', 'bolt'],
  next: ['Skip to the next track', 'arrow'],
  previous: ['Return to the previous track', 'arrow'],
  play: ['Start playback', 'arrow'],
  play_youtube: ['YouTube playback via yt-dlp', 'globe'],
  query_time: ['Time query · local command', 'clock'],
};

const FALLBACK_INTENTS = ['query_time', 'volume_up', 'wake', 'mute_mic'];

function Icon({ name, size = 18, className = '' }) {
  const paths = {
    home: <><path d="m3 10 9-7 9 7"/><path d="M5 9v11h14V9"/><path d="M9 20v-6h6v6"/></>,
    audio: <><rect x="8" y="3" width="8" height="12" rx="4"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6"/></>,
    transcript: <><path d="M8 5h12M8 10h12M8 15h7"/><path d="M4 4h.01M4 9h.01M4 14h.01M4 19h.01"/><path d="M8 20h12"/></>,
    command: <><path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3"/><path d="m9 9 6 6M15 9l-6 6"/></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.2 1-1.4 2.4-1.5-.6a8 8 0 0 1-1.7 1l-.3 1.6h-2.8l-.3-1.6a8 8 0 0 1-1.7-1l-1.5.6-1.4-2.4 1.2-1a8 8 0 0 1 0-2l-1.2-1 1.4-2.4 1.5.6a8 8 0 0 1 1.7-1l.3-1.6h2.8l.3 1.6a8 8 0 0 1 1.7 1l1.5-.6 1.4 2.4-1.2 1a8 8 0 0 1 0 2Z" transform="translate(-1 -1) scale(1.08)"/></>,
    wifi: <><path d="M5 12.5a11 11 0 0 1 14 0M8 15.5a6.5 6.5 0 0 1 8 0M11 18.5a2 2 0 0 1 2 0"/><circle cx="12" cy="21" r=".6" fill="currentColor" stroke="none"/></>,
    monitor: <><rect x="3" y="4" width="18" height="13" rx="1.5"/><path d="M8 21h8M12 17v4"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    arrow: <><path d="M4 12h15"/><path d="m13 5 7 7-7 7"/></>,
    mic: <><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6"/></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    pulse: <><path d="M2 12h4l3-7 5 14 3-7h5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    close: <><path d="m18 6-12 12M6 6l12 12"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    copy: <><rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
    refresh: <><path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.5 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 12.5-3"/></>,
    bolt: <path d="m13 2-3 8h7l-6 12 1-9H5Z"/>,
    cpu: <><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 9h6v6H9zM9 1v5M15 1v5M9 18v5M15 18v5M1 9h5M1 15h5M18 9h5M18 15h5"/></>,
    memory: <><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h2v4H7zM12 10h2v4h-2zM17 10h1v4h-1zM7 3v3M12 3v3M17 3v3M7 18v3M12 18v3M17 18v3"/></>,
    globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"/></>,
    folder: <><path d="M3 6a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></>,
    external: <><path d="M14 3h7v7M21 3l-9 9"/><path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  };
  return <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.bolt}</svg>;
}

function StatusPill({ status, children }) {
  return <span className={`status-pill status-${status}`}><span className="status-led" />{children}</span>;
}

function PanelTitle({ icon, title, right, rightClass = '' }) {
  return <div className="panel-title-row"><div className="panel-title"><span className="panel-title-icon">{icon}</span><h2>{title}</h2></div>{right && <span className={`panel-title-right ${rightClass}`}>{right}</span>}</div>;
}

function NavItem({ item, active, onClick }) {
  return (
    <button
      className={`nav-item ${active ? 'active' : ''}`}
      onClick={onClick}
      type="button"
      aria-current={active ? 'page' : undefined}
    >
      <Icon name={item.icon} size={19} />
      <span>{item.label}</span>
      {active && <span className="nav-active-line" />}
    </button>
  );
}

function MetricCard({ icon, label, value, detail, state = 'normal' }) {
  return (
    <article className="metric-card">
      <div className="metric-card-head"><span>{icon}</span><label>{label}</label><span className="metric-indicator" /></div>
      <div className="metric-card-body"><strong className={state === 'muted' ? 'muted-value' : ''}>{value}</strong><p>{detail}</p></div>
    </article>
  );
}

function PipelineRow({ number, title, description, status, state = 'configured' }) {
  return (
    <div className="pipeline-row">
      <div className={`pipeline-icon ${state}`}><span>{number}</span></div>
      <div className="pipeline-copy"><strong>{title}</strong><span>{description}</span></div>
      <span className={`pipeline-state ${state}`}>{status}</span>
    </div>
  );
}

function Message({ message }) {
  const role = message.role === 'user' ? 'user' : message.role === 'system' ? 'system' : 'assistant';
  return (
    <div className={`message message-${role}`}>
      <div className="message-meta">
        <span className="message-avatar">{role === 'user' ? 'YOU' : role === 'system' ? 'SYS' : 'AI'}</span>
        <span>{role === 'user' ? 'USER TRANSCRIPT' : role === 'system' ? 'SYSTEM EVENT' : 'ALVIN RESPONSE'}</span>
        <time dateTime={message.iso}>{message.time}</time>
      </div>
      <p>{message.text}</p>
      {Number.isFinite(message.confidence) && <span className="message-confidence">STT confidence {Math.round(message.confidence * 100)}%</span>}
    </div>
  );
}

function formatUptimeOrConnections(health) {
  if (!health) return '—';
  if (Number.isFinite(health.active_connections)) return String(health.active_connections).padStart(2, '0');
  return 'OK';
}

function formatGreeting(date) {
  const h = date.getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}

export default function App() {
  const alvin = useAlvin();
  const [activeNav, setActiveNav] = useState('Dashboard');
  const [showSettings, setShowSettings] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState(false);
  const [now, setNow] = useState(new Date());
  const messageListRef = useRef(null);
  const messageListMetricsRef = useRef({ scrollTop: 0, scrollHeight: 0, clientHeight: 0 });
  const navRef = useRef(null);
  const dialogRef = useRef(null);
  const cancelButtonRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  // Keep the transcript pinned to its own scroll container. Scrolling the page
  // instead would yank the whole dashboard to the conversation panel every
  // time a partial transcript or a reply arrives.
  const trackMessageListScroll = () => {
    const list = messageListRef.current;
    if (!list) return;
    messageListMetricsRef.current = {
      scrollTop: list.scrollTop,
      scrollHeight: list.scrollHeight,
      clientHeight: list.clientHeight,
    };
  };

  useEffect(() => {
    const list = messageListRef.current;
    if (!list) return;
    // Was the list at the bottom before this update? The metrics captured on
    // the previous pass describe the previous content, which is the only way
    // to tell "the reader scrolled back" from "the list just grew".
    const previous = messageListMetricsRef.current;
    const wasPinned = !previous.scrollHeight
      || (previous.scrollHeight - previous.scrollTop - previous.clientHeight) < 24;
    if (wasPinned) {
      list.scrollTo({ top: list.scrollHeight, behavior: 'auto' });
    }
    messageListMetricsRef.current = {
      scrollTop: list.scrollTop,
      scrollHeight: list.scrollHeight,
      clientHeight: list.clientHeight,
    };
  }, [alvin.messages, alvin.transcript]);

  // The settings panel is only mounted once it is enabled, so it can only be
  // scrolled to after the render that creates it.
  useEffect(() => {
    if (!showSettings) return;
    scrollToElement(document.getElementById('dashboard-settings'), { offset: stickyChromeHeight() });
  }, [showSettings]);

  // On narrow screens the nav is a horizontally scrollable strip; keep the
  // active entry on screen.
  useEffect(() => {
    const strip = navRef.current;
    if (!strip) return;
    if (strip.scrollWidth <= strip.clientWidth) return;
    const active = strip.querySelector('.nav-item.active');
    if (!active) return;
    const stripRect = strip.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    if (activeRect.left >= stripRect.left && activeRect.right <= stripRect.right) return;
    strip.scrollTo({
      left: active.offsetLeft - (strip.clientWidth - active.offsetWidth) / 2,
      behavior: 'smooth',
    });
  }, [activeNav]);

  const isConnected = alvin.connectionStatus === 'connected';
  const canDisconnect = ['connected', 'connecting'].includes(alvin.connectionStatus);
  const connectLabel = alvin.connectionStatus === 'connecting' ? 'CONNECTING' : isConnected ? 'DISCONNECT' : 'CONNECT SERVER';
  const serverOnline = alvin.serverStatus === 'online';
  const activeConnections = alvin.serverHealth?.active_connections;
  const maxConnections = alvin.serverHealth?.max_connections;
  const authRequired = alvin.serverHealth?.auth_required;
  const health = alvin.serverHealth;
  const sttProvider   = health?.provider;
  const llmEnabled    = health?.llm?.enabled;
  const llmModel      = health?.llm?.model;
  const ttsVoice      = health?.tts_voice;
  const sampleRateKhz = health?.sample_rate ? (health.sample_rate / 1000).toFixed(0) : null;
  const historyTurns  = health?.max_llm_history;
  const wakeWords     = health?.intents?.wake_words?.join(', ');
  const minConfidence = health?.intents?.min_confidence;
  const intentsEnabled = health?.intents?.enabled;
  const registeredIntents = Array.isArray(health?.intents?.commands) && health.intents.commands.length
    ? health.intents.commands
    : FALLBACK_INTENTS;
  const intentRows = registeredIntents.map((name) => {
    const [description, icon] = INTENT_DETAILS[name] || [name, 'bolt'];
    return { name, description, icon };
  });
  const echoMode      = health?.intents?.echo_mode;
  const echoThreshold = health?.intents?.echo_similarity_threshold;
  const echoCooldown  = health?.intents?.echo_cooldown_ms;

  function scrollToTarget(item) {
    setActiveNav(item.label);
    if (item.label === 'Settings') {
      setShowSettings((value) => !value);
      return;
    }
    scrollToElement(document.getElementById(item.target), { offset: stickyChromeHeight() });
  }

  function startSession() {
    if (!isConnected) {
      alvin.connect();
      return;
    }
    alvin.toggleMicrophone();
  }

  async function copyHealth() {
    const info = JSON.stringify(alvin.serverHealth || { status: alvin.serverStatus }, null, 2);
    try { await navigator.clipboard.writeText(info); } catch { /* Clipboard may be blocked in an insecure context. */ }
  }

  // Disconnect dialog: move focus in, trap it, close on Escape, restore it.
  useEffect(() => {
    if (!confirmDisconnect) return undefined;
    const previouslyFocused = document.activeElement;
    cancelButtonRef.current?.focus();

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setConfirmDisconnect(false);
        return;
      }
      if (event.key !== 'Tab') return;
      const focusables = dialogRef.current?.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusables || focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
    };
  }, [confirmDisconnect]);

  return (
    <div className="app-shell" id="dashboard-top">
      <header className="topbar">
        <div className="brand-block">
          <button
            type="button"
            className="brand-orb"
            aria-label="Back to top and refresh health"
            title="Back to top · refresh health"
            onClick={() => {
              alvin.fetchHealth();
              scrollToElement(document.getElementById('dashboard-top'), { offset: stickyChromeHeight() });
            }}
          >
            <span /><i />
          </button>
          <div className="brand-wordmark"><strong>ALVIN<span>AI</span></strong><small>VOICE INTELLIGENCE SYSTEM</small></div>
        </div>
        <div className="topbar-divider" />
        <div className="greeting-block">
          <strong>{formatGreeting(now)}, Alvin.</strong>
          <span>Your voice systems at a glance.</span>
        </div>
        <div className="topbar-spacer" />
        <div className="clock-block"><strong>{now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</strong><span>{now.toLocaleDateString([], { weekday: 'short', month: 'short', day: '2-digit', year: 'numeric' }).toUpperCase()}</span></div>
        <div className="topbar-actions">
          <button className="icon-button" type="button" aria-label="Check backend health" title="Check backend health" onClick={alvin.fetchHealth}><Icon name="refresh" /></button>
          <button className="icon-button" type="button" aria-label="Ping WebSocket" title="Ping WebSocket" onClick={alvin.sendPing} disabled={!isConnected}><Icon name="wifi" /></button>
          <button className={`icon-button ${showSettings ? 'active' : ''}`} type="button" aria-label="Settings" title="Settings" onClick={() => setShowSettings((v) => !v)}><Icon name="settings" /></button>
        </div>
      </header>

      <aside className="sidebar" ref={navRef}>
        <div className="sidebar-kicker">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => <NavItem key={item.label} item={item} active={activeNav === item.label} onClick={() => scrollToTarget(item)} />)}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-voice-card">
          <div className="sidebar-voice-header"><span className="status-led" /><span>VOICE SESSION</span></div>
          <strong>{alvin.isMicActive ? (alvin.isPushToTalk ? 'Push-to-talk active' : 'Microphone streaming') : 'Voice standby'}</strong>
          <span className="sidebar-voice-hint">
            {alvin.isPushToTalk ? 'PTT ACTIVE — RELEASE TO STOP' : (<>HOLD <kbd>SHIFT</kbd> + <kbd>Z</kbd> · PTT</>)}
          </span>
          <button type="button" className="sidebar-small-button" onClick={() => alvin.setVoiceReplies((v) => !v)}><span className={`toggle-dot ${alvin.voiceReplies ? 'on' : ''}`} /> Voice replies {alvin.voiceReplies ? 'ON' : 'OFF'}</button>
        </div>
        <div className="sidebar-footer">
          <a href="https://github.com/alwincasagan549-oss" target="_blank" rel="noopener noreferrer" style={{ color: 'inherit', textDecoration: 'none' }}>
            ALVIN AI · GITHUB
          </a>
          <span className="sidebar-footer-dot" />
        </div>
      </aside>

      <main className="main-content">
        <div className="page-heading">
          <div><p className="eyebrow">NEURAL INTERFACE / OVERVIEW</p><h1>Command <span>Center</span></h1><p className="heading-subtitle">Monitor the voice pipeline. Keep every signal in sync.</p></div>
          <div className="connection-actions">
            <StatusPill status={serverOnline ? 'online' : alvin.serverStatus === 'checking' ? 'checking' : 'offline'}>{serverOnline ? 'API HEALTHY' : alvin.serverStatus === 'checking' ? 'CHECKING API' : 'API OFFLINE'}</StatusPill>
            <StatusPill status={isConnected ? 'online' : alvin.connectionStatus === 'connecting' ? 'checking' : 'offline'}>{isConnected ? 'WS CONNECTED' : alvin.connectionStatus === 'connecting' ? 'WS CONNECTING' : 'WS DISCONNECTED'}</StatusPill>
            <button className="connect-button" onClick={() => canDisconnect ? setConfirmDisconnect(true) : alvin.connect()} type="button" disabled={alvin.connectionStatus === 'connecting'}>
              <span className="connect-button-led" />{connectLabel}
            </button>
          </div>
        </div>

        {alvin.lastError && <div className="error-banner" role="alert"><Icon name="activity" size={16} /><span>{alvin.lastError}</span><button onClick={alvin.clearError} type="button" aria-label="Dismiss error"><Icon name="close" size={15} /></button></div>}

        {showSettings && <section className="settings-panel" id="dashboard-settings"><div><strong>Dashboard settings</strong><p>Voice responses are {alvin.voiceReplies ? 'enabled' : 'disabled'}. The local development proxy expects ALVIN's FastAPI server on port 8000.</p></div><button className="outline-button small" onClick={() => alvin.setVoiceReplies((v) => !v)} type="button">Voice replies {alvin.voiceReplies ? 'ON' : 'OFF'}</button><button className="outline-button small" onClick={copyHealth} type="button"><Icon name="copy" size={14}/> Copy health JSON</button><span className="settings-endpoint">{alvin.wsUrl}</span></section>}

{alvin.lastYoutube && alvin.lastYoutube.videoId && (
  <section
    className="panel"
    id="now-playing"
    style={{ padding: 14, marginBottom: 16, display: 'flex', flexDirection: 'column', gap: 12 }}
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ color: alvin.lastYoutube.playing ? 'var(--green)' : 'var(--dim)', font: '11px var(--display)', letterSpacing: '.05em' }}>
        {alvin.lastYoutube.playing ? 'NOW PLAYING' : 'PAUSED'}
      </span>
      <span style={{ color: 'var(--text)', fontSize: 14, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {alvin.lastYoutube.title || alvin.lastYoutube.query || 'YouTube Video'}
      </span>
      {alvin.lastYoutube.playing ? (
        <>
          <button className="outline-button small" onClick={alvin.stopYoutube} type="button">STOP</button>
          <button className="outline-button small" onClick={alvin.unmuteYoutube} type="button" title="Unmute YouTube audio">
            🔇 Unmute
          </button>
        </>
      ) : (
        <button className="outline-button small" onClick={alvin.resumeYoutube} type="button">▶ RESUME</button>
      )}
      <a href={alvin.lastYoutube.url} target="_blank" rel="noopener noreferrer" className="outline-button small">
        Open ↗
      </a>
    </div>
    <iframe
      width="100%"
      height="340"
      // The src is stable across play/pause — only the YouTube IFrame API
      // (postMessage with playVideo/pauseVideo) controls playback, so the
      // player preserves its position when toggling play state.
      // mute=1 + autoplay=1 are required for Chrome/Edge/Safari to allow
      // autoplay without a direct user gesture; the Unmute button above
      // restores audio on click.
      src={`https://www.youtube.com/embed/${alvin.lastYoutube.videoId}?autoplay=1&mute=1&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}`}
      title="YouTube Player"
      frameBorder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowFullScreen
      style={{ borderRadius: 8, border: '1px solid var(--line)' }}
    />
  </section>
)}
{alvin.lastYoutube && !alvin.lastYoutube.videoId && (
  <section
    className="panel"
    id="now-playing"
    style={{ padding: 14, marginBottom: 16, display: 'flex', flexDirection: 'column', gap: 12 }}
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ color: 'var(--dim)', font: '11px var(--display)', letterSpacing: '.05em' }}>
        SEARCH FAILED
      </span>
      <span style={{ color: 'var(--text)', fontSize: 14, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {alvin.lastYoutube.query || 'YouTube Video'}
      </span>
      <span style={{ color: 'var(--dim)', fontSize: 12 }}>No video ID</span>
      <a href={alvin.lastYoutube.url} target="_blank" rel="noopener noreferrer" className="outline-button small">
        Open ↗
      </a>
    </div>
  </section>
)}

        <section className="top-grid" aria-label="Voice overview">
          <article className="panel hero-panel" id="voice-console">
            <div className="hero-panel-header"><PanelTitle icon={<Icon name="bolt" size={17} />} title="VOICE CONTROL CENTER" /></div>
            <div className="hero-body">
              <div className="hero-copy">
                <span className="live-eyebrow"><span className="pulse-dot" /> REAL-TIME VOICE INTERFACE</span>
                <h2>{alvin.isSpeaking ? 'ALVIN is speaking.' : alvin.isMicActive ? 'I’m listening.' : 'Your voice, in control.'}</h2>
                <p>{alvin.isSpeaking ? 'Streaming audio response from the voice engine.' : alvin.isMicActive ? 'Speak naturally. I’m listening for your next request.' : 'Speak naturally. ALVIN transcribes, thinks, and responds in real time.'}</p>
                <div className="hero-actions">
                  <button className={`primary-button ${alvin.isMicActive ? 'stop-action' : ''}`} onClick={startSession} type="button">
                    <Icon name={alvin.isMicActive ? 'stop' : isConnected ? 'mic' : 'bolt'} size={16} />
                    {alvin.isMicActive ? 'STOP LISTENING' : isConnected ? 'START LISTENING' : 'START SESSION'}
                  </button>
                  {alvin.isSpeaking && (
                    <button className="outline-button interrupt-action" onClick={alvin.interruptSpeech} type="button" title="Interrupt speech playback">
                      <Icon name="stop" size={15} />
                      INTERRUPT
                    </button>
                  )}
                  <button className={`outline-button ptt-button ${alvin.isPushToTalk ? 'holding' : ''}`} type="button" onPointerDown={(event) => { event.currentTarget.setPointerCapture?.(event.pointerId); if (isConnected && alvin.isMicActive) alvin.sendPtt(true); }} onPointerUp={() => alvin.sendPtt(false)} onPointerCancel={() => alvin.sendPtt(false)} onPointerLeave={() => alvin.sendPtt(false)} onContextMenu={(event) => event.preventDefault()} title="Use Shift+Z on your keyboard, or start the microphone before using push-to-talk." onClick={() => { if (!isConnected) alvin.connect(); else if (!alvin.isMicActive) alvin.toggleMicrophone(); }}>
                    <kbd>SHIFT</kbd> + <kbd>Z</kbd>
                  </button>
                  <button
                    className={`audio-mode-button ${alvin.voiceReplies ? 'enabled' : ''}`}
                    type="button"
                    onClick={() => alvin.setVoiceReplies((v) => !v)}
                    title="Toggle spoken replies"
                    aria-pressed={alvin.voiceReplies}
                  >
                    <span className={`toggle-dot ${alvin.voiceReplies ? 'on' : ''}`} />
                    {alvin.voiceReplies ? 'VOICE ON' : 'VOICE OFF'}
                  </button>
                </div>
                <div className="hero-footnote"><span className="tiny-live-dot" />{alvin.lastEvent}</div>
              </div>
              <Visualizer connected={isConnected} listening={alvin.isMicActive} speaking={alvin.isSpeaking} pushToTalk={alvin.isPushToTalk} pingLatency={alvin.pingLatency} />
            </div>
          </article>

          <article className="panel pipeline-panel">
            <PanelTitle icon={<Icon name="activity" size={17} />} title="REAL-TIME PIPELINE" right="4 STAGES" />
            <div className="pipeline-flow">
              <PipelineRow number="01" title="VOICE ACTIVITY" description="Silero VAD · 32 ms frames" status={isConnected ? 'LIVE' : 'READY'} state={isConnected ? 'active' : 'configured'} />
              <div className="pipeline-connector"><span /></div>
              <PipelineRow number="02" title="SPEECH TO TEXT" description={sttProvider === 'groq' ? `Groq ${health?.model || ''} · local fallback` : sttProvider === 'local' ? `Local ${health?.fallback_model || 'Whisper'} only` : 'Awaiting /health'} status={sttProvider ? sttProvider.toUpperCase() : '—'} state="configured" />
              <div className="pipeline-connector"><span /></div>
              <PipelineRow number="03" title="LLM BRAIN" description={llmEnabled ? (llmModel || 'Cloudflare / Kilo gateway') : 'Disabled on this deployment'} status={llmEnabled === true ? 'ENABLED' : llmEnabled === false ? 'DISABLED' : '—'} state="configured" />
              <div className="pipeline-connector"><span /></div>
              <PipelineRow number="04" title="VOICE OUTPUT" description={ttsVoice ? `edge-tts · ${ttsVoice}` : 'edge-tts · 16 kHz PCM'} status={alvin.isSpeaking ? 'SPEAKING' : 'AVAILABLE'} state={alvin.isSpeaking ? 'active' : 'configured'} />
            </div>
            <div className="pipeline-footer"><span className="status-led" /> Provider setup comes from the backend configuration.</div>
          </article>
        </section>

        <section className="metrics-grid" aria-label="Runtime metrics">
          <MetricCard icon={<Icon name="pulse" size={17} />} label="AUDIO SAMPLE RATE" value={sampleRateKhz ? `${sampleRateKhz} kHz` : '—'} detail="Mono PCM input / output" />
          <MetricCard icon={<Icon name="wifi" size={17} />} label="LIVE CONNECTIONS" value={formatUptimeOrConnections(alvin.serverHealth)} detail={alvin.serverHealth ? `Maximum ${maxConnections ?? 'not reported'} sessions` : 'Awaiting /health response'} state={serverOnline ? 'normal' : 'muted'} />
          <MetricCard
            icon={<Icon name="activity" size={17} />}
            label="ECHO GUARD"
            value={echoMode ? echoMode.toUpperCase() : '—'}
            detail={echoMode ? `≥ ${echoThreshold ?? '—'} · ${echoCooldown ?? '—'} ms` : 'Not reported by /health'}
            state={serverOnline ? 'normal' : 'muted'}
          />
          <MetricCard icon={<Icon name="memory" size={17} />} label="CONVERSATION HISTORY" value={historyTurns ? `${historyTurns} turns` : '—'} detail="Configured per connection" />
        </section>

        <section className="bottom-grid">
          <article className="panel conversation-panel" id="conversation">
            <div className="panel-title-row"><PanelTitle icon={<Icon name="audio" size={17} />} title="LIVE CONVERSATION" /><span className="panel-title-right live-state"><span className="tiny-live-dot" />{isConnected ? 'LIVE SOCKET' : 'PREVIEW'}</span></div>
            <div className="conversation-route"><span>REALTIME SOCKET</span><b>/ws/transcribe</b><button type="button" onClick={alvin.clearConversation} aria-label="Clear conversation history" title="Clear conversation history"><Icon name="close" size={13} /></button><button type="button" onClick={alvin.sendPing} disabled={!isConnected} aria-label="Send ping" title="Send ping"><Icon name="refresh" size={13} /></button></div>
            <div className="message-list" aria-live="polite" ref={messageListRef} onScroll={trackMessageListScroll}>
              {alvin.messages.map((message) => <Message key={message.id} message={message} />)}
              {alvin.transcript && <div className="message message-partial"><div className="message-meta"><span className="message-avatar">...</span><span>LIVE TRANSCRIPTION</span><span className="listening-animation"><i /><i /><i /></span></div><p>{alvin.transcript}<span className="typing-caret" /></p></div>}
            </div>
            <div className="conversation-composer">
              <div className="composer-leading"><Icon name="mic" size={16} /></div>
              <div className="composer-text"><strong>{alvin.isMicActive ? 'MICROPHONE STREAMING' : 'MICROPHONE STANDBY'}</strong><span>{alvin.isMicActive ? `${sampleRateKhz || 16} kHz · mono PCM` : 'Start a session to send audio'}</span></div>
              <button type="button" className={alvin.isMicActive ? 'round-action active' : 'round-action'} onClick={alvin.toggleMicrophone} aria-label={alvin.isMicActive ? 'Stop microphone' : 'Start microphone'} title={alvin.isMicActive ? 'Stop microphone' : 'Start microphone'} disabled={!isConnected}><Icon name={alvin.isMicActive ? 'stop' : 'arrow'} size={17} /></button>
            </div>
          </article>

<article className="panel intents-panel" id="intent-router">
            <PanelTitle icon={<Icon name="command" size={17} />} title="INTENT ROUTER" right={intentsEnabled === false ? 'DISABLED' : 'DETERMINISTIC'} />
            <p className="panel-intro">Device commands can be routed without involving the LLM.</p>
            <div className="intent-list">
              {intentRows.map(({ name, description, icon }) => <div className="intent-row" key={name}><div className="intent-icon"><Icon name={icon} size={17} /></div><div className="intent-copy"><strong>{name.toUpperCase()}</strong><span>{description}</span></div><span className={alvin.lastIntent?.intent === name ? 'intent-status active' : 'intent-status'}>
  {alvin.lastIntent?.intent === name
    ? 'MATCHED'
    : intentsEnabled === false
    ? 'DISABLED'
    : 'SUPPORTED'}</span></div>)}
            </div>
            <div className="intent-bottom">
              <div><span>LAST MATCH</span><strong>{alvin.lastIntent?.intent || 'No command yet'}</strong></div>
              <div><span>WAKE WORD</span><strong>{wakeWords || '—'}</strong></div>
              <div><span>MIN CONFIDENCE</span><strong>{minConfidence != null ? minConfidence.toFixed(2) : '—'}</strong></div>
            </div>
            <p className="panel-intro" style={{ marginTop: 8, marginBottom: 0, fontSize: 11 }}>
              {intentRows.length} deterministic command{intentRows.length === 1 ? '' : 's'} registered on the server.
            </p>
          </article>

          <article className="panel diagnostics-panel" id="system-diagnostics">
            <PanelTitle icon={<Icon name="cpu" size={17} />} title="SYSTEM DIAGNOSTICS" right="LIVE HEALTH" />
            <p className="panel-intro">Values below are based on the current backend health response and documented configuration.</p>
            <div className="diagnostic-status"><span className={`diagnostic-icon ${serverOnline ? 'online' : 'offline'}`}><Icon name={serverOnline ? 'check' : 'activity'} size={18} /></span><div><strong>{serverOnline ? 'API endpoint reachable' : 'Waiting for backend'}</strong><span>{serverOnline ? 'GET /health returned successfully' : 'Start the FastAPI server on port 8000'}</span></div><StatusPill status={serverOnline ? 'online' : 'offline'}>{serverOnline ? 'ONLINE' : 'OFFLINE'}</StatusPill></div>
            <div className="diagnostic-rows">
              <div><span>HTTP HEALTH</span><strong>/health</strong></div>
              <div><span>WEBSOCKET</span><strong>/ws/transcribe</strong></div>
              <div><span>ACTIVE CONNECTIONS</span><strong>{Number.isFinite(activeConnections) ? activeConnections : 'Not reported'}</strong></div>
              <div><span>MAX CONNECTIONS</span><strong>{Number.isFinite(maxConnections) ? maxConnections : 'Not reported'}</strong></div>
              <div><span>AUTH REQUIRED</span><strong>{typeof authRequired === 'boolean' ? (authRequired ? 'YES' : 'NO') : 'Not reported'}</strong></div>
              <div><span>ACTIVE LLM</span><strong>{alvin.modelName || 'No reply yet'}</strong></div>
            </div>
            <div className="diagnostic-config">
              <div className="config-row"><span className={`config-led ${sttProvider ? '' : 'dim'}`} /><div><strong>Groq Whisper</strong><small>{sttProvider === 'groq' ? 'Cloud STT · local fallback available' : sttProvider === 'local' ? `Local ${health?.fallback_model || 'Whisper'} only` : 'Not reported by /health'}</small></div><span>{sttProvider === 'groq' ? 'CONFIGURED' : sttProvider === 'local' ? 'LOCAL ONLY' : 'UNKNOWN'}</span></div>
              <div className="config-row"><span className={`config-led ${llmEnabled ? '' : 'dim'}`} /><div><strong>LLM providers</strong><small>{llmEnabled ? (llmModel || 'Cloudflare / Kilo gateway') : 'Set LLM_ENABLED=true to enable'}</small></div><span>{llmEnabled === true ? 'ENABLED' : llmEnabled === false ? 'DISABLED' : 'UNKNOWN'}</span></div>
              <div className="config-row"><span className="config-led" /><div><strong>edge-tts</strong><small>{ttsVoice || 'Voice not reported'}</small></div><span>{alvin.isSpeaking ? 'SPEAKING' : 'AVAILABLE'}</span></div>
              <div className={`config-row ${alvin.currentSpeaker ? '' : (health?.speaker?.enabled ? 'dim' : '')}`}><span className={`config-led ${alvin.currentSpeaker ? '' : 'dim'}`} /><div><strong>Speaker ID</strong><small>{alvin.currentSpeaker ? alvin.currentSpeaker.name : (health?.speaker?.enabled ? 'Listening for enrolled voice' : 'Disabled')}</small></div><span>{alvin.currentSpeaker ? 'MATCHED' : (health?.speaker?.enabled ? 'IDLE' : 'DISABLED')}</span></div>
            </div>
            <div className="diagnostic-footer"><span>LAST EVENT</span><strong>{alvin.lastEvent}</strong><button type="button" onClick={alvin.fetchHealth} aria-label="Refresh health" title="Refresh health"><Icon name="refresh" size={14} /></button></div>
          </article>
        </section>

        <footer className="app-footer">
          <span><i className="footer-signal" /> ALVIN AI / VOICE INTELLIGENCE</span>
          <span>FASTAPI <b>·</b> WEBSOCKET <b>·</b> PCM {sampleRateKhz ? `${sampleRateKhz} KHZ` : '—'}</span>
          <span>{isConnected ? 'LIVE SESSION' : 'STANDBY'}</span>
        </footer>
      </main>

      {confirmDisconnect && <div className="modal-backdrop" role="presentation" onClick={() => setConfirmDisconnect(false)}><div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="disconnect-title" ref={dialogRef} onClick={(e) => e.stopPropagation()}><div className="dialog-icon"><Icon name="close" size={23} /></div><h2 id="disconnect-title">Disconnect ALVIN?</h2><p>This closes the live WebSocket and stops microphone capture. Your conversation shown here will remain until the page is refreshed.</p><div className="dialog-actions"><button className="outline-button" ref={cancelButtonRef} onClick={() => setConfirmDisconnect(false)} type="button">Cancel</button><button className="primary-button danger-button" onClick={() => { alvin.disconnect(); setConfirmDisconnect(false); }} type="button">Disconnect</button></div></div></div>}
    </div>
  );
}
