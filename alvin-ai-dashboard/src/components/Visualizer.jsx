import { useEffect, useMemo, useState } from 'react';

const W = 440;
const H = 260;
const CX = 220;
const CY = 116;

function polar(radius, angle) {
  const rad = (angle * Math.PI) / 180;
  return [CX + Math.cos(rad) * radius, CY + Math.sin(rad) * radius];
}

function Ring({ radius, stroke = '#25d9ff', width = 1, opacity = 0.5, dash, rotate = 0, className = '' }) {
  return (
    <circle
      className={className}
      cx={CX}
      cy={CY}
      r={radius}
      fill="none"
      stroke={stroke}
      strokeWidth={width}
      strokeOpacity={opacity}
      strokeDasharray={dash || undefined}
      transform={`rotate(${rotate} ${CX} ${CY})`}
    />
  );
}

export default function Visualizer({ connected, listening, speaking, pushToTalk, pingLatency }) {
  const [tick, setTick] = useState(0);
  const particles = useMemo(() => Array.from({ length: 132 }, (_, i) => {
    const angle = i * 2.399963;
    const radius = 74 + ((Math.sin(i * 1.67) + 1) * 12.5) + (i % 4) * 1.7;
    return {
      x: CX + Math.cos(angle) * radius,
      y: CY + Math.sin(angle) * radius * 0.88,
      r: 0.4 + (i % 5) * 0.18,
      opacity: 0.11 + (i % 6) * 0.07,
      color: i % 7 === 0 ? '#eaffff' : i % 3 === 0 ? '#69cfff' : '#7ceeff',
    };
  }), []);

  useEffect(() => {
    const timer = window.setInterval(() => setTick((v) => v + 1), 150);
    return () => window.clearInterval(timer);
  }, []);

  const waveform = useMemo(() => Array.from({ length: 82 }, (_, i) => {
    const env = Math.min(1, (i + 2) / 9, (82 - i) / 9);
    const wave = 4 + Math.abs(Math.sin(i * 0.36 + tick * (speaking ? 0.16 : listening ? 0.06 : 0.012))) * (speaking ? 14 : listening ? 10 : 7)
      + Math.abs(Math.sin(i * 0.11 + 0.7)) * 7;
    return Math.max(2, wave * env);
  }), [tick, speaking, listening]);

  const mode = speaking ? 'SPEAKING' : pushToTalk ? 'PUSH TO TALK' : listening ? 'LISTENING' : connected ? 'ONLINE' : 'STANDBY';
  const statusClass = speaking ? 'is-speaking' : pushToTalk || listening ? 'is-listening' : connected ? 'is-online' : 'is-idle';

  return (
    <div className={`visualizer ${statusClass}`} aria-label={`ALVIN visualizer: ${mode.toLowerCase()}`}>
      <div className="visualizer-status-pill">
        <span className="status-led" />
        <span>{mode}</span>
      </div>
      <svg className="visualizer-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Holographic AI orbital animation and audio spectrum">
        <defs>
          <radialGradient id="orbital-atmosphere">
            <stop offset="0" stopColor="#0b8199" stopOpacity=".23" />
            <stop offset=".58" stopColor="#064866" stopOpacity=".09" />
            <stop offset="1" stopColor="#04111b" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="ai-core">
            <stop offset="0" stopColor="#d9ffff" stopOpacity=".28" />
            <stop offset=".28" stopColor="#29daff" stopOpacity=".12" />
            <stop offset=".78" stopColor="#0c4560" stopOpacity=".6" />
            <stop offset="1" stopColor="#041a27" stopOpacity=".96" />
          </radialGradient>
          <linearGradient id="spectrum-gradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#ee66ff" />
            <stop offset=".17" stopColor="#9874ff" />
            <stop offset=".34" stopColor="#3999ff" />
            <stop offset=".49" stopColor="#2be1ff" />
            <stop offset=".64" stopColor="#25f0b2" />
            <stop offset=".78" stopColor="#d0ee55" />
            <stop offset=".9" stopColor="#ffb14a" />
            <stop offset="1" stopColor="#ff5d73" />
          </linearGradient>
          <filter id="cyan-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="3.4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="core-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <ellipse cx={CX} cy={CY} rx="202" ry="124" fill="url(#orbital-atmosphere)" />

        {/* Ambient star field */}
        {particles.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={p.r} fill={p.color} opacity={p.opacity} />)}
        {Array.from({ length: 20 }, (_, i) => {
          const x = (i * 47.37 + 11) % 430;
          const y = (i * 31.18 + 9) % 202;
          return <path key={`streak-${i}`} d={`M ${x} ${y} l ${3 + i % 4} ${((i % 3) - 1) * 1.2}`} stroke="#53d5fa" strokeOpacity=".15" strokeWidth=".7" />;
        })}

        {/* Holographic orbital rings */}
        <g className="orbital-ring slow-orbit"><Ring radius={107} stroke="#248cff" width={1.1} opacity={0.6} dash="2 8" /></g>
        <Ring radius={101} stroke="#248cff" width={1.1} opacity={0.4} />
        <g className="orbital-ring fast-orbit"><Ring radius={91} stroke="#2bdfff" width={1.1} opacity={0.65} dash="38 5 3 6" rotate={-16} /></g>
        <Ring radius={82} stroke="#49bfff" width={1.4} opacity={0.85} />
        <g className="orbital-ring reverse-orbit"><Ring radius={73} stroke="#5aafff" width={1.1} opacity={0.65} dash="64 5 14 4 6 9" rotate={19} /></g>
        <Ring radius={62} stroke="#92f7ff" width={0.8} opacity={0.34} dash="1 2" />
        <Ring radius={53} stroke="#28dfff" width={2.3} opacity={0.92} className="ring-bloom" />
        <Ring radius={45} stroke="#188dff" width={1.3} opacity={0.82} />

        {/* Scattered luminous filaments */}
        {Array.from({ length: 11 }, (_, j) => {
          const points = Array.from({ length: 85 }, (_, i) => {
            const a = (i / 84) * Math.PI * 2;
            const radius = 79 + j * 1.08 + Math.sin(a * (2 + j % 4) + j * 0.8) * 3.2 + Math.sin(a * 9 - j * 1.3) * 1.5;
            return `${CX + Math.cos(a) * radius},${CY + Math.sin(a) * radius * .88}`;
          }).join(' ');
          return <polyline key={`filament-${j}`} points={points} fill="none" stroke={j % 3 === 0 ? '#d7ffff' : '#55dfff'} strokeOpacity={j % 3 === 0 ? '.28' : '.12'} strokeWidth={j % 4 === 0 ? 1 : .55} />;
        })}

        {/* Radar ticks */}
        {Array.from({ length: 64 }, (_, i) => {
          const angle = -90 + i * (360 / 64);
          const major = i % 8 === 0;
          const medium = i % 4 === 0;
          const [x, y] = polar(108, angle);
          const len = major ? 8 : medium ? 5 : 2.8;
          return <line key={`tick-${i}`} x1={x} y1={y - len / 2} x2={x} y2={y + len / 2} stroke={major ? '#d6fcff' : medium ? '#2bdfff' : '#188dff'} strokeWidth={major ? 1.6 : .8} opacity={major ? .95 : medium ? .66 : .35} transform={`rotate(${angle + 90} ${x} ${y})`} />;
        })}

        {/* Side data rails */}
        <g className="telemetry-rail">
          <path d="M 0 116 H 98 M 342 116 H 440" stroke="#39dbff" strokeOpacity=".65" strokeWidth="1" />
          <circle cx="98" cy="116" r="3" fill="#45e6ff" filter="url(#cyan-glow)" />
          <circle cx="342" cy="116" r="3" fill="#45b6ff" filter="url(#cyan-glow)" />
          <path d="M 12 131 h 27 M 12 135 h 20 M 401 131 h 26 M 401 135 h 18" stroke="#72d9f0" strokeOpacity=".45" strokeWidth=".8" />
          <text x="7" y="91" className="tiny-hud">
            PING: {connected && pingLatency != null ? `${pingLatency}ms` : '—'}
          </text>
          <text x="7" y="106" className="micro-hud">
            {connected ? (pingLatency != null ? 'WS LATENCY' : 'MEASURING') : 'NO SOCKET'}
          </text>
          <text x="358" y="91" className="tiny-hud">
            LINK: {connected ? 'UP' : 'DOWN'}
          </text>
          <text x="358" y="106" className="micro-hud">
            {speaking ? 'AUDIO OUT' : listening ? 'AUDIO IN' : 'IDLE'}
          </text>
          <text x="329" y="41" className="micro-hud">{connected ? 'ONLINE' : 'OFFLINE'}</text>
          <text x="329" y="54" className="micro-hud">STATUS</text>
        </g>

        {/* Main AI glass core */}
        <circle className="core-back-glow" cx={CX} cy={CY} r="39" fill="#32ddff" opacity=".17" filter="url(#core-glow)" />
        <circle cx={CX} cy={CY} r="36" fill="url(#ai-core)" stroke="#b7fbff" strokeOpacity=".75" strokeWidth="1.1" />
        <circle cx={CX} cy={CY} r="29" fill="none" stroke="#4fe9ff" strokeOpacity=".45" strokeWidth=".8" />
        <path d={`M ${CX - 21} ${CY - 21} Q ${CX} ${CY - 36} ${CX + 18} ${CY - 23}`} fill="none" stroke="#efffff" strokeOpacity=".8" strokeWidth="1.4" />
        <text x={CX} y={CY + 10} textAnchor="middle" className="ai-core-label">AI</text>
        <text x={CX} y="184" textAnchor="middle" className="alvin-online-label">ALVIN / {connected ? 'ONLINE' : 'OFFLINE'}</text>

        {/* Metallic HUD base */}
        <g className="metal-platform">
          <path d="M 6 226 L 39 210 L 401 210 L 434 226 L 434 239 L 6 239 Z" fill="#102b3a" fillOpacity=".82" stroke="#9eefff" strokeOpacity=".68" strokeWidth="1" />
          <path d="M 17 230 L 49 216 L 391 216 L 423 230" fill="none" stroke="#4cddff" strokeOpacity=".5" strokeWidth=".8" />
          <path d="M 26 235 L 64 222 L 112 222 L 128 227 L 181 227 L 194 222 L 246 222 L 263 227 L 311 227 L 327 222 L 376 222 L 414 235" fill="none" stroke="#70d9fa" strokeOpacity=".32" strokeWidth=".6" />
          <path d="M 0 226 l 25 -12 h 24 M 391 214 h 24 l 25 12" fill="none" stroke="#d9fcff" strokeOpacity=".75" strokeWidth="1.1" />
        </g>

        {/* Audio spectrum */}
        {waveform.map((h, i) => {
          const x = 45 + i * 4.28;
          return <rect key={`wave-${i}`} x={x} y={214 - h / 2} width="1.9" height={h} rx=".8" fill="url(#spectrum-gradient)" opacity={listening || speaking ? .98 : .78} />;
        })}
        {Array.from({ length: 45 }, (_, i) => {
          const x = 58 + i * 7.2;
          const major = i % 5 === 0;
          return <line key={`cal-${i}`} x1={x} x2={x} y1="235" y2={major ? 240 : 238} stroke="#d7faff" strokeOpacity={major ? .52 : .21} strokeWidth={major ? 1 : .6} />;
        })}
      </svg>
      <div className="visualizer-footer">
        <span><i className="tiny-live-dot" /> {connected ? 'CORE LINK ESTABLISHED' : 'CORE LINK STANDBY'}</span>
        <span>{speaking ? 'AUDIO OUT' : listening ? 'AUDIO IN' : 'NEURAL INTERFACE'}</span>
      </div>
    </div>
  );
}
