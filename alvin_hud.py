"""ALVIN AI - J.A.R.V.I.S.-style holographic voice interface (PySide6).

Controls
    Click the core / click the mic button / press SPACE  -> mic on / off
    F2 / SYS button                                      -> system settings window
    F3 / CAM button                                      -> camera preview on / off
    F11                                                  -> fullscreen
    ESC                                                  -> leave fullscreen

Voice
    Say "Alvin open your system" to open the embedded system settings
    window. Say "open the camera" / "close the camera" to control the
    camera preview. The HUD is the main frontend; the React dashboard is
    loaded inside it as the system-settings view.

Optional
    * pip install psutil      -> real CPU / memory gauges
    * Drop Orbitron or Rajdhani .ttf files in a "fonts" folder next to this
      script for the most authentic sci-fi look (picked up automatically).
"""
import asyncio
import base64
import html
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from array import array
from collections import deque
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import websockets
from PySide6.QtCore import QPointF, QRectF, QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

try:
    import psutil
except ImportError:  # CPU / memory gauges simply show N/A
    psutil = None

try:
    import cv2
except ImportError:  # camera preview is optional
    cv2 = None

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # embedded system window degrades to a browser tab
    QWebEngineView = None

# Set to False to use the normal OS window frame instead of the custom title bar.
FRAMELESS = True
WS_URL = "ws://127.0.0.1:8000/ws/transcribe"
DASHBOARD_URL = "http://127.0.0.1:5173"

# --------------------------------------------------------------------------
# Palette & helpers
# --------------------------------------------------------------------------
CYAN = QColor("#00e5ff")
ICE = QColor("#dff8ff")
BLUE = QColor("#1e90ff")
AMBER = QColor("#ffb020")
RED = QColor("#ff3b5c")
GREEN = QColor("#2dffb3")
BG_DEEP = QColor("#020912")
BG_MID = QColor("#07203a")

FONT_DISPLAY = "Segoe UI"
FONT_MONO = "Consolas"


def rgba(color, alpha=255):
    c = QColor(color)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


def make_font(kind, px, bold=False, spacing=0.0):
    f = QFont(FONT_DISPLAY if kind == "display" else FONT_MONO)
    f.setPixelSize(px)
    f.setBold(bold)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


def init_fonts():
    """Load optional bundled fonts, then pick the best available families."""
    global FONT_DISPLAY, FONT_MONO
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    if fonts_dir.is_dir():
        for path in list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf")):
            QFontDatabase.addApplicationFont(str(path))
    families = set(QFontDatabase.families())

    def pick(candidates, fallback):
        return next((c for c in candidates if c in families), fallback)

    FONT_DISPLAY = pick(
        ["Orbitron", "Rajdhani", "Bahnschrift", "Segoe UI", "Helvetica Neue", "DejaVu Sans"],
        FONT_DISPLAY,
    )
    FONT_MONO = pick(
        ["Cascadia Mono", "JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono"],
        FONT_MONO,
    )


def rms_level(raw: bytes) -> float:
    """Cheap 0..1 loudness estimate of 16-bit mono PCM."""
    n = len(raw) // 2
    if n == 0:
        return 0.0
    samples = array("h")
    samples.frombytes(raw[: n * 2])
    step = max(1, n // 64)
    picked = samples[::step]
    energy = sum(s * s for s in picked) / len(picked)
    return min(1.0, math.sqrt(energy) / 32768.0 * 6.0)


# --------------------------------------------------------------------------
# Audio & WebSocket worker (protocol unchanged, now with levels + auto-retry)
# --------------------------------------------------------------------------
# Longest a single spoken turn may keep the client-side mic gate closed before
# the watchdog reopens it. Replies are short; this only fires on a lost frame.
_SPEAKING_WATCHDOG_SEC = 60.0


class BackendWorker(QObject):
    status_changed = Signal(str, str)
    partial_transcript = Signal(str)
    final_transcript = Signal(str)
    llm_received = Signal(str)
    mic_level = Signal(float)
    out_level = Signal(float)
    link_changed = Signal(bool)
    #: ``command`` events from the server: (intent, slots).
    command_received = Signal(str, object)
    #: The server wants a fresh camera frame (``what do you see``).
    camera_frame_requested = Signal()

    def __init__(self, ws_url=WS_URL):
        super().__init__()
        self.ws_url = ws_url
        self.loop = asyncio.new_event_loop()
        self.is_listening = False
        self.is_speaking = False
        self._speaking_since = 0.0
        self.mic_stream = None
        self.play_stream = None
        self.ws = None
        self._running = True
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self.loop_thread.start()

    def stop(self):
        self._running = False
        self._close_streams()

    def send_json(self, payload: dict) -> None:
        """Queue a JSON control frame (camera frames, etc.) from the GUI thread."""
        ws = self.ws
        if ws is None or not self.loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(payload)), self.loop)
        except Exception:
            pass

    def _close_streams(self):
        for name in ("mic_stream", "play_stream"):
            stream = getattr(self, name)
            setattr(self, name, None)
            try:
                if stream:
                    stream.stop()
                    stream.close()
            except Exception:
                pass

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._supervise())
        except Exception:
            pass

    async def _supervise(self):
        """Keep trying to reach the backend so the HUD survives a server restart."""
        while self._running:
            await self._connect()
            self.ws = None
            self.is_speaking = False
            self.link_changed.emit(False)
            if not self._running:
                break
            await asyncio.sleep(3)

    async def _connect(self):
        try:
            async with websockets.connect(self.ws_url) as ws:
                self.ws = ws
                self.link_changed.emit(True)
                self.status_changed.emit("LISTENING" if self.is_listening else "CONNECTED", "#00e5ff")
                await ws.send(
                    json.dumps({"type": "config", "language": "en", "client": "hud"})
                )

                self.play_stream = sd.RawOutputStream(samplerate=16000, channels=1, dtype="int16")
                self.play_stream.start()

                send_queue = asyncio.Queue()

                async def sender():
                    while True:
                        data = await send_queue.get()
                        try:
                            await ws.send(data)
                        except Exception:
                            pass

                sender_task = asyncio.create_task(sender())

                def mic_callback(indata, frames, time_info, status):
                    # Safety valve: if a tts_end/tts_stop frame is ever lost, the
                    # mic gate must not stay closed for the rest of the session.
                    if (
                        self.is_speaking
                        and time.monotonic() - self._speaking_since > _SPEAKING_WATCHDOG_SEC
                    ):
                        self.is_speaking = False
                    if status:
                        return
                    if self.is_listening and not self.is_speaking and self.ws is not None:
                        raw = bytes(indata)
                        self.mic_level.emit(rms_level(raw))
                        asyncio.run_coroutine_threadsafe(send_queue.put(raw), self.loop)

                self.mic_stream = sd.RawInputStream(
                    samplerate=16000, channels=1, dtype="int16", callback=mic_callback
                )
                self.mic_stream.start()

                try:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            if not self.is_speaking:
                                self._speaking_since = time.monotonic()
                            self.is_speaking = True
                            self.status_changed.emit("SPEAKING", "#ffb020")
                            self.out_level.emit(rms_level(msg))
                            self.play_stream.write(msg)
                        else:
                            self._handle_json(json.loads(msg))
                finally:
                    sender_task.cancel()
        except Exception:
            pass
        finally:
            self._close_streams()
            if self._running:
                self.status_changed.emit("OFFLINE", "#ff3b5c")

    def _handle_json(self, data):
        if "transcript" in data:
            if data.get("is_partial"):
                self.partial_transcript.emit(data["transcript"])
            elif data.get("is_final"):
                self.partial_transcript.emit("")
                if data["transcript"].strip():
                    self.final_transcript.emit(data["transcript"])
        elif data.get("type") == "llm_response":
            self.llm_received.emit(data.get("text", ""))
        elif data.get("type") == "command":
            self.command_received.emit(data.get("intent", ""), data.get("slots") or {})
        elif data.get("type") == "request_camera_frame":
            self.camera_frame_requested.emit()
        elif data.get("type") in ("tts_end", "tts_stop"):
            # tts_stop means the server cancelled playback (barge-in, stop,
            # stop_speak) and no tts_end will follow. The mic gate must reopen
            # here, otherwise the HUD silently stops sending microphone audio
            # for the rest of the session.
            self.is_speaking = False
            self.status_changed.emit("LISTENING" if self.is_listening else "STANDBY", "#00e5ff")


# --------------------------------------------------------------------------
# Backdrop: deep-blue gradient, hex grid, scanlines, vignette (cached)
# --------------------------------------------------------------------------
class Stage(QWidget):
    def __init__(self, framed):
        super().__init__()
        self.framed = framed
        self._cache = None
        self._cache_key = None

    def _build(self, w, h, dpr):
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        bg = QRadialGradient(w / 2, h * 0.46, max(w, h) * 0.72)
        bg.setColorAt(0, BG_MID)
        bg.setColorAt(1, BG_DEEP)
        p.fillRect(0, 0, w, h, bg)

        # Hex grid
        s = 30.0
        dx, dy = 1.5 * s, math.sqrt(3) * s
        rng = random.Random(7)
        p.setPen(QPen(rgba(CYAN, 14), 1))
        p.setBrush(Qt.NoBrush)
        x, col = 0.0, 0
        lit = []
        while x < w + s:
            y = (dy / 2 if col % 2 else 0.0) - dy
            while y < h + dy:
                poly = QPolygonF(
                    [
                        QPointF(x + s * math.cos(math.radians(60 * i)), y + s * math.sin(math.radians(60 * i)))
                        for i in range(6)
                    ]
                )
                p.drawPolygon(poly)
                if rng.random() < 0.012:
                    lit.append(poly)
                y += dy
            x += dx
            col += 1
        p.setPen(Qt.NoPen)
        for poly in lit:
            p.setBrush(rgba(CYAN, rng.randint(10, 22)))
            p.drawPolygon(poly)

        # Scanlines
        p.setPen(QPen(rgba(CYAN, 5), 1))
        for yy in range(0, h, 4):
            p.drawLine(0, yy, w, yy)

        # Vignette
        vg = QRadialGradient(w / 2, h / 2, max(w, h) * 0.78)
        vg.setColorAt(0.55, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 190))
        p.setPen(Qt.NoPen)
        p.fillRect(0, 0, w, h, vg)

        if self.framed:
            p.setPen(QPen(rgba(CYAN, 70), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(0, 0, w - 1, h - 1)
        p.end()
        return pm

    def paintEvent(self, _):
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        key = (w, h, dpr)
        if key != self._cache_key:
            self._cache = self._build(w, h, dpr)
            self._cache_key = key
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)


# --------------------------------------------------------------------------
# Reusable HUD panel with chamfered corners and bracket accents
# --------------------------------------------------------------------------
class HoloPanel(QWidget):
    def __init__(self, title=None):
        super().__init__()
        self.title = title
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(16, 50 if title else 12, 16, 16 if title else 12)
        self.lay.setSpacing(10)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, cut = self.width(), self.height(), 16
        poly = QPolygonF(
            [
                QPointF(cut, 0.5),
                QPointF(w - 0.5, 0.5),
                QPointF(w - 0.5, h - cut),
                QPointF(w - cut, h - 0.5),
                QPointF(0.5, h - 0.5),
                QPointF(0.5, cut),
            ]
        )
        fill = QLinearGradient(0, 0, 0, h)
        fill.setColorAt(0, QColor(8, 40, 68, 175))
        fill.setColorAt(1, QColor(3, 14, 26, 160))
        p.setPen(QPen(rgba(CYAN, 70), 1))
        p.setBrush(fill)
        p.drawPolygon(poly)

        bright = QPen(rgba(CYAN, 235), 2, Qt.SolidLine, Qt.SquareCap)
        p.setPen(bright)
        p.drawLine(QPointF(w - 24, 1), QPointF(w - 1, 1))
        p.drawLine(QPointF(w - 1, 1), QPointF(w - 1, 24))
        p.drawLine(QPointF(1, h - 24), QPointF(1, h - 1))
        p.drawLine(QPointF(1, h - 1), QPointF(24, h - 1))
        p.setPen(QPen(rgba(CYAN, 200), 1.5))
        p.drawLine(QPointF(0.5, cut), QPointF(cut, 0.5))
        p.drawLine(QPointF(w - 0.5, h - cut), QPointF(w - cut, h - 0.5))

        if self.title:
            p.setPen(Qt.NoPen)
            p.setBrush(CYAN)
            p.drawRect(QRectF(18, 19, 6, 6))
            p.setFont(make_font("display", 12, True, 3))
            p.setPen(rgba(ICE, 235))
            p.drawText(QRectF(32, 10, w - 100, 26), Qt.AlignLeft | Qt.AlignVCenter, self.title)
            line = QLinearGradient(16, 0, w - 16, 0)
            line.setColorAt(0, rgba(CYAN, 170))
            line.setColorAt(1, rgba(CYAN, 0))
            p.setPen(QPen(line, 1))
            p.drawLine(QPointF(16, 40), QPointF(w - 16, 40))
            p.setPen(Qt.NoPen)
            for i, a in enumerate((230, 140, 70)):
                p.setBrush(rgba(CYAN, a))
                p.drawRect(QRectF(w - 62 + i * 16, 21, 12, 3))


# --------------------------------------------------------------------------
# Small data widgets
# --------------------------------------------------------------------------
class KeyValue(QWidget):
    def __init__(self, key):
        super().__init__()
        self.key, self.value, self.color = key, "--", QColor(ICE)
        self.setFixedHeight(26)

    def set_value(self, value, color=None):
        self.value = value
        if color is not None:
            self.color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setFont(make_font("mono", 11, False, 1.5))
        p.setPen(rgba(ICE, 120))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignLeft | Qt.AlignVCenter, self.key)
        p.setFont(make_font("mono", 12, True, 1))
        p.setPen(self.color)
        p.drawText(QRectF(0, 0, w, h), Qt.AlignRight | Qt.AlignVCenter, self.value)
        p.setPen(QPen(rgba(CYAN, 28), 1))
        p.drawLine(0, h - 1, w, h - 1)


class SegBar(QWidget):
    def __init__(self, label, segments=22):
        super().__init__()
        self.label, self.segments = label, segments
        self.value, self.display = 0.0, 0.0
        self.setFixedHeight(34)

    def set_value(self, value):
        self.value = value  # None -> N/A

    def tick(self, dt):
        target = 0.0 if self.value is None else self.value
        new = self.display + (target - self.display) * min(1.0, dt * 10)
        if abs(new - self.display) > 0.002:
            self.display = new
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w = self.width()
        p.setFont(make_font("mono", 10, False, 1.5))
        p.setPen(rgba(ICE, 150))
        p.drawText(QRectF(0, 0, w, 14), Qt.AlignLeft | Qt.AlignVCenter, self.label)
        text = "N/A" if self.value is None else f"{int(round(self.display * 100)):3d}%"
        p.setPen(rgba(ICE, 210))
        p.drawText(QRectF(0, 0, w, 14), Qt.AlignRight | Qt.AlignVCenter, text)
        n, gap = self.segments, 2
        sw = (w - gap * (n - 1)) / n
        lit = int(round(self.display * n))
        for i in range(n):
            if i < lit:
                col = mix(CYAN, AMBER, ((i / (n - 1)) - 0.7) / 0.3)
                col.setAlpha(235)
            else:
                col = rgba(CYAN, 28)
            p.fillRect(QRectF(i * (sw + gap), 19, sw, 9), col)


class Radar(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.sweep, self.blips, self._spawn = 0.0, [], 0.0
        self.accent = QColor(CYAN)

    def tick(self, dt):
        self.sweep = (self.sweep + 80 * dt) % 360
        self._spawn -= dt
        if self._spawn <= 0:
            self._spawn = random.uniform(0.4, 1.4)
            self.blips.append([self.sweep - random.uniform(0, 25), random.uniform(0.2, 0.92), 1.0])
        for b in self.blips:
            b[2] -= dt / 2.5
        self.blips = [b for b in self.blips if b[2] > 0]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = max(10, min(w, h) / 2 - 4)
        p.translate(w / 2, h / 2)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(rgba(self.accent, 55), 1))
        for f in (1.0, 0.66, 0.33):
            p.drawEllipse(QPointF(0, 0), r * f, r * f)
        p.drawLine(QPointF(-r, 0), QPointF(r, 0))
        p.drawLine(QPointF(0, -r), QPointF(0, r))

        p.save()
        p.rotate(self.sweep)
        cg = QConicalGradient(0, 0, 0)
        cg.setColorAt(0.0, rgba(self.accent, 110))
        cg.setColorAt(0.18, rgba(self.accent, 0))
        cg.setColorAt(1.0, rgba(self.accent, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(cg)
        p.drawEllipse(QPointF(0, 0), r, r)
        p.setPen(QPen(rgba(self.accent, 210), 1.5))
        p.drawLine(QPointF(0, 0), QPointF(r, 0))
        p.restore()

        p.setPen(Qt.NoPen)
        for ang, dist, life in self.blips:
            a = math.radians(ang)
            p.setBrush(rgba(AMBER, 230 * life))
            p.drawEllipse(QPointF(math.cos(a) * r * dist, math.sin(a) * r * dist), 2.6, 2.6)


class WaveStrip(QWidget):
    """Rolling audio-level history, drawn as mirrored bars."""

    def __init__(self):
        super().__init__()
        self.samples = deque([0.0] * 400, maxlen=400)
        self.accent = QColor(CYAN)
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def push(self, v):
        self.samples.append(v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        step = 5
        n = max(1, w // step)
        data = list(self.samples)[-n:]
        mid = h / 2
        p.setPen(QPen(rgba(self.accent, 40), 1))
        p.drawLine(0, int(mid), w, int(mid))
        for i, v in enumerate(data):
            age = (i + 1) / len(data)
            bar = max(2.0, v * (h - 6))
            x = w - (len(data) - i) * step
            p.fillRect(QRectF(x, mid - bar / 2, 3, bar), rgba(self.accent, 40 + 190 * age))


class MicButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(68, 68)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("Toggle microphone (Space)")
        self.accent, self.phase = QColor(CYAN), 0.0

    def tick(self, dt):
        self.phase += dt * 4
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        on = self.isChecked()
        pulse = 0.5 + 0.5 * math.sin(self.phase)
        if on:
            g = QRadialGradient(c, 34)
            g.setColorAt(0.5, rgba(self.accent, 70 + 50 * pulse))
            g.setColorAt(1.0, rgba(self.accent, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawEllipse(c, 34, 34)
        p.setBrush(rgba(self.accent, 40) if on else rgba(BG_DEEP, 200))
        p.setPen(QPen(rgba(self.accent, 255 if on else 110), 2))
        p.drawEllipse(c, 26, 26)
        p.setPen(QPen(rgba(self.accent, 120), 1, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, 31, 31)

        icon = QPen(ICE if on else rgba(ICE, 150), 2.2, Qt.SolidLine, Qt.RoundCap)
        p.setPen(icon)
        p.drawRoundedRect(QRectF(c.x() - 6, c.y() - 15, 12, 19), 6, 6)
        p.drawArc(QRectF(c.x() - 11, c.y() - 9, 22, 20), 180 * 16, 180 * 16)
        p.drawLine(QPointF(c.x(), c.y() + 11), QPointF(c.x(), c.y() + 17))
        p.drawLine(QPointF(c.x() - 6, c.y() + 17), QPointF(c.x() + 6, c.y() + 17))
        if not on:
            p.setPen(QPen(rgba(RED, 230), 2.4, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(c.x() - 13, c.y() - 15), QPointF(c.x() + 13, c.y() + 15))


class ElidedLabel(QWidget):
    """Single-line text that keeps the *end* visible, so live speech never overflows."""

    def __init__(self):
        super().__init__()
        self.text, self.color, self.bold, self.size = "", QColor(CYAN), False, 18
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def set_text(self, text, color, bold=False):
        self.text, self.color, self.bold = text, QColor(color), bold
        self.update()

    def sizeHint(self):
        return QSize(100, 28)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setFont(make_font("mono", self.size, self.bold))
        p.setPen(self.color)
        shown = p.fontMetrics().elidedText(self.text, Qt.ElideLeft, self.width())
        p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignLeft | Qt.AlignVCenter, shown)


# --------------------------------------------------------------------------
# The reactor core
# --------------------------------------------------------------------------
STATE_COLORS = {
    "standby": (CYAN, AMBER),
    "listening": (CYAN, AMBER),
    "speaking": (AMBER, CYAN),
    "error": (RED, AMBER),
}


class HUDCanvas(QWidget):
    clicked = Signal()
    BARS = 72

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(360, 360)
        self.status_text = "SYSTEM STANDBY"
        self.status_color = QColor(CYAN)
        self.state = "standby"
        self.main, self.sub = QColor(CYAN), QColor(AMBER)
        self.t = 0.0
        self.a_slow = self.a_fast = self.a_rev = self.a_sweep = 0.0
        self.level = 0.0
        self.bars = [0.0] * self.BARS
        self.bar_phase = [random.uniform(0, 6.28) for _ in range(self.BARS)]
        self.nodes = [
            (
                random.uniform(0, 2 * math.pi),
                random.uniform(0.78, 1.08),
                random.uniform(1.0, 2.5),
                random.uniform(0, 6.28),
                random.uniform(0.6, 2.2),
            )
            for _ in range(120)
        ]

    # -- state -------------------------------------------------------------
    def set_status(self, text, color_hex):
        self.status_text = text
        self.status_color = QColor(color_hex)
        t = text.upper()
        if "LISTEN" in t:
            self.state = "listening"
        elif "SPEAK" in t:
            self.state = "speaking"
        elif "ERROR" in t or "OFFLINE" in t:
            self.state = "error"
        else:
            self.state = "standby"
        self.update()

    def tick(self, dt, level):
        self.t += dt
        mult = {"listening": 2.0, "speaking": 1.5}.get(self.state, 0.5) * (1 + level * 0.8)
        self.a_slow = (self.a_slow + 12 * mult * dt) % 360
        self.a_fast = (self.a_fast - 30 * mult * dt) % 360
        self.a_rev = (self.a_rev + 48 * mult * dt) % 360
        self.a_sweep = (self.a_sweep + 70 * dt) % 360

        tgt_main, tgt_sub = STATE_COLORS[self.state]
        k = min(1.0, dt * 6)
        self.main = mix(self.main, tgt_main, k)
        self.sub = mix(self.sub, tgt_sub, k)
        self.level += (level - self.level) * min(1.0, dt * 14)

        for i in range(self.BARS):
            idle = 0.05 * (0.5 + 0.5 * math.sin(self.t * 1.5 + self.bar_phase[i]))
            live = self.level * (0.35 + 0.65 * abs(math.sin(self.t * 7 + self.bar_phase[i] * 3)))
            self.bars[i] += (max(idle, live) - self.bars[i]) * min(1.0, dt * 18)
        self.update()

    # -- geometry / input ----------------------------------------------------
    def _geom(self):
        cx, cy = self.width() / 2, self.height() / 2
        max_r = max(40.0, min(cx, cy) - 8)
        return cx, cy, max_r, max_r * 0.66

    def _inside_core(self, pos):
        cx, cy, _, base_r = self._geom()
        return math.hypot(pos.x() - cx, pos.y() - cy) <= base_r * 0.75

    def mouseMoveEvent(self, e):
        self.setCursor(Qt.PointingHandCursor if self._inside_core(e.position()) else Qt.ArrowCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._inside_core(e.position()):
            self.clicked.emit()

    # -- painting ------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, max_r, base_r = self._geom()
        k = max(0.6, base_r / 190.0)
        main, sub, lvl = self.main, self.sub, self.level

        # halo
        halo = base_r * 1.28 + math.sin(self.t * 1.6) * 4 + lvl * 14
        g = QRadialGradient(cx, cy, halo)
        g.setColorAt(0.0, rgba(main, 0))
        g.setColorAt(0.62, rgba(main, 0))
        g.setColorAt(0.80, rgba(main, 22 + lvl * 45))
        g.setColorAt(1.0, rgba(main, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(cx, cy), halo, halo)

        # drifting particles
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_slow * 0.2)
        for ang, dist, size, ph, sp in self.nodes:
            a = 25 + 75 * (0.5 + 0.5 * math.sin(self.t * sp + ph))
            p.setBrush(rgba(main, a))
            p.drawEllipse(QPointF(math.cos(ang) * max_r * dist, math.sin(ang) * max_r * dist), size, size)
        p.restore()

        # radial voice bars
        r0 = base_r * 1.12
        bar_max = max(8.0, max_r - r0 - 6)
        p.save()
        p.translate(cx, cy)
        for i, b in enumerate(self.bars):
            p.save()
            p.rotate(i * 360.0 / self.BARS)
            p.setPen(QPen(rgba(main, 55 + 190 * b), 2.2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(r0, 0), QPointF(r0 + 3 + b * bar_max, 0))
            p.restore()
        p.restore()

        # outer segmented ring
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_slow)
        rect = QRectF(-base_r, -base_r, base_r * 2, base_r * 2)
        segs = [(10, 70), (100, 80), (210, 60), (290, 50)]
        p.setPen(QPen(rgba(BLUE, 60), 14 * k, Qt.SolidLine, Qt.FlatCap))
        for s, ln in segs:
            p.drawArc(rect, int(s * 16), int(ln * 16))
        p.setPen(QPen(rgba(main, 200), 3 * k, Qt.SolidLine, Qt.RoundCap))
        for s, ln in segs:
            p.drawArc(rect, int(s * 16), int(ln * 16))
        p.restore()

        # tick ring
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_fast)
        r_t = base_r + 15 * k
        for i in range(100):
            p.save()
            p.rotate(i * 3.6)
            if i % 10 == 0:
                p.setPen(QPen(rgba(ICE, 180), 2))
                ln = 15
            elif i % 2 == 0:
                p.setPen(QPen(rgba(main, 120), 1.5))
                ln = 8
            else:
                p.setPen(QPen(rgba(BLUE, 90), 1))
                ln = 4
            p.drawLine(QPointF(r_t, 0), QPointF(r_t - ln * k, 0))
            p.restore()
        p.restore()

        # dashed ring
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_rev)
        pen = QPen(rgba(main, 100), 1.5)
        pen.setDashPattern([2, 6])
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), base_r * 0.94, base_r * 0.94)
        p.restore()

        # radar sweep wedge
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_sweep)
        cg = QConicalGradient(0, 0, 0)
        cg.setColorAt(0.0, rgba(main, 60))
        cg.setColorAt(0.2, rgba(main, 0))
        cg.setColorAt(1.0, rgba(main, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(cg)
        p.drawEllipse(QPointF(0, 0), base_r * 0.9, base_r * 0.9)
        p.restore()

        # accent arcs
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_slow * 0.8)
        ri = base_r * 0.84
        p.setPen(QPen(rgba(sub, 190), 2.5 * k, Qt.SolidLine, Qt.RoundCap))
        rect_i = QRectF(-ri, -ri, ri * 2, ri * 2)
        for s, ln in ((200, 25), (20, 15), (300, 45)):
            p.drawArc(rect_i, int(s * 16), int(ln * 16))
        p.restore()

        # inner core arcs
        p.save()
        p.translate(cx, cy)
        p.rotate(self.a_rev * 1.5)
        rc = base_r * 0.72
        rect_c = QRectF(-rc, -rc, rc * 2, rc * 2)
        arcs = ((45, 90), (180, 60), (300, 45))
        p.setPen(QPen(rgba(main, 40 + 40 * lvl), 8 * k, Qt.SolidLine, Qt.FlatCap))
        for s, ln in arcs:
            p.drawArc(rect_c, int(s * 16), int(ln * 16))
        p.setPen(QPen(rgba(ICE, 255), 2.5 * k, Qt.SolidLine, Qt.RoundCap))
        for s, ln in arcs:
            p.drawArc(rect_c, int(s * 16), int(ln * 16))
        p.restore()

        # core disc
        core_r = base_r * 0.6
        cg2 = QRadialGradient(cx, cy, core_r)
        cg2.setColorAt(0, QColor("#000000"))
        cg2.setColorAt(0.9, QColor("#020e18"))
        cg2.setColorAt(1, rgba(main, 60 + 60 * lvl))
        p.setPen(QPen(rgba(main, 120), 1))
        p.setBrush(cg2)
        p.drawEllipse(QPointF(cx, cy), core_r, core_r)

        # name + status
        fs = max(14, int(core_r * 0.3))
        p.setFont(make_font("display", fs, True, fs * 0.2))
        box = QRectF(cx - core_r, cy - fs * 1.1, core_r * 2, fs * 2)
        p.setPen(rgba(main, 40))
        for i in range(1, 4):
            p.drawText(box.adjusted(-i, -i, i, i), Qt.AlignCenter, "ALVIN")
        p.setPen(ICE)
        p.drawText(box, Qt.AlignCenter, "ALVIN")

        p.setFont(make_font("mono", max(10, int(fs * 0.36)), True, 1.5))
        p.setPen(self.status_color)
        p.drawText(
            QRectF(cx - core_r, cy + fs * 0.8, core_r * 2, 30),
            Qt.AlignCenter,
            self.status_text,
        )


# --------------------------------------------------------------------------
# Side panels
# --------------------------------------------------------------------------
class SystemPanel(HoloPanel):
    def __init__(self):
        super().__init__("SYSTEM STATUS")
        self.started = time.time()
        self.k_link = KeyValue("LINK")
        self.k_mic = KeyValue("MICROPHONE")
        self.k_mode = KeyValue("MODE")
        self.k_up = KeyValue("UPTIME")
        for w in (self.k_link, self.k_mic, self.k_mode, self.k_up):
            self.lay.addWidget(w)
        self.lay.addSpacing(6)
        self.bar_mic = SegBar("MIC INPUT")
        self.bar_out = SegBar("VOICE OUTPUT")
        self.bar_cpu = SegBar("PROCESSOR")
        self.bar_mem = SegBar("MEMORY")
        for b in (self.bar_mic, self.bar_out, self.bar_cpu, self.bar_mem):
            self.lay.addWidget(b)
        self.lay.addSpacing(6)
        self.radar = Radar()
        self.lay.addWidget(self.radar, 1)
        self.set_link(False)
        self.set_mic(False)
        self.set_mode("STANDBY", CYAN)
        self.refresh_slow()
        if psutil:
            psutil.cpu_percent(None)

    def set_link(self, ok):
        self.k_link.set_value("ONLINE" if ok else "OFFLINE", GREEN if ok else RED)

    def set_mic(self, live):
        self.k_mic.set_value("LIVE" if live else "MUTED", GREEN if live else rgba(ICE, 160))

    def set_mode(self, text, color):
        self.k_mode.set_value(text, color)

    def refresh_slow(self):
        s = int(time.time() - self.started)
        self.k_up.set_value(f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}")
        if psutil:
            self.bar_cpu.set_value(psutil.cpu_percent(None) / 100)
            self.bar_mem.set_value(psutil.virtual_memory().percent / 100)
        else:
            self.bar_cpu.set_value(None)
            self.bar_mem.set_value(None)

    def tick(self, dt, mic, out, accent):
        self.bar_mic.set_value(mic)
        self.bar_out.set_value(out)
        for b in (self.bar_mic, self.bar_out, self.bar_cpu, self.bar_mem):
            b.tick(dt)
        self.radar.accent = QColor(accent)
        self.radar.tick(dt)


class LogPanel(HoloPanel):
    MAX_MESSAGES = 200

    def __init__(self):
        super().__init__("COMMUNICATION LOG")
        self.messages = []  # [role, text, timestamp]
        self.view = QTextBrowser()
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.view.setOpenLinks(False)
        self.view.setFont(make_font("display", 14))
        self.view.viewport().setAutoFillBackground(False)
        self.view.setStyleSheet(
            """
            QTextBrowser { background: transparent; border: none; color: #dff8ff; }
            QScrollBar:vertical { border: none; background: transparent; width: 6px; margin: 0; }
            QScrollBar::handle:vertical { background: rgba(0,229,255,110); min-height: 30px; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )
        self.lay.addWidget(self.view)
        self._render()

    def add_user(self, text):
        self.messages.append(["user", text, datetime.now().strftime("%H:%M:%S")])
        self._trim_and_render()

    def add_alvin(self, text):
        if not text.strip():
            return
        if self.messages and self.messages[-1][0] == "alvin":
            self.messages[-1][1] += " " + text.strip()  # streamed chunks join one reply
        else:
            self.messages.append(["alvin", text.strip(), datetime.now().strftime("%H:%M:%S")])
        self._trim_and_render()

    def _trim_and_render(self):
        del self.messages[: -self.MAX_MESSAGES]
        self._render()

    def _render(self):
        sb = self.view.verticalScrollBar()
        stick = sb.value() >= sb.maximum() - 4
        if not self.messages:
            body = (
                f'<p style="color:#4f7a8c;font-family:\'{FONT_MONO}\';font-size:12px;">'
                "Awaiting input. Click the core or press Space to start talking.</p>"
            )
        else:
            parts = []
            for role, text, ts in self.messages:
                tag, col, tcol = ("YOU", "#ffb020", "#ffffff") if role == "user" else ("ALVIN", "#00e5ff", "#bfefff")
                parts.append(
                    f'<table width="100%" cellspacing="0" cellpadding="0"><tr>'
                    f'<td width="3" bgcolor="{col}"></td>'
                    f'<td style="padding-left:10px;">'
                    f'<span style="color:{col};font-family:\'{FONT_MONO}\';font-size:11px;font-weight:bold;">{tag}</span>'
                    f'&nbsp;&nbsp;<span style="color:#4f7a8c;font-family:\'{FONT_MONO}\';font-size:10px;">{ts}</span><br>'
                    f'<span style="color:{tcol};font-size:14px;">{html.escape(text)}</span>'
                    f"</td></tr><tr><td colspan=\"2\" height=\"14\"></td></tr></table>"
                )
            body = "".join(parts)
        self.view.setHtml(body)
        if stick:
            sb.setValue(sb.maximum())


class CommandBar(HoloPanel):
    def __init__(self):
        super().__init__(None)
        row = QHBoxLayout()
        row.setSpacing(18)
        self.mic_btn = MicButton()
        row.addWidget(self.mic_btn)

        col = QVBoxLayout()
        col.setSpacing(2)
        self.caption = QLabel("LIVE TRANSCRIPT")
        self.caption.setFont(make_font("mono", 10, False, 3))
        self.caption.setStyleSheet("color: rgba(0,229,255,150); background: transparent;")
        self.partial = ElidedLabel()
        col.addStretch(1)
        col.addWidget(self.caption)
        col.addWidget(self.partial)
        col.addStretch(1)
        row.addLayout(col, 3)

        self.wave = WaveStrip()
        row.addWidget(self.wave, 2)
        self.lay.addLayout(row)

    def set_idle(self, listening):
        if listening:
            self.partial.set_text("Listening...", CYAN)
        else:
            self.partial.set_text("Click the core or press Space to talk.", rgba(CYAN, 170))

    def tick(self, dt, level, accent):
        self.mic_btn.accent = QColor(accent)
        self.wave.accent = QColor(accent)
        self.mic_btn.tick(dt)
        self.wave.push(level)


# --------------------------------------------------------------------------
# System settings window: the React dashboard embedded in the HUD
# --------------------------------------------------------------------------
_SYSTEM_ERROR_HTML = """
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body { height: 100%%; margin: 0; background: #020912; }
  body { display: flex; align-items: center; justify-content: center;
         color: #9fe9f5; font: 13px Consolas, monospace; letter-spacing: .08em; }
  .box { text-align: center; border: 1px solid rgba(0,229,255,.35);
         padding: 28px 36px; background: rgba(7,32,58,.55); }
  h1 { color: #00e5ff; font-size: 15px; margin: 0 0 10px; letter-spacing: .2em; }
  p { margin: 6px 0; color: rgba(223,248,255,.65); }
  b { color: #ffb020; }
</style></head>
<body><div class="box">
  <h1>ALVIN SYSTEM</h1>
  <p>Waiting for the dashboard server…</p>
  <p><b>%s</b></p>
  <p>Retrying automatically. Start the backend with <b>python run.py</b>.</p>
</div></body></html>
"""


class SystemWindow(QMainWindow):
    """The React dashboard (system settings) hosted inside the HUD."""

    def __init__(self, url=DASHBOARD_URL):
        super().__init__()
        self._url = QUrl(url)
        self._loading_error_page = False
        self.setWindowTitle("ALVIN AI - System Settings")
        self.resize(1280, 840)
        self.setMinimumSize(900, 620)
        self.setStyleSheet("QMainWindow { background: #020912; }")

        self.view = QWebEngineView(self)
        self.view.page().setBackgroundColor(QColor("#020912"))
        self.setCentralWidget(self.view)

        self._retry = QTimer(self)
        self._retry.setInterval(2500)
        self._retry.timeout.connect(self._retry_load)
        self.view.loadFinished.connect(self._on_loaded)
        self.view.load(self._url)

        QShortcut(QKeySequence("F5"), self).activated.connect(self.view.reload)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.view.reload)

    def _on_loaded(self, ok):
        if self._loading_error_page:
            return
        if ok:
            self._retry.stop()
        else:
            # Show a themed placeholder while the dev server comes up.
            self._loading_error_page = True
            self.view.setHtml(_SYSTEM_ERROR_HTML % str(self._url.toString()))
            self._retry.start()

    def _retry_load(self):
        self._loading_error_page = False
        self.view.load(self._url)

    def reload_if_stale(self):
        """Reload immediately when the last load failed and retries are pending."""
        if self._loading_error_page:
            self._retry_load()


# --------------------------------------------------------------------------
# Camera window: live preview + JPEG frames for vision turns
# --------------------------------------------------------------------------
class CameraWindow(QWidget):
    """Live webcam preview; streams JPEG frames to the server for vision."""

    closed = Signal()

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.capture = None
        self._frame = None
        self._send_accum = 1.5  # send the first frame as soon as it arrives
        self._last = time.perf_counter()
        self._closing = False
        self.setWindowTitle("ALVIN AI - Camera Feed")
        self.setMinimumSize(720, 600)
        self.setStyleSheet("background: #020912;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        header = QLabel("CAMERA FEED · LIVE")
        header.setFont(make_font("mono", 11, True, 2))
        header.setStyleSheet("color: #00e5ff; background: transparent;")
        lay.addWidget(header)

        self.transcript = QLabel("")
        self.transcript.setWordWrap(True)
        self.transcript.setFont(make_font("mono", 11))
        self.transcript.setStyleSheet(
            "color: #ffb020; background: transparent; min-height: 18px;"
        )
        lay.addWidget(self.transcript)

        self.video = QLabel("Starting camera...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet(
            "background: #000; color: #4f7a8c;"
            " border: 1px solid rgba(0,229,255,80);"
        )
        lay.addWidget(self.video, 1)

        row = QHBoxLayout()
        self.status = QLabel("Streaming frames to ALVIN")
        self.status.setFont(make_font("mono", 10, False, 1))
        self.status.setStyleSheet(
            "color: rgba(223,248,255,120); background: transparent;"
        )
        row.addWidget(self.status, 1)
        close_btn = QPushButton("CLOSE CAMERA")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setFocusPolicy(Qt.NoFocus)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #9fe9f5;"
            " border: 1px solid rgba(0,229,255,110); padding: 8px 16px;"
            " font-size: 11px; letter-spacing: 2px; }"
            "QPushButton:hover { background: rgba(0,229,255,40); }"
        )
        close_btn.clicked.connect(self.close_camera)
        row.addWidget(close_btn)
        lay.addLayout(row)

        self._open_capture()
        self.timer = QTimer(self)
        self.timer.setInterval(66)  # ~15 fps preview
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _open_capture(self):
        if cv2 is None:
            self.video.setText("OpenCV is not installed (pip install opencv-python).")
            self.status.setText("Camera unavailable")
            return
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                self.capture = cap
                return
            cap.release()
        self.video.setText("Camera not available.")
        self.status.setText("Camera unavailable")

    def _tick(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return
        self._frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        now = time.perf_counter()
        dt = min(0.5, now - self._last)
        self._last = now
        self._send_accum += dt
        if self._send_accum >= 1.5:
            self._send_accum = 0.0
            self.send_frame_now()

    def send_frame_now(self):
        """Send the latest frame to the server for vision analysis."""
        if self._frame is None or cv2 is None:
            return
        try:
            ok, buffer = cv2.imencode(
                ".jpg", self._frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if not ok:
                return
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            self.worker.send_json({"type": "camera_frame", "image": encoded})
        except Exception:
            pass

    def set_transcript(self, text):
        self.transcript.setText(f"HEARING: {text}" if text else "")

    def close_camera(self):
        if self._closing:
            return
        self._closing = True
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.worker.send_json({"type": "camera_close"})
        self.closed.emit()
        self.close()

    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)


# --------------------------------------------------------------------------
# Title bar (logo, clock, link chip, window buttons)
# --------------------------------------------------------------------------
class TitleBar(QWidget):
    def __init__(self, frameless):
        super().__init__()
        self.frameless = frameless
        self.link_ok = False
        self.btn_w = 0
        self.setFixedHeight(58)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(0)
        lay.addStretch(1)

        self.system_btn = QPushButton("SYS")
        self.system_btn.setObjectName("sysbtn")
        self.system_btn.setToolTip("Open system settings (F2)")
        self.system_btn.setFocusPolicy(Qt.NoFocus)
        self.system_btn.setCursor(Qt.PointingHandCursor)
        self.system_btn.setFixedSize(52, 34)
        lay.addWidget(self.system_btn)

        self.camera_btn = QPushButton("CAM")
        self.camera_btn.setObjectName("cambtn")
        self.camera_btn.setToolTip("Toggle camera (F3)")
        self.camera_btn.setFocusPolicy(Qt.NoFocus)
        self.camera_btn.setCursor(Qt.PointingHandCursor)
        self.camera_btn.setFixedSize(52, 34)
        lay.addWidget(self.camera_btn)

        if frameless:
            for glyph, slot, name in (
                ("–", self._minimize, "min"),
                ("□", self._toggle_max, "max"),
                ("×", self._close, "close"),
            ):
                b = QPushButton(glyph)
                b.setObjectName(name)
                b.setFocusPolicy(Qt.NoFocus)
                b.setCursor(Qt.PointingHandCursor)
                b.setFixedSize(42, 34)
                b.clicked.connect(slot)
                lay.addWidget(b)
        self.btn_w = 104 + (126 if frameless else 0)
        self.setStyleSheet(
            """
            QPushButton { background: transparent; border: none; color: #9fe9f5; font-size: 18px; }
            QPushButton:hover { background: rgba(0,229,255,40); }
            QPushButton#close:hover { background: rgba(255,59,92,170); color: white; }
            QPushButton#sysbtn, QPushButton#cambtn {
                border: 1px solid rgba(0,229,255,90);
                color: #9fe9f5; font-size: 11px; letter-spacing: 1px;
            }
            QPushButton#sysbtn:hover, QPushButton#cambtn:hover {
                background: rgba(0,229,255,40);
            }
            """
        )

    def set_link(self, ok):
        self.link_ok = ok
        self.update()

    def _minimize(self):
        self.window().showMinimized()

    def _toggle_max(self):
        w = self.window()
        w.showNormal() if w.isMaximized() else w.showMaximized()

    def _close(self):
        self.window().close()

    def mousePressEvent(self, e):
        if self.frameless and e.button() == Qt.LeftButton:
            handle = self.window().windowHandle()
            if handle:
                handle.startSystemMove()

    def mouseDoubleClickEvent(self, e):
        if self.frameless and e.button() == Qt.LeftButton:
            self._toggle_max()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # logo hexagon
        c = QPointF(40, h / 2)

        def hexagon(r):
            return QPolygonF(
                [QPointF(c.x() + r * math.cos(math.radians(60 * i + 30)), c.y() + r * math.sin(math.radians(60 * i + 30))) for i in range(6)]
            )

        p.setPen(QPen(CYAN, 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(hexagon(15))
        p.setPen(Qt.NoPen)
        p.setBrush(rgba(CYAN, 220))
        p.drawPolygon(hexagon(6.5))

        p.setFont(make_font("display", 20, True, 6))
        p.setPen(ICE)
        p.drawText(QRectF(68, 8, 260, 26), Qt.AlignLeft | Qt.AlignVCenter, "ALVIN")
        p.setFont(make_font("mono", 10, False, 3))
        p.setPen(rgba(CYAN, 160))
        p.drawText(QRectF(69, 33, 300, 16), Qt.AlignLeft | Qt.AlignVCenter, "VOICE INTERFACE")

        # clock
        now = datetime.now()
        p.setFont(make_font("mono", 22, True, 3))
        p.setPen(ICE)
        p.drawText(QRectF(w / 2 - 150, 6, 300, 28), Qt.AlignCenter, now.strftime("%H:%M:%S"))
        p.setFont(make_font("mono", 10, False, 3))
        p.setPen(rgba(CYAN, 160))
        p.drawText(QRectF(w / 2 - 150, 33, 300, 16), Qt.AlignCenter, now.strftime("%a %d %b %Y").upper())

        # link chip
        chip_w = 142
        x = w - 8 - self.btn_w - 16 - chip_w
        col = GREEN if self.link_ok else RED
        p.setPen(QPen(rgba(col, 150), 1))
        p.setBrush(rgba(col, 24))
        p.drawRoundedRect(QRectF(x, h / 2 - 14, chip_w, 28), 4, 4)
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(x + 16, h / 2), 4, 4)
        p.setFont(make_font("mono", 11, True, 2))
        p.setPen(col)
        p.drawText(
            QRectF(x + 28, h / 2 - 14, chip_w - 30, 28),
            Qt.AlignLeft | Qt.AlignVCenter,
            "LINK ONLINE" if self.link_ok else "LINK OFFLINE",
        )

        line = QLinearGradient(0, 0, w, 0)
        line.setColorAt(0, rgba(CYAN, 0))
        line.setColorAt(0.15, rgba(CYAN, 150))
        line.setColorAt(0.85, rgba(CYAN, 150))
        line.setColorAt(1, rgba(CYAN, 0))
        p.setPen(QPen(line, 1))
        p.drawLine(0, h - 1, w, h - 1)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class AlvinWindow(QMainWindow):
    def __init__(self, worker, frameless=FRAMELESS):
        super().__init__()
        self.worker = worker
        self.frameless = frameless
        if frameless:
            self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowTitle("ALVIN AI - Voice Interface")
        self.resize(1320, 840)
        self.setMinimumSize(1100, 680)
        self.setStyleSheet("QMainWindow { background: #020912; }")

        stage = Stage(framed=frameless)
        self.setCentralWidget(stage)
        root = QVBoxLayout(stage)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.titlebar = TitleBar(frameless)
        root.addWidget(self.titlebar)

        body = QHBoxLayout()
        body.setContentsMargins(20, 12, 20, 12)
        body.setSpacing(18)
        self.system = SystemPanel()
        self.system.setFixedWidth(260)
        self.hud = HUDCanvas()
        self.log = LogPanel()
        self.log.setFixedWidth(380)
        body.addWidget(self.system)
        body.addWidget(self.hud, 1)
        body.addWidget(self.log)
        root.addLayout(body, 1)

        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(20, 0, 20, 20)
        self.cmd = CommandBar()
        self.cmd.setFixedHeight(96)
        bar_row.addWidget(self.cmd)
        root.addLayout(bar_row)

        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(16, 16)
        self.grip.setStyleSheet("background: transparent;")
        self.grip.setVisible(frameless)

        self.system_window = None
        self.camera_window = None
        self._youtube_url = None
        self._youtube_stream_url = None
        self._youtube_audio_only = True
        self._media_proc = None
        # Set when the server announces "goodbye". The HUD stays up until the
        # goodbye TTS has played and the server drops the socket, then closes
        # itself -- only this AI frontend, never the operating system.
        self._shutdown_pending = False

        # wiring
        w = worker
        w.status_changed.connect(self.update_status)
        w.partial_transcript.connect(self.update_partial)
        w.final_transcript.connect(self.log.add_user)
        w.llm_received.connect(self.log.add_alvin)
        w.mic_level.connect(self._on_mic_level)
        w.out_level.connect(self._on_out_level)
        w.link_changed.connect(self._on_link)
        w.command_received.connect(self._on_command)
        w.camera_frame_requested.connect(self._on_camera_frame_requested)
        self.hud.clicked.connect(self.toggle_mic)
        self.cmd.mic_btn.clicked.connect(self.toggle_mic)
        self.titlebar.system_btn.clicked.connect(self.open_system_window)
        self.titlebar.camera_btn.clicked.connect(self.toggle_camera)

        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_mic)
        QShortcut(QKeySequence(Qt.Key_F2), self).activated.connect(self.open_system_window)
        QShortcut(QKeySequence(Qt.Key_F3), self).activated.connect(self.toggle_camera)
        QShortcut(QKeySequence(Qt.Key_F11), self).activated.connect(self.toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self._leave_fullscreen)

        self._mic_t = self._out_t = 0.0
        self._mic = self._out = 0.0
        self._last = time.perf_counter()
        # Hidden start.bat mode arms the mic before the window exists; reflect
        # the real state so the button does not look muted after reveal.
        self.cmd.mic_btn.setChecked(worker.is_listening)
        self.cmd.set_idle(worker.is_listening)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self.slow = QTimer(self)
        self.slow.timeout.connect(self._slow_tick)
        self.slow.start(1000)

    # -- animation -----------------------------------------------------------
    def _tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._last)
        self._last = now
        self._mic_t *= max(0.0, 1 - dt * 6)
        self._out_t *= max(0.0, 1 - dt * 6)
        self._mic += (self._mic_t - self._mic) * min(1.0, dt * 18)
        self._out += (self._out_t - self._out) * min(1.0, dt * 18)
        level = max(self._mic, self._out)
        self.hud.tick(dt, level)
        self.system.tick(dt, self._mic, self._out, self.hud.main)
        self.cmd.tick(dt, level, self.hud.main)

    def _slow_tick(self):
        self.system.refresh_slow()
        self.titlebar.update()

    # -- signals -------------------------------------------------------------
    def _on_mic_level(self, v):
        self._mic_t = v

    def _on_out_level(self, v):
        self._out_t = v

    def _on_link(self, ok):
        self.titlebar.set_link(ok)
        self.system.set_link(ok)
        if not ok and self._shutdown_pending:
            # The server has finished the shutdown acknowledgement and gone
            # away: close the AI frontend. OS shutdown is never involved.
            self.close()

    def reveal(self):
        """Bring the HUD on screen (it starts hidden under start.bat)."""
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def go_home(self):
        """Return to the assistant's main window ("go back to homepage")."""
        if self.system_window is not None:
            self.system_window.close()
        if self.camera_window is not None:
            self.close_camera_window()
        self.reveal()

    def update_status(self, text, color):
        self.hud.set_status(text, color)
        self.system.set_mode(text, color)

    def update_partial(self, text):
        if text:
            self.cmd.partial.set_text(text, ICE, bold=True)
        else:
            self.cmd.set_idle(self.worker.is_listening)
        if self.camera_window is not None:
            self.camera_window.set_transcript(text)

    def toggle_mic(self):
        on = not self.worker.is_listening
        self.worker.is_listening = on
        self.cmd.mic_btn.setChecked(on)
        self.system.set_mic(on)
        if not self.worker.is_speaking:
            self.worker.status_changed.emit("LISTENING" if on else "STANDBY", "#00e5ff")
        if on:
            self.cmd.set_idle(True)
        else:
            self.cmd.partial.set_text("Microphone muted.", rgba(CYAN, 170))

    # -- system settings / camera -------------------------------------------
    def _on_command(self, intent, slots):
        if intent == "wake_up":
            # start.bat launches the HUD hidden; it appears on "Alvin, wake up".
            self.reveal()
        elif intent == "show_home":
            # "go back to homepage": dismiss pop-ups, return to the main UI.
            self.go_home()
        elif intent in ("open_system", "show_dashboard"):
            self._show_system_window(request_start=False)
        elif intent == "play_youtube":
            self.play_youtube(slots)
        elif intent in ("play", "resume"):
            self.resume_youtube()
        elif intent == "stop":
            self.stop_local_media()
        elif intent == "open_camera":
            self.open_camera_window()
        elif intent == "close_camera":
            self.close_camera_window()
        elif intent == "shutdown":
            # Keep the window alive for the spoken acknowledgement; _on_link
            # closes the HUD once the server has actually stopped.
            self._shutdown_pending = True
            self.update_status("SHUTTING DOWN", RED)

    def play_youtube(self, slots):
        """Play the resolved song ad-free, falling back to the browser.

        The server resolves a direct yt-dlp stream URL (no YouTube ads) for HUD
        clients. If a local player is available we use it; otherwise the normal
        watch page opens in the browser.
        """
        video_id = slots.get("video_id")
        query = (slots.get("query") or "").strip()
        url = slots.get("url") or ""
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
        elif not url:
            from urllib.parse import quote_plus

            url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        self._youtube_url = url

        stream_url = (slots.get("stream_url") or "").strip()
        audio_only = bool(slots.get("audio_only", True))
        if stream_url:
            self._youtube_stream_url = stream_url
            self._youtube_audio_only = audio_only
            if self._play_media(stream_url, audio_only):
                self.worker.send_json({"type": "youtube_playback", "local": True})
                return
        # Browser fallback: tell the server so "stop the music" knows a tab
        # exists and may be closed (and only then).
        self.worker.send_json({"type": "youtube_playback", "local": False})
        webbrowser.open(url, new=1)

    def resume_youtube(self):
        """Replay the last song when the user says "play"/"resume"."""
        if self._youtube_stream_url and self._play_media(
            self._youtube_stream_url, self._youtube_audio_only
        ):
            self.worker.send_json({"type": "youtube_playback", "local": True})
            return
        if self._youtube_url:
            self.worker.send_json({"type": "youtube_playback", "local": False})
            webbrowser.open(self._youtube_url, new=1)

    @staticmethod
    def _find_media_player():
        """First installed player able to open a yt-dlp direct stream."""
        for name in ("mpv", "vlc", "ffplay"):
            found = shutil.which(name)
            if found:
                return found
        candidates = []
        program_files = os.environ.get("ProgramFiles")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if program_files:
            candidates.append(Path(program_files) / "VideoLAN" / "VLC" / "vlc.exe")
        if program_files_x86:
            candidates.append(Path(program_files_x86) / "VideoLAN" / "VLC" / "vlc.exe")
        for path in candidates:
            if path.is_file():
                return str(path)
        return None

    def _play_media(self, stream_url, audio_only=True):
        """Play a direct stream in a local player. Returns False if none exists."""
        player = self._find_media_player()
        if not player:
            return False

        name = Path(player).name.lower()
        if name.startswith("vlc"):
            if audio_only:
                # Dummy interface: play in the background with no VLC window.
                args = [
                    "--intf", "dummy", "--dummy-quiet", "--no-video",
                    "--no-one-instance", "--play-and-exit", "--quiet",
                ]
            else:
                args = [
                    "--no-one-instance", "--play-and-exit", "--quiet",
                    "--no-video-title-show",
                ]
            args.append(stream_url)
        elif name.startswith("mpv"):
            args = ["--no-terminal"]
            if audio_only:
                args.extend(["--no-video", "--force-window=no"])
            args.append(stream_url)
        elif name.startswith("ffplay"):
            args = ["-autoexit", "-loglevel", "error"]
            if audio_only:
                args.append("-nodisp")
            args.append(stream_url)
        else:
            return False

        self.stop_local_media()
        popen_kwargs = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self._media_proc = subprocess.Popen(
                [player, *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **popen_kwargs,
            )
            return True
        except Exception:
            self._media_proc = None
            return False

    def stop_local_media(self):
        """Stop the ad-free player started by :meth:`_play_media`."""
        proc = self._media_proc
        self._media_proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def open_system_window(self):
        self._show_system_window(request_start=True)

    def _show_system_window(self, request_start: bool):
        if self.system_window is not None:
            self.system_window.showNormal()
            self.system_window.raise_()
            self.system_window.activateWindow()
            self.system_window.reload_if_stale()
            return
        if QWebEngineView is None:
            # No QtWebEngine: fall back to the default browser.
            webbrowser.open(DASHBOARD_URL, new=1)
            return
        self.system_window = SystemWindow(DASHBOARD_URL + "#settings")
        self.system_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.system_window.destroyed.connect(self._on_system_window_closed)
        self.system_window.show()
        if request_start:
            # Ask the server to start the dashboard dev server if needed.
            self.worker.send_json({"type": "open_system"})

    def _on_system_window_closed(self, *_):
        self.system_window = None

    def open_camera_window(self):
        if self.camera_window is not None:
            self.camera_window.showNormal()
            self.camera_window.raise_()
            self.camera_window.activateWindow()
            return
        if cv2 is None:
            return
        self.camera_window = CameraWindow(self.worker)
        self.camera_window.closed.connect(self._on_camera_window_closed)
        self.camera_window.show()

    def _on_camera_window_closed(self):
        self.camera_window = None

    def close_camera_window(self):
        if self.camera_window is not None:
            self.camera_window.close_camera()

    def toggle_camera(self):
        if self.camera_window is not None:
            self.close_camera_window()
        else:
            self.open_camera_window()

    def _on_camera_frame_requested(self):
        if self.camera_window is not None:
            self.camera_window.send_frame_now()

    # -- window --------------------------------------------------------------
    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _leave_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.grip.move(self.width() - 18, self.height() - 18)
        self.grip.raise_()

    def closeEvent(self, event):
        self.stop_local_media()
        if self.camera_window is not None:
            self.camera_window.close_camera()
        if self.system_window is not None:
            self.system_window.close()
        self.worker.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    init_fonts()

    hidden = "--hidden" in sys.argv[1:]
    worker = BackendWorker()
    if hidden:
        # start.bat mode: the window is off-screen, but the microphone must be
        # armed so "Alvin, wake up" can be heard. The server answers with a
        # ``wake_up`` command, which calls AlvinWindow.reveal().
        worker.is_listening = True
    window = AlvinWindow(worker)  # connect signals first...
    worker.start()  # ...then start, so the first CONNECTED signal isn't lost
    if hidden:
        window.hide()
    else:
        window.show()

    sys.exit(app.exec())
