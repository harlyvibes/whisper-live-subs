"""Always-on-top caption overlay window."""
from __future__ import annotations

import ctypes
import sys
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen,
                           QTextLayout, QTextOption)
from PySide6.QtWidgets import QApplication, QWidget

from .config import Config

EDGE = 10  # px grab zone for resizing


@dataclass
class _Line:
    runs: list[tuple[str, QColor]]  # consecutive pieces drawn in different colours
    italic: bool

    @property
    def text(self) -> str:
        return "".join(t for t, _ in self.runs)


def _line(text: str, color: QColor, italic: bool) -> _Line:
    return _Line([(text, color)], italic)


@dataclass
class _Entry:
    text: str
    lang: str
    translation: str = ""
    translated: bool = False
    id: int = 0
    stable: int = -1   # partials: chars confirmed by two passes (-1 = all)


def _tag(lang: str, translated: bool) -> str:
    return f"[{lang.upper()}→EN] " if translated else f"[{lang.upper()}] "


class CaptionOverlay(QWidget):
    geometry_changed = Signal(int, int, int)   # x, bottom, width
    font_size_changed = Signal(int)
    context_menu_requested = Signal(QPoint)
    settings_requested = Signal()

    def __init__(self, cfg: Config):
        super().__init__(None)
        self.setWindowTitle("Whisper Live Subs Overlay")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(True)

        self.cfg = cfg.copy()
        self.entries: deque[_Entry] = deque(maxlen=30)
        self.partial: _Entry | None = None
        self.status_text = ""
        self._lines: list[_Line] = []
        self._drag_mode: str | None = None
        self._drag_origin = QPoint()
        self._drag_geom = (0, 0, 0)

        self._hide_timer = QTimer(self, singleShot=True, timeout=self._auto_clear)
        self._status_timer = QTimer(self, singleShot=True, timeout=self._clear_status)
        self._topmost_timer = QTimer(self, interval=3000, timeout=self._reassert_topmost)
        self._topmost_timer.start()

        self.apply_config(cfg)

    # ---------- public API ----------
    def apply_config(self, cfg: Config) -> None:
        locked_changed = cfg.locked != self.cfg.locked
        self.cfg = cfg.copy()
        self.setWindowOpacity(max(10, cfg.window_opacity) / 100)
        if locked_changed or not self.testAttribute(Qt.WA_WState_Created):
            visible = self.isVisible()
            self.setWindowFlag(Qt.WindowTransparentForInput, cfg.locked)
            if visible:
                self.show()
        self._relayout()

    def add_final(self, fid: int, text: str, lang: str, translation: str, translated: bool) -> None:
        self.partial = None
        self.entries.append(_Entry(text, lang, translation, translated, fid))
        self._touch()

    def revise(self, fid: int, text: str, lang: str, translation: str, translated: bool) -> None:
        """Correct a caption that is (still) on screen."""
        for e in self.entries:
            if e.id == fid:
                e.text, e.lang, e.translation, e.translated = text, lang, translation, translated
                self._relayout()
                return

    def set_partial(self, text: str, lang: str, stable: int = -1) -> None:
        self.partial = _Entry(text, lang, stable=stable) if text else None
        self._touch()

    def show_status(self, text: str, seconds: float = 4.0) -> None:
        self.status_text = text
        if seconds > 0:
            self._status_timer.start(int(seconds * 1000))
        else:
            self._status_timer.stop()
        self._relayout()

    def clear(self) -> None:
        self.entries.clear()
        self.partial = None
        self._relayout()

    # ---------- layout ----------
    def _touch(self) -> None:
        if self.cfg.auto_hide_s > 0:
            self._hide_timer.start(int(self.cfg.auto_hide_s * 1000))
        self._relayout()

    def _auto_clear(self) -> None:
        if self.partial is None:
            self.entries.clear()
            self._relayout()
        elif self.cfg.auto_hide_s > 0:
            self._hide_timer.start(int(self.cfg.auto_hide_s * 1000))

    def _clear_status(self) -> None:
        self.status_text = ""
        self._relayout()

    def _font(self, italic: bool = False) -> QFont:
        f = QFont(self.cfg.font_family)
        f.setPixelSize(max(6, int(self.cfg.font_size)))
        f.setBold(self.cfg.font_bold)
        f.setItalic(italic)
        f.setHintingPreference(QFont.PreferNoHinting)
        f.setStyleStrategy(QFont.PreferAntialias)
        return f

    def _wrap(self, text: str, font: QFont, width: float) -> list[str]:
        return [t for _, t in self._wrap_spans(text, font, width)]

    def _wrap_spans(self, text: str, font: QFont, width: float) -> list[tuple[int, str]]:
        layout = QTextLayout(text, font)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(opt)
        out = []
        layout.beginLayout()
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            raw = text[line.textStart():line.textStart() + line.textLength()]
            lead = len(raw) - len(raw.lstrip())
            out.append((line.textStart() + lead, raw.strip()))
        layout.endLayout()
        return [(i, t) for i, t in out if t]

    def _entry_lines(self, e: _Entry, partial: bool, width: float) -> list[_Line]:
        c = self.cfg
        tags = c.show_language_tags
        if e.translated:
            color = QColor(c.translation_color)
        elif e.lang == "ja" and c.separate_ja_color:
            color = QColor(c.ja_text_color)
        else:
            color = QColor(c.text_color)
        italic = partial and c.partial_style == "italic"
        tag = _tag(e.lang, e.translated) if tags and e.lang else ""
        text = tag + e.text
        font = self._font(italic)
        if partial and c.partial_style == "dim":
            # Words two passes agree on are solid; the newest, still-changing words are faded.
            faded = QColor(color)
            faded.setAlphaF(color.alphaF() * 0.55)
            stable = len(text) if e.stable < 0 else len(tag) + e.stable
            lines = []
            for start, t in self._wrap_spans(text, font, width):
                cut = max(0, min(len(t), stable - start))
                runs = [(t[:cut], color), (t[cut:], faded)]
                lines.append(_Line([r for r in runs if r[0]], italic))
        else:
            lines = [_line(t, color, italic) for t in self._wrap(text, font, width)]
        if e.translation:
            tcolor = QColor(c.translation_color)
            ttext = ("[EN] " if tags else "") + e.translation
            lines += [_line(t, tcolor, italic) for t in self._wrap(ttext, font, width)]
        return lines

    def _content_width(self) -> float:
        c = self.cfg
        return max(50.0, c.overlay_width - 2 * c.padding - 2 * c.outline_width - 4)

    def _relayout(self) -> None:
        c = self.cfg
        width = self._content_width()
        lines: list[_Line] = []
        for e in self.entries:
            lines += self._entry_lines(e, False, width)
        if self.partial is not None:
            lines += self._entry_lines(self.partial, True, width)
        lines = lines[-max(1, c.max_lines):]
        if self.status_text:
            col = QColor("#ffcfd8dc")
            lines += [_line(t, col, True) for t in self._wrap(self.status_text, self._font(True), width)]
        if not lines and not c.locked:
            hint = ("Captions appear here · drag to move · drag sides to resize · "
                    "Ctrl+scroll for size · lock from tray")
            col = QColor("#ccffffff")
            lines = [_line(t, col, True) for t in self._wrap(hint, self._font(True), width)]
        self._lines = lines

        fm = QFontMetricsF(self._font())
        line_h = fm.height() * 1.05
        h = int(len(lines) * line_h + 2 * c.padding + 2 * c.outline_width + 4) if lines else 1
        self._place(h)
        self.update()

    def _place(self, height: int) -> None:
        c = self.cfg
        scr = QApplication.primaryScreen().availableGeometry()
        w = max(200, c.overlay_width)
        x = c.overlay_x if c.overlay_x >= 0 else scr.x() + (scr.width() - w) // 2
        bottom = c.overlay_bottom if c.overlay_bottom >= 0 else scr.y() + int(scr.height() * 0.92)
        self.setGeometry(x, bottom - height, w, height)

    # ---------- painting ----------
    def paintEvent(self, _ev) -> None:
        if not self._lines:
            return
        c = self.cfg
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        fm_cache = {}

        def metrics(italic: bool) -> QFontMetricsF:
            if italic not in fm_cache:
                fm_cache[italic] = QFontMetricsF(self._font(italic))
            return fm_cache[italic]

        base = metrics(False)
        line_h = base.height() * 1.05
        widths = [metrics(l.italic).horizontalAdvance(l.text) for l in self._lines]
        pad = c.padding + c.outline_width
        block_w = min(self.width(), max(widths) + 2 * pad + 4)

        # Background box
        if c.bg_fit == "full":
            bg = QRectF(0, 0, self.width(), self.height())
        else:
            bx = {"left": 0, "right": self.width() - block_w}.get(c.alignment, (self.width() - block_w) / 2)
            bg = QRectF(bx, 0, block_w, self.height())
        bgc = QColor(c.bg_color)
        if bgc.alpha() > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(bgc)
            p.drawRoundedRect(bg, c.corner_radius, c.corner_radius)

        if not c.locked:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 140), 1, Qt.DashLine))
            p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1),
                              c.corner_radius, c.corner_radius)

        outline_pen = QPen(QColor(c.outline_color), c.outline_width * 2,
                           Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        shadow_col = QColor(c.shadow_color)
        shadow_pen = QPen(shadow_col, max(1.0, c.outline_width * 2), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        y = pad + 2
        for line, w in zip(self._lines, widths):
            font = self._font(line.italic)
            if c.alignment == "left":
                x = pad + 2
            elif c.alignment == "right":
                x = self.width() - pad - 2 - w
            else:
                x = (self.width() - w) / 2
            fm = metrics(line.italic)
            for run_text, run_color in line.runs:
                path = QPainterPath()
                path.addText(QPointF(x, y + base.ascent()), font, run_text)
                if c.shadow and shadow_col.alpha() > 0:
                    sp = path.translated(c.shadow_offset, c.shadow_offset)
                    if c.outline_width > 0:
                        p.strokePath(sp, shadow_pen)
                    p.fillPath(sp, shadow_col)
                if c.outline_width > 0 and QColor(c.outline_color).alpha() > 0:
                    p.strokePath(path, outline_pen)
                p.fillPath(path, run_color)
                x += fm.horizontalAdvance(run_text)
            y += line_h
        p.end()

    # ---------- interaction (only when unlocked) ----------
    def _edge(self, x: int) -> str | None:
        if x <= EDGE:
            return "left"
        if x >= self.width() - EDGE:
            return "right"
        return None

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._drag_mode = self._edge(int(ev.position().x())) or "move"
            self._drag_origin = ev.globalPosition().toPoint()
            self._drag_geom = (self.x(), self.y() + self.height(), self.width())
        elif ev.button() == Qt.RightButton:
            self.context_menu_requested.emit(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev) -> None:
        if self._drag_mode is None:
            edge = self._edge(int(ev.position().x()))
            self.setCursor(Qt.SizeHorCursor if edge else Qt.SizeAllCursor)
            return
        d = ev.globalPosition().toPoint() - self._drag_origin
        x0, b0, w0 = self._drag_geom
        c = self.cfg
        if self._drag_mode == "move":
            c.overlay_x, c.overlay_bottom = x0 + d.x(), b0 + d.y()
        elif self._drag_mode == "right":
            c.overlay_x, c.overlay_width = x0, max(200, w0 + d.x())
        else:
            nw = max(200, w0 - d.x())
            c.overlay_x, c.overlay_width = x0 + w0 - nw, nw
        if c.overlay_bottom < 0:
            c.overlay_bottom = b0
        self._relayout()

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag_mode is not None:
            self._drag_mode = None
            c = self.cfg
            self.geometry_changed.emit(self.x(), self.y() + self.height(), c.overlay_width)

    def mouseDoubleClickEvent(self, ev) -> None:
        self.settings_requested.emit()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.ControlModifier:
            step = 1 if ev.angleDelta().y() > 0 else -1
            size = min(200, max(8, self.cfg.font_size + step * 2))
            self.font_size_changed.emit(size)

    # ---------- topmost ----------
    def _reassert_topmost(self) -> None:
        if sys.platform != "win32" or not self.isVisible():
            return
        HWND_TOPMOST = -1
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x1, 0x2, 0x10
        ctypes.windll.user32.SetWindowPos(int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0,
                                          SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
