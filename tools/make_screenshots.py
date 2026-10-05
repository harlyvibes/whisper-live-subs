"""Regenerate the README screenshots in docs/ from the real widgets.

    .venv\\Scripts\\python tools\\make_screenshots.py

Uses default settings (not your saved ones) and never touches the desktop, so the
images contain no personal windows. Works while the app itself is running.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter,  # noqa: E402
                           QRadialGradient, QRegion)
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget  # noqa: E402

from livesubs import app as app_mod  # noqa: E402
from livesubs.config import Config  # noqa: E402
from livesubs.overlay import CaptionOverlay  # noqa: E402
from livesubs.settings_dialog import SettingsDialog  # noqa: E402

DOCS = ROOT / "docs"


def backdrop(w: int, h: int, dpr: float) -> QImage:
    """An abstract 'stream' scene to show the overlay against."""
    img = QImage(int(w * dpr), int(h * dpr), QImage.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, w, h)
    g.setColorAt(0, QColor("#1d2b64"))
    g.setColorAt(0.55, QColor("#5b3a8c"))
    g.setColorAt(1, QColor("#f8a07e"))
    p.fillRect(QRectF(0, 0, w, h), g)
    for x, y, r, c in ((0.2, 0.3, 220, "#ffd6e7"), (0.78, 0.25, 160, "#9be7ff"), (0.6, 0.8, 260, "#ffe29a")):
        rg = QRadialGradient(QPointF(w * x, h * y), r)
        col = QColor(c)
        col.setAlpha(110)
        rg.setColorAt(0, col)
        col.setAlpha(0)
        rg.setColorAt(1, col)
        p.setBrush(rg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(w * x, h * y), r, r)
    f = QFont("Segoe UI", 13)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255, 200))
    p.drawText(QRectF(24, 18, 300, 30), Qt.AlignLeft, "● LIVE")
    p.end()
    return img


def overlay_shot(cfg: Config, name: str, lines, partial=None, w=1280, h=520) -> None:
    ov = CaptionOverlay(cfg)
    for text, lang, trans, translated in lines:
        ov.add_final(text, lang, trans, translated)
    if partial:
        ov.set_partial(*partial)
    dpr = ov.devicePixelRatioF() or 1.0
    img = backdrop(w, h, dpr)
    p = QPainter(img)
    x = (w - ov.width()) / 2
    ov.render(p, QPointF(x, h - ov.height() - 36).toPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    p.end()
    img.scaledToWidth(1600, Qt.SmoothTransformation).save(str(DOCS / name))
    ov.deleteLater()


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    app_mod.setup_style(app)
    app_mod.apply_theme("light")  # consistent README images whatever the Windows theme

    cfg = Config()
    cfg.locked = True             # no dashed "unlocked" frame
    cfg.overlay_width = 1100
    cfg.auto_hide_s = 0

    overlay_shot(cfg, "overlay-bilingual.png", [
        ("みんなこんばんは！今日もよろしくね", "ja", "", False),
        ("OK chat, let's go — we're doing the English challenge today!", "en", "", False),
    ], partial=("えっ、ちょっと待って…", "ja"))

    both = cfg.copy()
    both.separate_ja_color = True
    both.output_mode = "both"
    both.show_language_tags = True
    both.max_lines = 4
    overlay_shot(both, "overlay-translation.png", [
        ("みんなこんばんは！今日もよろしくね", "ja",
         "Good evening everyone! Let's have a good time today", False),
        ("Thank you so much for the super chat!", "en", "", False),
    ])

    styled = cfg.copy()
    styled.font_family = "Meiryo"
    styled.bg_color = "#00000000"
    styled.outline_width = 4.5
    styled.outline_color = "#ff3a1f5d"
    styled.text_color = "#ffffffff"
    styled.font_size = 34
    overlay_shot(styled, "overlay-outline-only.png", [
        ("今日はマインクラフトを一緒にやっていきます！", "ja", "", False),
    ], h=300)

    # Settings window, one image per tab
    dlg = SettingsDialog(Config())
    dlg.resize(620, 760)
    tabs = dlg.findChild(QTabWidget)
    for i, slug in enumerate(["model", "audio", "language", "appearance", "behaviour"]):
        tabs.setCurrentIndex(i)
        dlg.adjustSize()
        dlg.grab().save(str(DOCS / f"settings-{slug}.png"))

    # Tray menu, built by the real controller with side effects disabled
    app_mod.GlobalHotkeys.set_hotkeys = lambda self, m: None
    app_mod.Controller._set_launch_at_login = lambda self, e: None
    app_mod.Config.load = classmethod(lambda cls: Config(start_on_launch=False))
    ctl = app_mod.Controller(app)
    ctl.running, ctl.state, ctl.model_info = True, "listening", "small · CPU int8"
    ctl._update_ui()
    ctl.menu.adjustSize()
    ctl.menu.grab().save(str(DOCS / "tray-menu.png"))
    ctl.overlay.hide()
    ctl.tray.hide()
    print("Saved screenshots to", DOCS)


if __name__ == "__main__":
    main()
