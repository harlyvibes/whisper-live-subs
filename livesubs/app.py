"""Application controller: system tray, overlay, engine and settings."""
from __future__ import annotations

import ctypes
import os
import sys
import winreg
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRectF, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QMenu, QPlainTextEdit,
                               QPushButton, QSystemTrayIcon, QVBoxLayout)

from .config import APP_ID, APP_NAME, ENGINE_KEYS, MODELS, PROJECT_DIR, Config
from .engine import CaptionEngine
from .hotkeys import GlobalHotkeys
from .overlay import CaptionOverlay
from .settings_dialog import SettingsDialog

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def make_icon(state: str) -> QIcon:
    colors = {"listening": "#2e9cff", "loading": "#f5a623", "paused": "#7a7f87",
              "error": "#e5484d", "stopped": "#7a7f87"}
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(colors.get(state, "#2e9cff")))
        r = size * 0.18
        p.drawRoundedRect(QRectF(0.5, 0.5, size - 1, size - 1), r, r)
        f = QFont("Yu Gothic UI")
        f.setPixelSize(int(size * 0.72))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("white"))
        p.drawText(QRectF(0, -size * 0.04, size, size), Qt.AlignCenter, "字")
        p.end()
        icon.addPixmap(pm)
    return icon


class HistoryWindow(QDialog):
    def __init__(self, open_folder):
        super().__init__(None)
        self.setWindowTitle(f"{APP_NAME} — Caption history")
        self.resize(640, 420)
        self.text = QPlainTextEdit(readOnly=True)
        self.text.setFont(QFont("Yu Gothic UI", 11))
        self.text.setMaximumBlockCount(5000)
        copy = QPushButton("Copy all")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        clear = QPushButton("Clear")
        clear.clicked.connect(self.text.clear)
        folder = QPushButton("Open transcript folder")
        folder.clicked.connect(open_folder)
        row = QHBoxLayout()
        for b in (copy, clear, folder):
            row.addWidget(b)
        row.addStretch()
        lay = QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addLayout(row)

    def append(self, line: str) -> None:
        sb = self.text.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.text.appendPlainText(line)
        if at_bottom:
            sb.setValue(sb.maximum())


class Controller(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.cfg = Config.load()
        self.engine: CaptionEngine | None = None
        self.running = False
        self.paused = False
        self.state = "stopped"
        self.model_info = ""
        self.status = ""
        self.dialog: SettingsDialog | None = None
        self.transcript_path: Path | None = None

        self.overlay = CaptionOverlay(self.cfg)
        self.overlay.geometry_changed.connect(self._on_overlay_moved)
        self.overlay.font_size_changed.connect(self._on_font_wheel)
        self.overlay.context_menu_requested.connect(lambda pos: self.menu.popup(pos))
        self.overlay.settings_requested.connect(self.open_settings)
        self.overlay.show()

        self.history = HistoryWindow(self.open_transcript_folder)

        self.hotkeys = GlobalHotkeys()
        self.hotkeys.triggered.connect(self._on_hotkey)
        self.hotkeys.failed.connect(self._on_hotkey_failed)
        self._apply_hotkeys()

        self.tray = QSystemTrayIcon(make_icon("stopped"))
        self.menu = QMenu()
        self._build_menu()
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()
        self._update_ui()

        self._set_launch_at_login(self.cfg.launch_at_login)
        if self.cfg.start_on_launch:
            self.start()
        else:
            self.overlay.show_status("Captioning is off — start it from the tray icon.", 6)

    # ---------- menu ----------
    def _build_menu(self) -> None:
        m = self.menu
        self.act_status = m.addAction("")
        self.act_status.setEnabled(False)
        m.addSeparator()
        self.act_run = m.addAction("Start captioning", self.toggle_running)
        self.act_pause = m.addAction("Pause", self.toggle_pause)
        self.act_overlay = m.addAction("Hide overlay", self.toggle_overlay)
        self.act_lock = m.addAction("Lock overlay (click-through)", self.toggle_lock)
        self.act_lock.setCheckable(True)
        m.addAction("Clear captions", self.overlay.clear)
        m.addSeparator()

        self.model_menu = m.addMenu("Model")
        self.model_group = QActionGroup(self)
        for key, label, size, _ in MODELS:
            if key == "custom":
                continue
            a = self.model_menu.addAction(f"{label}  ({size})")
            a.setCheckable(True)
            a.setData(key)
            self.model_group.addAction(a)
        self.model_group.triggered.connect(lambda a: self._quick_set(model=a.data()))

        self.lang_menu = m.addMenu("Spoken language")
        self.lang_group = QActionGroup(self)
        for key, label in (("auto", "Auto-detect (Japanese + English)"), ("ja", "Japanese only"),
                           ("en", "English only")):
            a = self.lang_menu.addAction(label)
            a.setCheckable(True)
            a.setData(key)
            self.lang_group.addAction(a)
        self.lang_group.triggered.connect(lambda a: self._quick_set(language_mode=a.data()))

        self.out_menu = m.addMenu("Captions show")
        self.out_group = QActionGroup(self)
        for key, label in (("original", "What was said"), ("translate", "English translation"),
                           ("both", "Original + English")):
            a = self.out_menu.addAction(label)
            a.setCheckable(True)
            a.setData(key)
            self.out_group.addAction(a)
        self.out_group.triggered.connect(lambda a: self._quick_set(output_mode=a.data()))

        m.addSeparator()
        m.addAction("Caption history…", self.show_history)
        m.addAction("Settings…", self.open_settings)
        m.addSeparator()
        m.addAction("Quit", self.quit)

    def _update_ui(self) -> None:
        c = self.cfg
        if not self.running:
            state = "stopped"
        elif self.paused and self.state == "listening":
            state = "paused"
        else:
            state = self.state
        label = {"stopped": "Off", "loading": "Loading…", "listening": "Listening",
                 "paused": "Paused", "error": "Error"}.get(state, state)
        info = f" — {self.model_info}" if self.model_info and self.running else ""
        self.act_status.setText(f"● {label}{info}")
        self.tray.setIcon(make_icon(state))
        tip = f"{APP_NAME}: {label}{info}"
        if self.status:
            tip += f"\n{self.status}"
        self.tray.setToolTip(tip[:127])
        self.act_run.setText("Stop captioning" if self.running else "Start captioning")
        self.act_pause.setText("Resume" if self.paused else "Pause")
        self.act_pause.setEnabled(self.running)
        self.act_overlay.setText("Hide overlay" if self.overlay.isVisible() else "Show overlay")
        self.act_lock.setChecked(c.locked)
        for group, value in ((self.model_group, c.model), (self.lang_group, c.language_mode),
                             (self.out_group, c.output_mode)):
            for a in group.actions():
                a.setChecked(a.data() == value)

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self.open_settings()
        elif reason == QSystemTrayIcon.MiddleClick:
            self.toggle_pause()

    def _notify(self, text: str) -> None:
        self.tray.showMessage(APP_NAME, text, QSystemTrayIcon.Information, 4000)

    # ---------- engine ----------
    def start(self) -> None:
        old = self.engine
        old_thread = None
        if old is not None:
            self._disconnect(old)
            old.stop()
            old_thread = old.thread if old.thread.is_alive() else None
        self.engine = CaptionEngine(self.cfg, wait_for=old_thread)
        e = self.engine
        e.status.connect(self._on_status)
        e.state.connect(self._on_state)
        e.partial.connect(self.overlay.set_partial)
        e.final.connect(self._on_final)
        e.model_info.connect(self._on_model_info)
        e.set_paused(self.paused)
        self.running = True
        self.state = "loading"
        self.model_info = ""
        e.start()
        self._update_ui()

    def stop(self) -> None:
        if self.engine is not None:
            self._disconnect(self.engine)
            self.engine.stop()
            self.engine = None
        self.running = False
        self.overlay.set_partial("", "")
        self.overlay.show_status("Captioning stopped.", 3)
        self._update_ui()

    @staticmethod
    def _disconnect(e: CaptionEngine) -> None:
        for sig in (e.status, e.state, e.partial, e.final, e.model_info):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass

    def _on_status(self, text: str) -> None:
        self.status = text
        persistent = text.startswith(("Downloading", "Loading"))
        self.overlay.show_status(text, 0 if persistent else 5)
        if text.startswith(("Could not", "Engine error", "GPU failed")):
            self._notify(text)
        self._update_ui()

    def _on_state(self, state: str) -> None:
        self.state = state
        if state == "listening" and self.overlay.status_text.startswith(("Downloading", "Loading")):
            self.overlay.show_status(self.status, 3)
        self._update_ui()

    def _on_model_info(self, info: str) -> None:
        self.model_info = info
        self._update_ui()

    def _on_final(self, text: str, lang: str, translation: str, translated: bool) -> None:
        self.overlay.add_final(text, lang, translation, translated)
        stamp = datetime.now().strftime("%H:%M:%S")
        tag = f"{lang.upper()}→EN" if translated else lang.upper()
        line = f"[{stamp}] [{tag}] {text}"
        if translation:
            line += f"\n{' ' * 11}[EN] {translation}"
        self.history.append(line)
        if self.cfg.save_transcript:
            self._write_transcript(line)

    def _write_transcript(self, line: str) -> None:
        try:
            if self.transcript_path is None or self.transcript_path.parent != Path(self.cfg.transcript_dir):
                folder = Path(self.cfg.transcript_dir)
                folder.mkdir(parents=True, exist_ok=True)
                self.transcript_path = folder / f"captions-{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
            with self.transcript_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as e:
            self._on_status(f"Could not write transcript: {e}")

    # ---------- actions ----------
    def toggle_running(self) -> None:
        self.stop() if self.running else self.start()

    def toggle_pause(self) -> None:
        if not self.running:
            self.start()
            return
        self.paused = not self.paused
        if self.engine:
            self.engine.set_paused(self.paused)
        self.overlay.show_status("Paused" if self.paused else "Resumed", 0 if self.paused else 2)
        self._update_ui()

    def toggle_overlay(self) -> None:
        self.overlay.setVisible(not self.overlay.isVisible())
        self._update_ui()

    def toggle_lock(self) -> None:
        self._quick_set(locked=not self.cfg.locked)
        self.overlay.show_status("Overlay locked (click-through)" if self.cfg.locked
                                 else "Overlay unlocked — drag to move", 2.5)

    def _on_hotkey(self, action: str) -> None:
        {"pause": self.toggle_pause, "overlay": self.toggle_overlay, "lock": self.toggle_lock}[action]()

    def _on_hotkey_failed(self, text: str) -> None:
        self._notify(f"Hotkey {text} is already used by another app.")

    def _quick_set(self, **changes) -> None:
        cfg = self.cfg.copy()
        for k, v in changes.items():
            setattr(cfg, k, v)
        self.apply_config(cfg)
        if self.dialog is not None:
            self.dialog.close()

    def show_history(self) -> None:
        self.history.show()
        self.history.raise_()
        self.history.activateWindow()

    def open_transcript_folder(self) -> None:
        folder = Path(self.cfg.transcript_dir)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def open_settings(self) -> None:
        if self.dialog is not None:
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        self.dialog = SettingsDialog(self.cfg)
        self.dialog.setWindowIcon(make_icon("listening"))
        self.dialog.preview.connect(self.overlay.apply_config)
        self.dialog.applied.connect(self.apply_config)
        self.dialog.sample_requested.connect(self._show_sample)
        self.dialog.reset_position.connect(self._reset_position)
        self.dialog.finished.connect(self._dialog_closed)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _dialog_closed(self, *_):
        self.dialog.deleteLater()
        self.dialog = None
        self.overlay.apply_config(self.cfg)

    def _show_sample(self) -> None:
        self.overlay.add_final("みんなこんばんは！今日もよろしくね", "ja",
                               "Good evening everyone! Let's have a good time today", False)
        self.overlay.add_final("OK chat, let's go — we're doing the English challenge today!", "en", "", False)
        self.overlay.set_partial("えっ、ちょっと待って…", "ja")

    def _reset_position(self) -> None:
        cfg = self.dialog.collect() if self.dialog else self.cfg.copy()
        cfg.overlay_x = cfg.overlay_bottom = -1
        self.cfg.overlay_x = self.cfg.overlay_bottom = -1
        self.cfg.save()
        self.overlay.apply_config(cfg)
        if self.dialog:
            self.dialog.sync_geometry(cfg)

    def _on_overlay_moved(self, x: int, bottom: int, width: int) -> None:
        self.cfg.overlay_x, self.cfg.overlay_bottom, self.cfg.overlay_width = x, bottom, width
        self.cfg.save()
        if self.dialog:
            self.dialog.sync_geometry(self.cfg)

    def _on_font_wheel(self, size: int) -> None:
        self.cfg.font_size = size
        self.cfg.save()
        self.overlay.apply_config(self.cfg)
        if self.dialog:
            self.dialog.sync_geometry(self.cfg)

    def apply_config(self, cfg: Config) -> None:
        old = self.cfg
        self.cfg = cfg.copy()
        self.cfg.save()
        self.overlay.apply_config(self.cfg)
        if (old.hotkey_pause, old.hotkey_overlay, old.hotkey_lock) != \
                (cfg.hotkey_pause, cfg.hotkey_overlay, cfg.hotkey_lock):
            self._apply_hotkeys()
        if old.launch_at_login != cfg.launch_at_login:
            self._set_launch_at_login(cfg.launch_at_login)
        if old.save_transcript != cfg.save_transcript or old.transcript_dir != cfg.transcript_dir:
            self.transcript_path = None
        restart = any(getattr(old, k) != getattr(cfg, k) for k in ENGINE_KEYS)
        if self.running:
            if restart:
                self.overlay.clear()
                self.start()
            elif self.engine:
                self.engine.update_settings(self.cfg)
        self._update_ui()

    def _apply_hotkeys(self) -> None:
        self.hotkeys.set_hotkeys({"pause": self.cfg.hotkey_pause, "overlay": self.cfg.hotkey_overlay,
                                  "lock": self.cfg.hotkey_lock})

    def _set_launch_at_login(self, enabled: bool) -> None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_ALL_ACCESS) as key:
                if enabled:
                    pyw = Path(sys.executable).with_name("pythonw.exe")
                    exe = pyw if pyw.exists() else Path(sys.executable)
                    cmd = f'"{exe}" "{PROJECT_DIR / "WhisperLiveSubs.pyw"}"'
                    winreg.SetValueEx(key, APP_ID, 0, winreg.REG_SZ, cmd)
                else:
                    try:
                        winreg.DeleteValue(key, APP_ID)
                    except FileNotFoundError:
                        pass
        except OSError as e:
            self._notify(f"Could not change startup setting: {e}")

    def quit(self) -> None:
        self.hotkeys.stop()
        if self.engine:
            self._disconnect(self.engine)
            self.engine.stop()
            self.engine.thread.join(3)
        self.cfg.save()
        self.tray.hide()
        self.app.quit()


def _setup_cuda_dlls() -> None:
    """Make pip-installed NVIDIA cuBLAS/cuDNN DLLs (requirements-gpu.txt) visible to CTranslate2."""
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if nv.is_dir():
            for b in nv.glob("*/bin"):
                os.add_dll_directory(str(b))
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _setup_cuda_dlls()

    # Single instance
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, f"Local\\{APP_ID}")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        ctypes.windll.user32.MessageBoxW(None, f"{APP_NAME} is already running (see the system tray).",
                                         APP_NAME, 0x40)
        return 0
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(make_icon("listening"))
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("System tray not available")
    ctl = Controller(app)
    app.aboutToQuit.connect(lambda: ctl.cfg.save())
    code = app.exec()
    del mutex
    return code
