"""Options window."""
from __future__ import annotations

import os
import threading
from dataclasses import fields

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFontComboBox, QFormLayout,
                               QHBoxLayout, QKeySequenceEdit, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

from .config import FROZEN, MODELS, MODELS_DIR, Config, model_is_downloaded


class ColorButton(QPushButton):
    changed = Signal()

    def __init__(self, color: str, alpha: bool = True):
        super().__init__()
        self._alpha = alpha
        self.setFixedWidth(110)
        self.clicked.connect(self._pick)
        self.set_color(color)

    def color(self) -> str:
        return self._color.name(QColor.HexArgb)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        pm = QPixmap(28, 16)
        pm.fill(self._color)
        self.setIcon(QIcon(pm))
        a = round(self._color.alphaF() * 100)
        self.setText(self._color.name(QColor.HexRgb).upper() + (f" {a}%" if a < 100 else ""))

    def _pick(self) -> None:
        opts = QColorDialog.ShowAlphaChannel if self._alpha else QColorDialog.ColorDialogOption(0)
        c = QColorDialog.getColor(self._color, self, "Choose colour", opts)
        if c.isValid():
            self.set_color(c.name(QColor.HexArgb))
            self.changed.emit()


def _combo(items: list[tuple[str, str]]) -> QComboBox:
    cb = QComboBox()
    for value, label in items:
        cb.addItem(label, value)
    cb.setMinimumWidth(cb.fontMetrics().horizontalAdvance(max((l for _, l in items), key=len)) + 48)
    return cb


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
    return lab


class SettingsDialog(QDialog):
    preview = Signal(object)          # Config (appearance preview while editing)
    applied = Signal(object)          # Config
    sample_requested = Signal()
    reset_position = Signal()
    _download_done = Signal(str, str)  # model, error

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper Live Subs — Settings")
        self.setMinimumWidth(560)
        self.original = cfg.copy()
        self.cfg = cfg.copy()
        self._binds: dict[str, tuple] = {}
        self._loading = True

        tabs = QTabWidget()
        tabs.addTab(self._model_tab(), "Model")
        tabs.addTab(self._audio_tab(), "Audio")
        tabs.addTab(self._language_tab(), "Language")
        tabs.addTab(self._appearance_tab(), "Appearance")
        tabs.addTab(self._behaviour_tab(), "Behaviour")

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
                              | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(bb)

        self._download_done.connect(self._on_download_done)
        self._load(self.cfg)
        self._loading = False

    # ---------- binding helpers ----------
    def _bind(self, key: str, w):
        if isinstance(w, QCheckBox):
            get, put, sig = w.isChecked, w.setChecked, w.toggled
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            get, put, sig = w.value, w.setValue, w.valueChanged
        elif isinstance(w, QFontComboBox):
            get = lambda: w.currentFont().family()
            put = lambda v: w.setCurrentFont(QFont(v))
            sig = w.currentFontChanged
        elif isinstance(w, QComboBox):
            def put(v, w=w):
                i = w.findData(v)
                w.setCurrentIndex(i if i >= 0 else 0)
            get, sig = w.currentData, w.currentIndexChanged
        elif isinstance(w, QLineEdit):
            get, put, sig = w.text, w.setText, w.textChanged
        elif isinstance(w, QPlainTextEdit):
            get, put, sig = w.toPlainText, w.setPlainText, w.textChanged
        elif isinstance(w, ColorButton):
            get, put, sig = w.color, w.set_color, w.changed
        elif isinstance(w, QKeySequenceEdit):
            get = lambda: w.keySequence().toString(QKeySequence.PortableText)
            put = lambda v: w.setKeySequence(QKeySequence(v))
            sig = w.keySequenceChanged
        else:
            raise TypeError(w)
        self._binds[key] = (get, put)
        sig.connect(self._changed)
        return w

    def _load(self, cfg: Config) -> None:
        self._loading = True
        for key, (_, put) in self._binds.items():
            put(getattr(cfg, key))
        self._loading = False
        self._refresh_states()

    def collect(self) -> Config:
        cfg = self.cfg.copy()
        types = {f.name: type(getattr(cfg, f.name)) for f in fields(cfg)}
        for key, (get, _) in self._binds.items():
            v = get()
            setattr(cfg, key, types[key](v) if v is not None else getattr(cfg, key))
        return cfg

    def _changed(self, *_):
        if self._loading:
            return
        self._refresh_states()
        self.preview.emit(self.collect())

    def _refresh_states(self) -> None:
        self.custom_model.setEnabled(self.model_cb.currentData() == "custom")
        m = next((m for m in MODELS if m[0] == self.model_cb.currentData()), None)
        self.model_desc.setText(m[3] if m else "")
        self._update_model_status()
        self.allowed.setEnabled(self.lang_cb.currentData() == "auto")
        self.ja_color.setEnabled(self.sep_ja.isChecked())
        self.blocklist.setEnabled(self.halluc.isChecked())
        self.transcript_dir.setEnabled(self.save_tr.isChecked())

    # ---------- tabs ----------
    def _model_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.model_cb = self._bind("model", _combo([(m[0], f"{m[1]}  {m[2]}".strip()) for m in MODELS]))
        f.addRow("Whisper model:", self.model_cb)
        self.model_desc = _hint("")
        f.addRow("", self.model_desc)
        self.custom_model = self._bind("custom_model", QLineEdit())
        self.custom_model.setPlaceholderText("e.g. kotoba-tech/kotoba-whisper-v2.0-faster  or  C:\\models\\my-model")
        f.addRow("Custom model:", self.custom_model)

        row = QHBoxLayout()
        self.model_status = QLabel()
        self.dl_btn = QPushButton("Download now")
        self.dl_btn.clicked.connect(self._download)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(lambda: (MODELS_DIR.mkdir(exist_ok=True), os.startfile(MODELS_DIR)))
        row.addWidget(self.model_status, 1)
        row.addWidget(self.dl_btn)
        row.addWidget(open_btn)
        f.addRow("Status:", row)

        self.device_cb = self._bind("device", _combo([
            ("auto", "Auto (NVIDIA GPU if available)"), ("cpu", "CPU"), ("cuda", "NVIDIA GPU (CUDA)")]))
        f.addRow("Device:", self.device_cb)
        try:
            import ctranslate2
            n = ctranslate2.get_cuda_device_count()
        except Exception:  # noqa: BLE001
            n = 0
        f.addRow("", _hint(f"CUDA GPUs detected: {n}. " + (
            "GPU mode is available." if n else
            "Running on CPU. With an NVIDIA GPU, " + (
                "use the Python version (setup.bat) for GPU acceleration." if FROZEN else
                "install requirements-gpu.txt for 5-20× speed."))))
        f.addRow("Precision:", self._bind("compute_type", _combo([
            ("auto", "Auto (float16 on GPU, int8 on CPU)"), ("int8", "int8 (fast, low memory)"),
            ("int8_float16", "int8_float16 (GPU)"), ("float16", "float16 (GPU)"),
            ("float32", "float32 (slow, max precision)")])))
        threads = self._bind("cpu_threads", QSpinBox())
        threads.setRange(0, os.cpu_count() or 64)
        threads.setSpecialValueText("All cores")
        f.addRow("CPU threads:", threads)
        beam = self._bind("beam_size", QSpinBox())
        beam.setRange(1, 5)
        f.addRow("Beam size:", beam)
        f.addRow("", _hint("Beam size 1 is fastest. 3-5 is slightly more accurate for final captions "
                           "but slower. Changing model/device restarts captioning."))
        return w

    def _audio_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.source_cb = self._bind("audio_source", _combo([
            ("loopback", "System audio — what you hear (streams, videos, games)"),
            ("microphone", "Microphone / input device")]))
        self.source_cb.currentIndexChanged.connect(lambda *_: self._fill_devices())
        f.addRow("Source:", self.source_cb)
        row = QHBoxLayout()
        self.device_list = QComboBox()
        self._binds["audio_device"] = (self.device_list.currentData, self._set_device)
        self.device_list.currentIndexChanged.connect(self._changed)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._fill_devices)
        row.addWidget(self.device_list, 1)
        row.addWidget(refresh)
        f.addRow("Device:", row)
        gain = self._bind("gain", QDoubleSpinBox())
        gain.setRange(0.1, 20.0)
        gain.setSingleStep(0.25)
        gain.setSuffix(" ×")
        f.addRow("Input gain:", gain)
        f.addRow("", _hint("System audio captures everything playing on the selected output device. "
                           "Tip: route the stream to its own output device in Windows' "
                           "'App volume and device preferences' to ignore other sounds."))
        self._fill_devices()
        return w

    def _fill_devices(self) -> None:
        from .audio import list_devices
        current = self.device_list.currentData() if self.device_list.count() else self.cfg.audio_device
        loading = self._loading
        self._loading = True
        self.device_list.clear()
        self.device_list.addItem("Windows default device", "")
        try:
            for name in list_devices(self.source_cb.currentData() or "loopback"):
                self.device_list.addItem(name, name)
        except Exception as e:  # noqa: BLE001
            self.device_list.addItem(f"(error listing devices: {e})", "")
        self._set_device(current)
        self._loading = loading

    def _set_device(self, name: str) -> None:
        i = self.device_list.findData(name)
        self.device_list.setCurrentIndex(i if i >= 0 else 0)

    def _language_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.lang_cb = self._bind("language_mode", _combo([
            ("auto", "Auto-detect each phrase (bilingual)"),
            ("ja", "Japanese only"), ("en", "English only")]))
        f.addRow("Spoken language:", self.lang_cb)
        self.allowed = self._bind("allowed_languages", QLineEdit())
        f.addRow("Allowed languages:", self.allowed)
        f.addRow("", _hint("Detection is restricted to these Whisper language codes, so short Japanese "
                           "phrases are never mistaken for Chinese/Korean. e.g. 'ja, en' or 'ja, en, ko'."))
        f.addRow("Captions show:", self._bind("output_mode", _combo([
            ("original", "What was said (Japanese stays Japanese)"),
            ("translate", "English (Japanese is translated)"),
            ("both", "Both: original + English translation underneath")])))
        f.addRow("", _hint("Translation uses Whisper's built-in translator. large-v3-turbo translates poorly; "
                           "use small/medium/large-v2/large-v3 for it. 'Both' needs ~2× the compute."))
        f.addRow("", self._bind("show_language_tags", QCheckBox("Prefix lines with a language tag ([JA] / [EN])")))
        f.addRow("", self._bind("live_partials", QCheckBox("Live-updating captions while a phrase is being spoken")))
        prompt = self._bind("initial_prompt", QLineEdit())
        prompt.setPlaceholderText("Streamer / game names, slang… e.g. ぺこら, Hololive, Minecraft")
        f.addRow("Vocabulary hint:", prompt)
        vad = self._bind("vad_threshold", QDoubleSpinBox())
        vad.setRange(0.1, 0.95)
        vad.setSingleStep(0.05)
        f.addRow("Voice detection threshold:", vad)
        f.addRow("", _hint("Raise if background music/game sounds produce junk captions; lower if quiet speech is missed."))
        sil = self._bind("silence_ms", QSpinBox())
        sil.setRange(200, 3000)
        sil.setSingleStep(50)
        sil.setSuffix(" ms")
        f.addRow("Pause that ends a phrase:", sil)
        mx = self._bind("max_phrase_s", QDoubleSpinBox())
        mx.setRange(3, 28)
        mx.setSuffix(" s")
        f.addRow("Max phrase length:", mx)
        self.halluc = self._bind("hallucination_filter", QCheckBox("Filter known Whisper hallucinations (one per line):"))
        f.addRow("", self.halluc)
        self.blocklist = self._bind("blocklist", QPlainTextEdit())
        self.blocklist.setFixedHeight(90)
        f.addRow("", self.blocklist)
        return w

    def _appearance_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        row = QHBoxLayout()
        row.addWidget(self._bind("font_family", QFontComboBox()), 1)
        size = self._bind("font_size", QSpinBox())
        size.setRange(8, 200)
        size.setSuffix(" px")
        row.addWidget(size)
        row.addWidget(self._bind("font_bold", QCheckBox("Bold")))
        f.addRow("Font:", row)
        f.addRow("", _hint("Pick a font with Japanese glyphs (Yu Gothic UI, Meiryo, MS Gothic, Noto Sans JP…)."))
        f.addRow("Text colour:", self._bind("text_color", ColorButton(self.cfg.text_color)))
        row = QHBoxLayout()
        self.sep_ja = self._bind("separate_ja_color", QCheckBox("Different colour for Japanese:"))
        self.ja_color = self._bind("ja_text_color", ColorButton(self.cfg.ja_text_color))
        row.addWidget(self.sep_ja)
        row.addWidget(self.ja_color)
        row.addStretch()
        f.addRow("", row)
        f.addRow("Translation colour:", self._bind("translation_color", ColorButton(self.cfg.translation_color)))
        f.addRow("In-progress text:", self._bind("partial_style", _combo([
            ("dim", "Slightly faded"), ("italic", "Italic"), ("same", "Same as final")])))
        row = QHBoxLayout()
        ow = self._bind("outline_width", QDoubleSpinBox())
        ow.setRange(0, 12)
        ow.setSingleStep(0.5)
        ow.setSuffix(" px")
        row.addWidget(ow)
        row.addWidget(self._bind("outline_color", ColorButton(self.cfg.outline_color)))
        row.addStretch()
        f.addRow("Outline:", row)
        row = QHBoxLayout()
        row.addWidget(self._bind("shadow", QCheckBox("Drop shadow")))
        row.addWidget(self._bind("shadow_color", ColorButton(self.cfg.shadow_color)))
        so = self._bind("shadow_offset", QSpinBox())
        so.setRange(0, 20)
        so.setSuffix(" px offset")
        row.addWidget(so)
        row.addStretch()
        f.addRow("Shadow:", row)
        row = QHBoxLayout()
        row.addWidget(self._bind("bg_color", ColorButton(self.cfg.bg_color)))
        row.addWidget(self._bind("bg_fit", _combo([("text", "Fit to text"), ("full", "Full width")])))
        row.addStretch()
        f.addRow("Background:", row)
        f.addRow("", _hint("Set background opacity to 0% in the colour picker for no box."))
        row = QHBoxLayout()
        cr = self._bind("corner_radius", QSpinBox())
        cr.setRange(0, 60)
        cr.setPrefix("Corners ")
        pad = self._bind("padding", QSpinBox())
        pad.setRange(0, 80)
        pad.setPrefix("Padding ")
        row.addWidget(cr)
        row.addWidget(pad)
        row.addStretch()
        f.addRow("Box shape:", row)
        ml = self._bind("max_lines", QSpinBox())
        ml.setRange(1, 20)
        f.addRow("Lines shown:", ml)
        f.addRow("Alignment:", self._bind("alignment", _combo([
            ("center", "Centre"), ("left", "Left"), ("right", "Right")])))
        row = QHBoxLayout()
        ww = self._bind("overlay_width", QSpinBox())
        ww.setRange(200, 8000)
        ww.setSuffix(" px")
        row.addWidget(ww)
        rp = QPushButton("Reset position")
        rp.clicked.connect(self.reset_position)
        row.addWidget(rp)
        row.addStretch()
        f.addRow("Overlay width:", row)
        op = self._bind("window_opacity", QSpinBox())
        op.setRange(10, 100)
        op.setSuffix(" %")
        f.addRow("Overall opacity:", op)
        sample = QPushButton("Show sample captions")
        sample.clicked.connect(self.sample_requested)
        f.addRow("", sample)
        return w

    def _behaviour_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        ah = self._bind("auto_hide_s", QDoubleSpinBox())
        ah.setRange(0, 120)
        ah.setSuffix(" s")
        ah.setSpecialValueText("Never")
        f.addRow("Clear captions after silence:", ah)
        f.addRow("", self._bind("locked", QCheckBox("Lock overlay (click-through, cannot be moved)")))
        f.addRow("", self._bind("start_on_launch", QCheckBox("Start captioning when the app launches")))
        f.addRow("", self._bind("launch_at_login", QCheckBox("Launch when I sign in to Windows")))
        self.save_tr = self._bind("save_transcript", QCheckBox("Save transcripts to text files"))
        f.addRow("", self.save_tr)
        row = QHBoxLayout()
        self.transcript_dir = self._bind("transcript_dir", QLineEdit())
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.transcript_dir, 1)
        row.addWidget(browse)
        f.addRow("Transcript folder:", row)
        for key, label in (("hotkey_pause", "Pause / resume:"), ("hotkey_overlay", "Show / hide overlay:"),
                           ("hotkey_lock", "Lock / unlock overlay:")):
            row = QHBoxLayout()
            ed = self._bind(key, QKeySequenceEdit())
            ed.setMaximumSequenceLength(1)
            clr = QPushButton("Clear")
            clr.clicked.connect(ed.clear)
            row.addWidget(ed, 1)
            row.addWidget(clr)
            f.addRow(label, row)
        f.addRow("", _hint("Global hotkeys work while other apps (games, browsers) are focused."))
        return w

    # ---------- actions ----------
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Transcript folder", self.transcript_dir.text())
        if d:
            self.transcript_dir.setText(d)

    def _update_model_status(self) -> None:
        mid = self.custom_model.text().strip() if self.model_cb.currentData() == "custom" else self.model_cb.currentData()
        if getattr(self, "_downloading", None) == mid:
            self.model_status.setText("Downloading…")
            self.dl_btn.setEnabled(False)
            return
        ok = model_is_downloaded(mid) if mid else False
        self.model_status.setText("✔ Downloaded" if ok else "Not downloaded (downloads on first use)")
        self.dl_btn.setEnabled(bool(mid) and not ok and not getattr(self, "_downloading", None))

    def _download(self) -> None:
        mid = self.collect().model_id()
        if not mid:
            return
        self._downloading = mid
        self._update_model_status()

        def work():
            err = ""
            try:
                from faster_whisper.utils import download_model
                download_model(mid, cache_dir=str(MODELS_DIR))
            except Exception as e:  # noqa: BLE001
                err = str(e)
            self._download_done.emit(mid, err)

        threading.Thread(target=work, daemon=True).start()

    def _on_download_done(self, mid: str, err: str) -> None:
        self._downloading = None
        self._update_model_status()
        if err:
            self.model_status.setText(f"Download failed: {err[:80]}")

    def _defaults(self) -> None:
        d = Config()
        # keep window placement
        d.overlay_x, d.overlay_bottom, d.overlay_width = self.cfg.overlay_x, self.cfg.overlay_bottom, self.cfg.overlay_width
        self._load(d)
        self.preview.emit(self.collect())

    def _apply(self) -> None:
        self.cfg = self.collect()
        self.original = self.cfg.copy()
        self.applied.emit(self.cfg.copy())

    def _ok(self) -> None:
        self._apply()
        self.accept()

    def reject(self) -> None:
        self.preview.emit(self.original.copy())  # undo un-applied previews
        super().reject()

    def sync_geometry(self, cfg: Config) -> None:
        """Overlay was moved/resized by mouse while the dialog is open."""
        self.cfg.overlay_x, self.cfg.overlay_bottom = cfg.overlay_x, cfg.overlay_bottom
        self.original.overlay_x, self.original.overlay_bottom = cfg.overlay_x, cfg.overlay_bottom
        self.original.overlay_width = cfg.overlay_width
        if "overlay_width" in self._binds:
            self._loading = True
            self._binds["overlay_width"][1](cfg.overlay_width)
            self._binds["font_size"][1](cfg.font_size)
            self.original.font_size = cfg.font_size
            self._loading = False
