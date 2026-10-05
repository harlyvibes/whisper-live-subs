"""Persistent settings for Whisper Live Subs."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "Whisper Live Subs"
APP_ID = "WhisperLiveSubs"

FROZEN = getattr(sys, "frozen", False)  # running as the PyInstaller-built exe
# Portable: in the exe build, models/ lives next to WhisperLiveSubs.exe.
PROJECT_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_ID
CONFIG_PATH = CONFIG_DIR / "settings.json"
DEFAULT_TRANSCRIPT_DIR = str(Path.home() / "Documents" / APP_ID)

# (key, label, approx download, note). Only multilingual models are listed:
# the ".en" / distil models cannot hear Japanese.
MODELS = [
    ("tiny", "tiny", "75 MB", "Fastest, weakest. Good for slow PCs."),
    ("base", "base", "145 MB", "Fast, rough accuracy."),
    ("small", "small", "485 MB", "Good balance on CPU. Recommended without a GPU."),
    ("medium", "medium", "1.5 GB", "Accurate. Needs a strong CPU or any NVIDIA GPU."),
    ("large-v3-turbo", "large-v3-turbo", "1.6 GB",
     "Near large-v3 accuracy, much faster. Best choice with a GPU. Poor at translation."),
    ("large-v2", "large-v2", "3 GB", "Very accurate, good translation. GPU recommended."),
    ("large-v3", "large-v3", "3 GB", "Most accurate. GPU strongly recommended."),
    ("custom", "Custom…", "", "A Hugging Face repo id or a local folder with a CTranslate2 Whisper model."),
]

# Common Whisper hallucinations on silence / music (matched after stripping punctuation).
DEFAULT_BLOCKLIST = "\n".join([
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "最後までご視聴いただきありがとうございました",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
    "おやすみなさい",
    "字幕視聴ありがとうございました",
    "Thank you for watching",
    "Thanks for watching",
    "Thank you for watching!",
    "Please subscribe",
    "Subtitles by the Amara.org community",
    "you",
])


@dataclass
class Config:
    # --- Model ---
    model: str = "small"
    custom_model: str = ""
    device: str = "auto"            # auto | cpu | cuda
    compute_type: str = "auto"      # auto | int8 | int8_float16 | float16 | float32
    cpu_threads: int = 0            # 0 = all cores
    beam_size: int = 1              # 1 = greedy (fastest)

    # --- Audio ---
    audio_source: str = "loopback"  # loopback (what you hear) | microphone
    audio_device: str = ""          # device name, "" = Windows default
    gain: float = 1.0

    # --- Language ---
    language_mode: str = "auto"     # auto | ja | en | <any whisper code>
    allowed_languages: str = "ja, en"
    output_mode: str = "original"   # original | translate | both
    show_language_tags: bool = False
    initial_prompt: str = ""
    live_partials: bool = True
    vad_threshold: float = 0.5
    silence_ms: int = 600
    max_phrase_s: float = 10.0
    hallucination_filter: bool = True
    blocklist: str = DEFAULT_BLOCKLIST

    # --- Appearance ---
    font_family: str = "Yu Gothic UI"
    font_size: int = 28
    font_bold: bool = True
    text_color: str = "#ffffffff"
    separate_ja_color: bool = False
    ja_text_color: str = "#ffffe9a8"
    translation_color: str = "#ffa8e6ff"
    partial_style: str = "dim"      # dim | italic | same
    outline_width: float = 3.0
    outline_color: str = "#ff000000"
    shadow: bool = True
    shadow_color: str = "#b4000000"
    shadow_offset: int = 2
    bg_color: str = "#8c000000"     # #AARRGGBB
    bg_fit: str = "text"            # text | full
    corner_radius: int = 10
    padding: int = 12
    max_lines: int = 3
    alignment: str = "center"       # left | center | right
    window_opacity: int = 100

    # --- Overlay window (bottom anchored) ---
    overlay_x: int = -1             # -1 = centre on primary screen
    overlay_bottom: int = -1
    overlay_width: int = 1100
    locked: bool = False

    # --- Behaviour ---
    auto_hide_s: float = 6.0        # 0 = never clear
    start_on_launch: bool = True
    launch_at_login: bool = False
    save_transcript: bool = False
    transcript_dir: str = DEFAULT_TRANSCRIPT_DIR
    hotkey_pause: str = "Ctrl+Alt+C"
    hotkey_overlay: str = "Ctrl+Alt+H"
    hotkey_lock: str = "Ctrl+Alt+L"

    def copy(self) -> "Config":
        return Config(**asdict(self))

    def allowed_list(self) -> list[str]:
        return [c.strip().lower() for c in self.allowed_languages.replace(";", ",").split(",") if c.strip()]

    def model_id(self) -> str:
        return self.custom_model.strip() if self.model == "custom" else self.model

    @classmethod
    def load(cls) -> "Config":
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name: f for f in fields(cls)}
        cfg = cls()
        for k, v in data.items():
            if k in known:
                default = getattr(cfg, k)
                try:
                    setattr(cfg, k, type(default)(v))
                except (TypeError, ValueError):
                    pass
        return cfg

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CONFIG_PATH)


# Keys that require the transcription engine to restart when changed.
ENGINE_KEYS = {
    "model", "custom_model", "device", "compute_type", "cpu_threads",
    "audio_source", "audio_device",
}


def model_is_downloaded(model_id: str) -> bool:
    if not model_id:
        return False
    p = Path(model_id)
    if p.is_dir():
        return (p / "model.bin").exists()
    from faster_whisper.utils import _MODELS
    repo = _MODELS.get(model_id, model_id)
    snap = MODELS_DIR / ("models--" + repo.replace("/", "--")) / "snapshots"
    return any((s / "model.bin").exists() for s in snap.glob("*")) if snap.exists() else False
