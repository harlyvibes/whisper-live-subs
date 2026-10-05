"""Build the standalone Windows app (no Python needed) into dist/.

    .venv\\Scripts\\python -m pip install pyinstaller
    .venv\\Scripts\\python tools\\build_exe.py

Produces dist/WhisperLiveSubs/ (one-folder build: fast start-up) and
dist/WhisperLiveSubs-<version>-win64.zip. Whisper models are not bundled; they
download on first use into models/ next to the exe.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
NAME = "WhisperLiveSubs"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "dev"


def make_icon() -> Path:
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication

    from livesubs.app import make_icon as qicon

    QApplication.instance() or QApplication([])  # needed before using QPixmap
    ico = BUILD / "icon.ico"
    ico.parent.mkdir(parents=True, exist_ok=True)
    qicon("listening").pixmap(256, 256).toImage().save(str(ico))
    return ico


def main() -> None:
    ico = make_icon()
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
        "--name", NAME, "--icon", str(ico),
        "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller"), "--specpath", str(BUILD),
        "--collect-data", "faster_whisper",     # Silero VAD model
        "--collect-binaries", "ctranslate2",
        "--hidden-import", "pyaudiowpatch",
        # Qt modules the app never uses (keeps the download smaller)
        *[a for m in ("QtWebEngineCore", "QtWebEngineWidgets", "QtQuick", "QtQml", "Qt3DCore",
                      "QtMultimedia", "QtPdf", "QtCharts", "QtDataVisualization", "QtSql",
                      "QtNetwork", "QtOpenGL", "QtSvg")
          for a in ("--exclude-module", f"PySide6.{m}")],
        "--exclude-module", "tkinter",
        str(ROOT / "WhisperLiveSubs.pyw"),
    ], check=True, cwd=ROOT)

    app_dir = DIST / NAME
    shutil.copy(ROOT / "README.md", app_dir / "README.md")
    (app_dir / "models").mkdir(exist_ok=True)

    zpath = DIST / f"{NAME}-{VERSION}-win64.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(app_dir.rglob("*")):
            z.write(f, Path(NAME) / f.relative_to(app_dir))
    print(f"Built {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
