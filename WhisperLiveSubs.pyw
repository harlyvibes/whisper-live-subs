"""Windowless launcher (double-click, or used by "Launch when I sign in")."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pythonw has no console: send output to a log file for troubleshooting.
log_dir = Path(os.environ.get("APPDATA", Path.home())) / "WhisperLiveSubs"
log_dir.mkdir(parents=True, exist_ok=True)
if sys.stdout is None or sys.stderr is None:
    log = open(log_dir / "log.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log

from livesubs.app import main  # noqa: E402

sys.exit(main())
