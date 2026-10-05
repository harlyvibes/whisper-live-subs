"""Model downloads with progress reporting, and memory checks."""
from __future__ import annotations

import ctypes
import fnmatch
import threading
import time
from pathlib import Path
from typing import Callable

from .config import MODELS_DIR

# Same files faster_whisper.utils.download_model fetches.
ALLOW_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

# Measured peak RAM (GB) while loading each model on CPU as int8 (CT2_USE_MKL=0):
# roughly the download size + ~0.35 GB. large-* values are extrapolated.
RAM_NEEDED_GB = {"tiny": 0.4, "base": 0.5, "small": 0.8, "medium": 1.9,
                 "large-v3-turbo": 2.0, "large-v2": 3.5, "large-v3": 3.5}


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram_gb() -> tuple[float, float]:
    """(total, available) physical RAM in GB."""
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullTotalPhys / 1024 ** 3, s.ullAvailPhys / 1024 ** 3


def repo_id(model_id: str) -> str:
    from faster_whisper.utils import _MODELS
    return _MODELS.get(model_id, model_id)


def cache_dir(model_id: str) -> Path:
    return MODELS_DIR / ("models--" + repo_id(model_id).replace("/", "--"))


def _dir_size(p: Path) -> int:
    total = 0
    if p.exists():
        for f in p.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    total += f.stat().st_size
            except OSError:  # file renamed/moved mid-scan
                pass
    return total


def remote_size(model_id: str) -> int | None:
    """Total bytes of the files we download, or None if the Hub can't be reached."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id(model_id), files_metadata=True)
        return sum(s.size or 0 for s in info.siblings
                   if any(fnmatch.fnmatch(s.rfilename, pat) for pat in ALLOW_PATTERNS)) or None
    except Exception:  # noqa: BLE001
        return None


def format_progress(name: str, done: int, total: int | None, speed: float) -> str:
    mb = 1024 * 1024
    spd = f" · {speed / mb:.1f} MB/s" if speed > 0 else ""
    if total:
        pct = min(100, int(done * 100 / total))
        return f"Downloading '{name}': {pct}% · {done // mb:,} / {total // mb:,} MB{spd}"
    return f"Downloading '{name}': {done // mb:,} MB{spd}"


def download(model_id: str, on_progress: Callable[[int, int | None, float], None],
             should_stop: Callable[[], bool] = lambda: False) -> None:
    """Download a model into MODELS_DIR, calling on_progress(done, total, bytes/s) ~2×/s.

    Progress is measured from bytes on disk, so it works for both the plain HTTP
    and the hf_xet download paths. Returns early (download keeps going in the
    background) if should_stop() becomes true.
    """
    if Path(model_id).is_dir():
        return
    from faster_whisper.utils import download_model

    total = remote_size(model_id)
    err: list[BaseException] = []

    def work():
        try:
            download_model(model_id, cache_dir=str(MODELS_DIR))
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=work, daemon=True, name=f"download-{model_id}")
    t.start()
    folder = cache_dir(model_id)
    last_t, last_b, speed = time.monotonic(), _dir_size(folder), 0.0
    while t.is_alive():
        t.join(0.5)
        if should_stop():
            return
        now, done = time.monotonic(), _dir_size(folder)
        if now - last_t >= 1.0:
            inst = max(0.0, (done - last_b) / (now - last_t))
            speed = inst if speed == 0 else 0.6 * speed + 0.4 * inst
            last_t, last_b = now, done
        on_progress(min(done, total) if total else done, total, speed)
    if err:
        raise err[0]
    if total:
        on_progress(total, total, 0.0)


def is_memory_error(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, MemoryError) or "alloc" in text or "out of memory" in text
