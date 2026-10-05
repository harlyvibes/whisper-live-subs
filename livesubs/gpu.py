"""NVIDIA GPU detection, diagnostics and on-demand cuBLAS install.

What CTranslate2's Windows build (4.8) actually needs for the GPU, verified from
ctranslate2.dll: the CUDA runtime is linked in, cuDNN is not used at all, and the
only NVIDIA library it loads at runtime is cublas64_12.dll (+ cublasLt64_12.dll),
looked up *by name*, once per process. So:

  * a CUDA 13 toolkit doesn't help (it ships cublas64_13.dll),
  * a failed first attempt can't be fixed without restarting the app,
  * the reliable fix is to preload a cublas64_12.dll we found by *full path*
    before CTranslate2 asks for it (Windows then reuses the loaded module).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import re
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Callable

from .config import PROJECT_DIR

CUDA_DIR = PROJECT_DIR / "cuda"
WHEELS = [("nvidia-cublas-cu12", "12.9.2.10")]
CUBLAS_DLLS = ["cublasLt64_12.dll", "cublas64_12.dll"]   # load order (cublas depends on cublasLt)
INSTALL_SIZE_TEXT = "about 530 MB download"

LAST_ERROR = ""          # full traceback of the last GPU failure, for the diagnostics report
_preloaded: dict[str, ctypes.CDLL] = {}
_dll_dirs_added: set[str] = set()


# ---------------------------------------------------------------- driver / devices
def _nvcuda():
    try:
        return ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return None


def devices() -> list[tuple[str, tuple[int, int]]]:
    """[(name, (cc_major, cc_minor))] for NVIDIA GPUs the installed driver can see."""
    cu = _nvcuda()
    if cu is None:
        return []
    try:
        if cu.cuInit(0) != 0:
            return []
        count = ctypes.c_int()
        if cu.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        out = []
        for i in range(count.value):
            dev = ctypes.c_int()
            buf = ctypes.create_string_buffer(256)
            major, minor = ctypes.c_int(), ctypes.c_int()
            if cu.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue
            cu.cuDeviceGetName(buf, 256, dev)
            cu.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # COMPUTE_CAPABILITY_MAJOR
            cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # COMPUTE_CAPABILITY_MINOR
            out.append((buf.value.decode(errors="replace"), (major.value, minor.value)))
        return out
    except (AttributeError, OSError):
        return []


def device_names() -> list[str]:
    return [n for n, _ in devices()]


def driver_cuda_version() -> tuple[int, int] | None:
    cu = _nvcuda()
    try:
        v = ctypes.c_int()
        if cu is not None and cu.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return v.value // 1000, (v.value % 1000) // 10
    except (AttributeError, OSError):
        pass
    return None


# ---------------------------------------------------------------- finding cuBLAS 12
def _candidate_dirs() -> list[Path]:
    dirs = [CUDA_DIR]
    for base in map(Path, sys.path):  # pip's nvidia-cublas-cu12 (source install)
        if (base / "nvidia").is_dir():
            dirs += sorted((base / "nvidia").glob("*/bin"))
    for key, val in sorted(os.environ.items()):  # CUDA 12 toolkits (CUDA_PATH_V12_x)
        if key.upper().startswith("CUDA_PATH") and val:
            dirs += [Path(val) / "bin", Path(val) / "bin" / "x64"]
    dirs += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p.strip()]
    seen, out = set(), []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_cublas_dir() -> Path | None:
    for d in _candidate_dirs():
        try:
            if all((d / name).is_file() for name in CUBLAS_DLLS):
                return d
        except OSError:
            pass
    return None


def setup_dll_paths() -> None:
    """Also expose the folder on PATH / the DLL search path (belt and braces)."""
    d = find_cublas_dir()
    if d is not None and str(d) not in _dll_dirs_added:
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        _dll_dirs_added.add(str(d))


def preload() -> str:
    """Load cuBLAS 12 by full path so CTranslate2's by-name lookup finds it. '' on success."""
    if all(n in _preloaded for n in CUBLAS_DLLS):
        return ""
    d = find_cublas_dir()
    if d is None:
        return "cublas64_12.dll (CUDA 12 cuBLAS) not found"
    setup_dll_paths()
    for name in CUBLAS_DLLS:
        try:
            _preloaded[name] = ctypes.WinDLL(str(d / name))
        except OSError as e:
            return f"{d / name} could not be loaded: {e}"
    return ""


# ---------------------------------------------------------------- status / advice
def status() -> tuple[str, str]:
    """(kind, explanation). kind: none | driver_old | libs_missing | ready"""
    devs = devices()
    if not devs:
        return "none", ("No NVIDIA GPU found. Windows doesn't see an NVIDIA card/driver on this PC "
                        "(a virtual machine can't use the host's GPU).")
    name = devs[0][0]
    ver = driver_cuda_version()
    if ver and ver < (12, 0):
        return "driver_old", (f"{name} found, but its driver only supports CUDA {ver[0]}.{ver[1]}. "
                              "Update the NVIDIA driver (CUDA 12 or newer is needed).")
    d = find_cublas_dir()
    if d is None:
        return "libs_missing", (f"{name} found, but cuBLAS for CUDA 12 (cublas64_12.dll) isn't installed — "
                                "a CUDA 13 toolkit doesn't include it. "
                                f"Click 'Install GPU support' ({INSTALL_SIZE_TEXT}).")
    where = "app's cuda folder" if d == CUDA_DIR else str(d)
    return "ready", f"{name} ready (driver CUDA {ver[0]}.{ver[1]}, cuBLAS from {where})." if ver \
        else f"{name} ready (cuBLAS from {where})."


def why_not_cuda() -> str:
    """Explanation when CTranslate2 reports no CUDA device; '' if there's simply no NVIDIA card."""
    kind, text = status()
    if kind == "none":
        return ""
    if kind == "ready":
        return "The NVIDIA driver is present but CUDA reported no usable device. Try updating the driver."
    return text


def best_compute_type(requested: str) -> str:
    """A compute type the GPU actually supports (GTX 10-series can't do float16, etc.)."""
    import ctranslate2
    try:
        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:  # noqa: BLE001
        return "float32"
    if requested not in ("auto", "default") and requested in supported:
        return requested
    for ct in ("float16", "int8_float16", "bfloat16", "int8_float32", "int8", "float32"):
        if ct in supported:
            return ct
    return "float32"


def explain_cuda_error(e: BaseException) -> str:
    msg = str(e)
    low = msg.lower()
    if "cublas" in low and ("not found" in low or "cannot be loaded" in low or "load" in low):
        return (f"cuBLAS for CUDA 12 couldn't be loaded. Open Settings → Model and click "
                f"'Install GPU support' ({INSTALL_SIZE_TEXT}), then restart the app.")
    if "float16" in low or "compute type" in low:
        return f"This GPU doesn't support the requested precision ({msg[:120]}). Set Precision to Auto."
    if "out of memory" in low:
        return "The GPU ran out of video memory. Pick a smaller model or close other GPU apps."
    if "driver" in low or "insufficient" in low:
        return "The NVIDIA driver is too old for CUDA 12. Update it from nvidia.com."
    return f"GPU error: {msg[:200]}"


def record_error(e: BaseException) -> None:
    global LAST_ERROR
    LAST_ERROR = "".join(traceback.format_exception(e)).strip()


def report() -> str:
    """Plain-text diagnostics the user can copy into a bug report."""
    lines = ["Whisper Live Subs — GPU report",
             f"Windows {platform.version()} | Python {platform.python_version()} | "
             f"{'exe' if getattr(sys, 'frozen', False) else 'source'} at {PROJECT_DIR}"]
    try:
        import ctranslate2
        lines.append(f"ctranslate2 {ctranslate2.__version__}; CUDA devices: {ctranslate2.get_cuda_device_count()}")
        try:
            lines.append(f"supported compute types (cuda): {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"supported compute types (cuda): error: {e}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"ctranslate2 import failed: {e}")
    ver = driver_cuda_version()
    lines.append(f"driver CUDA version: {f'{ver[0]}.{ver[1]}' if ver else 'none (no NVIDIA driver)'}")
    for n, cc in devices() or [("no NVIDIA GPU visible", (0, 0))]:
        lines.append(f"GPU: {n} (compute capability {cc[0]}.{cc[1]})")
    d = find_cublas_dir()
    lines.append(f"cuBLAS 12 folder: {d or 'NOT FOUND'}")
    lines.append(f"preloaded: {sorted(_preloaded) or 'nothing yet'}")
    for key, val in sorted(os.environ.items()):
        if key.upper().startswith("CUDA_PATH"):
            lines.append(f"{key}={val}")
    hits = [p for p in os.environ.get("PATH", "").split(os.pathsep)
            if p and any(Path(p).glob("cublas64_*.dll"))]
    lines.append(f"PATH folders with cublas: {hits or 'none'}")
    kind, text = status()
    lines.append(f"status: {kind} — {text}")
    lines.append("last GPU error:\n" + (LAST_ERROR or "none"))
    return "\n".join(lines)


# ---------------------------------------------------------------- installer
def _wheel_info(pkg: str, version: str) -> dict:
    import httpx
    j = httpx.get(f"https://pypi.org/pypi/{pkg}/{version}/json", timeout=30, follow_redirects=True).json()
    for f in j["urls"]:
        if f["filename"].endswith("win_amd64.whl"):
            return {"url": f["url"], "size": f["size"], "sha256": f["digests"]["sha256"], "name": f["filename"]}
    raise RuntimeError(f"No Windows build of {pkg} {version} on PyPI")


def install(on_progress: Callable[[int, int, float], None],
            should_stop: Callable[[], bool] = lambda: False) -> None:
    """Download NVIDIA's cuBLAS 12 wheel from PyPI and unpack its DLLs into cuda/."""
    import httpx
    infos = [_wheel_info(p, v) for p, v in WHEELS]
    total = sum(i["size"] for i in infos)
    staging = PROJECT_DIR / "cuda.download"
    staging.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.monotonic()
    for info in infos:
        dest = staging / info["name"]
        h = hashlib.sha256()
        with httpx.stream("GET", info["url"], timeout=60, follow_redirects=True) as r, dest.open("wb") as fh:
            r.raise_for_status()
            last = 0.0
            for chunk in r.iter_bytes(1 << 20):
                if should_stop():
                    raise RuntimeError("cancelled")
                fh.write(chunk)
                h.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last > 0.3:
                    on_progress(done, total, done / max(0.001, now - t0))
                    last = now
        if h.hexdigest() != info["sha256"]:
            raise RuntimeError(f"Download of {info['name']} was corrupted (checksum mismatch). Try again.")
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    for old in CUDA_DIR.glob("cudnn*.dll"):  # left by v1.1.0's installer; never used
        old.unlink(missing_ok=True)
    for info in infos:
        with zipfile.ZipFile(staging / info["name"]) as z:
            for member in z.namelist():
                if re.fullmatch(r"nvidia/[^/]+/bin/[^/]+\.dll", member):
                    with z.open(member) as src, (CUDA_DIR / Path(member).name).open("wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
    (CUDA_DIR / "VERSIONS.txt").write_text("\n".join(f"{p}=={v}" for p, v in WHEELS) + "\n")
    shutil.rmtree(staging, ignore_errors=True)
    on_progress(total, total, 0.0)
    setup_dll_paths()
