"""NVIDIA GPU detection, diagnostics and on-demand CUDA library install.

CTranslate2 ships with GPU support but loads NVIDIA's cuBLAS and cuDNN DLLs at
runtime. They are ~1.2 GB, so instead of bundling them the app can download the
official NVIDIA wheels from PyPI and unpack just the DLLs into cuda/ next to the
app (works for both the exe and the source version).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

from .config import PROJECT_DIR

CUDA_DIR = PROJECT_DIR / "cuda"
# cuDNN must match the cudnn64_9.dll dispatcher bundled with ctranslate2 (9.10.2.21
# for ctranslate2 4.8); mixing in sub-libraries from another 9.x release can fail.
WHEELS = [("nvidia-cublas-cu12", "12.9.2.10"), ("nvidia-cudnn-cu12", "9.10.2.21")]
REQUIRED_DLLS = ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll",
                 "cudnn_cnn64_9.dll"]
INSTALL_SIZE_TEXT = "about 1.2 GB download"

_dll_dirs_added: set[str] = set()


def setup_dll_paths() -> None:
    """Make CUDA DLLs from cuda/ (in-app install) or pip's nvidia-* packages loadable."""
    candidates = [CUDA_DIR]
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if nv.is_dir():
            candidates += list(nv.glob("*/bin"))
    for d in candidates:
        s = str(d)
        if d.is_dir() and s not in _dll_dirs_added:
            os.add_dll_directory(s)
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
            _dll_dirs_added.add(s)


def device_names() -> list[str]:
    """NVIDIA GPUs the installed driver can see (empty if no card/driver)."""
    try:
        cu = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return []
    try:
        if cu.cuInit(0) != 0:
            return []
        count = ctypes.c_int()
        if cu.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        names = []
        for i in range(count.value):
            dev = ctypes.c_int()
            buf = ctypes.create_string_buffer(256)
            if cu.cuDeviceGet(ctypes.byref(dev), i) == 0 and cu.cuDeviceGetName(buf, 256, dev) == 0:
                names.append(buf.value.decode(errors="replace"))
        return names
    except (AttributeError, OSError):
        return []


def driver_cuda_version() -> tuple[int, int] | None:
    try:
        cu = ctypes.WinDLL("nvcuda.dll")
        v = ctypes.c_int()
        if cu.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return v.value // 1000, (v.value % 1000) // 10
    except (AttributeError, OSError):
        pass
    return None


def missing_dlls() -> list[str]:
    setup_dll_paths()
    missing = []
    for name in REQUIRED_DLLS:
        try:
            ctypes.WinDLL(name)
        except OSError:
            missing.append(name)
    return missing


def status() -> tuple[str, str]:
    """(kind, explanation). kind: none | driver_old | libs_missing | ready"""
    names = device_names()
    if not names:
        return "none", ("No NVIDIA GPU found. Windows doesn't see an NVIDIA card/driver on this PC "
                        "(a virtual machine can't use the host's GPU).")
    ver = driver_cuda_version()
    if ver and ver < (12, 0):
        return "driver_old", (f"{names[0]} found, but its driver only supports CUDA {ver[0]}.{ver[1]}. "
                              "Update the NVIDIA driver (CUDA 12 or newer is needed).")
    if missing_dlls():
        return "libs_missing", (f"{names[0]} found, but the GPU libraries (cuBLAS/cuDNN) aren't installed. "
                                f"Click 'Install GPU support' ({INSTALL_SIZE_TEXT}).")
    return "ready", f"{names[0]} ready (CUDA {ver[0]}.{ver[1]} driver)." if ver else f"{names[0]} ready."


def why_not_cuda() -> str:
    """Explanation when CTranslate2 reports no CUDA device; '' if there's simply no NVIDIA card."""
    kind, text = status()
    if kind == "none":
        return ""
    if kind == "ready":
        return "The NVIDIA driver is present but CUDA reported no usable device. Try updating the driver."
    return text


def explain_cuda_error(e: BaseException) -> str:
    msg = str(e)
    m = re.search(r"(cublas\w*|cudnn\w*|cudart\w*)\.dll", msg, re.I)
    if m or "Could not load library" in msg or "cannot load" in msg.lower():
        lib = m.group(0) if m else "a CUDA library"
        return (f"GPU libraries missing ({lib}). Open Settings → Model and click "
                f"'Install GPU support' ({INSTALL_SIZE_TEXT}).")
    if "out of memory" in msg.lower():
        return "The GPU ran out of video memory. Pick a smaller model or close other GPU apps."
    if "driver" in msg.lower() or "insufficient" in msg.lower():
        return "The NVIDIA driver is too old for CUDA 12. Update it from nvidia.com."
    return f"GPU error: {msg[:160]}"


def _wheel_info(pkg: str, version: str) -> dict:
    import httpx
    j = httpx.get(f"https://pypi.org/pypi/{pkg}/{version}/json", timeout=30, follow_redirects=True).json()
    for f in j["urls"]:
        if f["filename"].endswith("win_amd64.whl"):
            return {"url": f["url"], "size": f["size"], "sha256": f["digests"]["sha256"], "name": f["filename"]}
    raise RuntimeError(f"No Windows build of {pkg} {version} on PyPI")


def install(on_progress: Callable[[int, int, float], None],
            should_stop: Callable[[], bool] = lambda: False) -> None:
    """Download NVIDIA's cuBLAS + cuDNN wheels from PyPI and unpack their DLLs into cuda/."""
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
