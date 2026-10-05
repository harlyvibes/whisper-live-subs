"""System-wide hotkeys via Win32 RegisterHotKey (work while a game/browser has focus)."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading

from PySide6.QtCore import QObject, Signal

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

MODS = {"ctrl": 0x2, "control": 0x2, "alt": 0x1, "shift": 0x4, "meta": 0x8, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
NAMED_KEYS = {
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08, "insert": 0x2D, "ins": 0x2D, "del": 0x2E, "delete": 0x2E,
    "home": 0x24, "end": 0x23, "pgup": 0x21, "pageup": 0x21, "pgdown": 0x22, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "pause": 0x13,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, "\\": 0xDC,
}


def parse_hotkey(text: str) -> tuple[int, int] | None:
    """'Ctrl+Alt+C' -> (modifiers, virtual key). None if empty/invalid."""
    text = text.strip()
    if not text:
        return None
    parts = [p.strip().lower() for p in text.replace("++", "+plus").split("+") if p.strip()]
    mods, vk = 0, None
    for p in parts:
        if p in MODS:
            mods |= MODS[p]
        elif len(p) == 1 and p.isalnum():
            vk = ord(p.upper())
        elif p.startswith("f") and p[1:].isdigit() and 1 <= int(p[1:]) <= 24:
            vk = 0x70 + int(p[1:]) - 1
        elif p.startswith("num") and p[3:].isdigit():
            vk = 0x60 + int(p[3:])
        elif p in NAMED_KEYS:
            vk = NAMED_KEYS[p]
        elif p == "plus":
            vk = 0xBB
        else:
            return None
    return (mods, vk) if vk is not None else None


class GlobalHotkeys(QObject):
    triggered = Signal(str)       # action name
    failed = Signal(str)          # hotkey text that could not be registered

    def __init__(self):
        super().__init__()
        self._thread: threading.Thread | None = None
        self._thread_id = 0

    def set_hotkeys(self, mapping: dict[str, str]) -> None:
        self.stop()
        parsed = {}
        for action, text in mapping.items():
            hk = parse_hotkey(text)
            if hk:
                parsed[action] = (text, hk)
        if not parsed:
            return
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(parsed, ready), daemon=True,
                                        name="GlobalHotkeys")
        self._thread.start()
        ready.wait(2)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2)
        self._thread = None

    def _run(self, parsed: dict, ready: threading.Event) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        msg = wt.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # create the message queue
        ids = {}
        for i, (action, (text, (mods, vk))) in enumerate(parsed.items(), start=1):
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                ids[i] = action
            else:
                self.failed.emit(text)
        ready.set()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in ids:
                self.triggered.emit(ids[msg.wParam])
        for i in ids:
            user32.UnregisterHotKey(None, i)
