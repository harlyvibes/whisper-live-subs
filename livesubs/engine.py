"""Streaming caption engine.

Audio is gated with Silero VAD and buffered into phrases. While a phrase is being
spoken the buffer is re-transcribed periodically ("partial" captions that update
live); when the speaker pauses (or the phrase gets too long) it is committed as a
"final" caption.

Mixed Japanese/English: the language is detected *per phrase*, restricted to the
allowed languages (so short Japanese clips are never misread as Chinese, etc.),
with a little hysteresis so ambiguous one-word phrases ("うん", "yeah") stick with
the language the streamer was just speaking.
"""
from __future__ import annotations

import threading
import time
import traceback
import unicodedata

import numpy as np
from PySide6.QtCore import QObject, Signal

from .audio import SAMPLE_RATE as SR
from .audio import AudioCapture
from .config import MODELS, MODELS_DIR, Config, model_is_downloaded

VAD_WINDOW = 512                     # samples per Silero frame (32 ms)
NO_SPACE_LANGS = {"ja", "zh", "yue", "th", "lo", "my", "km"}


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower()
                   if not unicodedata.category(ch)[0] in "PSZC")


class CaptionEngine(QObject):
    status = Signal(str)                 # human readable status
    state = Signal(str)                  # loading | listening | error | stopped
    partial = Signal(str, str)           # text, lang ("" text clears)
    final = Signal(str, str, str, bool)  # text, lang, translation, text_is_translated
    model_info = Signal(str)             # e.g. "small · CPU int8"

    def __init__(self, cfg: Config, wait_for: threading.Thread | None = None):
        super().__init__()
        self.cfg = cfg.copy()
        self._wait_for = wait_for
        self._stop = threading.Event()
        self._paused = False
        self._prev_lang: str | None = None
        self.model = None
        self.thread = threading.Thread(target=self._run, name="CaptionEngine", daemon=True)

    # ---- control (called from the GUI thread) ----
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def update_settings(self, cfg: Config) -> None:
        self.cfg = cfg.copy()  # atomic swap; read once per loop iteration

    # ---- worker thread ----
    def _run(self) -> None:
        if self._wait_for is not None:
            self._wait_for.join()  # let the previous engine release its model first
        if self._stop.is_set():
            return
        self.state.emit("loading")
        try:
            self._load_model(self.cfg)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status.emit(f"Could not load model: {e}")
            self.state.emit("error")
            return
        if self._stop.is_set():
            return

        cap = AudioCapture(self.cfg.audio_source, self.cfg.audio_device)
        try:
            cap.start()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status.emit(f"Could not open audio device: {e}")
            self.state.emit("error")
            return

        self.status.emit(f"Listening to: {cap.device_label}")
        self.state.emit("listening")
        try:
            self._loop(cap)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status.emit(f"Engine error: {e}")
            self.state.emit("error")
        finally:
            cap.stop()
            self.model = None
        if self._stop.is_set():
            self.state.emit("stopped")

    def _load_model(self, cfg: Config) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        model_id = cfg.model_id()
        if not model_id:
            raise ValueError("no custom model set (Settings → Model)")
        device = cfg.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        ctype = cfg.compute_type
        if ctype == "auto":
            ctype = "float16" if device == "cuda" else "int8"

        if not model_is_downloaded(model_id):
            size = next((m[2] for m in MODELS if m[0] == model_id), "")
            self.status.emit(f"Downloading model '{model_id}' {f'({size}) ' if size else ''}— first run only…")
        else:
            self.status.emit(f"Loading model '{model_id}' on {device.upper()}…")

        def load(dev: str, ct: str):
            m = WhisperModel(model_id, device=dev, compute_type=ct,
                             cpu_threads=cfg.cpu_threads, download_root=str(MODELS_DIR))
            # Warm up: forces CUDA/cuDNN libraries to load so failures surface now.
            segs, _ = m.transcribe(np.zeros(SR, dtype=np.float32), language="en", beam_size=1)
            list(segs)
            return m

        try:
            self.model = load(device, ctype)
        except Exception as e:  # noqa: BLE001
            if device != "cuda":
                raise
            traceback.print_exc()
            self.status.emit(f"GPU failed ({e.__class__.__name__}); falling back to CPU…")
            device, ctype = "cpu", "int8"
            self.model = load(device, ctype)
        self.model_info.emit(f"{model_id} · {device.upper()} {ctype}")

    def _speech_probs(self, audio: np.ndarray) -> np.ndarray:
        from faster_whisper.vad import get_vad_model
        pad = (-len(audio)) % VAD_WINDOW
        return get_vad_model()(np.pad(audio, (0, pad)))

    def _loop(self, cap: AudioCapture) -> None:
        buf = np.zeros(0, dtype=np.float32)
        last_audio = time.monotonic()
        last_partial_len = 0
        utt_lang: str | None = None
        showing_partial = False

        def clear_partial():
            nonlocal showing_partial
            if showing_partial:
                self.partial.emit("", "")
                showing_partial = False

        while not self._stop.is_set():
            new = cap.read(0.1)
            cfg = self.cfg
            now = time.monotonic()

            if self._paused:
                buf = buf[:0]
                utt_lang, last_partial_len = None, 0
                clear_partial()
                continue

            if new.size:
                if cfg.gain != 1.0:
                    new = np.clip(new * cfg.gain, -1.0, 1.0)
                buf = np.concatenate([buf, new])
                last_audio = now
            # WASAPI loopback delivers nothing while nothing is playing.
            stalled = (now - last_audio) * 1000 >= cfg.silence_ms
            if buf.size < SR * 0.3 and not (stalled and buf.size):
                continue

            probs = self._speech_probs(buf)
            speech = np.flatnonzero(probs >= cfg.vad_threshold)
            if speech.size == 0:
                clear_partial()
                buf = buf[-int(SR * 0.4):] if not stalled else buf[:0]
                utt_lang, last_partial_len = None, 0
                continue

            # Drop leading non-speech (keep 200 ms of lead-in).
            start = max(0, speech[0] * VAD_WINDOW - int(SR * 0.2))
            if start > 0:
                buf = buf[start:]
                last_partial_len = max(0, last_partial_len - start)
            # A pause *inside* the buffer (new speech began while we were busy
            # transcribing): commit the phrase before the pause on its own.
            gaps = np.flatnonzero(np.diff(speech) * VAD_WINDOW * 1000 / SR >= cfg.silence_ms)
            if gaps.size:
                g = gaps[0]
                end = min(len(buf), (speech[g] + 1) * VAD_WINDOW - start + int(SR * 0.15))
                nxt = max(end, speech[g + 1] * VAD_WINDOW - start - int(SR * 0.2))
                self._finalize(buf[:end], cfg, split=False)
                showing_partial = False
                buf = buf[nxt:]
                utt_lang, last_partial_len = None, 0
                continue

            speech_end = min(len(buf), (speech[-1] + 1) * VAD_WINDOW - start + int(SR * 0.15))
            trailing_ms = (len(buf) - speech_end) * 1000 / SR

            if trailing_ms >= cfg.silence_ms - 150 or stalled:
                self._finalize(buf[:speech_end], cfg, split=False)
                showing_partial = False
                buf = buf[speech_end:][-int(SR * 0.3):]
                utt_lang, last_partial_len = None, 0
            elif len(buf) >= cfg.max_phrase_s * SR:
                cut = self._finalize(buf, cfg, split=True)
                showing_partial = False
                buf = buf[cut:]
                last_partial_len = 0
            elif (cfg.live_partials and len(buf) >= SR * 1.0
                  and len(buf) - last_partial_len >= SR * 0.5):
                if utt_lang is None:
                    utt_lang = self._pick_language(buf, cfg)
                task = "translate" if cfg.output_mode == "translate" and utt_lang != "en" else "transcribe"
                segs = self._whisper(buf, utt_lang, task, cfg, final=False)
                text = self._join(segs, "en" if task == "translate" else utt_lang)
                last_partial_len = len(buf)
                if text and not self._stop.is_set() and not self._paused:
                    self.partial.emit(text, utt_lang)
                    showing_partial = True

    # ---- whisper helpers ----
    def _pick_language(self, audio: np.ndarray, cfg: Config) -> str:
        mode = cfg.language_mode
        if mode != "auto":
            return mode
        _, _, probs = self.model.detect_language(audio=audio)
        allowed = cfg.allowed_list()
        scores = {lang: p for lang, p in probs if not allowed or lang in allowed}
        if not scores:
            return allowed[0] if allowed else probs[0][0]
        if self._prev_lang in scores:  # hysteresis for short, ambiguous phrases
            scores[self._prev_lang] *= 1.3
        return max(scores, key=scores.get)

    def _whisper(self, audio, lang, task, cfg: Config, final: bool):
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            task=task,
            beam_size=max(1, cfg.beam_size) if final else 1,
            best_of=max(1, cfg.beam_size) if final else 1,
            temperature=[0.0, 0.2, 0.4, 0.6] if final else 0.0,
            condition_on_previous_text=False,
            initial_prompt=cfg.initial_prompt.strip() or None,
            vad_filter=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )
        block = {_norm(b) for b in cfg.blocklist.splitlines() if b.strip()} if cfg.hallucination_filter else set()
        kept = []
        for s in segments:
            t = s.text.strip()
            n = _norm(t)
            if not n:
                continue
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.7:
                continue
            if s.compression_ratio > 2.6:  # repetition loop
                continue
            if n in block:
                continue
            kept.append(s)
        return kept

    @staticmethod
    def _join(segs, lang: str) -> str:
        sep = "" if lang in NO_SPACE_LANGS else " "
        return sep.join(s.text.strip() for s in segs).strip()

    def _finalize(self, audio: np.ndarray, cfg: Config, split: bool) -> int:
        """Commit a phrase. Returns how many samples were consumed."""
        if len(audio) < SR * 0.25:
            self.partial.emit("", "")
            return len(audio)
        lang = self._pick_language(audio, cfg)
        translated = cfg.output_mode == "translate" and lang != "en"
        segs = self._whisper(audio, lang, "translate" if translated else "transcribe", cfg, final=True)
        cut = len(audio)
        if split and len(segs) >= 2:
            last_start = int(segs[-1].start * SR)
            if SR < last_start < len(audio):
                cut, segs = last_start, segs[:-1]
        text = self._join(segs, "en" if translated else lang)
        trans = ""
        if text and cfg.output_mode == "both" and lang != "en":
            trans = self._join(self._whisper(audio[:cut], lang, "translate", cfg, final=True), "en")
        if self._stop.is_set() or self._paused:
            return cut
        if text:
            self._prev_lang = lang
            self.final.emit(text, lang, trans, translated)
        else:
            self.partial.emit("", "")
        return cut
