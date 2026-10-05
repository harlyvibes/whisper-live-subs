"""Streaming caption engine.

Audio is gated with Silero VAD and buffered into phrases. While a phrase is being
spoken it is re-decoded every ~0.25 s of new audio ("partial" captions that grow
word by word; words that two consecutive passes agree on are marked stable, the
rest are shown faded). When the speaker pauses, the latest partial becomes the
"final" caption immediately, and a second, more careful pass (beam search plus
the previous caption as context) runs in the background; if it hears something
different, the caption already on screen is corrected in place.

Mixed Japanese/English: the language is detected *per phrase* (from the same
encoder pass as the transcription), restricted to the allowed languages, with
hysteresis so short ambiguous phrases stick with the current language.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal

from . import gpu, modelstore
from .audio import SAMPLE_RATE as SR
from .audio import AudioCapture
from .config import MODELS_DIR, Config, model_is_downloaded
from .decoder import NO_SPACE_LANGS, FastDecoder, Result, norm

VAD_WINDOW = 512                     # samples per Silero frame (32 ms)
PARTIAL_STEP_S = 0.25                # re-decode after this much new audio
MAX_PENDING_REFINES = 4


@dataclass
class _Refine:
    id: int
    audio: np.ndarray
    lang: str
    text: str
    translation: str
    translated: bool
    context: str
    queued_at: float


def _stable_prefix(old: str, new: str, lang: str) -> int:
    """Length of the prefix of `new` that `old` agrees with, cut at a word boundary."""
    n = 0
    for a, b in zip(old, new):
        if a != b:
            break
        n += 1
    if n == len(new):
        return n
    if lang not in NO_SPACE_LANGS:  # don't mark half a word as stable
        cut = new.rfind(" ", 0, n + 1)
        n = cut if cut > 0 else 0
    return n


class CaptionEngine(QObject):
    status = Signal(str)                          # human readable status
    state = Signal(str)                           # loading | listening | error | stopped
    partial = Signal(str, str, int)               # text, lang, stable_chars ("" text clears)
    final = Signal(int, str, str, str, bool)      # id, text, lang, translation, text_is_translated
    revised = Signal(int, str, str, str, bool)    # id, corrected text, lang, translation, translated
    settled = Signal(int, str, str, str, bool)    # id + text that will no longer change (transcripts)
    model_info = Signal(str)                      # e.g. "small · CPU int8"
    load_failed = Signal(str, bool)               # message, was_out_of_memory
    gpu_problem = Signal(str)                     # why the GPU isn't being used

    _ids = itertools.count(1)

    def __init__(self, cfg: Config, wait_for: threading.Thread | None = None):
        super().__init__()
        self.cfg = cfg.copy()
        self._wait_for = wait_for
        self._stop = threading.Event()
        self._paused = False
        self._prev_lang: str | None = None
        self._last_final_text = ""      # context for the next caption's second pass
        self._last_final_id = 0
        self.model = None
        self.dec: FastDecoder | None = None
        self.on_gpu = False
        self._refines: deque[_Refine] = deque()
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
            oom = modelstore.is_memory_error(e)
            if oom:
                total, free = modelstore.ram_gb()
                msg = (f"Not enough memory to load '{self.cfg.model_id()}' "
                       f"({free:.1f} of {total:.0f} GB RAM free). Close other apps or pick a smaller model.")
            else:
                msg = f"Could not load model '{self.cfg.model_id()}': {e}"
            self.status.emit(msg)
            self.load_failed.emit(msg, oom)
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
            while True:
                try:
                    self._loop(cap)
                    break
                except Exception as e:  # noqa: BLE001
                    if not modelstore.is_memory_error(e) or self._stop.is_set():
                        raise
                    # Out of memory mid-phrase: drop it and keep captioning.
                    traceback.print_exc()
                    self.partial.emit("", "", 0)
                    self.status.emit("Low on memory — skipped a phrase. Close other apps or pick a smaller model.")
                    time.sleep(1)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status.emit(f"Engine error: {e}")
            self.state.emit("error")
        finally:
            cap.stop()
            self._settle_all()
            self.model = self.dec = None
        if self._stop.is_set():
            self.state.emit("stopped")

    def _load_model(self, cfg: Config) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        model_id = cfg.model_id()
        if not model_id:
            raise ValueError("no custom model set (Settings → Model)")
        device = cfg.device
        n_cuda = ctranslate2.get_cuda_device_count()
        if device == "auto":
            device = "cuda" if n_cuda > 0 else "cpu"
            if n_cuda == 0:
                reason = gpu.why_not_cuda()
                if reason:
                    self.gpu_problem.emit(reason)
        elif device == "cuda" and n_cuda == 0:
            self.gpu_problem.emit(gpu.why_not_cuda() or "CUDA is not available.")
            device = "cpu"
        if device == "cuda":
            # CTranslate2 looks cublas64_12.dll up by name once per process; load the copy
            # we found (app's cuda folder, pip, CUDA 12 toolkit, PATH) by full path first.
            problem = gpu.preload()
            if problem:
                print("GPU preload:", problem, file=sys.stderr)
        ctype = cfg.compute_type
        if device == "cuda":
            ctype = gpu.best_compute_type(ctype)  # e.g. GTX 10-series can't use float16
        elif ctype == "auto" or "float16" in ctype:
            ctype = "int8"

        if not model_is_downloaded(model_id):
            self.status.emit(f"Downloading '{model_id}': starting… (first use only)")
            modelstore.download(
                model_id,
                lambda done, total, speed: self.status.emit(
                    modelstore.format_progress(model_id, done, total, speed)),
                should_stop=self._stop.is_set)
            if self._stop.is_set():
                return

        msg = f"Loading '{model_id}' on {'GPU' if device == 'cuda' else 'CPU'}…"
        need = modelstore.RAM_NEEDED_GB.get(model_id)
        if device == "cpu" and need:
            total, free = modelstore.ram_gb()
            if free < need:
                msg += f" (needs ~{need:.1f} GB RAM, {free:.1f} GB free — may be slow or fail)"
        self.status.emit(msg)

        def load(dev: str, ct: str):
            m = WhisperModel(model_id, device=dev, compute_type=ct,
                             cpu_threads=cfg.cpu_threads, download_root=str(MODELS_DIR))
            d = FastDecoder(m)
            # Warm up: forces CUDA/cuBLAS to load now so failures surface here.
            a = np.zeros(SR, dtype=np.float32)
            d.decode(d.encode(a), 1.0, "en", "transcribe")
            return m, d

        try:
            self.model, self.dec = load(device, ctype)
        except Exception as e:  # noqa: BLE001
            if device != "cuda":
                raise
            traceback.print_exc()
            gpu.record_error(e)
            self.gpu_problem.emit(gpu.explain_cuda_error(e))
            self.status.emit("GPU failed; using the CPU instead…")
            device, ctype = "cpu", "int8"
            self.model, self.dec = load(device, ctype)
        self.on_gpu = device == "cuda"
        name = gpu.device_names()[0] if self.on_gpu and gpu.device_names() else ""
        self.model_info.emit(f"{model_id} · {'GPU' if self.on_gpu else 'CPU'} {ctype}"
                             + (f" ({name})" if name else ""))

    def _speech_probs(self, audio: np.ndarray) -> np.ndarray:
        from faster_whisper.vad import get_vad_model
        pad = (-len(audio)) % VAD_WINDOW
        return get_vad_model()(np.pad(audio, (0, pad)))

    # ---- main loop ----
    def _loop(self, cap: AudioCapture) -> None:
        buf = np.zeros(0, dtype=np.float32)
        last_audio = time.monotonic()
        last_partial_len = 0
        utt_lang: str | None = None
        partial_text = ""          # latest partial shown for the current phrase
        partial_lang = ""

        def reset_phrase():
            nonlocal last_partial_len, utt_lang, partial_text, partial_lang
            last_partial_len, utt_lang, partial_text, partial_lang = 0, None, "", ""

        def clear_partial():
            if partial_text:
                self.partial.emit("", "", 0)
            reset_phrase()

        while not self._stop.is_set():
            new = cap.read(0.05)
            cfg = self.cfg
            now = time.monotonic()

            if self._paused:
                buf = buf[:0]
                clear_partial()
                self._settle_all()
                continue

            if new.size:
                if cfg.gain != 1.0:
                    new = np.clip(new * cfg.gain, -1.0, 1.0)
                buf = np.concatenate([buf, new])
                last_audio = now
            # WASAPI loopback delivers nothing while nothing is playing.
            stalled = (now - last_audio) * 1000 >= cfg.silence_ms
            if buf.size < SR * 0.3 and not (stalled and buf.size):
                self._maybe_refine(idle=True)
                continue

            probs = self._speech_probs(buf)
            speech = np.flatnonzero(probs >= cfg.vad_threshold)
            if speech.size == 0:
                clear_partial()
                buf = buf[-int(SR * 0.4):] if not stalled else buf[:0]
                self._maybe_refine(idle=True)
                continue

            # Drop leading non-speech (keep 200 ms of lead-in).
            start = max(0, speech[0] * VAD_WINDOW - int(SR * 0.2))
            if start > 0:
                buf = buf[start:]
                last_partial_len = max(0, last_partial_len - start)
            # A pause *inside* the buffer (new speech began while we were busy
            # decoding): commit the phrase before the pause on its own.
            gaps = np.flatnonzero(np.diff(speech) * VAD_WINDOW * 1000 / SR >= cfg.silence_ms)
            if gaps.size:
                g = gaps[0]
                end = min(len(buf), (speech[g] + 1) * VAD_WINDOW - start + int(SR * 0.15))
                nxt = max(end, speech[g + 1] * VAD_WINDOW - start - int(SR * 0.2))
                usable = partial_text if last_partial_len >= end - int(SR * 0.1) else ""
                self._finalize(buf[:end], cfg, partial=(usable, partial_lang))
                buf = buf[nxt:]
                reset_phrase()
                continue

            speech_end = min(len(buf), (speech[-1] + 1) * VAD_WINDOW - start + int(SR * 0.15))
            trailing_ms = (len(buf) - speech_end) * 1000 / SR

            if trailing_ms >= cfg.silence_ms - 150 or stalled:
                # End of phrase. If the last partial already covered all the speech,
                # promote it instantly; the background pass will polish it.
                usable = partial_text if last_partial_len >= speech_end - int(SR * 0.1) else ""
                self._finalize(buf[:speech_end], cfg, partial=(usable, partial_lang))
                buf = buf[speech_end:][-int(SR * 0.3):]
                reset_phrase()
            elif len(buf) >= cfg.max_phrase_s * SR:
                cut = self._finalize(buf, cfg, partial=("", ""), split=True)
                buf = buf[cut:]
                reset_phrase()
            elif (cfg.live_partials and len(buf) >= SR * 0.5
                  and len(buf) - last_partial_len >= SR * PARTIAL_STEP_S):
                text, lang = self._decode_partial(buf, cfg, utt_lang)
                utt_lang = utt_lang or lang
                last_partial_len = len(buf)
                if text and not self._stop.is_set() and not self._paused:
                    stable = _stable_prefix(partial_text, text, lang) if partial_lang == lang else 0
                    self.partial.emit(text, lang, stable)
                    partial_text, partial_lang = text, lang
                # On a GPU there's time to polish captions between partials.
                self._maybe_refine(idle=False)
            else:
                self._maybe_refine(idle=False)

    # ---- decoding steps ----
    def _lang_and_task(self, cfg: Config, lang: str) -> tuple[str, bool]:
        translated = cfg.output_mode == "translate" and lang != "en"
        return ("translate" if translated else "transcribe"), translated

    def _pick_lang(self, enc, cfg: Config, locked: str | None) -> str:
        if cfg.language_mode != "auto":
            return cfg.language_mode
        if locked:
            return locked
        return self.dec.detect(enc, cfg.allowed_list(), self._prev_lang)

    def _filter(self, res: Result, cfg: Config) -> str:
        """Drop silence/music hallucinations and repetition loops."""
        if not norm(res.text):
            return ""
        if res.no_speech_prob > 0.6 and res.avg_logprob < -0.7:
            return ""
        if res.compression > 2.6:
            return ""
        if cfg.hallucination_filter:
            block = {norm(b) for b in cfg.blocklist.splitlines() if b.strip()}
            if norm(res.text) in block:
                return ""
            if res.segments:
                kept = [s for s in res.segments if norm(s[2]) and norm(s[2]) not in block]
                if len(kept) != len(res.segments):
                    sep = "" if res.lang in NO_SPACE_LANGS else " "
                    return sep.join(s[2] for s in kept).strip()
        return res.text

    def _decode_partial(self, audio: np.ndarray, cfg: Config, locked: str | None) -> tuple[str, str]:
        enc = self.dec.encode(audio)
        lang = self._pick_lang(enc, cfg, locked)
        task, _ = self._lang_and_task(cfg, lang)
        res = self.dec.robust_decode(audio, enc, lang, task, beam=1, prompt=cfg.initial_prompt)
        return self._filter(res, cfg), lang

    def _finalize(self, audio: np.ndarray, cfg: Config, partial: tuple[str, str], split: bool = False) -> int:
        """Commit a phrase. Returns how many samples were consumed."""
        if len(audio) < SR * 0.25:
            self.partial.emit("", "", 0)
            return len(audio)
        text, lang = partial
        cut = len(audio)
        if not text or split:
            enc = self.dec.encode(audio)
            lang = self._pick_lang(enc, cfg, None)
            task, _ = self._lang_and_task(cfg, lang)
            res = self.dec.robust_decode(audio, enc, lang, task, beam=max(1, cfg.beam_size),
                                         prompt=cfg.initial_prompt, timestamps=split, careful=True)
            if split and len(res.segments) >= 2:
                last_start = int(res.segments[-1][0] * SR)
                if SR < last_start < len(audio):
                    cut = last_start
                    res.segments = res.segments[:-1]
                    sep = "" if (lang in NO_SPACE_LANGS and task == "transcribe") else " "
                    res.text = sep.join(s[2] for s in res.segments)
            text = self._filter(res, cfg)
        _, translated = self._lang_and_task(cfg, lang)
        trans = ""
        if text and cfg.output_mode == "both" and lang != "en" and not cfg.refine_captions:
            enc = self.dec.encode(audio[:cut])
            trans = self._filter(self.dec.robust_decode(audio[:cut], enc, lang, "translate", beam=1,
                                                        prompt=""), cfg)
        if self._stop.is_set() or self._paused:
            return cut
        if not text:
            self.partial.emit("", "", 0)
            return cut
        fid = next(self._ids)
        context = self._last_final_text if self._prev_lang == lang else ""
        self._prev_lang, self._last_final_text, self._last_final_id = lang, text, fid
        self.final.emit(fid, text, lang, trans, translated)
        job = _Refine(fid, audio[:cut].copy(), lang, text, trans, translated, context, time.monotonic())
        if cfg.refine_captions:
            self._refines.append(job)
            while len(self._refines) > MAX_PENDING_REFINES:  # falling behind: keep what's shown
                self._settle(self._refines.popleft())
        else:
            self._settle(job)
        return cut

    # ---- second pass: correct captions already on screen ----
    def _maybe_refine(self, idle: bool) -> None:
        if not self._refines or self.dec is None:
            return
        overdue = time.monotonic() - self._refines[0].queued_at > 4.0
        if idle or self.on_gpu or overdue:
            self._refine(self._refines.popleft())

    def _refine(self, job: _Refine) -> None:
        cfg = self.cfg
        audio = job.audio
        enc = self.dec.encode(audio)
        lang = job.lang
        if cfg.language_mode == "auto":  # longer, settled audio: re-check the language
            lang = self.dec.detect(enc, cfg.allowed_list(), job.lang)
        task, translated = self._lang_and_task(cfg, lang)
        prompt = " ".join(p for p in (cfg.initial_prompt.strip(), job.context if lang == job.lang else "") if p)
        beam = max(5, cfg.beam_size)
        res = self.dec.robust_decode(audio, enc, lang, task, beam=beam, prompt=prompt, careful=True)
        if job.context and res.looks_bad:  # context occasionally derails Whisper; retry without it
            res = self.dec.robust_decode(audio, enc, lang, task, beam=beam, prompt=cfg.initial_prompt,
                                         careful=True)
        text = self._filter(res, cfg) or job.text
        trans = job.translation
        if cfg.output_mode == "both" and lang != "en":
            trans = self._filter(self.dec.robust_decode(audio, enc, lang, "translate", beam=beam, prompt="",
                                                        careful=True), cfg) or trans
        elif cfg.output_mode != "both":
            trans = ""
        if self._stop.is_set():
            return
        if (norm(text), norm(trans), lang) != (norm(job.text), norm(job.translation), job.lang):
            self.revised.emit(job.id, text, lang, trans, translated)
        if job.id == self._last_final_id:  # next caption's context uses the corrected text
            self._last_final_text = text
        self.settled.emit(job.id, text, lang, trans, translated)

    def _settle(self, job: _Refine) -> None:
        self.settled.emit(job.id, job.text, job.lang, job.translation, job.translated)

    def _settle_all(self) -> None:
        while self._refines:
            self._settle(self._refines.popleft())
