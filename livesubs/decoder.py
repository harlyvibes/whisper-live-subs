"""Low-latency Whisper decoding on top of faster-whisper's CTranslate2 model.

faster-whisper always pads audio to a 30 s window before running the encoder,
so a 2 s phrase costs as much as 30 s of audio. Here the encoder runs on a
window sized to the phrase (min 5 s), which measured 3-4x faster on CPU with
identical output, and one encoder pass is shared by language detection,
transcription and translation. If a short window ever produces a repetition
loop, the pass is retried on the full 30 s window.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import ctranslate2
import numpy as np
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens

SR = 16000
MIN_WINDOW_S = 5      # shorter windows made the decoder repeat itself on ~1 s clips
FULL_WINDOW_S = 30
NO_SPACE_LANGS = {"ja", "zh", "yue", "th", "lo", "my", "km"}


def norm(text: str) -> str:
    """Lower-case text without punctuation/spaces, for comparisons and the blocklist."""
    return "".join(ch for ch in text.lower() if unicodedata.category(ch)[0] not in "PSZC")


@dataclass
class Result:
    text: str = ""
    lang: str = ""
    segments: list[tuple[float, float, str]] = field(default_factory=list)  # (start, end, text)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression: float = 1.0

    @property
    def looks_bad(self) -> bool:
        """Repetition loop or very unsure decode: worth retrying."""
        return self.compression > 2.4 or (self.avg_logprob < -1.0 and self.no_speech_prob < 0.6)


class FastDecoder:
    def __init__(self, model):
        self.model = model                  # faster_whisper.WhisperModel
        self.ct2 = model.model              # ctranslate2.models.Whisper
        self.fe = model.feature_extractor
        self._tokenizers: dict[tuple[str, str], Tokenizer] = {}

    def tokenizer(self, task: str, lang: str) -> Tokenizer:
        key = (task, lang)
        if key not in self._tokenizers:
            self._tokenizers[key] = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual,
                                              task=task, language=lang)
        return self._tokenizers[key]

    # ---- encoder ----
    def encode(self, audio: np.ndarray, full: bool = False):
        feats = self.fe(audio)                                   # (n_mels, frames), 100 frames/s
        secs = len(audio) / SR
        win = FULL_WINDOW_S if full else min(FULL_WINDOW_S, max(MIN_WINDOW_S, math.ceil(secs + 1.0)))
        frames = win * 100
        f = np.zeros((1, feats.shape[0], frames), dtype=np.float32)
        n = min(frames, feats.shape[1])
        f[0, :, :n] = feats[:, :n]
        return self.ct2.encode(ctranslate2.StorageView.from_array(f), to_cpu=False)

    # ---- language ----
    def detect(self, enc, allowed: list[str], prefer: str | None) -> str:
        probs = {tok[2:-2]: p for tok, p in self.ct2.detect_language(enc)[0]}
        scores = {lang: p for lang, p in probs.items() if not allowed or lang in allowed}
        if not scores:
            return allowed[0] if allowed else max(probs, key=probs.get)
        if prefer in scores:  # hysteresis: short "うん"/"yeah" stays in the current language
            scores[prefer] *= 1.3
        return max(scores, key=scores.get)

    # ---- decoder ----
    def decode(self, enc, secs: float, lang: str, task: str, *, beam: int = 1, prompt: str = "",
               timestamps: bool = False, temperature: float = 0.0) -> Result:
        tk = self.tokenizer(task, lang)
        previous = tk.encode(" " + prompt.strip())[-200:] if prompt.strip() else []
        tokens = self.model.get_prompt(tk, previous, without_timestamps=not timestamps)
        max_new = min(220, int(secs * 14) + 24)  # generous for Japanese, stops runaway loops early
        kw = dict(beam_size=max(1, beam), patience=1.0, length_penalty=1.0,
                  max_length=len(tokens) + max_new, return_scores=True, return_no_speech_prob=True,
                  suppress_blank=True, suppress_tokens=list(get_suppressed_tokens(tk, [-1])))
        if temperature > 0:
            kw.update(beam_size=1, sampling_temperature=temperature, sampling_topk=0)
        r = self.ct2.generate(enc, [tokens], **kw)[0]
        ids = r.sequences_ids[0]

        segments: list[tuple[float, float, str]] = []
        if timestamps:
            cur: list[int] = []
            start = 0.0
            for t in ids:
                if t >= tk.timestamp_begin:
                    ts = (t - tk.timestamp_begin) * 0.02
                    if cur:
                        segments.append((start, ts, tk.decode(cur).strip()))
                        cur = []
                    start = ts
                elif t < tk.eot:
                    cur.append(t)
            if cur:
                segments.append((start, secs, tk.decode(cur).strip()))
            segments = [s for s in segments if s[2]]
            sep = "" if (lang in NO_SPACE_LANGS and task == "transcribe") else " "
            text = sep.join(s[2] for s in segments)
        else:
            text = tk.decode([t for t in ids if t < tk.eot]).strip()
        return Result(text=text, lang=lang, segments=segments,
                      avg_logprob=r.scores[0] if r.scores else 0.0,
                      no_speech_prob=r.no_speech_prob,
                      compression=get_compression_ratio(text) if text else 1.0)

    def robust_decode(self, audio: np.ndarray, enc, lang: str, task: str, *, beam: int, prompt: str,
                      timestamps: bool = False, careful: bool = False) -> Result:
        """Decode; on a repetition loop/unsure result retry with the full window (and, if
        careful, with temperature fallback like Whisper's own transcribe())."""
        secs = len(audio) / SR
        res = self.decode(enc, secs, lang, task, beam=beam, prompt=prompt, timestamps=timestamps)
        if not res.looks_bad:
            return res
        full = self.encode(audio, full=True)
        res = self.decode(full, secs, lang, task, beam=beam, prompt=prompt, timestamps=timestamps)
        if careful:
            for t in (0.2, 0.4, 0.6):
                if not res.looks_bad:
                    break
                res = self.decode(full, secs, lang, task, prompt=prompt, timestamps=timestamps, temperature=t)
        return res
