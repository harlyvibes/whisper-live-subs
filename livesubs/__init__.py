"""Whisper Live Subs: live Japanese/English captions for anything playing on Windows."""
import os

# Must be set before huggingface_hub is imported.
# Plain HTTP downloads write to disk steadily, so the progress % is smooth (hf_xet
# writes in large bursts, which made big downloads look frozen) and speed is similar.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Must be set before ctranslate2 is imported. CTranslate2's Intel MKL backend reserves
# ~5x more memory than it uses (small: 2.4 GB vs 0.47 GB; medium: 4.5 GB vs 1 GB), which
# made medium fail to load on 4-8 GB PCs. The default backend is only ~6% slower on CPU.
# GPU (CUDA) is unaffected.
os.environ.setdefault("CT2_USE_MKL", "0")
