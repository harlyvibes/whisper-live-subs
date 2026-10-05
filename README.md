# Whisper Live Subs

Live captions for anything playing on your PC, powered by OpenAI's Whisper models
(via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), the same weights running on CTranslate2, about 4× faster).
It was built for Japanese streamers who switch between Japanese and English: the language
is detected **per phrase**, so a stream can go back and forth freely.

The app lives in the **system tray** (the blue 字 icon) and draws captions in an always-on-top
overlay that floats over every window, including browsers, players and borderless-fullscreen games.

- **Near word-by-word:** captions update about every 0.3 s while someone is talking, even on a CPU.
  Words two updates agree on are drawn solid and locked in; at most a few newer words are shown faded, and a
  word that is still being spoken is held back until it's finished (Whisper's guesses at half-heard words are
  where most "nonsense" came from). Prefer no flicker at all? Settings → Appearance → *Hide unconfirmed words*.
- **No repeat loops:** Whisper sometimes repeats a phrase over and over; loops are detected and redone or
  collapsed, while real repeats ("wait, wait", "let's go, let's go") are kept.
- **Self-correcting:** when a phrase ends it becomes a caption immediately, then a second, more careful pass
  re-listens to it in the background. If it hears something different, the caption already on screen is
  corrected in place (and in the caption history and transcript).

![Bilingual captions: Japanese and English lines, with the in-progress phrase faded](docs/overlay-bilingual.png)

<table>
<tr>
<td><img src="docs/overlay-translation.png" alt="Original + English translation mode with language tags"></td>
<td><img src="docs/overlay-outline-only.png" alt="Outline-only style with no background box"></td>
</tr>
<tr>
<td align="center"><em>"Original + English" mode with language tags</em></td>
<td align="center"><em>Restyled: no box, coloured outline, Meiryo</em></td>
</tr>
</table>

## Download (no Python needed)

Grab `WhisperLiveSubs-<version>-win64.zip` from [Releases](../../releases), extract it anywhere, and run
**`WhisperLiveSubs.exe`**. It's portable: models download into the `models` folder next to the exe on first use.
Windows SmartScreen may warn because the exe is unsigned: click *More info* → *Run anyway*.

With an NVIDIA card, open Settings → Model and click **Install GPU support** (one-time ~530 MB download of
NVIDIA's cuBLAS for CUDA 12 into a `cuda` folder next to the exe), then let the app restart.

## Run from source

Double-click **`Whisper Live Subs.bat`**. On a fresh machine it runs `setup.bat` first, which creates
`.venv` and installs the dependencies.

- **Move** the overlay: drag it. **Resize**: drag any edge or corner. Dragging the top/bottom edge sets a fixed
  height (captions stay at the bottom, as many lines as fit); tray → *Auto-fit overlay height* undoes it.
  **Font size**: Ctrl+mouse wheel.
- **Lock** it (tray → *Lock overlay*, or `Ctrl+Alt+L`) to make it click-through so it never gets in the way.
- **Double-click** the tray icon or overlay to open Settings. Right-click either one for the menu.

| Default hotkey | Action |
|---|---|
| `Ctrl+Alt+C` | Pause / resume captions |
| `Ctrl+Alt+H` | Show / hide the overlay |
| `Ctrl+Alt+L` | Lock / unlock the overlay |

## Choosing a model

| Model | Download | Notes |
|---|---|---|
| tiny / base | 75 / 145 MB | Fast but rough, especially for Japanese |
| **small** | 485 MB | Default. Best accuracy that keeps up live on a CPU |
| medium | 1.5 GB | Clearly better Japanese. Needs a fast CPU or any NVIDIA GPU |
| **large-v3-turbo** | 1.6 GB | Best choice with an NVIDIA GPU. Weak at *translation* |
| large-v2 / large-v3 | 3 GB | Most accurate. GPU strongly recommended |

Models download automatically on first use into `models/`, with live progress (%, MB, speed) on the overlay,
the tray tooltip and Settings → Model, where you can also pre-download them. The Model tab shows how much free
RAM each model needs on this PC. If a model can't load (e.g. not enough memory), the app says so and switches back
to the last model that worked.

CPU memory use is about the download size + 0.35 GB (e.g. medium ≈ 1.9 GB), so medium runs on a 4 GB PC. *Custom* accepts any CTranslate2 Whisper model, either a Hugging Face repo id
or a local folder.

### NVIDIA GPU

The speech engine (CTranslate2 4.8) has the CUDA runtime built in. The only extra NVIDIA file it needs is
**cuBLAS for CUDA 12** (`cublas64_12.dll`); it doesn't use cuDNN. Having "CUDA installed" isn't always enough:
**a CUDA 13 toolkit ships `cublas64_13.dll`, which this engine can't use.**

- Settings → Model shows a **GPU** line explaining what it found: no NVIDIA card/driver, a driver too old for
  CUDA 12, cuBLAS 12 missing (with an **Install GPU support** button), or ready (and where cuBLAS was found).
- The app looks for cuBLAS 12 in its own `cuda` folder, pip's `nvidia-cublas-cu12`, any CUDA 12 toolkit
  (`CUDA_PATH_V12_*`) and `PATH`, and loads it before the engine starts, so no PATH editing is needed.
- The precision is chosen from what the card supports (e.g. GTX 10-series can't use float16).
- If the GPU still isn't used, click **Copy GPU report** and include it in an issue: it lists the driver, GPU,
  compute capability, where cuBLAS was found and the exact error.

From source, the setup installs cuBLAS automatically when it finds an NVIDIA GPU. By hand:

```bash
.venv\Scripts\python -m pip install -r requirements-gpu.txt
```

A virtual machine (VirtualBox, etc.) can't use the host's graphics card, so run the app on the host to use the GPU.

## Language options (Settings → Language)

- **Spoken language**: *Auto-detect each phrase* (bilingual), *Japanese only*, or *English only*.
  Auto-detect only considers the **allowed languages** (`ja, en`), so short Japanese clips are never
  mistaken for Chinese. It also slightly favours the language the streamer was just speaking, which
  keeps one-word replies like "うん" or "yeah" stable.
- **Captions show**: what was said; English (Japanese gets translated); or both, with the
  translation underneath in its own colour (added by the second pass, about a second later).
- **Correct captions after they appear**: the second pass described above. Turn it off to save CPU.
- **Vocabulary hint**: streamer, game and member names help Whisper spell them correctly.
- **Voice detection threshold / pause / max phrase length**: tune these for music-heavy streams.
- **Hallucination filter**: drops Whisper's classic junk on silence or music
  (e.g. 「ご視聴ありがとうございました」). The list is editable.

Other options include font, size, bold, text/Japanese/translation colours, outline, shadow,
background colour and opacity, box shape, lines shown, alignment, width, auto-clear delay, app theme
(follows Windows light/dark automatically, or force Light/Dark),
transcript saving, caption history window, launch at Windows sign-in, and the global hotkeys.

## Screenshots

| Tray menu | Appearance | Language |
|---|---|---|
| <img src="docs/tray-menu.png" width="220"> | <img src="docs/settings-appearance.png" width="300"> | <img src="docs/settings-language.png" width="300"> |

| Model | Audio | Behaviour |
|---|---|---|
| <img src="docs/settings-model.png" width="300"> | <img src="docs/settings-audio.png" width="300"> | <img src="docs/settings-behaviour.png" width="300"> |

### Updating the screenshots

Everything in `docs/` is rendered from the real widgets by a script, not captured from the screen. It uses
default settings, so the images never show your desktop, other windows or your saved preferences. It works
while the app is running. After changing the UI, regenerate them:

```bash
.venv\Scripts\python tools\make_screenshots.py
```

To add a new shot, add a call to `overlay_shot(...)` (captions on the demo backdrop) or a `widget.grab().save(...)`
in [tools/make_screenshots.py](tools/make_screenshots.py), then reference the file from this README.

## Tips

- To caption only the stream and not Discord or game audio, send the browser to a separate output
  device (Windows *App volume and device preferences*). Then pick that device in Settings → Audio.
- Settings live in `%APPDATA%\WhisperLiveSubs\settings.json`. The log for the windowless
  launcher is in `log.txt` in the same folder.
- Exclusive-fullscreen games draw over every overlay. Use borderless/windowed fullscreen.

## Files

```
Whisper Live Subs.bat   launcher (no console)
setup.bat               one-time dependency install
livesubs/engine.py      VAD + streaming captions, instant finals, second-pass corrections
livesubs/decoder.py     fast Whisper decoding (encoder window sized to the phrase)
livesubs/gpu.py         NVIDIA detection, cuBLAS preload/installer, GPU report
livesubs/audio.py       WASAPI loopback / microphone capture → 16 kHz mono
livesubs/overlay.py     caption window (outlined text, click-through, drag/resize)
livesubs/settings_dialog.py, app.py, hotkeys.py, config.py
tools/make_screenshots.py   regenerates docs/*.png
tools/build_exe.py          builds the standalone exe + zip (pip install pyinstaller first)
```
