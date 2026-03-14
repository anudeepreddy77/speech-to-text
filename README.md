---
title: Transcribe Studio
emoji: 🎙️
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Transcribe Studio

> **Open-source speech-to-text** powered by [OpenAI Whisper](https://github.com/openai/whisper) —  
> upload any audio or video file and get an accurate, timestamped, exportable transcript.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📁 Large files | Up to **1 GB** uploads supported |
| 🎬 Video & Audio | MP4, MOV, MKV, AVI, WebM · MP3, WAV, M4A, OGG, FLAC … |
| 🕐 Timestamps | Every segment is time-coded; click to seek the player |
| 🌍 Multilingual | 99 languages + auto-detection; optional translate → English |
| ⬇️ Export | **SRT** · **VTT** · **TXT** · **JSON** |
| 🔒 Private | All processing is local — no audio leaves your machine |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  Upload → Preview → Transcribe → Timestamped Transcript  │
└─────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   ffmpeg (audio          openai-whisper (speech → text)
    extraction)           ┌────────────────────────────┐
                          │  Models (choose one):       │
                          │  tiny   ~39 MB  ~32× speed  │
                          │  base   ~74 MB  ~16× speed  │
                          │  small  ~244 MB  ~6× speed  │
                          │  medium ~769 MB  ~2× speed  │
                          │  large  ~1.5 GB  ~1× speed  │
                          └────────────────────────────┘
```

**Key design decisions:**

- **Whisper** is used directly via the `openai-whisper` Python package — it runs entirely locally, no API key required.
- **ffmpeg** extracts and normalises audio from video files (to 16 kHz mono WAV) before passing it to Whisper.
- `@st.cache_resource` keeps the model in memory across reruns so it's only loaded once.
- Uploaded files are written to a `tempfile` on disk (not held in RAM) to support large files.
- `st.video(path, start_time=N)` enables the click-to-seek feature — clicking a timestamp reruns the app with the new `start_time`.

---

## 🚀 Local Development

### Prerequisites

```bash
# Python 3.9+
pip install openai-whisper streamlit torch torchaudio

# ffmpeg — required for video files
# Ubuntu / Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
# Windows — download from https://ffmpeg.org/download.html
```

### Run

```bash
git clone <your-repo-url>
cd transcribe-studio
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

---

## ☁️ Deployment

### Option A — Hugging Face Spaces (recommended · free)

1. Create an account at https://huggingface.co
2. Click **New Space** → SDK: **Streamlit** → Hardware: **CPU basic** (free)
3. Push this repository:

```bash
git remote add hf https://huggingface.co/spaces/<your-user>/<your-space>
git push hf main
```

Hugging Face reads `packages.txt` (system deps) and `requirements.txt` automatically.  
For faster transcription upgrade to **CPU upgrade** or **T4 GPU** (paid, ~$0.60/hr).

### Option B — Streamlit Community Cloud (free)

1. Push the repo to GitHub (public or private)
2. Go to https://share.streamlit.io → **New app**
3. Select your repo · branch · `app.py`
4. **Important:** Streamlit Community Cloud does not auto-install system packages.  
   Add a startup script or use the `packages.txt` mechanism if supported.
5. The free tier limits RAM to ~1 GB — use `tiny` or `base` model only.

### Option C — Railway / Render / Fly.io (free tiers)

All three support Docker. Use this `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🎛️ Model Guide

| Model | Size | Speed (CPU) | Best for |
|---|---|---|---|
| `tiny` | 39 MB | ~32× realtime | Quick demos, very short clips |
| `base` | 74 MB | ~16× realtime | Clear speech, short meetings |
| `small` | 244 MB | ~6× realtime | **Recommended starting point** |
| `medium` | 769 MB | ~2× realtime | Accented speech, interviews |
| `large` | 1.5 GB | ~1× realtime | Maximum accuracy, requires GPU |

> A 10-minute file takes ~3 min on CPU with `small`, ~30 sec with a T4 GPU.

---

## 📦 Tech Stack

| Layer | Tool | License |
|---|---|---|
| UI | Streamlit | Apache 2.0 |
| STT | OpenAI Whisper | MIT |
| Audio | ffmpeg | LGPL 2.1 |
| ML runtime | PyTorch | BSD |
| Deployment | Hugging Face Spaces | — |

---

## 📄 License

MIT — free to use, modify, and deploy.
