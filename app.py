"""
Transcribe Studio — Open-source speech-to-text powered by OpenAI Whisper.
Supports audio/video up to 1 GB with timestamped, clickable transcripts.
"""

import streamlit as st
import whisper
import tempfile
import os
import json
import subprocess
import shutil
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transcribe Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:         #0a0a0f;
    --surface:    #111118;
    --surface2:   #18181f;
    --surface3:   #1f1f28;
    --border:     #26262f;
    --border2:    #32323d;
    --accent:     #f0b429;
    --accent2:    #e8830a;
    --accent-dim: rgba(240,180,41,0.10);
    --accent-mid: rgba(240,180,41,0.20);
    --text:       #eeeef2;
    --text-dim:   #8888a0;
    --text-faint: #44445a;
    --success:    #34d399;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
    --radius: 10px;
}

html, body, [class*="css"] { font-family: var(--sans) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 2rem !important; }

/* ── Global background ── */
.stApp, .stApp > div { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Logo block ── */
.logo-block {
    display: flex; align-items: center; gap: 14px;
    padding: 8px 0 20px;
}
.logo-icon {
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
}
.logo-text { line-height: 1.2; }
.logo-title {
    font-weight: 700; font-size: 1.05rem; color: var(--text) !important;
}
.logo-sub {
    font-size: 0.72rem; color: var(--text-dim) !important;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Page header ── */
.page-header { margin-bottom: 6px; }
.page-title {
    font-family: var(--sans);
    font-size: 2rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.5px; line-height: 1.1;
}
.page-title em { font-style: normal; color: var(--accent); }
.page-sub { font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; margin-bottom: 28px; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 14px !important;
    padding: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] * { color: var(--text) !important; }
[data-testid="stFileUploader"] small { color: var(--text-dim) !important; }

/* ── Stat cards ── */
.stat-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 24px;
}
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 16px;
}
.stat-val {
    font-family: var(--mono); font-size: 1.3rem;
    font-weight: 700; color: var(--accent); letter-spacing: -0.5px;
}
.stat-lbl {
    font-size: 0.68rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}

/* ── Transcript container ── */
.transcript-wrap {
    max-height: 58vh; overflow-y: auto;
    padding-right: 4px; margin-top: 2px;
}
.transcript-wrap::-webkit-scrollbar { width: 3px; }
.transcript-wrap::-webkit-scrollbar-track { background: transparent; }
.transcript-wrap::-webkit-scrollbar-thumb {
    background: var(--border2); border-radius: 3px;
}

/* ── Segment row ── */
.seg-row {
    display: flex; gap: 12px; padding: 9px 12px;
    border-radius: 8px; margin-bottom: 3px;
    border: 1px solid transparent;
    transition: background 0.15s, border-color 0.15s;
}
.seg-row:hover {
    background: var(--surface2); border-color: var(--border);
}
.seg-row.active {
    background: var(--accent-dim); border-color: var(--accent);
}
.seg-time {
    font-family: var(--mono); font-size: 0.68rem;
    color: var(--accent); min-width: 52px;
    padding-top: 3px; flex-shrink: 0; cursor: pointer;
}
.seg-text { font-size: 0.88rem; color: var(--text); line-height: 1.55; }
.seg-row.active .seg-text { color: var(--text); }

/* ── Timestamp button override ── */
div[data-testid="column"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    min-height: 0 !important; height: auto !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
}

/* ── Primary action button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important; font-weight: 600 !important;
    border: none !important; border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
}

/* ── Select / radio ── */
.stSelectbox > div > div, .stSelectbox > label {
    color: var(--text) !important;
}
.stSelectbox > div > div > div {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
}
.stRadio > label, .stRadio div { color: var(--text) !important; }
[data-baseweb="select"] div { background: var(--surface2) !important; color: var(--text) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Info / warning ── */
.stAlert { background: var(--surface2) !important; border: 1px solid var(--border2) !important; }

/* ── Caption / label ── */
.stCaption, caption, small { color: var(--text-dim) !important; }
label { color: var(--text) !important; }
p, span, div { color: var(--text); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 14px 0 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Model badge ── */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-dim); border: 1px solid var(--accent-mid);
    border-radius: 6px; padding: 4px 10px; margin-top: 6px;
    font-family: var(--mono); font-size: 0.7rem; color: var(--accent);
}

/* ── File info card ── */
.file-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px; margin-bottom: 16px;
}
.file-card-name {
    font-family: var(--mono); font-size: 0.85rem;
    color: var(--text); word-break: break-all;
}
.file-card-meta {
    font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;
}

/* ── Export section ── */
.export-label {
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 20px; margin-bottom: 8px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 60px 20px;
    color: var(--text-dim); font-size: 0.9rem;
}
.empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }

/* ── Spinner text ── */
.stSpinner { color: var(--text-dim) !important; }

/* ── Video / audio player ── */
video, audio { border-radius: var(--radius) !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ── Utility Functions ─────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "Auto-detect": None,
    "English": "en", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Portuguese": "pt",
    "Russian": "ru", "Japanese": "ja", "Chinese": "zh",
    "Korean": "ko", "Arabic": "ar", "Hindi": "hi",
    "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
    "Swedish": "sv", "Norwegian": "no", "Danish": "da",
}

MODEL_INFO = {
    "tiny":   {"size": "~39 MB",  "speed": "~32× realtime", "note": "Fastest · basic accuracy"},
    "base":   {"size": "~74 MB",  "speed": "~16× realtime", "note": "Great for clear audio"},
    "small":  {"size": "~244 MB", "speed": "~6× realtime",  "note": "Recommended balance"},
    "medium": {"size": "~769 MB", "speed": "~2× realtime",  "note": "High accuracy · slower"},
    "large":  {"size": "~1.5 GB", "speed": "1× realtime",   "note": "Best accuracy · GPU recommended"},
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".ogv", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma", ".amr"}


def fmt_time(seconds: float, style: str = "display") -> str:
    """Convert float seconds to a formatted timestamp string."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if style == "srt":
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1)*1000):03d}"
    elif style == "vtt":
        return f"{h:02d}:{m:02d}:{int(s):02d}.{int((s % 1)*1000):03d}"
    else:  # display
        if h > 0:
            return f"{h}:{m:02d}:{int(s):02d}"
        return f"{m:02d}:{int(s):02d}"


def dur_human(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [
            str(i),
            f"{fmt_time(seg['start'], 'srt')} --> {fmt_time(seg['end'], 'srt')}",
            seg["text"].strip(), ""
        ]
    return "\n".join(lines)


def to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines += [
            f"{fmt_time(seg['start'], 'vtt')} --> {fmt_time(seg['end'], 'vtt')}",
            seg["text"].strip(), ""
        ]
    return "\n".join(lines)


def to_txt(segments) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)


def to_json(segments) -> str:
    out = [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
           for seg in segments]
    return json.dumps(out, indent=2, ensure_ascii=False)


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTS


def extract_audio(src: str, dst: str) -> tuple[bool, str]:
    """Use ffmpeg to extract & normalise audio to 16 kHz mono WAV."""
    cmd = [
        "ffmpeg", "-i", src,
        "-vn",                    # drop video stream
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16 kHz — Whisper's native rate
        "-ac", "1",               # mono
        "-y", dst,                # overwrite
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    """Load & cache a Whisper model (persists across reruns)."""
    return whisper.load_model(name)


def file_size_human(path: str) -> str:
    sz = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if sz < 1024:
            return f"{sz:.1f} {unit}"
        sz /= 1024
    return f"{sz:.1f} TB"


# ── Session State Defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "temp_path":   None,   # path to original uploaded file
    "audio_path":  None,   # path to extracted/converted audio for Whisper
    "segments":    None,   # list of dicts from Whisper result
    "seek_time":   0,      # seconds — current player position
    "is_video":    False,
    "file_name":   "",
    "file_size":   "",
    "duration":    0.0,
    "word_count":  0,
    "detected_lang": "",
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_state():
    for path_key in ("temp_path", "audio_path"):
        p = st.session_state.get(path_key)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-icon">🎙️</div>
        <div class="logo-text">
            <div class="logo-title">Transcribe Studio</div>
            <div class="logo-sub">Powered by Whisper</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Whisper model",
        list(MODEL_INFO.keys()),
        index=1,
        label_visibility="collapsed",
    )
    info = MODEL_INFO[model_choice]
    st.markdown(
        f'<div class="model-badge">⚡ {info["speed"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f'{info["size"]} · {info["note"]}')

    st.markdown("---")
    st.markdown('<div class="section-label">Language</div>', unsafe_allow_html=True)

    lang_choice = st.selectbox(
        "Language",
        list(LANGUAGE_MAP.keys()),
        index=0,
        label_visibility="collapsed",
        help="Set 'Auto-detect' to let Whisper identify the language automatically.",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Task</div>', unsafe_allow_html=True)

    task_choice = st.radio(
        "Task",
        ["transcribe", "translate"],
        format_func=lambda x: "Transcribe" if x == "transcribe" else "Translate → English",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Show reset only when a file is loaded
    if st.session_state.temp_path:
        if st.button("🗑️  Clear & start over", use_container_width=True):
            reset_state()
            st.rerun()

    st.markdown("---")
    st.caption("**System requirements**\n- ffmpeg (for video)\n- ~4 GB RAM for `medium`\n- GPU optional but speeds up larger models")


# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-title">Transcribe <em>Studio</em></div>
  <div class="page-sub">
    Upload any audio or video · Get an accurate, timestamped transcript ·
    Export as SRT, VTT, TXT or JSON
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 1 — Upload
# ══════════════════════════════════════════════════════════════════
if st.session_state.temp_path is None:

    accepted_types = (
        [e.lstrip(".") for e in sorted(AUDIO_EXTS)] +
        [e.lstrip(".") for e in sorted(VIDEO_EXTS)]
    )

    uploaded = st.file_uploader(
        "Drop your audio or video file here — up to 1 GB",
        type=accepted_types,
        help=(
            "**Audio:** MP3, WAV, M4A, OGG, FLAC, AAC, OPUS, WMA\n\n"
            "**Video:** MP4, MOV, AVI, MKV, WebM, FLV, WMV, M4V"
        ),
    )

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with st.spinner("Saving file to disk…"):
            shutil.copyfileobj(uploaded, tmp)
        tmp.close()

        st.session_state.temp_path = tmp.name
        st.session_state.is_video  = is_video(uploaded.name)
        st.session_state.file_name = uploaded.name
        st.session_state.file_size = file_size_human(tmp.name)
        st.rerun()

    # ── How it works ──
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("📁", "Upload", "Drop any audio or video file up to 1 GB"),
        ("🤖", "Transcribe", "Whisper AI converts speech to text locally"),
        ("🕐", "Timestamps", "Every segment is time-coded and clickable"),
        ("⬇️", "Export", "Download as SRT, VTT, plain text, or JSON"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);
                        border-radius:var(--radius);padding:18px 16px;text-align:center">
                <div style="font-size:1.6rem;margin-bottom:8px">{icon}</div>
                <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px">{title}</div>
                <div style="font-size:0.78rem;color:var(--text-dim)">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — File loaded, ready to transcribe
# ══════════════════════════════════════════════════════════════════
elif st.session_state.segments is None:

    col_preview, col_action = st.columns([3, 2], gap="large")

    with col_preview:
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        if st.session_state.is_video:
            st.video(st.session_state.temp_path)
        else:
            st.audio(st.session_state.temp_path)

    with col_action:
        st.markdown('<div class="section-label">File</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="file-card">
            <div class="file-card-name">📄 {st.session_state.file_name}</div>
            <div class="file-card-meta">
                {'🎬 Video' if st.session_state.is_video else '🎵 Audio'}
                · {st.session_state.file_size}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:var(--radius);padding:14px 16px;
                    font-size:0.82rem;color:var(--text-dim);margin-bottom:16px">
            <div>Model <span style="color:var(--accent);font-family:var(--mono)">{model_choice}</span>
                 &nbsp;·&nbsp; {MODEL_INFO[model_choice]['size']}</div>
            <div style="margin-top:4px">
                Language <span style="color:var(--accent)">{lang_choice}</span>
                &nbsp;·&nbsp;
                Task <span style="color:var(--accent)">{task_choice}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎙️  Start Transcription", use_container_width=True, type="primary"):

            progress = st.progress(0, text="Initialising…")

            # Step 1: Extract audio from video
            if st.session_state.is_video:
                progress.progress(15, text="Extracting audio from video…")
                audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_tmp.close()
                ok, err = extract_audio(st.session_state.temp_path, audio_tmp.name)
                if not ok:
                    st.error(f"**ffmpeg error** — could not extract audio.\n\n```\n{err}\n```")
                    st.info("Make sure ffmpeg is installed: `sudo apt install ffmpeg`")
                    st.stop()
                st.session_state.audio_path = audio_tmp.name
            else:
                st.session_state.audio_path = st.session_state.temp_path

            # Step 2: Load model
            progress.progress(30, text=f"Loading Whisper '{model_choice}' model…")
            model = load_model(model_choice)

            # Step 3: Transcribe
            lang_code = LANGUAGE_MAP[lang_choice]
            progress.progress(50, text="Transcribing… (may take a few minutes for long files)")

            result = model.transcribe(
                st.session_state.audio_path,
                language=lang_code,
                task=task_choice,
                verbose=False,
                word_timestamps=True,
            )

            progress.progress(95, text="Finalising results…")
            st.session_state.segments = result["segments"]
            st.session_state.detected_lang = result.get("language", "")
            if result["segments"]:
                st.session_state.duration = result["segments"][-1]["end"]
            st.session_state.word_count = sum(
                len(seg["text"].split()) for seg in result["segments"]
            )
            progress.progress(100, text="Done!")

            st.rerun()


# ══════════════════════════════════════════════════════════════════
# PHASE 3 — Show transcript
# ══════════════════════════════════════════════════════════════════
else:
    segments = st.session_state.segments
    stem     = Path(st.session_state.file_name).stem

    # ── Stats row ──────────────────────────────────────────────────
    detected = st.session_state.detected_lang.capitalize() if st.session_state.detected_lang else "—"
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">{len(segments)}</div>
            <div class="stat-lbl">Segments</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{st.session_state.word_count:,}</div>
            <div class="stat-lbl">Words</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{dur_human(st.session_state.duration)}</div>
            <div class="stat-lbl">Duration</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{detected}</div>
            <div class="stat-lbl">Detected language</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────────
    col_left, col_right = st.columns([5, 7], gap="large")

    # ── LEFT: Media player + export ───────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">Player</div>', unsafe_allow_html=True)
        seek = int(st.session_state.seek_time)

        if st.session_state.is_video:
            st.video(st.session_state.temp_path, start_time=seek)
        else:
            st.audio(st.session_state.temp_path, start_time=seek)

        if st.session_state.seek_time > 0:
            st.caption(f"⏱ Jumped to **{fmt_time(st.session_state.seek_time)}**")

        # ── Export ────────────────────────────────────────────────
        st.markdown('<div class="export-label">Export transcript</div>', unsafe_allow_html=True)

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.download_button(
                "SRT", to_srt(segments),
                file_name=f"{stem}.srt", mime="text/plain",
                use_container_width=True,
                help="SubRip subtitle format — compatible with most video players",
            )
        with ec2:
            st.download_button(
                "VTT", to_vtt(segments),
                file_name=f"{stem}.vtt", mime="text/vtt",
                use_container_width=True,
                help="WebVTT format — for web players and HTML5 video",
            )
        with ec3:
            st.download_button(
                "TXT", to_txt(segments),
                file_name=f"{stem}.txt", mime="text/plain",
                use_container_width=True,
                help="Plain text, no timestamps",
            )
        with ec4:
            st.download_button(
                "JSON", to_json(segments),
                file_name=f"{stem}.json", mime="application/json",
                use_container_width=True,
                help="Structured JSON with start/end times and text",
            )

    # ── RIGHT: Timestamped transcript ─────────────────────────────
    with col_right:
        st.markdown(
            '<div class="section-label">Transcript — click a timestamp to jump</div>',
            unsafe_allow_html=True,
        )

        # Open scroll container
        st.markdown('<div class="transcript-wrap">', unsafe_allow_html=True)

        for i, seg in enumerate(segments):
            t_start = seg["start"]
            t_end   = seg["end"]
            text    = seg["text"].strip()
            active  = (st.session_state.seek_time >= t_start and
                       st.session_state.seek_time < t_end)

            # Each segment: [timestamp button] [text]
            c_time, c_text = st.columns([1, 5], gap="small")

            with c_time:
                if st.button(fmt_time(t_start), key=f"ts_{i}"):
                    st.session_state.seek_time = t_start
                    st.rerun()

            with c_text:
                colour = "#f0b429" if active else "#eeeef2"
                st.markdown(
                    f'<div style="font-size:0.88rem;line-height:1.55;'
                    f'padding:4px 0;color:{colour}">{text}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
