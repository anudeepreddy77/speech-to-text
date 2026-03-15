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
    initial_sidebar_state="collapsed",   
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:         #FFFFFF;
    --surface:    ##C9BFBF;
    --surface2:   #a8a8eb;
    --surface3:   #1f1f28;
    --border:     #26262f;
    --border2:    #32323d;
    --accent:     #3b82f6;
    --accent2:    #2563eb;
    --accent-dim: rgba(59,130,246,0.10);
    --accent-mid: rgba(59,130,246,0.20);
    --text:       #111118;
    --text-dim:   #111118;
    --text-faint: #111118;
    --success:    #34d399;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
    --radius: 10px;
}

html, body, [class*="css"] { font-family: var(--sans) !important; }

/* ── Hide Streamlit chrome + completely hide sidebar ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }   /* ← Sidebar fully removed */
.block-container { padding-top: 2rem !important; }

/* ── Global background ── */
.stApp, .stApp > div { background: var(--bg) !important; }

/* ── Logo block (now in main) ── */
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

/* Rest of your original styling (unchanged) */
.page-header { margin-bottom: 6px; }
.page-title {
    font-family: var(--sans);
    font-size: 2rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.5px; line-height: 1.1;
}
.page-title em { font-style: normal; color: var(--accent); }
.page-sub { font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; margin-bottom: 28px; }

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

.section-label {
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}

.transcript-wrap {
    max-height: 58vh; overflow-y: auto;
    padding-right: 4px; margin-top: 2px;
}
.transcript-wrap::-webkit-scrollbar { width: 3px; }
.transcript-wrap::-webkit-scrollbar-track { background: transparent; }
.transcript-wrap::-webkit-scrollbar-thumb {
    background: var(--border2); border-radius: 3px;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important; font-weight: 600 !important;
    border: none !important; border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    width: 100%;
}

.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-dim); border: 1px solid var(--accent-mid);
    border-radius: 6px; padding: 4px 10px; margin-top: 6px;
    font-family: var(--mono); font-size: 0.7rem; color: var(--accent);
}

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

.export-label {
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 20px; margin-bottom: 8px;
}

video, audio { border-radius: var(--radius) !important; width: 100% !important; }
</style>

<script>
// Sidebar completely removed — controls are now next to upload
</script>
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
    "tiny": {
        "size": "~39 MB",
        "speed": "~32× realtime",
        "note": "Very fast but lower accuracy — best for quick drafts or testing"
    },
    "base": {
        "size": "~74 MB",
        "speed": "~16× realtime",
        "note": "Balanced speed and accuracy — good for most recordings"
    },
    "small": {
        "size": "~244 MB",
        "speed": "~6× realtime",
        "note": "Slower but more accurate — better for longer or complex audio"
    },
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".ogv", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma", ".amr"}


def fmt_time(seconds: float, style: str = "display") -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if style == "srt":
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1)*1000):03d}"
    elif style == "vtt":
        return f"{h:02d}:{m:02d}:{int(s):02d}.{int((s % 1)*1000):03d}"
    else:
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
    cmd = [
        "ffmpeg", "-i", src,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


@st.cache_resource(show_spinner=False)
def load_model(name: str):
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
    "temp_path":   None,
    "audio_path":  None,
    "segments":    None,
    "seek_time":   0,
    "is_video":    False,
    "file_name":   "",
    "file_size":   "",
    "duration":    0.0,
    "word_count":  0,
    "detected_lang": "",
    # ← New: persist settings without sidebar
    "model_choice": "base",
    "lang_choice": "Auto-detect",
    "task_choice": "transcribe",
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


# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.page-header{
    padding: 20px 20px;
}

.logo-block{
    display:flex;
    align-items:center;
    gap:20px;
}

.logo-icon{
    font-size:30px;
}

.logo-title{
    font-size:20px;
    font-weight:500;
}

.logo-title em{
    font-size:15px;
    font-style:normal;
    color:#888;
}

.logo-sub{
    font-size:15px;
    color:#666;
}

.page-sub{
    margin-top:15px;
    font-size:15px;
    color:#444;
}
</style>

<div class="page-header">
  <div class="logo-block">
      <div class="logo-icon">🎙️</div>
      <div class="logo-text">
          <div class="logo-title">Transcribe Studio <em> Developed by Anudeep Reddy</em></div>
          <div class="logo-sub">Powered by Whisper</div>
      </div>
  </div>
  <div class="page-sub">
    Upload any audio or video · Get an accurate, timestamped transcript ·
    Export as SRT, VTT, TXT or JSON
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 1 — Upload + Settings moved next to uploader
# ══════════════════════════════════════════════════════════════════
if st.session_state.temp_path is None:

    st.markdown('<div class="section-label">Transcription Settings</div>', unsafe_allow_html=True)

    set_col, up_col = st.columns([2, 3], gap="large")

    with set_col:
        # Model
        model_choice = st.selectbox(
            "Whisper model",
            list(MODEL_INFO.keys()),
            index=list(MODEL_INFO.keys()).index(st.session_state.model_choice),
            label_visibility="collapsed",
            key="model_select"
        )
        st.session_state.model_choice = model_choice
        info = MODEL_INFO[model_choice]
        st.markdown(
            f'<div class="model-badge">⚡ {info["speed"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f'{info["size"]} · {info["note"]}')

        # Language
        lang_choice = st.selectbox(
            "Language",
            list(LANGUAGE_MAP.keys()),
            index=list(LANGUAGE_MAP.keys()).index(st.session_state.lang_choice),
            label_visibility="collapsed",
            help="Set 'Auto-detect' to let Whisper identify the language automatically.",
            key="lang_select"
        )
        st.session_state.lang_choice = lang_choice

        # Task
        task_choice = st.radio(
            "Task",
            ["transcribe", "translate"],
            format_func=lambda x: "Transcribe" if x == "transcribe" else "Translate → English",
            index=0 if st.session_state.task_choice == "transcribe" else 1,
            label_visibility="collapsed",
            key="task_select"
        )
        st.session_state.task_choice = task_choice

        # st.caption("**System requirements**\n- ffmpeg (for video)\n- ~4 GB RAM for `medium`\n- GPU optional")

    with up_col:
        accepted_types = (
            [e.lstrip(".") for e in sorted(AUDIO_EXTS)] +
            [e.lstrip(".") for e in sorted(VIDEO_EXTS)]
        )

        uploaded = st.file_uploader(
            "Drop your audio or video file here — up to 200 MB",
            type=accepted_types,
            help=(
                "**Audio:** MP3, WAV, M4A, OGG, FLAC, AAC, OPUS, WMA\n\n"
                "**Video:** MP4, MOV, AVI, MKV, WebM, FLV, WMV, M4V"
            ),
        )

        st.markdown("""
            <div class="privacy-note">
            🔒 Privacy: Your files are processed locally using Whisper. Nothing is uploaded or permanently saved.
            </div>
            """, unsafe_allow_html=True)

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
        ("📁", "Upload", "Drop any audio or video file up to 200 MB"),
        ("🤖", "Transcribe", "Whisper AI converts Video/Audio to text locally"),
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

        st.markdown('<div class="section-label">Settings (used for transcription)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:var(--radius);padding:14px 16px;
                    font-size:0.82rem;color:var(--text-dim);margin-bottom:16px">
            <div>Model <span style="color:var(--accent);font-family:var(--mono)">{st.session_state.model_choice}</span>
                 &nbsp;·&nbsp; {MODEL_INFO[st.session_state.model_choice]['size']}</div>
            <div style="margin-top:4px">
                Language <span style="color:var(--accent)">{st.session_state.lang_choice}</span>
                &nbsp;·&nbsp;
                Task <span style="color:var(--accent)">{st.session_state.task_choice}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎙️  Start Transcription", use_container_width=True, type="primary"):

            progress = st.progress(0, text="Initialising…")

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

            progress.progress(30, text=f"Loading Whisper '{st.session_state.model_choice}' model…")
            model = load_model(st.session_state.model_choice)

            lang_code = LANGUAGE_MAP[st.session_state.lang_choice]
            progress.progress(50, text="Transcribing… (may take a few minutes for long files)")

            result = model.transcribe(
                st.session_state.audio_path,
                language=lang_code,
                task=st.session_state.task_choice,
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

        # Reset button (moved from old sidebar)
        st.markdown("---")
        if st.button("🗑️  Clear & start over", use_container_width=True):
            reset_state()
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
            )
        with ec2:
            st.download_button(
                "VTT", to_vtt(segments),
                file_name=f"{stem}.vtt", mime="text/vtt",
                use_container_width=True,
            )
        with ec3:
            st.download_button(
                "TXT", to_txt(segments),
                file_name=f"{stem}.txt", mime="text/plain",
                use_container_width=True,
            )
        with ec4:
            st.download_button(
                "JSON", to_json(segments),
                file_name=f"{stem}.json", mime="application/json",
                use_container_width=True,
            )

        # Reset button in final view too
        st.markdown("---")
        if st.button("🗑️  Clear & start over", use_container_width=True):
            reset_state()
            st.rerun()

    # ── RIGHT: Timestamped transcript ─────────────────────────────
    with col_right:
        st.markdown(
            '<div class="section-label">Transcript — click a timestamp to jump</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="transcript-wrap">', unsafe_allow_html=True)

        for i, seg in enumerate(segments):
            t_start = seg["start"]
            t_end   = seg["end"]
            text    = seg["text"].strip()
            active  = (st.session_state.seek_time >= t_start and
                       st.session_state.seek_time < t_end)

            c_time, c_text = st.columns([1, 5], gap="small")

            with c_time:
                if st.button(fmt_time(t_start), key=f"ts_{i}"):
                    st.session_state.seek_time = t_start
                    st.rerun()

            with c_text:
                colour = "#2563eb" if active else "#374151"
                st.markdown(
                    f'<div style="font-size:0.88rem;line-height:1.55;'
                    f'padding:4px 0;color:{colour}">{text}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)