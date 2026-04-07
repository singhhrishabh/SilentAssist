"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Streamlit Application                        ║
║  ──────────────────────────────────────────────────────────  ║
║  Two modes:                                                  ║
║    📹 Video Upload — upload .mp4, extract lips, match cmd    ║
║    📷 Live Camera  — real-time webcam lip reading via WebRTC ║
╚══════════════════════════════════════════════════════════════╝

Launch:
    streamlit run app.py
"""

import os
import streamlit as st
import torch
import time
import numpy as np
import threading
from collections import deque
from PIL import Image

# ── WebRTC for live camera ───────────────────────────────────
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# ── Local modules ────────────────────────────────────────────
from processor import (
    preprocess_video,
    extract_lip_roi_from_frame,
    build_tensor_from_buffer,
    get_landmarker,
    MAX_FRAMES,
)
from model import load_model, get_device, ctc_greedy_decode, demo_inference
from decoder import decode_intent
from executor import execute_tool_call

# ══════════════════════════════════════════════════════════════
#  Page Configuration & Custom CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SilentAssist — Silent Voice Assistant",
    page_icon="🤫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject dark-themed CSS ───────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global overrides ──────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #0a0a0f 0%, #0d1117 40%, #101820 100%);
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(88, 166, 255, 0.12);
    }

    /* ── Cards ─────────────────────────────────────────────── */
    .glass-card {
        background: rgba(22, 27, 34, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.35);
    }

    /* ── Accent heading ────────────────────────────────────── */
    .accent-heading {
        background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f778ba 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }

    .sub-heading {
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 28px;
    }

    /* ── Status badges ─────────────────────────────────────── */
    .badge-success {
        display: inline-block;
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
        border-radius: 999px;
        padding: 6px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
    }

    .badge-fail {
        display: inline-block;
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
        border-radius: 999px;
        padding: 6px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
    }

    /* ── Command output box ────────────────────────────────── */
    .command-box {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08) 0%, rgba(163, 113, 247, 0.08) 100%);
        border: 1px solid rgba(88, 166, 255, 0.25);
        border-radius: 12px;
        padding: 22px 28px;
        margin-top: 12px;
    }

    .command-text {
        color: #e6edf3;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .confidence-text {
        color: #8b949e;
        font-size: 0.9rem;
        margin-top: 6px;
    }

    /* ── Metric cards ──────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-top: 16px;
    }
    .metric-card {
        flex: 1;
        background: rgba(22, 27, 34, 0.55);
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── Use-case cards (sidebar) ──────────────────────────── */
    .usecase-card {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .usecase-title {
        color: #e6edf3;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .usecase-desc {
        color: #8b949e;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    /* ── Live indicator ────────────────────────────────────── */
    @keyframes pulse-live {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #f85149;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-live 1.2s ease-in-out infinite;
    }

    /* ── Lip ROI mini card ─────────────────────────────────── */
    .roi-mini {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    /* ── Hide Streamlit branding ───────────────────────────── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
#  Sidebar — Project info & use-cases
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <span style="font-size:3rem;">🤫</span>
            <h2 style="color:#e6edf3; margin:8px 0 2px 0; font-weight:800;
                        letter-spacing:-0.02em;">SilentAssist</h2>
            <p style="color:#8b949e; font-size:0.88rem; margin:0;">
                Visual Speech Recognition<br>Silent Voice Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("##### ⚙️ Configuration")
    
    use_weights = st.toggle("🤖 Pre-trained Weights", value=True, help="Auto-downloads LipNet weights from Hugging Face if no custom file provided.")
    weights_file = None
    if use_weights:
        weights_file = st.file_uploader(
            "Custom .pt / .pth checkpoint (Optional)",
            type=["pt", "pth"],
        )
    
    st.markdown("---")

    st.markdown("##### 🎯 Hardware")
    # Show active device
    device_info = str(get_device()).upper()
    if device_info == "MPS":
        device_display = "🍏 Apple Silicon (MPS)"
    elif device_info == "CUDA":
        device_display = "🟢 NVIDIA (CUDA)"
    else:
        device_display = "💻 CPU"
        
    st.markdown(f"<div style='color: #8b949e; font-size: 0.9rem;'>Active Device: <b>{device_display}</b></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#484f58; font-size:0.75rem; padding:8px;">
            Built with ❤️ for the 24-hr Hackathon<br>
            PyTorch · MediaPipe · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  Shared: Render results
# ══════════════════════════════════════════════════════════════
def render_results(result, raw_text, preprocess_time=0, inference_time=0, n_frames=0):
    """Render the tool execution results with metrics."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if result.tool_name:
        st.markdown(
            '<span class="badge-success">✓ INTENT PARSED</span>',
            unsafe_allow_html=True,
        )
        
        # Execute tool mapping OS integrations
        with st.spinner(f"Executing Agentic Tool: {result.tool_name}..."):
            exec_res = execute_tool_call(result.tool_name, result.tool_args)
            
        exec_status = "✅ " if exec_res["status"] == "success" else "❌ "
        st.markdown(
            f"""
            <div class="command-box">
                <div class="command-text">
                    {exec_status} {exec_res['message']}
                </div>
                <div class="confidence-text">
                    Tool Executed: {result.tool_name}({result.tool_args})&nbsp;&nbsp;|&nbsp;&nbsp;
                    Solver: 🧠 Agentic LLM&nbsp;&nbsp;|&nbsp;&nbsp;
                    Raw VSR Out: "{raw_text}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result.reasoning:
            st.info(f"**Agent Reasoning:** {result.reasoning}")

    else:
        st.markdown(
            '<span class="badge-fail">✗ NOT RECOGNISED</span>',
            unsafe_allow_html=True,
        )
        st.error(f"Reason: {result.reasoning}")

    # Metrics row
    total_time = preprocess_time + inference_time
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{n_frames}</div>
                <div class="metric-label">Frames Processed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{preprocess_time:.2f}s</div>
                <div class="metric-label">Preprocessing</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{inference_time:.2f}s</div>
                <div class="metric-label">Inference</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_time:.2f}s</div>
                <div class="metric-label">Total Pipeline</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Main Content — Header
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="accent-heading">SilentAssist</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-heading">'
    "Read lips, execute commands natively — no audio needed. "
    "Choose a mode below to get started."
    "</p>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
#  Mode Tabs
# ══════════════════════════════════════════════════════════════
tab_upload, tab_live = st.tabs(["📹  Video Upload", "📷  Live Camera"])

# Helper for inference
@st.cache_resource(show_spinner=False)
def get_cached_model(weights_path=None, auto_download=True):
    device = get_device()
    return load_model(weights_path=weights_path, device=device, auto_download=auto_download), device

# ──────────────────────────────────────────────────────────────
#  TAB 1: Video Upload Mode
# ──────────────────────────────────────────────────────────────
with tab_upload:
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm", "mpeg4"],
        label_visibility="collapsed",
        key="video_uploader",
    )

    if uploaded_video is not None:
        # Show the uploaded video
        col_vid, col_info = st.columns([2, 1])
        with col_vid:
            st.video(uploaded_video)
        with col_info:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="color:#e6edf3; font-weight:600; margin-bottom:8px;">
                        📹 Video Info
                    </div>
                    <div style="color:#8b949e; font-size:0.88rem;">
                        <b>File:</b> {uploaded_video.name}<br>
                        <b>Size:</b> {uploaded_video.size / 1024:.1f} KB
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Run pipeline ─────────────────────────────────────
        if st.button("🚀  Analyse Lip Movements", use_container_width=True, type="primary"):

            # ── Step 1: Preprocess ───────────────────────────
            with st.status("🔬 Extracting lip kinematics…", expanded=True) as status:
                st.write("Detecting face landmarks with MediaPipe Face Mesh…")
                t0 = time.time()

                try:
                    uploaded_video.seek(0)
                    tensor, roi_frames = preprocess_video(uploaded_video)
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()

                preprocess_time = time.time() - t0
                st.write(f"✅ Extracted **{len(roi_frames)}** lip ROI frames in **{preprocess_time:.2f}s**")

                # Show lip ROI preview (first 10 frames)
                st.write("**Lip ROI Preview:**")
                preview_frames = roi_frames[: min(10, len(roi_frames))]
                cols = st.columns(len(preview_frames))
                for i, (col, frame) in enumerate(zip(cols, preview_frames)):
                    with col:
                        img = Image.fromarray(frame)
                        st.image(img, caption=f"F{i+1}", use_container_width=True)

                status.update(label="✅ Lip kinematics extracted", state="complete")

            # ── Step 2: Model Inference ──────────────────────
            with st.status("🧠 Running VSR inference…", expanded=True) as status:
                st.write("Loading model…")
                t1 = time.time()

                weights_path = None
                if use_weights and weights_file is not None:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
                    tmp.write(weights_file.read())
                    tmp.flush()
                    weights_path = tmp.name

                model, device = get_cached_model(weights_path=weights_path, auto_download=use_weights)

                with torch.no_grad():
                    # Check if model has valid weights loaded
                    is_demo = sum(p.sum().item() for p in model.parameters()) == sum(p.sum().item() for p in get_cached_model(auto_download=False)[0].parameters())
                    
                    if not is_demo or (weights_path is None and not use_weights):
                        # Move tensor to correct device
                        tensor = tensor.to(device)
                        log_probs = model(tensor)
                        raw_text = ctc_greedy_decode(log_probs)
                        st.write(f"Used **pre-trained weights** on **{device}**.")
                    else:
                        st.write("Using **Cloud Demo Agent** for parsing execution.")
                        raw_text = demo_inference(tensor)

                inference_time = time.time() - t1
                st.write(f"Raw VSR output: `{raw_text}`")

                if weights_path is not None:
                    os.unlink(weights_path)

                status.update(label="✅ Inference complete", state="complete")

            # ── Step 3: Intent Decoding & Agent Execution ───────────────
            with st.status("🎯 Mapping to tool logic…", expanded=True) as status:
                st.write("Prompting local Agentic LLM (Ollama) to parse garbled text intent...")
                result = decode_intent(raw_text)
                status.update(label="✅ Tool identified", state="complete")

            # ── Results ──────────────────────────────────────
            render_results(result, raw_text, preprocess_time, inference_time, len(roi_frames))

    else:
        # ── Empty state ──────────────────────────────────────
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:60px 32px;">
                <div style="font-size:4rem; margin-bottom:16px;">🎥</div>
                <div style="color:#e6edf3; font-size:1.2rem; font-weight:600;
                            margin-bottom:8px;">
                    Upload a video to get started
                </div>
                <div style="color:#8b949e; font-size:0.92rem; max-width:480px;
                            margin:0 auto;">
                    Record yourself silently mouthing a command, then upload
                    the clip. SilentAssist will extract lip movements, run
                    visual speech recognition, and trigger the native Mac OS tool.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🛠️ Available Agentic Tools")
        st.markdown("""
        The LLM Agent automatically maps lip-read inputs to:
        - `set_volume`, `increase_volume`, `decrease_volume`
        - `toggle_media` (play/pause Music/Spotify)
        - `lock_screen` (macOS native lock)
        - `open_application` (Launches Safari, Maps, etc.)
        - `emergency_protocol` (Launches Messages)
        """)


# ──────────────────────────────────────────────────────────────
#  TAB 2: Live Camera Mode
# ──────────────────────────────────────────────────────────────
with tab_live:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; margin-bottom:12px;">
                <span class="live-dot"></span>
                <span style="color:#e6edf3; font-weight:700; font-size:1.1rem;">
                    Live Lip Reading
                </span>
            </div>
            <div style="color:#8b949e; font-size:0.88rem; line-height:1.5;">
                Start the camera, silently mouth a command, then click
                <b>"Analyse Captured Frames"</b>. The system collects lip ROI
                frames in real-time. Upon analysis, the Agentic LLM translates the 
                garbled inference into a physical tool execution on your MacOS device.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Shared state for the WebRTC callback ─────────────────
    if "live_roi_buffer" not in st.session_state:
        st.session_state.live_roi_buffer = deque(maxlen=MAX_FRAMES)
    if "live_frame_count" not in st.session_state:
        st.session_state.live_frame_count = 0
    if "live_landmarker" not in st.session_state:
        st.session_state.live_landmarker = None
    if "live_lock" not in st.session_state:
        st.session_state.live_lock = threading.Lock()
    if "latest_roi_preview" not in st.session_state:
        st.session_state.latest_roi_preview = None

    # ── Video frame callback ─────────────────────────────────
    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Lazy-init the landmarker in the callback thread
        if st.session_state.live_landmarker is None:
            try:
                st.session_state.live_landmarker = get_landmarker()
            except Exception:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        landmarker = st.session_state.live_landmarker

        try:
            roi, annotated = extract_lip_roi_from_frame(img, landmarker)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if roi is not None:
            with st.session_state.live_lock:
                st.session_state.live_roi_buffer.append(roi)
                st.session_state.live_frame_count += 1
                st.session_state.latest_roi_preview = roi

        # Draw frame counter on annotated frame
        n_buffered = len(st.session_state.live_roi_buffer)
        cv2_text = f"Buffered: {n_buffered}/{MAX_FRAMES}"

        import cv2
        cv2.putText(
            annotated, cv2_text, (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 166, 255), 2,
        )

        # Show progress bar at top of frame
        bar_w = int((n_buffered / MAX_FRAMES) * annotated.shape[1])
        cv2.rectangle(annotated, (0, 0), (bar_w, 4), (63, 185, 80), -1)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # ── WebRTC streamer ──────────────────────────────────────
    col_cam, col_status = st.columns([2, 1])

    with col_cam:
        webrtc_ctx = webrtc_streamer(
            key="silent-assist-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_callback,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            },
            async_processing=True,
        )

    with col_status:
        st.markdown(
            """
            <div class="glass-card">
                <div style="color:#e6edf3; font-weight:600; margin-bottom:10px;">
                    📊 Live Status
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Show live stats
        n_buf = len(st.session_state.live_roi_buffer)
        progress_pct = min(100, int((n_buf / MAX_FRAMES) * 100))

        st.markdown(
            f"""
                <div style="color:#8b949e; font-size:0.85rem; margin-bottom:8px;">
                    <b>Frames buffered:</b> {n_buf} / {MAX_FRAMES}<br>
                    <b>Total processed:</b> {st.session_state.live_frame_count}
                </div>
                <div style="background:#21262d; border-radius:4px; height:8px;
                            overflow:hidden; margin-bottom:12px;">
                    <div style="width:{progress_pct}%; height:100%;
                                background: linear-gradient(90deg, #58a6ff, #3fb950);
                                border-radius:4px; transition: width 0.3s;"></div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Show latest ROI preview
        if st.session_state.latest_roi_preview is not None:
            st.markdown(
                '<div style="color:#8b949e; font-size:0.78rem; margin-bottom:4px;">'
                "Latest Lip ROI:</div>",
                unsafe_allow_html=True,
            )
            roi_img = Image.fromarray(st.session_state.latest_roi_preview)
            st.image(roi_img, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Clear buffer button ──────────────────────────────
        if st.button("🗑️  Clear Buffer", use_container_width=True):
            with st.session_state.live_lock:
                st.session_state.live_roi_buffer.clear()
                st.session_state.live_frame_count = 0
                st.session_state.latest_roi_preview = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyse captured frames ──────────────────────────────
    if st.button(
        "🧠  Analyse Captured Frames",
        use_container_width=True,
        type="primary",
        disabled=(len(st.session_state.live_roi_buffer) == 0),
    ):
        with st.session_state.live_lock:
            captured_rois = list(st.session_state.live_roi_buffer)

        n_captured = len(captured_rois)

        if n_captured == 0:
            st.warning("No lip frames captured yet. Start the camera and mouth a command!")
        else:
            # ── Step 1: Build tensor ─────────────────────────
            with st.status("🔬 Processing captured lip frames…", expanded=True) as status:
                t0 = time.time()
                tensor = build_tensor_from_buffer(captured_rois)
                preprocess_time = time.time() - t0
                st.write(f"✅ Built tensor from **{n_captured}** captured frames in **{preprocess_time:.2f}s**")

                # Show lip ROI preview (first 10 frames)
                st.write("**Captured Lip ROI Preview:**")
                preview = captured_rois[: min(10, n_captured)]
                cols = st.columns(len(preview))
                for i, (col, frame) in enumerate(zip(cols, preview)):
                    with col:
                        img = Image.fromarray(frame)
                        st.image(img, caption=f"F{i+1}", use_container_width=True)

                status.update(label="✅ Frames processed", state="complete")

            # ── Step 2: Inference ────────────────────────────
            with st.status("🧠 Running VSR inference…", expanded=True) as status:
                t1 = time.time()

                weights_path = None
                if use_weights and weights_file is not None:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
                    tmp.write(weights_file.read())
                    tmp.flush()
                    weights_path = tmp.name

                model, device = get_cached_model(weights_path=weights_path, auto_download=use_weights)

                with torch.no_grad():
                    is_demo = sum(p.sum().item() for p in model.parameters()) == sum(p.sum().item() for p in get_cached_model(auto_download=False)[0].parameters())
                    
                    if not is_demo or (weights_path is None and not use_weights):
                        # Move to correct device
                        tensor = tensor.to(device)
                        log_probs = model(tensor)
                        raw_text = ctc_greedy_decode(log_probs)
                        st.write(f"Used **pre-trained weights** on **{device}**.")
                    else:
                        st.write("Using **Cloud Demo Agent** for parsing execution.")
                        raw_text = demo_inference(tensor)

                inference_time = time.time() - t1
                st.write(f"Raw VSR output: `{raw_text}`")

                if weights_path is not None:
                    os.unlink(weights_path)

                status.update(label="✅ Inference complete", state="complete")

            # ── Step 3: Intent Parsing & Execution ──────────────────────────
            with st.status("🎯 Mapping into System Command…", expanded=True) as status:
                result = decode_intent(raw_text)
                status.update(label="✅ Intent parsed", state="complete")

            # ── Results ──────────────────────────────────────
            render_results(result, raw_text, preprocess_time, inference_time, n_captured)

    # ── Instructions ─────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card" style="margin-top:8px;">
            <div style="color:#e6edf3; font-weight:600; margin-bottom:10px;">
                📖 How to use Live Mode
            </div>
            <div style="color:#8b949e; font-size:0.88rem; line-height:1.7;">
                <b>1.</b> Click <b>"START"</b> to activate your webcam<br>
                <b>2.</b> Position your face so the <span style="color:#3fb950;">green</span>
                lip ROI box is visible<br>
                <b>3.</b> Silently mouth a command (e.g. "Turn down the volume" or "Lock the screen")<br>
                <b>4.</b> Click <b>"Analyse Captured Frames"</b> to run inference<br>
                <b>5.</b> The Agentic LLM translates the raw output and natively executes the tool!
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
