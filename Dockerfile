# ══════════════════════════════════════════════════════════════
#  SilentAssist — Docker Container
# ══════════════════════════════════════════════════════════════
#  Multi-stage build:
#    Stage 1 — System deps (libGL for OpenCV, build tools)
#    Stage 2 — Python dependencies
#    Stage 3 — Application code
# ══════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# ── System dependencies ──────────────────────────────────────
#    libGL1 + libglib2 are required by opencv-python-headless
#    curl is needed for downloading the MediaPipe model
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create app directory ─────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download MediaPipe face model ────────────────────────────
RUN curl -L -o face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# ── Copy application code ────────────────────────────────────
COPY app.py .
COPY model.py .
COPY processor.py .
COPY decoder.py .
COPY dataset.py .
COPY train.py .

# ── Expose Streamlit port ────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Streamlit configuration ──────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# ── Launch ───────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]
