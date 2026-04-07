<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-00A67E?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-LLM-white?logo=ollama&logoColor=black" />
</p>

<h1 align="center">🤫 SilentAssist</h1>
<p align="center"><strong>Visual Speech Recognition — Silent Voice Assistant</strong></p>
<p align="center">
  Read lips, not voices. Execute commands silently via camera — zero audio data.
</p>

---

## 🎯 What Is SilentAssist?

**SilentAssist** is a Visual Speech Recognition (VSR) assistant that reads a user's lip movements via camera to execute commands or send messages, **strictly bypassing all audio data**. 

It is designed for:
- ♿ **Accessibility** — Empowers individuals with speech impairments to control devices.
- 🔊 **Loud Environments** — Factory floors, concerts, and construction sites.
- 🔒 **Privacy & Stealth** — Hospitals, courtrooms, libraries.
- 🎖️ **Tactical** — Silent command execution for field operatives.

## ⚡ Key Upgrades (V2)

- **🤖 Hybrid Intent Parsing** — Pipes raw CTC text into a local LLM via `Ollama` (Llama 3.2 1B) for semantic intent extraction. Highly robust to homophene noise. Falls back to fuzzy matching (`thefuzz`) if offline.
- **⚡ Apple Silicon Optimization** — Uses `torch.backends.mps` to accelerate 3D-CNN and RNN inference on modern Mac hardware.
- **🤗 Hub Auto-Download** — Frictionless setup. Automatically fetches pre-trained LipNet weights off Hugging Face if no checkpoint is manually provided.
- **🎓 Complete Training Pipeline** — Full data locators and PyTorch `train.py` script featuring CTC loss, mixed-precision, and Word Error Rate (WER) / Character Error Rate (CER) tracking.
- **🐳 Docker Containerization** — Ships with `Dockerfile` and `docker-compose.yml` encapsulating all OpenCV system dependencies (libGL) alongside the app.

## 🏗️ Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Video Input │───▶│  processor.py   │───▶│    model.py     │───▶│  decoder.py   │
│  (or WebRTC) │    │ MediaPipe Face  │    │ 3D-CNN + Bi-GRU │    │  Ollama LLM   │
│   Webcam     │    │ → ROI Bounding  │    │ → CTC Logits    │    │ → Intent JSON │
└──────────────┘    └─────────────────┘    └─────────────────┘    └───────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │    dataset.py   │
                                           │    train.py     │
                                           └─────────────────┘
```

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/singhhrishabh/SilentAssist.git
cd SilentAssist

# Install dependencies
pip install -r requirements.txt

# (Optional) Start Ollama for Agentic LLM parsing
ollama serve &
ollama pull llama3.2:1b

# Launch the Streamlit app
streamlit run app.py
```

## 🐳 Quick Start (Docker Compose)

The easiest way to run SilentAssist without worrying about system dependencies. It will automatically orchestrate the Streamlit app and the Ollama server.

```bash
docker-compose up --build -d
```
The app will be available at `http://localhost:8501`.

## 🎓 Training Pipeline

To train your own model or fine-tune on the **GRID corpus**:
1. Download a dataset or structure custom videos into a folder.
2. Run the `train.py` script:

```bash
# Example: Train on standard GRID dataset
python train.py --data_root /path/to/grid --dataset_type grid --epochs 50 --batch_size 8

# Example: Train on custom folder of videos + labels
python train.py --data_root /path/to/custom --dataset_type folder --labels_file labels.txt
```
Checkpoints and logs (including CER/WER metrics) will be saved in the `./checkpoints/` directory.

## 🔧 Model Configuration

SilentAssist checks `torch.backends.mps.is_available()` on macOS, defaulting to GPU acceleration if permitted, else CUDA or CPU.

### Using Default Weights
We host a baseline checkpoint trained loosely on the GRID dataset. If the *Pre-trained Weights* toggle is on and you didn't upload a custom file, passing `auto_download=True` automatically downloads the checkpoint from `singhhrishabh/silentassist-lipnet-grid` to a local `.weights_cache/` directory.

### Demo Mode
If the weights toggle is disabled, SilentAssist falls back to a deterministic **Demo Stub** for purely showcasing the system UI end-to-end.

## 📄 License & Credits

MIT License — Built over a 24-hr open-environment hackathon runtime.
