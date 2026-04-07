<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-00A67E?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<h1 align="center">🤫 SilentAssist</h1>
<p align="center"><strong>Visual Speech Recognition — Silent Voice Assistant</strong></p>
<p align="center">
  Read lips, not voices. Execute commands silently via camera — zero audio data.
</p>

---

## 🎯 What Is SilentAssist?

**SilentAssist** is a Visual Speech Recognition (VSR) assistant that reads a user's lip movements via camera to execute commands or send messages, **strictly bypassing all audio data**. It is designed for:

| Use Case | Description |
|----------|-------------|
| ♿ **Accessibility** | Empowers individuals with speech impairments to control devices using only lip movements |
| 🔊 **Loud Environments** | Factory floors, concerts, construction sites where audio assistants fail |
| 🔒 **Privacy & Stealth** | Hospitals, courtrooms, libraries — where speaking aloud is inappropriate |
| 🎖️ **Tactical** | Silent command execution for field operatives |

## ⚡ Features

- **📹 Video Upload Mode** — Upload a `.mp4` video and analyse lip movements offline
- **📷 Live Camera Mode** — Real-time lip reading from your webcam with live ROI extraction and command matching
- **🧠 Neural Network** — Spatiotemporal 3D-CNN + Bi-GRU with CTC decoding (accepts pre-trained LipNet/AV-HuBERT weights)
- **🎯 Fuzzy Command Matching** — Robust to VSR homophene noise via `thefuzz` token sort ratio
- **🌙 Dark-Themed UI** — Polished glassmorphism design with gradient accents

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌───────────────┐     ┌──────────────┐
│  Video/Camera│────▶│  processor.py   │────▶│   model.py    │────▶│  decoder.py  │
│   Input      │     │  MediaPipe Face │     │  3D-CNN +     │     │  thefuzz     │
│              │     │  Landmarker     │     │  Bi-GRU + CTC │     │  Fuzzy Match │
└──────────────┘     │  → Lip ROI      │     │  → Raw Text   │     │  → Command   │
                     └─────────────────┘     └───────────────┘     └──────────────┘
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/singhhrishabh/SilentAssist.git
cd SilentAssist

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe face model (auto-downloads if missing)
# Or manually:
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Launch the app
streamlit run app.py
```

## 📂 Project Structure

```
SilentAssist/
├── app.py              # Streamlit UI — video upload + live camera modes
├── processor.py        # MediaPipe lip ROI extraction pipeline
├── model.py            # 3D-CNN + Bi-GRU spatiotemporal network + CTC decode
├── decoder.py          # Fuzzy command matching (thefuzz)
├── requirements.txt    # Pinned dependencies
└── face_landmarker.task  # MediaPipe model bundle (downloaded at setup)
```

## 📋 Available Commands

The system recognises 20 pre-defined commands via fuzzy matching:

| | | | |
|---|---|---|---|
| Turn on the lights | Turn off the lights | Call for help | Send emergency text |
| Lock the doors | Unlock the doors | Open the window | Close the window |
| Play some music | Stop the music | Set an alarm | Cancel the alarm |
| Take a screenshot | Read my messages | Start recording | Stop recording |
| Call an ambulance | Increase the volume | Decrease the volume | Navigate home |

## 🔧 Using Pre-Trained Weights

SilentAssist is designed to accept pre-trained LipNet or AV-HuBERT weights:

1. Toggle **"Load pre-trained weights"** in the sidebar
2. Upload your `.pt` / `.pth` checkpoint file
3. The model will load the weights and run real inference with CTC decoding

Without weights, the app runs in **demo mode** with a deterministic stub for end-to-end pipeline demonstration.

## 🛠️ Tech Stack

- **UI**: [Streamlit](https://streamlit.io) + [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- **Computer Vision**: [OpenCV](https://opencv.org) + [MediaPipe](https://ai.google.dev/edge/mediapipe)
- **Deep Learning**: [PyTorch](https://pytorch.org) (3D-CNN + Bi-GRU + CTC)
- **NLP**: [thefuzz](https://github.com/seatgeek/thefuzz) (Fuzzy string matching)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for the 24-hr Hackathon</p>
