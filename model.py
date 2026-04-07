"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Spatiotemporal VSR Network + CTC Decoder    ║
║  ──────────────────────────────────────────────────────────  ║
║  Architecture                                                ║
║    • 3D-CNN front-end  (spatial feature extraction)          ║
║    • Bi-directional GRU back-end  (temporal modelling)       ║
║    • Linear classifier → CTC-decoded text                    ║
║                                                              ║
║  Designed to accept pre-trained LipNet-style weights.        ║
║  Includes greedy CTC beam decoding for inference.            ║
╚══════════════════════════════════════════════════════════════╝
"""

import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Character vocabulary (LipNet convention) ─────────────────
#    Index 0 = CTC blank token
#    Indices 1-26 = a-z
#    Index 27 = space
#    Index 28 = apostrophe  (optional, useful for contractions)
VOCAB = ["<blank>"] + list(string.ascii_lowercase) + [" ", "'"]
VOCAB_SIZE = len(VOCAB)            # 29

# Reverse look-up for decoding
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


# ══════════════════════════════════════════════════════════════
#  3D-CNN Front-End
# ══════════════════════════════════════════════════════════════
class SpatiotemporalFrontEnd(nn.Module):
    """
    Three 3D-convolutional blocks with BatchNorm, ReLU, and
    3D max-pooling.  Reduces spatial dimensions while preserving
    the temporal axis for the downstream RNN.

    Expected input shape:  (B, 1, T, 64, 128)
    Output shape:          (B, C_out, T, H', W')
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# ══════════════════════════════════════════════════════════════
#  Full LipNet-Style Model
# ══════════════════════════════════════════════════════════════
class SilentAssistNet(nn.Module):
    """
    End-to-end Visual Speech Recognition network.

    Pipeline:
        Input (B, 1, T, 64, 128)
          → 3D-CNN front-end      → (B, 96, T, H', W')
          → Flatten spatial dims  → (B, T, 96 * H' * W')
          → Bi-GRU (2 layers)     → (B, T, hidden*2)
          → Linear                → (B, T, vocab_size)

    The output logits are CTC-compatible (log-softmax over vocab).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.frontend = SpatiotemporalFrontEnd()

        # We compute the flattened spatial size dynamically in forward()
        # but also cache it here for the Linear layer initialisation.
        # For input (1, 1, 75, 64, 128):
        #   after conv1: (1, 32, 75, 16, 32)
        #   after conv2: (1, 64, 75,  8, 16)
        #   after conv3: (1, 96, 75,  4,  8)
        #   flattened spatial = 96 * 4 * 8 = 3072
        self._flat_features = 96 * 4 * 8   # 3072

        self.gru = nn.GRU(
            input_size=self._flat_features,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size * 2, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, 64, 128)

        Returns:
            log_probs: (B, T, vocab_size)
        """
        # ── Spatial feature extraction ───────────────────────
        B = x.size(0)
        cnn_out = self.frontend(x)          # (B, C, T, H', W')

        # Reshape: merge spatial dims, keep time axis
        _, C, T, H, W = cnn_out.shape
        cnn_out = cnn_out.permute(0, 2, 1, 3, 4)   # (B, T, C, H, W)
        cnn_out = cnn_out.contiguous().view(B, T, C * H * W)

        # ── Temporal modelling ───────────────────────────────
        gru_out, _ = self.gru(cnn_out)      # (B, T, hidden*2)

        # ── Classification ───────────────────────────────────
        logits = self.fc(gru_out)            # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


# ══════════════════════════════════════════════════════════════
#  Model Loading Utilities
# ══════════════════════════════════════════════════════════════
def load_model(weights_path: Optional[str] = None) -> SilentAssistNet:
    """
    Instantiate the model and optionally load pre-trained weights.

    Args:
        weights_path:  Path to a .pt / .pth checkpoint.  If None the
                       model is returned with random initialisation
                       (demo / stub mode).

    Returns:
        model in eval mode on CPU.
    """
    model = SilentAssistNet()

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Support both raw state_dict and {"model_state_dict": ...} format
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[SilentAssist] Loaded weights from {weights_path}")
    else:
        print("[SilentAssist] No weights provided — running in DEMO stub mode.")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
#  CTC Greedy Decoding
# ══════════════════════════════════════════════════════════════
def ctc_greedy_decode(log_probs: torch.Tensor) -> str:
    """
    Greedy (best-path) CTC decoding.

    Args:
        log_probs:  (1, T, vocab_size)  — output of model.forward()

    Returns:
        Decoded text string with collapsed repeats and blanks removed.
    """
    # Argmax along vocab axis → (T,)
    indices = torch.argmax(log_probs.squeeze(0), dim=-1).tolist()

    # Collapse consecutive duplicates, then strip blanks
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    chars = [IDX_TO_CHAR.get(i, "") for i in collapsed if i != 0]
    return "".join(chars).strip()


# ══════════════════════════════════════════════════════════════
#  Demo / Stub Inference  (when no real weights are available)
# ══════════════════════════════════════════════════════════════
_DEMO_PHRASES = [
    "turn on the lights",
    "call for help",
    "send emergency text",
    "lock the doors",
    "open the window",
    "play some music",
    "set an alarm",
]


def demo_inference(tensor: torch.Tensor) -> str:
    """
    Deterministic stub that returns a plausible phrase based on
    a hash of the input tensor — used when no trained weights
    are available so the demo pipeline still runs end-to-end.
    """
    # Use the sum of the tensor as a simple hash seed
    seed = int(tensor.sum().item() * 1000) % len(_DEMO_PHRASES)
    return _DEMO_PHRASES[seed]
