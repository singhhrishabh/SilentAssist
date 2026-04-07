"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Dataset Loaders for Training                 ║
║  ──────────────────────────────────────────────────────────  ║
║  Supports:                                                   ║
║    • GRID Corpus (standard lip-reading benchmark)            ║
║    • LRW (Lip Reading in the Wild) — word-level              ║
║    • Custom video datasets in a simple folder structure      ║
║                                                              ║
║  All loaders output (tensor, label, input_length, target_len)║
║  tuples compatible with CTC loss training.                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import glob
import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

# ── Re-use the processor for lip ROI extraction ──────────────
from processor import preprocess_video, ROI_HEIGHT, ROI_WIDTH, MAX_FRAMES
from model import VOCAB, VOCAB_SIZE

# ── Character-to-index mapping for CTC labels ───────────────
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}


def text_to_indices(text: str) -> List[int]:
    """
    Convert a text string to a list of vocabulary indices.

    Characters not in the vocabulary are silently skipped.

    Args:
        text: Lowercased text label for a video.

    Returns:
        List of integer indices into VOCAB.
    """
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def indices_to_text(indices: List[int]) -> str:
    """Convert a list of vocabulary indices back to text."""
    from model import IDX_TO_CHAR
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices if i != 0)


# ══════════════════════════════════════════════════════════════
#  GRID Corpus Dataset
# ══════════════════════════════════════════════════════════════
class GRIDDataset(Dataset):
    """
    Dataset loader for the GRID audio-visual speech corpus.

    Expected directory structure:
        grid_root/
        ├── videos/
        │   ├── s1/
        │   │   ├── bbaf2n.mpg
        │   │   ├── ...
        │   ├── s2/
        │   └── ...
        └── aligns/
            ├── s1/
            │   ├── bbaf2n.align
            │   └── ...
            ├── s2/
            └── ...

    Each .align file contains frame-level word alignments.
    We extract the full sentence from the align file.
    """

    def __init__(
        self,
        grid_root: str,
        speakers: Optional[List[str]] = None,
        max_frames: int = MAX_FRAMES,
        transform=None,
    ):
        """
        Args:
            grid_root:   Root directory of the GRID corpus.
            speakers:    List of speaker IDs (e.g. ["s1", "s2"]).
                         If None, uses all available speakers.
            max_frames:  Max temporal length (pads / truncates).
            transform:   Optional transform applied to the tensor.
        """
        self.grid_root = grid_root
        self.max_frames = max_frames
        self.transform = transform

        video_dir = os.path.join(grid_root, "videos")
        align_dir = os.path.join(grid_root, "aligns")

        if speakers is None:
            speakers = sorted(os.listdir(video_dir))

        self.samples: List[Tuple[str, str]] = []  # (video_path, label_text)

        for spk in speakers:
            spk_video_dir = os.path.join(video_dir, spk)
            spk_align_dir = os.path.join(align_dir, spk)

            if not os.path.isdir(spk_video_dir):
                continue

            for vf in sorted(os.listdir(spk_video_dir)):
                video_path = os.path.join(spk_video_dir, vf)
                base = os.path.splitext(vf)[0]
                align_path = os.path.join(spk_align_dir, f"{base}.align")

                if os.path.isfile(align_path):
                    label = self._parse_align(align_path)
                    if label:
                        self.samples.append((video_path, label))

        print(f"[GRIDDataset] Loaded {len(self.samples)} samples from {len(speakers)} speakers")

    @staticmethod
    def _parse_align(align_path: str) -> str:
        """Parse a .align file and extract the full sentence (excluding sil/sp)."""
        words = []
        with open(align_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2].lower()
                    if word not in ("sil", "sp"):
                        words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label_text = self.samples[idx]

        try:
            tensor, _ = preprocess_video(video_path, self.max_frames)
        except Exception:
            # Return a zero tensor on failure
            tensor = torch.zeros(1, 1, self.max_frames, ROI_HEIGHT, ROI_WIDTH)

        tensor = tensor.squeeze(0)  # Remove batch dim: (1, T, H, W)

        if self.transform:
            tensor = self.transform(tensor)

        # CTC label indices
        label_indices = text_to_indices(label_text)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        # Lengths for CTC loss
        input_length = torch.tensor([self.max_frames], dtype=torch.long)
        target_length = torch.tensor([len(label_indices)], dtype=torch.long)

        return tensor, label_tensor, input_length, target_length


# ══════════════════════════════════════════════════════════════
#  Simple Folder Dataset (Custom Videos)
# ══════════════════════════════════════════════════════════════
class FolderLipDataset(Dataset):
    """
    Simple dataset for custom videos with text labels.

    Expected structure:
        data_root/
        ├── video1.mp4
        ├── video2.mp4
        └── labels.txt

    labels.txt format (one per line, tab/comma separated):
        video1.mp4,turn on the lights
        video2.mp4,call for help
    """

    def __init__(
        self,
        data_root: str,
        labels_file: str = "labels.txt",
        max_frames: int = MAX_FRAMES,
        separator: str = ",",
    ):
        self.data_root = data_root
        self.max_frames = max_frames
        self.samples: List[Tuple[str, str]] = []

        labels_path = os.path.join(data_root, labels_file)
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    video_name, label = parts[0].strip(), parts[1].strip()
                    video_path = os.path.join(data_root, video_name)
                    if os.path.isfile(video_path):
                        self.samples.append((video_path, label))

        print(f"[FolderLipDataset] Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label_text = self.samples[idx]

        try:
            tensor, _ = preprocess_video(video_path, self.max_frames)
        except Exception:
            tensor = torch.zeros(1, 1, self.max_frames, ROI_HEIGHT, ROI_WIDTH)

        tensor = tensor.squeeze(0)

        label_indices = text_to_indices(label_text)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        input_length = torch.tensor([self.max_frames], dtype=torch.long)
        target_length = torch.tensor([len(label_indices)], dtype=torch.long)

        return tensor, label_tensor, input_length, target_length


# ══════════════════════════════════════════════════════════════
#  CTC-compatible Collate Function
# ══════════════════════════════════════════════════════════════
def ctc_collate_fn(batch):
    """
    Custom collate function for CTC training.

    Pads variable-length label sequences and stacks tensors.

    Returns:
        inputs:         (B, 1, T, H, W) float32
        targets:        (sum_of_target_lengths,) int64  — concatenated
        input_lengths:  (B,) int64
        target_lengths: (B,) int64
    """
    inputs, labels, input_lengths, target_lengths = zip(*batch)

    # Stack input tensors: all should be (1, T, H, W)
    inputs = torch.stack(inputs, dim=0)  # (B, 1, T, H, W)

    # Concatenate all labels into a single 1D tensor (CTC format)
    targets = torch.cat(labels)

    input_lengths = torch.cat(input_lengths)
    target_lengths = torch.cat(target_lengths)

    return inputs, targets, input_lengths, target_lengths


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split a dataset into train/val loaders with CTC-compatible collation.

    Args:
        dataset:      Full dataset.
        batch_size:   Training batch size.
        val_split:    Fraction reserved for validation.
        num_workers:  DataLoader workers.
        seed:         Random seed for reproducibility.

    Returns:
        (train_loader, val_loader)
    """
    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
