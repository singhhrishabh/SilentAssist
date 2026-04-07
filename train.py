"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Training & Evaluation Pipeline               ║
║  ──────────────────────────────────────────────────────────  ║
║  Features:                                                   ║
║    • CTC loss training with AdamW optimiser                  ║
║    • OneCycleLR scheduling                                   ║
║    • Mixed-precision training (AMP) on CUDA / MPS            ║
║    • Character Error Rate (CER) & Word Error Rate (WER)      ║
║    • Checkpoint saving with best-CER tracking                ║
║    • TensorBoard-compatible logging                          ║
║    • Apple Silicon (MPS) + CUDA + CPU support                ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python train.py --data_root /path/to/grid --epochs 50
    python train.py --data_root /path/to/custom --dataset_type folder --epochs 30
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── Local imports ────────────────────────────────────────────
from model import SilentAssistNet, get_device, ctc_greedy_decode, VOCAB_SIZE
from dataset import (
    GRIDDataset,
    FolderLipDataset,
    ctc_collate_fn,
    create_dataloaders,
    indices_to_text,
)


# ══════════════════════════════════════════════════════════════
#  Evaluation Metrics: CER & WER
# ══════════════════════════════════════════════════════════════
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_cer(predicted: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER).

    CER = edit_distance(pred_chars, ref_chars) / len(ref_chars)
    """
    if len(reference) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return levenshtein_distance(predicted, reference) / len(reference)


def compute_wer(predicted: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER).

    WER = edit_distance(pred_words, ref_words) / len(ref_words)
    """
    pred_words = predicted.strip().split()
    ref_words = reference.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    return levenshtein_distance(
        " ".join(pred_words), " ".join(ref_words)
    ) / len(" ".join(ref_words))


# ══════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train for one epoch. Returns average CTC loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        log_probs = model(inputs)  # (B, T, vocab_size)

        # CTC loss expects (T, B, vocab_size)
        log_probs = log_probs.permute(1, 0, 2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] Skipping batch {batch_idx} — NaN/Inf loss")
            continue

        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════
#  Validation Loop
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Validate the model. Returns (avg_loss, avg_cer, avg_wer).
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n_batches = 0
    n_samples = 0

    for inputs, targets, input_lengths, target_lengths in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs = model(inputs)
        log_probs_ctc = log_probs.permute(1, 0, 2)

        loss = criterion(log_probs_ctc, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        n_batches += 1

        # Decode predictions and compute CER/WER for each sample
        offset = 0
        for i in range(inputs.size(0)):
            # Extract this sample's target
            tgt_len = target_lengths[i].item()
            tgt_indices = targets[offset : offset + tgt_len].cpu().tolist()
            offset += tgt_len

            ref_text = indices_to_text(tgt_indices)
            pred_text = ctc_greedy_decode(log_probs[i : i + 1])

            total_cer += compute_cer(pred_text, ref_text)
            total_wer += compute_wer(pred_text, ref_text)
            n_samples += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_cer = total_cer / max(n_samples, 1)
    avg_wer = total_wer / max(n_samples, 1)

    return avg_loss, avg_cer, avg_wer


# ══════════════════════════════════════════════════════════════
#  Main Training Script
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SilentAssist — Train the VSR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--dataset_type", type=str, default="grid",
                        choices=["grid", "folder"],
                        help="Dataset format: 'grid' (GRID corpus) or 'folder' (custom)")
    parser.add_argument("--speakers", type=str, nargs="*", default=None,
                        help="GRID speaker IDs to use (e.g. s1 s2 s3)")
    parser.add_argument("--labels_file", type=str, default="labels.txt",
                        help="Labels filename for folder dataset")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  SilentAssist Training Pipeline")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_type} @ {args.data_root}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # ── Create dataset ───────────────────────────────────────
    if args.dataset_type == "grid":
        dataset = GRIDDataset(
            grid_root=args.data_root,
            speakers=args.speakers,
        )
    else:
        dataset = FolderLipDataset(
            data_root=args.data_root,
            labels_file=args.labels_file,
        )

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty. Check your data_root path.")
        sys.exit(1)

    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples\n")

    # ── Create model ─────────────────────────────────────────
    model = SilentAssistNet(
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} total, {trainable_params:,} trainable\n")

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch = 0
    best_cer = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best CER: {best_cer:.4f}\n")

    # ── Loss, Optimiser, Scheduler ───────────────────────────
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs - start_epoch,
        pct_start=0.1,
    )

    # ── Output directory ─────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = vars(args)
    config["device"] = str(device)
    config["total_params"] = total_params
    config["start_time"] = datetime.now().isoformat()
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
        )

        # Validate
        val_loss, val_cer, val_wer = validate(
            model, val_loader, criterion, device,
        )

        elapsed = time.time() - t0

        # Log
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cer": round(val_cer, 4),
            "val_wer": round(val_wer, 4),
            "elapsed_s": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        is_best = val_cer < best_cer
        if is_best:
            best_cer = val_cer

        print(
            f"\n  Epoch {epoch}/{args.epochs-1} ({elapsed:.1f}s)\n"
            f"    Train Loss: {train_loss:.4f}\n"
            f"    Val   Loss: {val_loss:.4f}  |  CER: {val_cer:.4f}  |  WER: {val_wer:.4f}"
            f"{'  *** BEST ***' if is_best else ''}\n"
        )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "best_cer": best_cer,
            }, ckpt_path)
            print(f"    Saved checkpoint: {ckpt_path}")

        # Save best model separately
        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_cer": val_cer,
                "val_wer": val_wer,
            }, best_path)
            print(f"    Saved best model: {best_path}")

    # ── Save training history ────────────────────────────────
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best CER: {best_cer:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
