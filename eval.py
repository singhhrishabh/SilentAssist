"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Evaluation Pipeline (MLOps)                  ║
║  ──────────────────────────────────────────────────────────  ║
║  Calculates Character Error Rate (CER) and Word Error Rate   ║
║  (WER) systematically. Can be run with real datasets or in   ║
║  'CI/CD Dummy Mode' to verify pipeline integrity via         ║
║  GitHub Actions.                                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import torch
import torch.nn as nn
from model import load_model, get_device, ctc_greedy_decode, VOCAB_SIZE
from dataset import FolderLipDataset, ctc_collate_fn
from train import validate, compute_cer, compute_wer
from torch.utils.data import DataLoader, TensorDataset

def run_dummy_ci_test():
    """
    Runs a pipeline verification test for MLOps GitHub Actions,
    bypassing the need for 3GB dataset downloads.
    """
    print("[mlops] Running Dummy CI/CD Pipeline Evaluation...")
    device = get_device()
    model = load_model(auto_download=False, device=device)  # Random init for CI
    
    # Create fake batch: (batch=1, channel=1, time=75, H=64, W=128)
    inputs = torch.randn(1, 1, 75, 64, 128, device=device)
    
    # Create fake targets: 'test' -> [20, 5, 19, 20]
    # Since model is random, CER/WER will be terrible, but we test the math.
    targets = torch.tensor([20, 5, 19, 20], dtype=torch.long, device=device)
    input_lengths = torch.tensor([75], dtype=torch.long, device=device)
    target_lengths = torch.tensor([4], dtype=torch.long, device=device)
    
    log_probs = model(inputs)
    
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
    
    pred_str = ctc_greedy_decode(log_probs)
    ref_str = "test"
    
    cer = compute_cer(pred_str, ref_str)
    wer = compute_wer(pred_str, ref_str)
    
    print(f"[mlops] Evaluated Batch 1 | Loss: {loss.item():.4f} | CER: {cer:.4f} | WER: {wer:.4f}")
    print("[mlops] Pipeline integrity verified. Exiting gracefully.")
    sys.exit(0)


def evaluate_dataset(data_root: str, labels_file: str, weights_path: str = None):
    print(f"[mlops] Evaluating dataset at {data_root}")
    device = get_device()
    model = load_model(weights_path=weights_path, device=device)
    
    dataset = FolderLipDataset(data_root=data_root, labels_file=labels_file)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, 
        collate_fn=ctc_collate_fn, num_workers=0
    )
    
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    avg_loss, avg_cer, avg_wer = validate(model, loader, criterion, device)
    
    print(f"\n{'='*40}")
    print(f"  Evaluation Results")
    print(f"  Total Samples : {len(dataset)}")
    print(f"  Average Loss  : {avg_loss:.4f}")
    print(f"  Average CER   : {avg_cer:.4f}")
    print(f"  Average WER   : {avg_wer:.4f}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SilentAssist Evaluation Script")
    parser.add_argument("--ci", action="store_true", help="Run CI dummy validation mode")
    parser.add_argument("--data_root", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--labels_file", type=str, default="labels.txt")
    parser.add_argument("--weights_path", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.ci:
        run_dummy_ci_test()
    elif args.data_root:
        evaluate_dataset(args.data_root, args.labels_file, args.weights_path)
    else:
        print("[ERROR] Must provide either --ci flag or --data_root path.")
        sys.exit(1)
