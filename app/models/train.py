# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import argparse
from torch.nn.utils.rnn import pad_sequence

from dataset_lightcurve import LightCurveDataset
from cnn_attention import CNN_Attention

# ------------------------------
# Utility functions
# ------------------------------
def compute_metrics(y_true, y_pred, threshold=0.5):
    y_bin = (y_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec = recall_score(y_true, y_bin)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return dict(acc=acc, f1=f1, precision=prec, recall=rec, auc=auc)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    xs = xs.unsqueeze(1)  # Add channel dim for CNN (batch, 1, seq_len)
    ys = torch.stack(ys)
    return xs, ys

def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(labels), np.array(preds))
    return avg_loss, metrics


# ------------------------------
# Training function
# ------------------------------
def train(
    train_dir="dataset/train",
    val_dir="dataset/test",
    seq_len=2000,
    batch_size=32,
    lr=1e-4,
    num_epochs=20,
    save_path="checkpoints",
    mixed_precision=True,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Dataset & DataLoader ---
    train_ds = LightCurveDataset(train_dir, seq_len=seq_len)
    val_ds = LightCurveDataset(val_dir, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Model, optimizer, loss ---
    model = CNN_Attention(in_channels=1, seq_len=seq_len).to(device)
    model.print_layer_summary()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=mixed_precision)

    # --- Checkpoint directory ---
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    # --- Training loop ---
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}]")
        for x, y in pbar:
            x, y = x.to(device), y.float().to(device)

            optimizer.zero_grad()
            with autocast(enabled=mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)

        # --- Evaluate ---
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | AUC: {metrics['auc']:.4f} | "
            f"Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}"
        )

        # --- Save best model ---
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            ckpt = save_dir / f"best_model_epoch{epoch}_auc{best_auc:.4f}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"[INFO] Saved best model to {ckpt}")

    print(f"[DONE] Training completed. Best AUC={best_auc:.4f}")


# ------------------------------
# CLI entry point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+Attention on exoplanet light curves")
    parser.add_argument("--train_dir", type=str, default="dataset/train")
    parser.add_argument("--val_dir", type=str, default="dataset/test")
    parser.add_argument("--seq_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        save_path=args.save_path,
        mixed_precision=not args.no_amp,
    )

"""
python app/models/train.py --train_dir data/lc_dataset/train --val_dir data/lc_dataset/test --epochs 30 --seq_len 2000
"""