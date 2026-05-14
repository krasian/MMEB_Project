"""
Train BirdEmbeddingModel on the known species using ArcFace loss.

Saves the best checkpoint (by centroid-based val accuracy) to
{cfg.checkpoint_directory}/best_model.pt.

Also writes per-epoch training history to:
    {cfg.checkpoint_directory}/training_history.json
    {cfg.checkpoint_directory}/training_history.csv

Use plot_training_history.py (or any plotting tool) to visualise the
loss and validation accuracy curves over epochs.

FIXES APPLIED:
  1. ArcFaceLoss parameters (the class anchors) are now in the optimizer
     so they actually get trained. Previously they were frozen at their
     random initialization, which silently crippled the loss and led to
     looser clusters / worse outlier detection.
  2. Weight decay is set to 0 for the ArcFace anchors -- they live on
     the unit sphere after normalization, so shrinking them toward 0 is
     meaningless.
  3. Train embeddings used for centroid-based val accuracy are now
     recomputed in a clean forward pass AFTER the epoch finishes,
     instead of being collected mid-epoch from a moving model. This
     gives an honest val number and so a more reliable best-checkpoint
     selection.
  4. Per-epoch metrics are logged to JSON + CSV for plotting.
"""
import os
import csv
import json
import numpy as np
import torch
from tqdm import tqdm

try:
    from model_pipelines.config import cfg
    from model_pipelines.data.visual_dataset import make_loader
    from model_pipelines.models.visual_encoder import BirdEmbeddingModel
    from model_pipelines.models.arcface import ArcFaceLoss
    from model_pipelines.outlier.mahalanobis import compute_centroids
except ImportError:
    from config import cfg
    from data.visual_dataset import make_loader
    from models.visual_encoder import BirdEmbeddingModel
    from models.arcface import ArcFaceLoss
    from outlier.mahalanobis import compute_centroids


@torch.no_grad()
def _extract_embeddings_raw(model, loader) -> tuple:
    """Extract embeddings silently -- used internally during training."""
    device = cfg.device()
    all_embs, all_labels = [], []
    for imgs, labels in loader:
        embs = model(imgs.to(device)).cpu().numpy()
        all_embs.append(embs)
        all_labels.append(labels.numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)


def _compute_val_accuracy(model, val_loader, cached_train_embs: np.ndarray,
                          cached_train_labels: np.ndarray) -> float:
    """Centroid-based classification accuracy on the validation set."""
    centroids = compute_centroids(
        cached_train_embs, cached_train_labels, cfg.number_of_classes)
    val_embs, val_labels = _extract_embeddings_raw(model, val_loader)
    dists = np.linalg.norm(
        val_embs[:, None, :] - centroids[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1)
    return float((preds == val_labels).mean())


def _save_history(history: list, out_dir: str):
    """Persist the training history to JSON and CSV."""
    json_path = os.path.join(out_dir, "training_history.json")
    csv_path = os.path.join(out_dir, "training_history.csv")

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # CSV: easy to load into Excel / pandas / matplotlib
    fieldnames = ["epoch", "train_loss", "val_acc",
                  "learning_rate", "is_best"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train_model(train_samples: list, val_samples: list):
    """Fine-tune BirdEmbeddingModel using ArcFace loss."""
    device = cfg.device()
    print("\n-- Step 1: Training BirdEmbeddingModel with ArcFace loss --")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {cfg.epoches}")
    print(f"  Batch size : {cfg.batch}")
    print(f"  ArcFace s  : {cfg.arcface_scaler}  m: {cfg.arcface_margin}")

    train_loader = make_loader(train_samples, "train", shuffle=True)
    val_loader = make_loader(val_samples, "val")
    # Separate loader for clean post-epoch embedding extraction (no shuffle,
    # no augmentation -- uses the "val" transforms for deterministic features).
    train_eval_loader = make_loader(train_samples, "val", shuffle=False)

    model = BirdEmbeddingModel().to(device)
    criterion = ArcFaceLoss(
        cfg.embedding_dim, cfg.number_of_classes,
        s=cfg.arcface_scaler, m=cfg.arcface_margin).to(device)

    # FIX: include criterion.parameters() so the ArcFace class anchors
    # actually get trained. Anchors get NO weight decay (they're normalized
    # to the unit sphere in forward, so decaying them toward 0 is moot).
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),
         "lr": cfg.learning_rate * 0.1},
        {"params": model.embedding.parameters(),
         "lr": cfg.learning_rate},
        {"params": criterion.parameters(),
         "lr": cfg.learning_rate,
         "weight_decay": 0.0},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epoches)
    best_val_acc = 0.0

    # Per-epoch metrics for plotting. Each entry is a dict; persisted to
    # JSON and CSV after every epoch so a crashed run still leaves a
    # partial history on disk.
    history = []

    for epoch in range(cfg.epoches):
        model.train()
        total_loss, total = 0.0, 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"  Epoch {epoch+1}/{cfg.epoches}",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = model(imgs)
            loss = criterion(embs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        # FIX: recompute train embeddings AFTER the epoch with the final
        # weights and deterministic (val) transforms. Previously we used
        # embeddings collected during training, which came from a model
        # whose weights changed every batch -- so the centroids were a
        # blur of early-epoch and late-epoch states.
        model.eval()
        cached_train_embs, cached_train_labels = _extract_embeddings_raw(
            model, train_eval_loader)

        val_acc = _compute_val_accuracy(
            model, val_loader, cached_train_embs, cached_train_labels)

        # Grab the current LR from the head group BEFORE stepping the
        # scheduler so the logged LR matches the one used this epoch.
        current_lr = optimizer.param_groups[1]["lr"]
        scheduler.step()

        avg_loss = total_loss / total
        is_best = val_acc > best_val_acc

        print(f"  Epoch {epoch+1:2d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val acc (centroid): {val_acc:.3f} | "
              f"LR: {current_lr:.2e}")

        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoint_directory, "best_model.pt"))
            print(f"    [+] Checkpoint saved (val_acc={val_acc:.3f})")

        # Append this epoch's row and flush history to disk every epoch
        # so a crash doesn't lose the curve.
        history.append({
            "epoch":         epoch + 1,
            "train_loss":    round(avg_loss, 6),
            "val_acc":       round(val_acc, 6),
            "learning_rate": float(current_lr),
            "is_best":       bool(is_best),
        })
        _save_history(history, cfg.checkpoint_directory)

    print(f"\n  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Training history saved to:")
    print(f"    {os.path.join(cfg.checkpoint_directory, 'training_history.json')}")
    print(f"    {os.path.join(cfg.checkpoint_directory, 'training_history.csv')}")
