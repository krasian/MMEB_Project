"""
Train BirdEmbeddingModel on the known species using ArcFace loss.

Saves the best checkpoint (by centroid-based val accuracy) to
{cfg.checkpoint_directory}/best_model.pt.
"""
import os
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from data.visual_dataset import make_loader
from models.visual_encoder import BirdEmbeddingModel
from models.arcface import ArcFaceLoss
from outlier.mahalanobis import compute_centroids


@torch.no_grad()
def _extract_embeddings_raw(model, loader) -> tuple:
    """Extract embeddings silently — used internally during training."""
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


def train_model(train_samples: list, val_samples: list):
    """Fine-tune BirdEmbeddingModel using ArcFace loss."""
    device = cfg.device()
    print("\n── Step 1: Training BirdEmbeddingModel with ArcFace loss ──")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {cfg.epoches}")
    print(f"  Batch size : {cfg.batch}")
    print(f"  ArcFace s  : {cfg.arcface_scaler}  m: {cfg.arcface_margin}")

    train_loader = make_loader(train_samples, "train", shuffle=True)
    val_loader = make_loader(val_samples, "val")

    model = BirdEmbeddingModel().to(device)
    criterion = ArcFaceLoss(
        cfg.embedding_dim, cfg.number_of_classes,
        s=cfg.arcface_scaler, m=cfg.arcface_margin).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.learning_rate * 0.1},
        {"params": model.embedding.parameters(), "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epoches)
    best_val_acc = 0.0

    for epoch in range(cfg.epoches):
        model.train()
        total_loss, total = 0.0, 0
        epoch_embs, epoch_labels = [], []

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
            epoch_embs.append(embs.detach().cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())

        model.eval()
        cached_train_embs = np.vstack(epoch_embs)
        cached_train_labels = np.concatenate(epoch_labels)
        val_acc = _compute_val_accuracy(
            model, val_loader, cached_train_embs, cached_train_labels)
        scheduler.step()

        print(f"  Epoch {epoch+1:2d} | "
              f"Loss: {total_loss/total:.4f} | "
              f"Val acc (centroid): {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoint_directory, "best_model.pt"))
            print(f"    ✓ Checkpoint saved (val_acc={val_acc:.3f})")

    print(f"\n  Best val accuracy: {best_val_acc:.3f}")
