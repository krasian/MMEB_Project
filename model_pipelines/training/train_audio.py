"""Training loop for the Bird-MAE prototypical audio probe."""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_pipelines.models import BirdMAEModel
from model_pipelines.models.audio_encoder import precompute_window_features, create_window_dataloader
from model_pipelines.utils.device_utils import get_device


def compute_validation_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = torch.argmax(model(features), dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def compute_multilabel_metrics(model, val_loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            probabilities = torch.sigmoid(model(features))
            all_predictions.append((probabilities > threshold).float().cpu())
            all_labels.append(labels.cpu())

    if not all_labels:
        return {"accuracy": 0.0, "iou": 0.0, "orthogonality_loss": 0.0}

    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    multi_hot_labels = torch.zeros(all_labels.size(0), model.num_classes)
    multi_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1)

    per_class_accuracy = (all_predictions == multi_hot_labels).float().mean(dim=0).mean().item()
    intersection = (all_predictions * multi_hot_labels).sum().item()
    union = (all_predictions + multi_hot_labels).clamp(0, 1).sum().item()

    return {
        "accuracy": per_class_accuracy,
        "iou": intersection / (union + 1e-8),
        "orthogonality_loss": model.get_orthogonality_loss().item(),
    }


def extract_window_features_for_training(samples, cfg):
    print("\n" + "=" * 70)
    print("Precomputing Window Features for Prototypical Probing")
    print("=" * 70)
    return precompute_window_features(samples, cfg)


def _load_window_metadata(cfg):
    label_path = os.path.join(cfg.checkpoint_directory, "window_to_label.npy")
    path_map_path = os.path.join(cfg.checkpoint_directory, "window_to_path.npy")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing cached label metadata: {label_path}")
    if not os.path.exists(path_map_path):
        raise FileNotFoundError(f"Missing cached path metadata: {path_map_path}")
    return (
        np.load(label_path, allow_pickle=True).item(),
        np.load(path_map_path, allow_pickle=True).item(),
    )


def train_audio_model(cfg, train_samples, val_samples, spatial_features=None):
    device = cfg.device()

    print("\n" + "=" * 70)
    print("Training Prototypical Probe (Bird-MAE Paper)")
    print("=" * 70)
    print(f"  Device:           {device}")
    print(f"  Epochs:           {cfg.epochs}")
    print(f"  Batch size:       {cfg.batch_size}")
    print(f"  Learning rate:    {cfg.learning_rate}")
    print(f"  Train samples:    {len(train_samples)}")
    print(f"  Val samples:      {len(val_samples)}")
    print(f"  Number classes:   {cfg.number_of_classes}")
    print(f"  Prototypes/class: {getattr(cfg, 'num_prototypes', 20)}")

    train_known = [(p, l) for p, l in train_samples if l >= 0]
    val_known = [(p, l) for p, l in val_samples if l >= 0]
    train_outliers = [(p, l) for p, l in train_samples if l < 0]
    val_outliers = [(p, l) for p, l in val_samples if l < 0]

    if not train_known:
        raise ValueError("No known-class training samples found after splitting out negative labels.")
    if not val_known:
        raise ValueError("No known-class validation samples found after splitting out negative labels.")

    if spatial_features is None:
        all_samples = train_known + val_known + train_outliers + val_outliers
        window_features, window_to_label, window_to_path = extract_window_features_for_training(all_samples, cfg)
    else:
        window_features = spatial_features
        window_to_label, window_to_path = _load_window_metadata(cfg)

    train_loader = create_window_dataloader(
        train_known, window_features, window_to_label, window_to_path, cfg,
        shuffle=True, include_outliers=False,
    )
    val_loader = create_window_dataloader(
        val_known, window_features, window_to_label, window_to_path, cfg,
        shuffle=False, include_outliers=False,
    )

    model = BirdMAEModel(cfg, num_classes=cfg.number_of_classes).to(device)
    print(f"\n  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = getattr(cfg, "checkpoint_directory", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_prototypical_probe.pt")
    best_val_acc = 0.0
    best_val_iou = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = total_bce = total_ortho = 0.0
        total_samples = 0
        last_batch_end = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        for batch_idx, (features, labels) in enumerate(pbar, start=1):
            batch_start = time.perf_counter()
            data_wait_s = batch_start - last_batch_end

            move_start = time.perf_counter()
            features = features.to(device)
            labels = labels.to(device)
            move_s = time.perf_counter() - move_start

            forward_start = time.perf_counter()
            logits = model(features)
            forward_s = time.perf_counter() - forward_start

            multi_hot_labels = torch.zeros(labels.size(0), cfg.number_of_classes, device=device)
            multi_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
            bce_loss = criterion(logits, multi_hot_labels)
            ortho_loss = model.get_orthogonality_loss()
            loss = bce_loss + getattr(cfg, "orthogonality_weight", 0.1) * ortho_loss

            backward_start = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            backward_s = time.perf_counter() - backward_start

            step_start = time.perf_counter()
            optimizer.step()
            step_s = time.perf_counter() - step_start

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_bce += bce_loss.item() * batch_size
            total_ortho += ortho_loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({
                "loss": f"{total_loss / total_samples:.4f}",
                "bce": f"{total_bce / total_samples:.4f}",
                "ortho": f"{total_ortho / total_samples:.4f}",
            })
            last_batch_end = time.perf_counter()

            if batch_idx <= 5 or batch_idx % 25 == 0:
                tqdm.write(
                    f"  Batch {batch_idx}/{len(train_loader)} timings | "
                    f"data_wait={data_wait_s:.2f}s move={move_s:.2f}s "
                    f"forward={forward_s:.2f}s backward={backward_s:.2f}s step={step_s:.2f}s",
                )

        val_metrics = compute_multilabel_metrics(model, val_loader, device)
        val_acc = compute_validation_accuracy(model, val_loader, device)
        scheduler.step()

        avg_loss = total_loss / total_samples if total_samples else 0.0
        avg_bce = total_bce / total_samples if total_samples else 0.0
        avg_ortho = total_ortho / total_samples if total_samples else 0.0
        print(
            f"Epoch {epoch + 1:2d}/{cfg.epochs} | "
            f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Ortho: {avg_ortho:.4f}) | "
            f"Val Acc: {val_acc:.3f} | mIoU: {val_metrics['iou']:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_iou = val_metrics["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "val_iou": best_val_iou,
                "num_classes": cfg.number_of_classes,
                "num_prototypes": getattr(cfg, "num_prototypes", 20),
            }, checkpoint_path)
            print(f"  New best model saved (acc={val_acc:.3f}, iou={val_metrics['iou']:.3f})")

    print(f"\n  Best validation accuracy: {best_val_acc:.3f}")
    print(f"  Best validation IoU: {best_val_iou:.3f}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
