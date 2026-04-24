import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image
from tqdm import tqdm

try:
    import torch_directml
    _DIRECTML_AVAILABLE = True
except ImportError:
    _DIRECTML_AVAILABLE = False

logger = logging.getLogger(__name__)



@dataclass
class Config:
    """
    All pipeline settings in one place.

    Values here are the fallback defaults. In practice, they are
    overwritten at startup by load_config.apply_yaml_config(), which
    reads config.yaml and patches this instance in-place. The env var
    data_root (or --data-root CLI flag) always takes final priority.
    """
    def __init__(self):
        self.data_root = str(os.environ.get("DATA_ROOT", ""))
        self.known_csv= {"Common Blackbird":  "updated_blackbird_data.csv",
        "Eurasian Blue Tit": "updated_EurasianBlueTit_data.csv",
        "Great Tit":         "updated_GreatTit_data.csv",
        "House Sparrow":     "updated_HouseSparrow_data.csv"}

        self.outlier_csv ={"European Starling": "updated_EuropeanStarling.csv"}
        self.training_split_ratio=float( 0.70)
        self.validation_split_ratio=float( 0.15)
        self.number_of_classes=0
        self.embedding_dim=512
        self.image_size=224
        self.batch= 16
        self.epoches=40
        self.learning_rate = float(1e-4)
        self.weight_decay= float(1e-4)
        self.arcface_scaler = float(64.0)
        self.arcface_margin = float(0.8)
        self.percentile_of_threshold= 75
        self.checkpoint_directory= "checkpoints"
        self.results_directory= "results"
        # Evaluation settings (evaluate.py)
        self.embeding_visulize_method= "tsne"
        self.result_dpi= 300
        # Inference settings (predict.py)
        self.image_extensions={".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.csv_out= "predictions.csv"
        self.number_of_classes = len(self.known_csv)


    def device(self) -> torch.device:
        if _DIRECTML_AVAILABLE:
            return torch.device(torch_directml.device())
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def make_dirs(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        os.makedirs(self.results_directory,    exist_ok=True)


cfg = Config()
"""I first tried to do this with a config class and then decided that we can
have a config.yamal that has all these
inside but didnt want to change the code so I tried to do it without removing the class"""
try:
    from load_config import apply_yaml_config
    apply_yaml_config(cfg=cfg)
except FileNotFoundError:
    pass

names         = list(cfg.known_csv.keys())
class_to_id        = {name: i for i, name in enumerate(names)}
outlier_names = list(cfg.outlier_csv.keys())




class birdcsvdata(Dataset):
    """
    PyTorch Dataset that reads images from a list of (path, label) pairs.

    Args:
        samples:   list of (image_path: str, label: int) tuples.
                   Use label < 0 for outlier images (ignored during training).
        transform: torchvision transform to apply to each image.
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, IOError) as exc:
            logger.warning(
                "Could not load image %s: %s — using grey placeholder.", path, exc)
            img = Image.new("RGB", (cfg.image_size, cfg.image_size), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label



def load_csv_paths(csv_filename: str, label: int) -> list:
    """Read a species CSV and return a list of (image_path, label) tuples."""
    csv_path = os.path.join(cfg.data_root, csv_filename)
    print(f"DEBUG: trying to open '{cfg.data_root}'")  # add this
    df       = pd.read_csv(csv_path)
    return [(str(p), label) for p in df["image_path"].dropna()]


def split_samples(samples: list, train_r: float, val_r: float,
                  seed: int = 42) -> tuple:
    """Reproducibly shuffle and split a sample list into train / val / test."""
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(len(samples))
    n_train = int(len(samples) * train_r)
    n_val   = int(len(samples) * val_r)

    train = [samples[i] for i in idx[:n_train]]
    val   = [samples[i] for i in idx[n_train : n_train + n_val]]
    test  = [samples[i] for i in idx[n_train + n_val :]]
    return train, val, test


def build_splits() -> tuple:
    """
    Load all CSVs, split each known species into train/val/test, and
    collect all outlier-species images for testing only.

    Returns:
        train_samples, val_samples, test_known, test_outlier
    """
    train_all, val_all, test_known_all = [], [], []

    print("\n── Dataset summary ──")
    for species_name, csv_file in cfg.known_csv.items():
        label   = class_to_id[species_name]
        samples = load_csv_paths(csv_file, label)
        tr, va, te = split_samples(samples, cfg.training_split_ratio, cfg.validation_split_ratio)
        train_all      += tr
        val_all        += va
        test_known_all += te
        print(f"  {species_name:<25} total={len(samples):>5}  "
              f"train={len(tr):>4}  val={len(va):>4}  test={len(te):>4}")

    outlier_samples = []
    for i, (species_name, csv_file) in enumerate(cfg.outlier_csv.items()):
        species_samples = load_csv_paths(csv_file, label=-(i + 1))
        outlier_samples += species_samples
        print(f"  {species_name + ' (outlier)':<25} total={len(species_samples):>5}  "
              f"(test only)")

    print(f"\n  Train total : {len(train_all):,}")
    print(f"  Val total   : {len(val_all):,}")
    print(f"  Test known  : {len(test_known_all):,}")
    print(f"  Test outlier: {len(outlier_samples):,}")

    return train_all, val_all, test_known_all, outlier_samples


def get_transforms(split: str):
    """
    Return image pre-processing pipeline for training or evaluation.

    Training uses random augmentation; val/test uses deterministic resize + crop.
    Both normalise to ImageNet mean/std.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(cfg.image_size * 1.14)),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])


def make_loader(samples: list, split: str, shuffle: bool = False) -> DataLoader:
    """Wrap a sample list in a birscsvdata and return a DataLoader."""
    ds = birdcsvdata(samples, transform=get_transforms(split))
    return DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=False)



class BirdEmbeddingModel(nn.Module):
    """
    EfficientNetB0 backbone with a learned embedding projection head.

    Architecture:
        EfficientNetB0 features  (pretrained on ImageNet)
            ↓
        Global Average Pool  → 1280-dim vector
            ↓
        Linear projection    → embedding-dim vector
            ↓
        L2 normalization     → unit-length embedding
    """

    def __init__(self, embedding_dim: int = None, pretrained: bool = True):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = cfg.embedding_dim
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.efficientnet_b0(weights=weights)
        self.backbone      = base.features
        self.pool          = base.avgpool
        self.embedding     = nn.Linear(1280, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = self.extract_features(x)
        emb = self.embedding(emb)
        emb = nn.functional.normalize(emb, dim=1)
        return emb

    def extract_features(self, x):
        return self.pool(self.backbone(x)).flatten(1)


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss.

    Adds an angular margin to the target class angle, forcing embeddings of
    the same species to cluster much more tightly than plain cross-entropy.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.7):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weights)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        weight = nn.functional.normalize(self.weights, dim=1)
        cos_theta = torch.matmul(embeddings, weight.t())
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits  = one_hot * target_logits + (1 - one_hot) * cos_theta
        logits *= self.s
        return nn.functional.cross_entropy(logits, labels)



def compute_centroids(embeddings: np.ndarray, labels: np.ndarray,
                      num_classes: int) -> np.ndarray:
    """Compute one centroid (mean embedding position) per class."""
    return np.vstack([
        embeddings[labels == c].mean(axis=0)
        for c in range(num_classes)
    ])


def compute_covariances(embeddings: np.ndarray, labels: np.ndarray,
                        num_classes: int, reg: float = 1e-5) -> np.ndarray:
    """
    Computes per-class covariance matrix for each class in a batch.
    """
    dim = embeddings.shape[1]
    covs = np.zeros((num_classes, dim, dim), dtype=np.float64)
    for c in range(num_classes):
        X = embeddings[labels == c]
        cov = np.cov(X, rowvar=False) if len(X) > 1 else np.eye(dim)
        covs[c] = cov + reg * np.eye(dim)
    return covs


def _mahalanobis_to_class(embeddings: np.ndarray, centroid: np.ndarray,
                           cov: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from every embedding to one class centroid."""
    inv = np.linalg.inv(cov)
    diff = embeddings - centroid
    return np.sqrt(np.einsum("nd,dd,nd->n", diff, inv, diff).clip(0))


def min_centroid_distances(embeddings: np.ndarray, centroids: np.ndarray,
                           covariances: np.ndarray = None) -> np.ndarray:
    """
    Mahalanobis distance to the nearest class centroid.
    Falls back to Euclidean if covariances is None (backwards compat).
    """
    if covariances is None:
        dists = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :], axis=2)
        return np.min(dists, axis=1)

    dists = np.stack([
        _mahalanobis_to_class(embeddings, centroids[c], covariances[c])
        for c in range(len(centroids))
    ], axis=1)
    return np.min(dists, axis=1)


def compute_distance_threshold(embeddings: np.ndarray, labels: np.ndarray,
                                centroids: np.ndarray,
                                covariances: np.ndarray = None) -> float:
    """Threshold at cfg.percentile_of_threshold of training Mahalanobis distances."""
    if covariances is None:
        own_centroids = centroids[labels]
        distances = np.linalg.norm(embeddings - own_centroids, axis=1)
    else:
        distances = np.concatenate([
            _mahalanobis_to_class(embeddings[labels == c],
                                   centroids[c], covariances[c])
            for c in range(len(centroids))
            if (labels == c).any()
        ])
    return np.float64(np.percentile(distances, cfg.percentile_of_threshold))


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
    """
    Fine-tune BirdEmbeddingModel on the known species using ArcFace loss.
    Saves the best checkpoint to checkpoints/best_model.pt.
    """
    device = cfg.device()
    print("\n── Step 1: Training BirdEmbeddingModel with ArcFace loss ──")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {cfg.epoches}")
    print(f"  Batch size : {cfg.batch}")
    print(f"  ArcFace s  : {cfg.arcface_scaler}  m: {cfg.arcface_margin}")

    train_loader = make_loader(train_samples, "train", shuffle=True)
    val_loader   = make_loader(val_samples,   "val")

    model     = BirdEmbeddingModel().to(device)
    criterion = ArcFaceLoss(
        cfg.embedding_dim, cfg.number_of_classes,
        s=cfg.arcface_scaler, m=cfg.arcface_margin).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": cfg.learning_rate * 0.1},
        {"params": model.embedding.parameters(), "lr": cfg.learning_rate}
    ], weight_decay=cfg.weight_decay)

    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
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
            total      += imgs.size(0)
            epoch_embs.append(embs.detach().cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())

        model.eval()
        cached_train_embs   = np.vstack(epoch_embs)
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



def load_model() -> BirdEmbeddingModel:
    """Load the best saved model weights from disk, with dim validation."""
    device     = cfg.device()
    checkpoint = os.path.join(cfg.checkpoint_directory, "best_model.pt")
    state_dict = torch.load(checkpoint, map_location=device)

    saved_dim = state_dict["embedding.weight"].shape[0]
    if saved_dim != cfg.embedding_dim:
        raise ValueError(
            f"Checkpoint embedding_dim={saved_dim} does not match "
            f"cfg.EMBEDDING_DIM={cfg.embedding_dim}. Either retrain the model "
            f"or update embedding_dim in config.yaml to match the checkpoint."
        )

    model = BirdEmbeddingModel(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader) -> tuple:
    """Pass all images through the model and collect (embeddings, labels)."""
    device = cfg.device()
    all_embs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  Extracting embeddings", leave=False):
        embs = model(imgs.to(device)).cpu().numpy()
        all_embs.append(embs)
        all_labels.append(labels.numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)



def evaluate_centroid_detector(model, test_known: list, test_outlier: list):
    """Evaluate the centroid-distance detector on the test set."""
    print("\n── Evaluating centroid detector ──")

    centroids   = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    threshold   = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))

    Xk, _ = extract_embeddings(model, make_loader(test_known,   "val"))
    Xo, _ = extract_embeddings(model, make_loader(test_outlier, "val"))

    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    tp = (dk <= threshold).sum()
    fn = (dk >  threshold).sum()
    tn = (do >  threshold).sum()
    fp = (do <= threshold).sum()

    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    scores  = np.concatenate([-dk, -do])
    y_true  = np.concatenate([np.ones(len(dk)), np.zeros(len(do))])
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr  = average_precision_score(y_true, scores)

    print(f"\n  Threshold (percentile={cfg.percentile_of_threshold}): {threshold:.4f}")
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print(f"\n  {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("  " + "-" * 50)
    print(f"  {tp:>5} {fn:>5} {tn:>5} {fp:>5} "
          f"{rec:>8.3f} {prec:>10.3f} {f1:>6.3f}")
    print("\n  TP = known correctly accepted  | FN = known wrongly rejected")
    print("  TN = outliers correctly blocked | FP = outliers wrongly accepted")

    results = {
        "AUC-ROC":    round(auc_roc, 4),
        "AUC-PR":     round(auc_pr,  4),
        "Recall":     round(float(rec),  4),
        "Precision":  round(float(prec), 4),
        "F1":         round(float(f1),   4),
        "threshold":  round(threshold,   6),
        "percentile": cfg.percentile_of_threshold
    }
    with open(os.path.join(cfg.results_directory, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Metrics saved to {cfg.results_directory}/metrics.json")

    return results




class BirdAnomalyDetector:
    """
    Ready-to-use inference interface — loads all saved artifacts from disk
    and scores individual images.

    Usage:
        detector = BirdAnomalyDetector()
        result   = detector.predict(r"D:\\path\\to\\image.jpg")
    """
    def __init__(self, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            checkpoint_dir = cfg.checkpoint_directory

        self.transform          = get_transforms("val")
        self.device             = cfg.device()
        self.classes            = names
        self.model              = load_model()
        self.centroids          = np.load(
            os.path.join(checkpoint_dir, "centroids.npy"))
        self.covariances        = np.load(
            os.path.join(checkpoint_dir, "covariances.npy"))
        self.centroid_threshold = float(np.load(
            os.path.join(checkpoint_dir, "centroid_threshold.npy")))
        self._inv_covs = np.stack([
            np.linalg.inv(self.covariances[c])
            for c in range(len(self.centroids))
        ])

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """Score a single image and return classification + outlier decision."""
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}

        img    = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb    = self.model(tensor).cpu().numpy()[0]

        dists = np.array([
            float(np.sqrt((diff := emb - self.centroids[c]) @ self._inv_covs[c] @ diff))
            for c in range(len(self.centroids))
        ])
        min_idx  = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        return {
            "predicted_class": self.classes[min_idx],
            "distance":        round(min_dist, 4),
            "threshold":       round(self.centroid_threshold, 4),
            "is_outlier":      min_dist > self.centroid_threshold,
        }




def run_full_pipeline(skip_training: bool = False):
    cfg.make_dirs()

    print("=" * 52)
    print("  Bird Anomaly Detection — Full Pipeline")
    print("=" * 52)

    train_samples, val_samples, test_known, test_outlier = build_splits()

    with open(os.path.join(cfg.checkpoint_directory, "classes.json"), "w") as f:
        json.dump(names, f, indent=2)

    if skip_training:
        print("\n── Step 1: Skipped (using existing checkpoint) ──")
    else:
        train_model(train_samples, val_samples)

    print("\n── Step 2: Extracting training embeddings ──")
    model        = load_model()
    train_loader = make_loader(train_samples, "val")
    train_embs, train_labels = extract_embeddings(model, train_loader)

    np.save(os.path.join(cfg.checkpoint_directory, "train_embeddings.npy"), train_embs)
    print(f"  Embeddings shape: {train_embs.shape}")

    centroids = compute_centroids(train_embs, train_labels, cfg.number_of_classes)
    np.save(os.path.join(cfg.checkpoint_directory, "centroids.npy"), centroids)
    print(f"  Centroids saved: {centroids.shape}")

    covariances = compute_covariances(train_embs, train_labels, cfg.number_of_classes)
    np.save(os.path.join(cfg.checkpoint_directory, "covariances.npy"), covariances)
    print(f"  Covariances saved: {covariances.shape}")

    threshold = compute_distance_threshold(train_embs, train_labels, centroids, covariances)
    np.save(os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy"), threshold)
    print(f"  Threshold ({cfg.percentile_of_threshold}th percentile): {threshold:.4f}")

    results = evaluate_centroid_detector(model, test_known, test_outlier)

    print("\n── Final Metrics ──")
    print(json.dumps(results, indent=2))
    print("\n Pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Bird anomaly detection pipeline")
    parser.add_argument("--skip-training", action="store_true",
        help="Skip training and recompute centroids + threshold only.")
    parser.add_argument("--data-root", default=None,
        help="Override data.data_root in config.yaml.")
    parser.add_argument("--config", default=None,
        help="Path to config.yaml (default: ./config.yaml).")
    args = parser.parse_args()


    # CLI --data-root takes final priority over config.yaml
    if args.data_root:
        cfg.data_root = args.data_root

    run_full_pipeline(skip_training=args.skip_training)
