"""
Visual dataset utilities: read iNaturalist CSVs, split into train/val/test,
apply image transforms, and build PyTorch DataLoaders.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from model_pipelines.config import cfg, class_to_id
except ImportError:
    from config import cfg, class_to_id

logger = logging.getLogger(__name__)


class BirdCSVDataset(Dataset):
    """
    PyTorch Dataset that reads images from a list of (path, label) pairs.

    Args:
        samples:   list of (image_path: str, label: int) tuples.
                   Use label < 0 for outlier images (ignored during training).
        transform: torchvision transform to apply to each image.
    """

    def __init__(self, samples: list, transform=None):
        self.samples = samples
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


# Backwards-compat alias for old code that used the original lowercase name.
birdcsvdata = BirdCSVDataset


def load_csv_paths(csv_filename: str, label: int) -> list:
    """Read a species CSV and return a list of (image_path, label) tuples."""
    csv_path = os.path.join(cfg.data_root, csv_filename)
    df = pd.read_csv(csv_path)
    return [(str(p), label) for p in df["image_path"].dropna()]


def split_samples(samples: list, train_r: float, val_r: float,
                  seed: int = 42) -> tuple:
    """Reproducibly shuffle and split a sample list into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_train = int(len(samples) * train_r)
    n_val = int(len(samples) * val_r)

    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train:n_train + n_val]]
    test = [samples[i] for i in idx[n_train + n_val:]]
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
        label = class_to_id[species_name]
        samples = load_csv_paths(csv_file, label)
        tr, va, te = split_samples(
            samples, cfg.train_split_ratio, cfg.validation_split_ratio)
        train_all += tr
        val_all += va
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
    Image pre-processing pipeline for training or evaluation.
    Training uses random augmentation; val/test uses deterministic resize + crop.
    Both normalise to ImageNet mean/std.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(cfg.image_size * 1.14)),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def make_loader(samples: list, split: str, shuffle: bool = False) -> DataLoader:
    """Wrap a sample list in a BirdCSVDataset and return a DataLoader."""
    ds = BirdCSVDataset(samples, transform=get_transforms(split))
    return DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=False,
    )
