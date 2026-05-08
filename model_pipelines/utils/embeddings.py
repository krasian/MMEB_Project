"""
Utilities for extracting and persisting model embeddings.
"""
import os
import numpy as np
import torch
from tqdm import tqdm

from config import cfg


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


def save_embeddings(embeddings: np.ndarray, filename: str,
                    directory: str = None) -> str:
    """Save an embeddings array to a .npy file. Returns the full path."""
    if directory is None:
        directory = cfg.checkpoint_directory
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    np.save(path, embeddings)
    return path


def load_embeddings(filename: str, directory: str = None) -> np.ndarray:
    """Load an embeddings array from a .npy file."""
    if directory is None:
        directory = cfg.checkpoint_directory
    return np.load(os.path.join(directory, filename))
