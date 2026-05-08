"""utils package — embeddings, metrics, and visualization helpers."""
from .embeddings import extract_embeddings, save_embeddings, load_embeddings
from .metrics import build_palette, confusion_counts, precision_recall_f1
from .visualization import (
    plot_roc_pr,
    plot_distance_distribution,
    plot_embedding_space,
)

__all__ = [
    "extract_embeddings",
    "save_embeddings",
    "load_embeddings",
    "build_palette",
    "confusion_counts",
    "precision_recall_f1",
    "plot_roc_pr",
    "plot_distance_distribution",
    "plot_embedding_space",
]
