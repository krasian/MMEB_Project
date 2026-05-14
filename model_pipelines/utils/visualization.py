"""
Visualization utilities for both visual and audio bird species pipelines.

Visual pipeline functions (centroid distance + Mahalanobis):
  - plot_roc_pr
  - plot_distance_distribution
  - plot_per_species_distributions
  - plot_embedding_space

Audio pipeline functions (prototypical probing):
  - plot_training_curves
  - plot_confusion_matrix
  - plot_roc_curve
  - plot_tsne

All plots save to file via the Agg backend — no GUI windows pop up.
"""
import os
import numpy as np

# Force non-interactive backend BEFORE pyplot is imported — never opens windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional dependency: only used by audio plotting helpers.
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              confusion_matrix)
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Visual-pipeline imports — guarded so audio-only setups still load.
try:
    try:
        from model_pipelines.config import cfg, names, outlier_names
        from model_pipelines.outlier.mahalanobis import min_centroid_distances
        from model_pipelines.utils.metrics import build_palette
        from model_pipelines.data.visual_dataset import load_csv_paths, make_loader
        from model_pipelines.utils.embeddings import extract_embeddings
    except ImportError:
        from config import cfg, names, outlier_names
        from outlier.mahalanobis import min_centroid_distances
        from utils.metrics import build_palette
        from data.visual_dataset import load_csv_paths, make_loader
        from utils.embeddings import extract_embeddings
    _VISUAL_OK = True
except ImportError:
    _VISUAL_OK = False
    cfg = None
    names = []
    outlier_names = []


N_KNOWN = len(names) if _VISUAL_OK else 0
PALETTE = build_palette() if _VISUAL_OK else {}


# ═════════════════════════════════════════════
# AUDIO HELPERS
# ═════════════════════════════════════════════

def set_plot_style():
    """Set a clean, readable plot style. Used by audio plotting helpers."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        pass
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_training_curves(history: Dict[str, List[float]],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 4)):
    """Plot training/validation loss + validation accuracy curves."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], 'g-',
                     label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved training curves to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str],
                           title: str = "Confusion Matrix",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           normalize: bool = True):
    """Plot a confusion matrix heatmap."""
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt, cbar_label = '.2f', 'Proportion'
    else:
        fmt, cbar_label = 'd', 'Count'

    fig, ax = plt.subplots(figsize=figsize)
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt=fmt,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', ax=ax, cbar_kws={'label': cbar_label})
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray,
                    title: str = "ROC Curve - Outlier Detection",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (8, 6)) -> float:
    """Plot ROC curve from confidence scores. Returns the AUC."""
    set_plot_style()
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved ROC curve to {save_path}")
    plt.close(fig)
    return float(auc)


def plot_tsne(features: np.ndarray, labels: np.ndarray,
               class_names: List[str],
               title: str = "t-SNE Visualization",
               save_path: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 8),
               perplexity: int = 30):
    """Visualise feature vectors via t-SNE projection coloured by class."""
    set_plot_style()
    print("  Computing t-SNE projection...")
    perplexity = min(perplexity, len(features) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    projections = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(unique_labels))
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            projections[mask, 0], projections[mask, 1],
            c=[colors[i]],
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.7, s=50,
        )

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved t-SNE visualization to {save_path}")
    plt.close(fig)


# ═════════════════════════════════════════════
# VISUAL HELPERS
# ═════════════════════════════════════════════

def plot_roc_pr(Xk: np.ndarray, Xo: np.ndarray,
                centroids: np.ndarray, covariances: np.ndarray,
                save_path: str):
    """ROC and PR curves for the centroid detector (all outliers pooled)."""
    if not _VISUAL_OK:
        raise ImportError("Visual pipeline modules unavailable")

    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)
    scores = np.concatenate([-dk, -do])
    y_true = np.concatenate([np.ones(len(dk)), np.zeros(len(do))])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if len(outlier_names) == 1:
        roc_title = f"ROC — Known species vs {outlier_names[0]}"
    else:
        roc_title = (f"ROC — Known species vs outliers "
                     f"({len(outlier_names)} species pooled)")

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    axes[0].plot(fpr, tpr, lw=2, color="#4C8BF5",
                 label=f"Centroid detector (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random")
    axes[0].set_title(roc_title)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, scores)
    axes[1].plot(rec, prec, lw=2, color="#4C8BF5", label="Centroid detector")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.result_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ ROC/PR curves → {save_path}")


def plot_distance_distribution(Xk: np.ndarray, Xo: np.ndarray,
                                centroids: np.ndarray, covariances: np.ndarray,
                                threshold: float, save_path: str):
    """Histogram of centroid distances: known species vs outliers pooled."""
    if not _VISUAL_OK:
        raise ImportError("Visual pipeline modules unavailable")

    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    bins = np.linspace(min(dk.min(), do.min()),
                       max(dk.max(), do.max()), 60)

    known_label = f"Known ({len(names)} species)"
    if len(outlier_names) == 1:
        outlier_label = outlier_names[0]
    else:
        outlier_label = f"Outliers ({len(outlier_names)} species pooled)"

    fig = plt.figure(figsize=(8, 5))
    plt.hist(dk, bins=bins, color="#4C8BF5", alpha=0.65, label=known_label)
    plt.hist(do, bins=bins, color="#E74C3C", alpha=0.65, label=outlier_label)
    plt.axvline(threshold, color="black", ls="--", lw=2,
                label=f"Threshold ({threshold:.3f}, p={cfg.percentile_of_threshold})")
    plt.title("Centroid Distance Distribution")
    plt.xlabel("Distance to nearest centroid  (lower = more like a known species)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.result_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distance distribution → {save_path}")


def plot_per_species_distributions(model, Xk: np.ndarray,
                                    centroids: np.ndarray,
                                    covariances: np.ndarray,
                                    threshold: float, save_path: str):
    """One distance histogram per outlier species — shows difficulty spectrum."""
    if not _VISUAL_OK:
        raise ImportError("Visual pipeline modules unavailable")

    dk = min_centroid_distances(Xk, centroids, covariances)

    all_distances = [dk]
    species_distances = {}

    for species_name, csv_file in cfg.outlier_csv.items():
        samples = load_csv_paths(csv_file, label=-1)
        if not samples:
            continue
        Xo_sp, _ = extract_embeddings(model, make_loader(samples, "val"))
        do_sp = min_centroid_distances(Xo_sp, centroids, covariances)
        species_distances[species_name] = do_sp
        all_distances.append(do_sp)

    if not species_distances:
        print("  ⚠ No outlier species found for per-species plot, skipping.")
        return

    all_concat = np.concatenate(all_distances)
    bins = np.linspace(all_concat.min(), all_concat.max(), 60)

    n_species = len(species_distances)
    ncols = min(3, n_species)
    nrows = (n_species + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows),
                              sharey=False)
    if n_species == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (species_name, do_sp) in zip(axes, species_distances.items()):
        ax.hist(dk, bins=bins, color="#4C8BF5", alpha=0.50,
                label=f"Known ({len(names)} sp.)")
        color = PALETTE.get(species_name, "#E74C3C")
        ax.hist(do_sp, bins=bins, color=color, alpha=0.70, label=species_name)
        ax.axvline(threshold, color="black", ls="--", lw=1.5,
                   label=f"Threshold ({threshold:.2f})")

        recall = (do_sp > threshold).sum() / len(do_sp) if len(do_sp) > 0 else 0.0
        ax.set_title(f"{species_name}\n(outlier recall = {recall:.1%})", fontsize=11)
        ax.set_xlabel("Distance to nearest centroid")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_species:]:
        ax.set_visible(False)

    fig.suptitle("Per-Species Distance Distributions vs Known Species",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.result_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Per-species distributions → {save_path}")


def plot_embedding_space(all_embs: np.ndarray, all_labels: np.ndarray,
                          save_path: str, method: str = "tsne",
                          max_per_class: int = 300):
    """
    2-D t-SNE/UMAP projection of test embeddings coloured by species.

    Subsamples up to `max_per_class` points per class so the plot stays
    readable. Outliers are drawn as a faint background layer; knowns are
    drawn on top with full opacity so the clusters actually show up.

    ARGUMENTS:
        all_embs:       [N, D] embedding matrix
        all_labels:     [N]    integer labels (0..N_KNOWN-1 = known,
                               N_KNOWN.. = outlier)
        save_path:      where to write the PNG
        method:         "tsne" or "umap"
        max_per_class:  cap on points per class for plotting
                        (does NOT affect the trained model, only the plot)
    """
    if not _VISUAL_OK:
        raise ImportError("Visual pipeline modules unavailable")

    label_names = names + outlier_names

    # ── Subsample, evenly across all classes ─────────────────────
    # The actual t-SNE projection only sees the subsampled points -- this
    # is also way faster than running t-SNE on tens of thousands of points.
    rng = np.random.default_rng(seed=42)
    keep_idx_list = []
    for i in range(len(label_names)):
        class_idx = np.where(all_labels == i)[0]
        if len(class_idx) == 0:
            continue
        if len(class_idx) > max_per_class:
            class_idx = rng.choice(class_idx, size=max_per_class, replace=False)
        keep_idx_list.append(class_idx)

    if not keep_idx_list:
        print("  ⚠ No points to plot, skipping embedding space figure.")
        return

    keep_idx = np.concatenate(keep_idx_list)
    sub_embs = all_embs[keep_idx]
    sub_labels = all_labels[keep_idx]

    print(f"  Running {method.upper()} on {len(sub_embs)} samples "
          f"(subsampled from {len(all_embs)}, up to {max_per_class}/class)...")

    if method == "umap" and HAS_UMAP:
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(sub_embs)
    else:
        if method == "umap":
            print("  umap-learn not installed, falling back to t-SNE")
        # Perplexity should be < N_samples; cap it to be safe.
        perp = min(30, max(5, len(sub_embs) // 4))
        proj = TSNE(n_components=2, random_state=42,
                    perplexity=perp, max_iter=1000).fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(11, 8))

    # ── Draw outliers FIRST (background layer) ───────────────────
    # Small, faint, and behind everything else so they don't visually
    # drown out the known clusters.
    for i, name in enumerate(label_names):
        if i < N_KNOWN:
            continue
        mask = sub_labels == i
        if not mask.any():
            continue
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=PALETTE.get(name, "#888888"),
            s=10,
            marker="x",
            alpha=0.25,
            label=name,
            linewidths=0.7,
            zorder=1,
        )

    # ── Draw knowns ON TOP (foreground layer) ────────────────────
    # Larger, more opaque, with an edge so even single points stand out.
    for i, name in enumerate(label_names):
        if i >= N_KNOWN:
            continue
        mask = sub_labels == i
        if not mask.any():
            continue
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=PALETTE.get(name, "#888888"),
            s=28,
            marker="o",
            alpha=0.85,
            label=name,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )

    title = (f"Embedding space ({method.upper()}) — "
             f"{len(names)} known + {len(outlier_names)} outlier species")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=10, markerscale=1.6,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.result_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Embedding space → {save_path}")


# ═════════════════════════════════════════════
# Demo block — only runs when executed directly, not on import.
# This is the fix for the popup figure that appeared every time.
# ═════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Visualization Functions")
    print("=" * 60)

    history = {
        'train_loss':   [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss':     [0.9, 0.7, 0.55, 0.45, 0.35],
        'val_accuracy': [0.6, 0.7, 0.75, 0.8, 0.82],
    }

    print("\n1. Testing training curves...")
    plot_training_curves(history, save_path="test_training_curves.png")

    print("\n✓ All visualization functions are ready!")