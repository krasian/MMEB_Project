"""
Plotting helpers for the visual pipeline:
  - ROC/PR curves
  - centroid-distance histograms
  - 2-D embedding-space projections (t-SNE / UMAP)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from config import cfg, names, outlier_names
from outlier.mahalanobis import min_centroid_distances
from utils.metrics import build_palette


N_KNOWN = len(names)
PALETTE = build_palette()


def plot_roc_pr(Xk: np.ndarray, Xo: np.ndarray,
                centroids: np.ndarray, covariances: np.ndarray,
                save_path: str):
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
    plt.close()
    print(f"  ✓ ROC/PR curves → {save_path}")


def plot_distance_distribution(Xk: np.ndarray, Xo: np.ndarray,
                                centroids: np.ndarray, covariances: np.ndarray,
                                threshold: float, save_path: str):
    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    bins = np.linspace(min(dk.min(), do.min()),
                       max(dk.max(), do.max()), 60)

    known_label = f"Known ({len(names)} species)"
    if len(outlier_names) == 1:
        outlier_label = outlier_names[0]
    else:
        outlier_label = f"Outliers ({len(outlier_names)} species pooled)"

    plt.figure(figsize=(8, 5))
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
    plt.close()
    print(f"  ✓ Distance distribution → {save_path}")


def plot_embedding_space(all_embs: np.ndarray, all_labels: np.ndarray,
                         save_path: str, method: str = "tsne"):
    """
    2-D projection of all test embeddings coloured by species.
    Known species drawn as circles; outliers drawn as x's.
    """
    label_names = names + outlier_names

    print(f"  Running {method.upper()} on {len(all_embs)} samples ...")

    if method == "umap" and HAS_UMAP:
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(all_embs)
    else:
        if method == "umap":
            print("  umap-learn not installed, falling back to t-SNE")
        proj = TSNE(n_components=2, random_state=42,
                    perplexity=30, max_iter=1000).fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(label_names):
        mask = all_labels == i
        if not mask.any():
            continue
        is_outlier = i >= N_KNOWN
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=PALETTE.get(name, "#888888"),
            s=30 if is_outlier else 15,
            marker="x" if is_outlier else "o",
            alpha=0.6,
            label=name,
        )

    title = (f"Embedding space ({method.upper()}) — "
             f"{len(names)} known + {len(outlier_names)} outlier species")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=11, markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.result_dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Embedding space → {save_path}")
