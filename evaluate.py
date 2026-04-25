import os
import json
import argparse
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

from pipeline import (
    cfg, names, outlier_names, build_splits,
    load_model, extract_embeddings, make_loader,
    min_centroid_distances
)

N_KNOWN = len(names)




def build_palette() -> dict:
    """
    Build a colour palette covering every known and outlier species.
    Known species get qualitative colours; outlier species get warm/red
    tones so they stand out on plots.
    """
    known_colors = [
        "#2C3E50", "#4C8BF5", "#F5A623", "#27AE60",
        "#8E44AD", "#16A085", "#D35400", "#2980B9",
        "#C0392B", "#7F8C8D", "#F39C12", "#1ABC9C"
    ]
    outlier_colors = [
        "#E74C3C", "#E91E63", "#FF5722", "#AD1457",
        "#BF360C", "#880E4F"
    ]

    palette = {}
    for i, name in enumerate(names):
        palette[name] = known_colors[i % len(known_colors)]
    for i, name in enumerate(outlier_names):
        palette[name] = outlier_colors[i % len(outlier_colors)]
    return palette


PALETTE = build_palette()


def load_artifacts() -> tuple:
    centroids   = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    threshold   = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))
    return centroids, covariances, threshold


def get_test_embeddings(model) -> tuple:
    """
    Extract embeddings for all test images (known + outlier species).

    Returns:
        Xk         — (N_known, D)   embeddings for known species
        Xo         — (N_outlier, D) embeddings for all outlier species pooled
        all_embs   — Xk and Xo stacked vertically (used for t-SNE)
        all_labels — integer labels: 0..N_KNOWN-1 = known, N_KNOWN.. = outlier
    """
    _, _, test_known, test_outlier = build_splits()

    Xk, known_labels = extract_embeddings(model, make_loader(test_known,   "val"))
    Xo, raw_outlier  = extract_embeddings(model, make_loader(test_outlier, "val"))

    outlier_labels = N_KNOWN + (-raw_outlier - 1)

    all_embs   = np.vstack([Xk, Xo])
    all_labels = np.concatenate([known_labels, outlier_labels])

    return Xk, Xo, all_embs, all_labels



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


def print_summary_table(Xk: np.ndarray, Xo: np.ndarray,
                         centroids: np.ndarray, covariances: np.ndarray,
                         threshold: float):
    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    tp   = (dk <= threshold).sum()
    fn   = (dk >  threshold).sum()
    tn   = (do >  threshold).sum()
    fp   = (do <= threshold).sum()
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print("\n── Detection Summary ──")
    print(f"  Threshold : {threshold:.4f}  "
          f"(percentile={cfg.percentile_of_threshold})")
    print(f"\n  {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("  " + "-" * 50)
    print(f"  {tp:>5} {fn:>5} {tn:>5} {fp:>5} "
          f"{rec:>8.3f} {prec:>10.3f} {f1:>6.3f}")
    print("\n  TP = known correctly accepted  | FN = known wrongly rejected")
    print("  TN = outliers correctly blocked | FP = outliers wrongly accepted")



def run_evaluation():
    cfg.make_dirs()

    print("── Loading artifacts ──")
    centroids, covariances, threshold = load_artifacts()
    model = load_model()

    print("── Extracting test embeddings ──")
    Xk, Xo, all_embs, all_labels = get_test_embeddings(model)

    plot_roc_pr(
        Xk, Xo, centroids, covariances,
        os.path.join(cfg.results_directory, "centroid_roc_pr.png"))

    plot_distance_distribution(
        Xk, Xo, centroids, covariances, threshold,
        os.path.join(cfg.results_directory, "centroid_distribution.png"))

    plot_embedding_space(
        all_embs, all_labels,
        os.path.join(cfg.results_directory, "embedding_space.png"),
        method=cfg.embeding_visulize_method)

    print_summary_table(Xk, Xo, centroids, covariances, threshold)

    metrics_path = os.path.join(cfg.results_directory, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            print("\n── Saved metrics ──")
            print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird anomaly detection — evaluation")
    parser.add_argument("--config", default=None,
        help="Path to config.yaml (default: ./config.yaml).")
    parser.add_argument("--data-root", default=None,
        help="Override data.data_root in config.yaml.")
    args = parser.parse_args()

    from load_config import apply_yaml_config
    apply_yaml_config(args.config)

    if args.data_root:
        cfg.data_root = args.data_root

    run_evaluation()
