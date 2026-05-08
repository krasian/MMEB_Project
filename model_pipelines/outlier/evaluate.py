"""
Numeric evaluation of the centroid-based outlier detector.

This file handles the metrics side: TP/FP/TN/FN counts, AUC-ROC, AUC-PR,
F1, and a JSON dump of the results. Plotting (ROC curves, distance
histograms, embedding-space scatter) lives in utils/visualization.py.
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from config import cfg
from utils.embeddings import extract_embeddings
from data.visual_dataset import make_loader
from .mahalanobis import min_centroid_distances


def evaluate_centroid_detector(model, test_known: list, test_outlier: list):
    """Evaluate the centroid-distance detector on the test set."""
    print("\n── Evaluating centroid detector ──")

    centroids = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    threshold = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))

    Xk, _ = extract_embeddings(model, make_loader(test_known, "val"))
    Xo, _ = extract_embeddings(model, make_loader(test_outlier, "val"))

    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    tp = (dk <= threshold).sum()
    fn = (dk > threshold).sum()
    tn = (do > threshold).sum()
    fp = (do <= threshold).sum()

    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    scores = np.concatenate([-dk, -do])
    y_true = np.concatenate([np.ones(len(dk)), np.zeros(len(do))])
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

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
        "AUC-ROC": round(auc_roc, 4),
        "AUC-PR": round(auc_pr, 4),
        "Recall": round(float(rec), 4),
        "Precision": round(float(prec), 4),
        "F1": round(float(f1), 4),
        "threshold": round(threshold, 6),
        "percentile": cfg.percentile_of_threshold,
    }
    with open(os.path.join(cfg.results_directory, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Metrics saved to {cfg.results_directory}/metrics.json")

    return results


def print_summary_table(Xk: np.ndarray, Xo: np.ndarray,
                        centroids: np.ndarray, covariances: np.ndarray,
                        threshold: float):
    """Compact TP/FN/TN/FP/precision/recall/F1 summary printed to stdout."""
    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)

    tp = (dk <= threshold).sum()
    fn = (dk > threshold).sum()
    tn = (do > threshold).sum()
    fp = (do <= threshold).sum()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

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


def load_artifacts() -> tuple:
        """
    Load saved centroids, threshold, and (for Mahalanobis) covariances.
    Returns (centroids, covariances_or_None, threshold).
    """
    centroids = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    if cfg.distance_metric == "mahalanobis":
        covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    else:
        covariances = None
    threshold = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))
    return centroids, covariances, threshold
