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

from data import load_csv_paths


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _binary_metrics(dk, do, threshold):
    """
    Given distances for known (dk) and outlier (do) samples and a threshold,
    return a dict with TP, FN, TN, FP, Recall, Precision, F1, AUC-ROC, AUC-PR.

    Convention:
        distance <= threshold → accepted as known  (TP if truly known, FP if outlier)
        distance >  threshold → flagged as outlier  (TN if truly outlier, FN if known)
    """
    tp = int((dk <= threshold).sum())
    fn = int((dk >  threshold).sum())
    tn = int((do >  threshold).sum())
    fp = int((do <= threshold).sum())

    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    scores = np.concatenate([-dk, -do])   # higher = more normal
    y_true = np.concatenate([np.ones(len(dk)), np.zeros(len(do))])

    auc_roc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr  = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "Recall":    round(rec,     4),
        "Precision": round(prec,    4),
        "F1":        round(f1,      4),
        "AUC-ROC":   round(auc_roc, 4),
        "AUC-PR":    round(auc_pr,  4),
    }


def _print_metrics_row(label, m, threshold):
    """Print one row of the detection summary table."""
    print(f"\n  [{label}]")
    print(f"    TP={m['TP']}  FN={m['FN']}  TN={m['TN']}  FP={m['FP']}")
    print(f"    Recall={m['Recall']:.3f}  Precision={m['Precision']:.3f}  "
          f"F1={m['F1']:.3f}  AUC-ROC={m['AUC-ROC']:.4f}")


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate_centroid_detector(model, test_known: list, test_outlier: list):
    """
    Evaluate the centroid-distance detector on the full test set.

    Computes:
      - Overall metrics (all outlier species pooled)
      - Per-species outlier metrics (one row per outlier species)

    test_outlier is expected as a flat list of (path, label) samples — the
    per-species breakdown is read directly from cfg.outlier_csvs so that
    species names are preserved.
    """
    print("\n── Evaluating centroid detector ──")

    centroids   = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    threshold   = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))

    # ── Known species embeddings ──
    Xk, _ = extract_embeddings(model, make_loader(test_known, "val"))
    dk     = min_centroid_distances(Xk, centroids, covariances)

    # ── All outliers pooled ──
    Xo, _ = extract_embeddings(model, make_loader(test_outlier, "val"))
    do     = min_centroid_distances(Xo, centroids, covariances)

    # ── Overall metrics ──
    overall = _binary_metrics(dk, do, threshold)
    overall["threshold"] = round(threshold, 6)
    overall["percentile"] = cfg.percentile_of_threshold

    print(f"\n  Threshold (percentile={cfg.percentile_of_threshold}): {threshold:.4f}")
    print(f"  AUC-ROC : {overall['AUC-ROC']:.4f}")
    print(f"  AUC-PR  : {overall['AUC-PR']:.4f}")
    print(f"\n  {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("  " + "-" * 50)
    print(f"  {overall['TP']:>5} {overall['FN']:>5} "
          f"{overall['TN']:>5} {overall['FP']:>5} "
          f"{overall['Recall']:>8.3f} {overall['Precision']:>10.3f} "
          f"{overall['F1']:>6.3f}")
    print("\n  TP = known correctly accepted  | FN = known wrongly rejected")
    print("  TN = outliers correctly blocked | FP = outliers wrongly accepted")

    # ── Per-species outlier metrics ──
    per_species = _evaluate_per_species(model, Xk, centroids, covariances, threshold)

    # ── Save results ──
    results = {
        "overall":     overall,
        "per_species": per_species,
    }
    os.makedirs(cfg.results_directory, exist_ok=True)
    with open(os.path.join(cfg.results_directory, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Metrics saved to {cfg.results_directory}/metrics.json")

    return results


def _evaluate_per_species(model, Xk, centroids, covariances, threshold):
    """
    For each outlier species defined in cfg.outlier_csv, load its test
    images, extract embeddings, and compute detection metrics independently.

    Args:
        model:       trained BirdEmbeddingModel
        Xk:          (N, D) raw known-species test embeddings (NOT distances)
        centroids:   (C, D) class centroids
        covariances: per-class covariance matrices (or None for Euclidean)
        threshold:   calibrated distance threshold

    This lets us see which species are easy/hard to detect as outliers.
    AUC-ROC = 1.0 means perfect separation from known species.
    AUC-ROC = 0.5 means random — indistinguishable from known species.
    """
    # Compute known-species distances once — used for AUC-ROC per species
    dk = min_centroid_distances(Xk, centroids, covariances).flatten()

    print("\n── Per-species outlier breakdown ──")
    print(f"  {'Species':<22} {'N':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall(outlier)':>16} {'AUC-ROC':>8}")
    print("  " + "-" * 65)

    per_species = {}

    for species_name, csv_file in cfg.outlier_csv.items():
        # Load this species' test samples (label=-1, not used for metrics)
        samples = load_csv_paths(csv_file, label=-1)

        if not samples:
            print(f"  {species_name:<22} — no samples found, skipping")
            continue

        Xo_sp, _ = extract_embeddings(model, make_loader(samples, "val"))
        do_sp    = min_centroid_distances(Xo_sp, centroids, covariances).flatten()

        # Outlier recall = fraction of this species correctly blocked
        tn_sp          = int((do_sp >  threshold).sum())
        fp_sp          = int((do_sp <= threshold).sum())
        recall_outlier = tn_sp / len(do_sp) if len(do_sp) > 0 else 0.0

        # AUC-ROC: separability of this outlier species vs all known species
        # Higher score = more normal → known=1 gets higher score (-dk larger)
        # Outlier=0 gets lower score (-do_sp smaller if do_sp is large)
        scores = np.concatenate([-dk, -do_sp])
        y_true = np.concatenate([np.ones(len(dk)), np.zeros(len(do_sp))])
        auc    = (float(roc_auc_score(y_true, scores))
                  if len(np.unique(y_true)) > 1 else 0.0)

        per_species[species_name] = {
            "n_samples":      len(do_sp),
            "TN":             tn_sp,
            "FP":             fp_sp,
            "outlier_recall": round(recall_outlier, 4),
            "AUC-ROC":        round(auc, 4),
        }

        print(f"  {species_name:<22} {len(do_sp):>5} {tn_sp:>5} {fp_sp:>5} "
              f"{recall_outlier:>16.3f} {auc:>8.4f}")

    print("\n  outlier_recall = fraction of this species correctly blocked")
    print("  AUC-ROC = separability from known species "
          "(1.0 = perfect, 0.5 = random)")

    return per_species


# ─────────────────────────────────────────────
# SUMMARY TABLE (called from pipeline.py)
# ─────────────────────────────────────────────

def print_summary_table(Xk: np.ndarray, Xo: np.ndarray,
                        centroids: np.ndarray, covariances: np.ndarray,
                        threshold: float):
    """Compact TP/FN/TN/FP/precision/recall/F1 summary printed to stdout."""
    dk = min_centroid_distances(Xk, centroids, covariances)
    do = min_centroid_distances(Xo, centroids, covariances)
    m  = _binary_metrics(dk, do, threshold)

    print("\n── Detection Summary ──")
    print(f"  Threshold : {threshold:.4f}  "
          f"(percentile={cfg.percentile_of_threshold})")
    print(f"\n  {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("  " + "-" * 50)
    print(f"  {m['TP']:>5} {m['FN']:>5} {m['TN']:>5} {m['FP']:>5} "
          f"{m['Recall']:>8.3f} {m['Precision']:>10.3f} {m['F1']:>6.3f}")
    print("\n  TP = known correctly accepted  | FN = known wrongly rejected")
    print("  TN = outliers correctly blocked | FP = outliers wrongly accepted")


# ─────────────────────────────────────────────
# ARTIFACT LOADER
# ─────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load saved centroids, threshold, and (for Mahalanobis) covariances.
    Returns (centroids, covariances_or_None, threshold).
    """
    centroids = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    if cfg.distance_metric == "mahalanobis":
        covariances = np.load(
            os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    else:
        covariances = None
    threshold = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))
    return centroids, covariances, threshold
