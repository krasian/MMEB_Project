"""
Outlier-detection evaluation for both visual and audio modalities.

Visual side (centroid distance + Mahalanobis):
    - evaluate_centroid_detector
    - print_summary_table
    - load_artifacts
    - _evaluate_per_species

Audio side (prototypical confidence):
    - evaluate_outlier_detector
    - evaluate_outlier_detector_prototypical
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, accuracy_score, precision_score,
                              recall_score)


# Visual imports — guarded so this file works even if visual deps are missing
_VISUAL_DEPS = None
_VISUAL_OK = None


def _load_visual_deps():
    """Import visual-only dependencies lazily so audio evaluation stays audio-only."""
    global _VISUAL_DEPS, _VISUAL_OK

    if _VISUAL_DEPS is not None:
        return _VISUAL_DEPS

    try:
        try:
            from model_pipelines.config import cfg
            from model_pipelines.utils.embeddings import extract_embeddings
            from model_pipelines.data.visual_dataset import make_loader, load_csv_paths
            from model_pipelines.outlier.mahalanobis import min_centroid_distances
        except ImportError:
            from config import cfg
            from utils.embeddings import extract_embeddings
            from data.visual_dataset import make_loader, load_csv_paths
            from outlier.mahalanobis import min_centroid_distances
    except ImportError:
        _VISUAL_OK = False
        raise

    _VISUAL_OK = True
    _VISUAL_DEPS = (
        cfg,
        extract_embeddings,
        make_loader,
        load_csv_paths,
        min_centroid_distances,
    )
    return _VISUAL_DEPS


# ═════════════════════════════════════════════
# VISUAL: CENTROID DISTANCE EVALUATION
# ═════════════════════════════════════════════

def _binary_metrics(dk, do, threshold):
    """TP/FN/TN/FP + Recall/Precision/F1 + AUC-ROC + AUC-PR."""
    tp = int((dk <= threshold).sum())
    fn = int((dk >  threshold).sum())
    tn = int((do >  threshold).sum())
    fp = int((do <= threshold).sum())

    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    scores = np.concatenate([-dk, -do])
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


def evaluate_centroid_detector(model, test_known: list, test_outlier: list):
    """Visual: evaluate the centroid-distance detector on the full test set."""
    cfg, extract_embeddings, make_loader, _, min_centroid_distances = _load_visual_deps()

    print("\n── Evaluating centroid detector ──")

    centroids = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    if cfg.distance_metric == "mahalanobis":
        covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    else:
        covariances = None
    threshold = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))

    Xk, _ = extract_embeddings(model, make_loader(test_known, "val"))
    dk    = min_centroid_distances(Xk, centroids, covariances).flatten()

    Xo, _ = extract_embeddings(model, make_loader(test_outlier, "val"))
    do    = min_centroid_distances(Xo, centroids, covariances).flatten()

    # Save fusion scores for late-fusion with audio
    os.makedirs(cfg.results_directory, exist_ok=True)
    np.save(os.path.join(cfg.results_directory, "visual_scores_known.npy"),   -dk)
    np.save(os.path.join(cfg.results_directory, "visual_scores_outlier.npy"), -do)
    np.save(os.path.join(cfg.results_directory, "visual_threshold.npy"),     threshold)

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

    per_species = _evaluate_per_species(model, Xk, centroids, covariances, threshold)

    results = {"overall": overall, "per_species": per_species}
    with open(os.path.join(cfg.results_directory, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Metrics saved to {cfg.results_directory}/metrics.json")

    return results


def _evaluate_per_species(model, Xk, centroids, covariances, threshold):
    """Per outlier-species detection breakdown."""
    cfg, extract_embeddings, make_loader, load_csv_paths, min_centroid_distances = _load_visual_deps()

    dk = min_centroid_distances(Xk, centroids, covariances).flatten()

    print("\n── Per-species outlier breakdown ──")
    print(f"  {'Species':<22} {'N':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall(outlier)':>16} {'AUC-ROC':>8}")
    print("  " + "-" * 65)

    per_species = {}

    for species_name, csv_file in cfg.outlier_csv.items():
        samples = load_csv_paths(csv_file, label=-1)
        if not samples:
            print(f"  {species_name:<22} — no samples found, skipping")
            continue

        Xo_sp, _ = extract_embeddings(model, make_loader(samples, "val"))
        do_sp    = min_centroid_distances(Xo_sp, centroids, covariances).flatten()

        tn_sp = int((do_sp >  threshold).sum())
        fp_sp = int((do_sp <= threshold).sum())
        recall_outlier = tn_sp / len(do_sp) if len(do_sp) > 0 else 0.0

        scores = np.concatenate([-dk, -do_sp])
        y_true = np.concatenate([np.ones(len(dk)), np.zeros(len(do_sp))])
        auc = (float(roc_auc_score(y_true, scores))
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
    print("  AUC-ROC = separability from known species (1.0 = perfect, 0.5 = random)")
    return per_species


def print_summary_table(Xk, Xo, centroids, covariances, threshold):
    """Compact TP/FN/TN/FP summary."""
    cfg, _, _, _, min_centroid_distances = _load_visual_deps()

    dk = min_centroid_distances(Xk, centroids, covariances).flatten()
    do = min_centroid_distances(Xo, centroids, covariances).flatten()
    m  = _binary_metrics(dk, do, threshold)

    print("\n── Detection Summary ──")
    print(f"  Threshold : {threshold:.4f}  (percentile={cfg.percentile_of_threshold})")
    print(f"\n  {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
          f"{'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("  " + "-" * 50)
    print(f"  {m['TP']:>5} {m['FN']:>5} {m['TN']:>5} {m['FP']:>5} "
          f"{m['Recall']:>8.3f} {m['Precision']:>10.3f} {m['F1']:>6.3f}")
    print("\n  TP = known correctly accepted  | FN = known wrongly rejected")
    print("  TN = outliers correctly blocked | FP = outliers wrongly accepted")


def load_artifacts():
    """Load saved centroids, threshold, and (for Mahalanobis) covariances."""
    cfg, _, _, _, _ = _load_visual_deps()

    centroids = np.load(os.path.join(cfg.checkpoint_directory, "centroids.npy"))
    if cfg.distance_metric == "mahalanobis":
        covariances = np.load(os.path.join(cfg.checkpoint_directory, "covariances.npy"))
    else:
        covariances = None
    threshold = float(np.load(
        os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy")))
    return centroids, covariances, threshold


# ═════════════════════════════════════════════
# AUDIO: PROTOTYPICAL CONFIDENCE EVALUATION
# ═════════════════════════════════════════════

def evaluate_outlier_detector(model, test_known, test_outlier, embeddings,
                               centroids, covariances, threshold, audio_cfg):
    """Audio: Mahalanobis-based outlier detection (legacy MLP+ArcFace path)."""
    # Implementation kept for backwards compatibility — unused in current
    # prototypical-probing pipeline. See evaluate_outlier_detector_prototypical.
    raise NotImplementedError(
        "Mahalanobis evaluation for audio is not used — "
        "current audio pipeline uses prototypical probing. "
        "Call evaluate_outlier_detector_prototypical instead."
    )


def evaluate_outlier_detector_prototypical(known_confidences, outlier_confidences,
                                            threshold, known_labels=None,
                                            class_mapping=None,
                                            results_directory=None):
    """Audio: confidence-based outlier detection from prototypical probe."""
    known_predictions   = (known_confidences   >= threshold).astype(int)
    outlier_predictions = (outlier_confidences >= threshold).astype(int)

    known_true   = np.ones(len(known_confidences))
    outlier_true = np.zeros(len(outlier_confidences))

    all_confidences = np.concatenate([known_confidences, outlier_confidences])
    all_true        = np.concatenate([known_true, outlier_true])
    all_pred        = np.concatenate([known_predictions, outlier_predictions])

    results = {
        'auc_roc':   float(roc_auc_score(all_true, all_confidences)),
        'auc_pr':    float(average_precision_score(all_true, all_confidences)),
        'f1':        float(f1_score(all_true, all_pred)),
        'accuracy':  float(accuracy_score(all_true, all_pred)),
        'precision': float(precision_score(all_true, all_pred)),
        'recall':    float(recall_score(all_true, all_pred)),
        'threshold': float(threshold),
        'num_known':       int(len(known_confidences)),
        'num_outlier':     int(len(outlier_confidences)),
        'known_correct':   int((known_predictions == 1).sum()),
        'outlier_correct': int((outlier_predictions == 0).sum())
    }

    # Per-class accuracy
    if known_labels is not None:
        from collections import defaultdict
        class_acc = defaultdict(list)
        class_confidences = defaultdict(list)

        for conf, label, pred in zip(known_confidences, known_labels, known_predictions):
            class_acc[label].append(1 if pred == 1 else 0)
            class_confidences[label].append(conf)

        per_class = {}
        for label, acc_list in class_acc.items():
            species_name = (class_mapping.get(label, f"Class_{label}")
                            if class_mapping else f"Class_{label}")
            per_class[species_name] = {
                'accuracy':        float(np.mean(acc_list)),
                'mean_confidence': float(np.mean(class_confidences[label])),
                'num_samples':     len(acc_list)
            }
        results['per_class_accuracy'] = per_class

    # Print
    print(f"\n  Outlier Detection Results (Prototypical Probing):")
    print(f"  {'Metric':<15} {'Score':<10}")
    print(f"  {'-'*25}")
    print(f"  {'AUC-ROC':<15} {results['auc_roc']:.4f}")
    print(f"  {'AUC-PR':<15} {results['auc_pr']:.4f}")
    print(f"  {'F1 Score':<15} {results['f1']:.4f}")
    print(f"  {'Accuracy':<15} {results['accuracy']:.4f}")
    print(f"  {'Precision':<15} {results['precision']:.4f}")
    print(f"  {'Recall':<15} {results['recall']:.4f}")

    print(f"\n  Confusion Matrix (threshold={threshold:.3f}):")
    print(f"                    Predicted Known    Predicted Outlier")
    print(f"  Actual Known:        {results['known_correct']:>6}               "
          f"{results['num_known'] - results['known_correct']:>6}")
    print(f"  Actual Outlier:      {results['num_outlier'] - results['outlier_correct']:>6}               "
          f"{results['outlier_correct']:>6}")

    if 'per_class_accuracy' in results and results['per_class_accuracy']:
        print(f"\n  Per-Class Accuracy:")
        for species, metrics in results['per_class_accuracy'].items():
            print(f"    {species:<30}: {metrics['accuracy']:.3f} "
                  f"(n={metrics['num_samples']})")

    # Save fusion scores for late-fusion with visual modality
    if results_directory is not None:
        os.makedirs(results_directory, exist_ok=True)
        np.save(os.path.join(results_directory, "audio_scores_known.npy"),
                known_confidences)
        np.save(os.path.join(results_directory, "audio_scores_outlier.npy"),
                outlier_confidences)
        np.save(os.path.join(results_directory, "audio_threshold.npy"),
                np.array(threshold))

    return results
