"""
Small evaluation-metric and palette-building helpers.

Most heavy-lifting evaluation lives in outlier/evaluate.py; this module
provides reusable atoms used by both the eval and visualisation code.
"""
import numpy as np

from config import names, outlier_names


def build_palette() -> dict:
    """
    Build a colour palette covering every known and outlier species.
    Known species get qualitative colours; outlier species get warm/red
    tones so they stand out on plots.
    """
    known_colors = [
        "#2C3E50", "#4C8BF5", "#F5A623", "#27AE60",
        "#8E44AD", "#16A085", "#D35400", "#2980B9",
        "#C0392B", "#7F8C8D", "#F39C12", "#1ABC9C",
    ]
    outlier_colors = [
        "#E74C3C", "#E91E63", "#FF5722", "#AD1457",
        "#BF360C", "#880E4F",
    ]

    palette = {}
    for i, name in enumerate(names):
        palette[name] = known_colors[i % len(known_colors)]
    for i, name in enumerate(outlier_names):
        palette[name] = outlier_colors[i % len(outlier_colors)]
    return palette


def confusion_counts(distances_known: np.ndarray, distances_outlier: np.ndarray,
                     threshold: float) -> dict:
    """Return TP/FN/TN/FP counts for the centroid detector at a given threshold."""
    return {
        "tp": int((distances_known <= threshold).sum()),
        "fn": int((distances_known > threshold).sum()),
        "tn": int((distances_outlier > threshold).sum()),
        "fp": int((distances_outlier <= threshold).sum()),
    }


def precision_recall_f1(tp: int, fn: int, fp: int) -> tuple:
    """Compute precision, recall, and F1 from confusion counts."""
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)
