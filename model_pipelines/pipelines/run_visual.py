"""
Visual pipeline orchestration:
  - run_full_pipeline(): build splits, train, compute centroids/threshold,
    evaluate, dump metrics.
  - run_evaluation():    re-evaluate an existing checkpoint and produce
    ROC/PR curves, distance histograms, and t-SNE embedding plots.
"""
import os
import json
import numpy as np

from config import cfg, names, outlier_names
from data.visual_dataset import build_splits, make_loader
from training.train_visual import train_model
from inference.visual_detector import load_model
from outlier.mahalanobis import (
    compute_centroids,
    compute_covariances,
    compute_distance_threshold,
)
from outlier.evaluate import (
    evaluate_centroid_detector,
    print_summary_table,
    load_artifacts,
)
from utils.embeddings import extract_embeddings
from utils.visualization import (
    plot_roc_pr,
    plot_distance_distribution,
    plot_embedding_space,
)


N_KNOWN = len(names)


def run_full_pipeline(skip_training: bool = False):
    """End-to-end: train (optional), compute centroids + threshold, evaluate."""
    cfg.make_dirs()

    print("=" * 52)
    print("  Bird Anomaly Detection — Visual Pipeline")
    print("=" * 52)

    train_samples, val_samples, test_known, test_outlier = build_splits()

    with open(os.path.join(cfg.checkpoint_directory, "classes.json"), "w") as f:
        json.dump(names, f, indent=2)

    if skip_training:
        print("\n── Step 1: Skipped (using existing checkpoint) ──")
    else:
        train_model(train_samples, val_samples)

    print("\n── Step 2: Extracting training embeddings ──")
    model = load_model()
    train_loader = make_loader(train_samples, "val")
    train_embs, train_labels = extract_embeddings(model, train_loader)

    np.save(os.path.join(cfg.checkpoint_directory, "train_embeddings.npy"), train_embs)
    print(f"  Embeddings shape: {train_embs.shape}")

    centroids = compute_centroids(train_embs, train_labels, cfg.number_of_classes)
    np.save(os.path.join(cfg.checkpoint_directory, "centroids.npy"), centroids)
    print(f"  Centroids saved: {centroids.shape}")

    if cfg.distance_metric == "mahalanobis":
        covariances = compute_covariances(train_embs, train_labels, cfg.number_of_classes)
        np.save(os.path.join(cfg.checkpoint_directory, "covariances.npy"), covariances)
        print(f"  Covariances saved: {covariances.shape}")
    else:
        covariances = None
        # Remove any stale covariances from a previous Mahalanobis run so the
        # detector won't accidentally pick them up.
        stale = os.path.join(cfg.checkpoint_directory, "covariances.npy")
        if os.path.exists(stale):
            os.remove(stale)
        print(f"  Distance metric: euclidean (no covariances needed)")

    threshold = compute_distance_threshold(train_embs, train_labels, centroids, covariances)
    np.save(os.path.join(cfg.checkpoint_directory, "centroid_threshold.npy"), threshold)
    print(f"  Threshold ({cfg.percentile_of_threshold}th percentile): {threshold:.4f}")

    results = evaluate_centroid_detector(model, test_known, test_outlier)

    print("\n── Final Metrics ──")
    print(json.dumps(results, indent=2))
    print("\n Pipeline complete.")


def _get_test_embeddings(model) -> tuple:
    """
    Extract embeddings for all test images (known + outlier species).

    Returns:
        Xk         — (N_known, D)   embeddings for known species
        Xo         — (N_outlier, D) embeddings for all outlier species pooled
        all_embs   — Xk and Xo stacked vertically (used for t-SNE)
        all_labels — integer labels: 0..N_KNOWN-1 = known, N_KNOWN.. = outlier
    """
    _, _, test_known, test_outlier = build_splits()

    Xk, known_labels = extract_embeddings(model, make_loader(test_known, "val"))
    Xo, raw_outlier = extract_embeddings(model, make_loader(test_outlier, "val"))

    outlier_labels = N_KNOWN + (-raw_outlier - 1)

    all_embs = np.vstack([Xk, Xo])
    all_labels = np.concatenate([known_labels, outlier_labels])

    return Xk, Xo, all_embs, all_labels


def run_evaluation():
    """Re-evaluate an existing checkpoint and produce all evaluation plots."""
    cfg.make_dirs()

    print("── Loading artifacts ──")
    centroids, covariances, threshold = load_artifacts()
    model = load_model()

    print("── Extracting test embeddings ──")
    Xk, Xo, all_embs, all_labels = _get_test_embeddings(model)

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
