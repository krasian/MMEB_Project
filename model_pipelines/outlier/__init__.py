"""outlier package — centroid-based outlier scoring and evaluation."""
from .mahalanobis import (
    compute_centroids,
    compute_covariances,
    min_centroid_distances,
    compute_distance_threshold,
)
from .evaluate import (
    evaluate_centroid_detector,
    print_summary_table,
    load_artifacts,
)

__all__ = [
    "compute_centroids",
    "compute_covariances",
    "min_centroid_distances",
    "compute_distance_threshold",
    "evaluate_centroid_detector",
    "print_summary_table",
    "load_artifacts",
]
