"""
Centroid-based outlier scoring with Mahalanobis distance.

For each known class we compute:
  - a centroid (mean embedding)
  - a covariance matrix (with small ridge regularisation)

A new embedding's outlier score is its minimum Mahalanobis distance to
any class centroid. The threshold is calibrated as a percentile of the
training-set distances (configurable in config.yaml).
"""
import numpy as np

from config import cfg


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray,
                      num_classes: int) -> np.ndarray:
    """One centroid (mean embedding position) per class."""
    return np.vstack([
        embeddings[labels == c].mean(axis=0)
        for c in range(num_classes)
    ])


def compute_covariances(embeddings: np.ndarray, labels: np.ndarray,
                        num_classes: int, reg: float = 1e-5) -> np.ndarray:
    """Per-class covariance matrix with ridge regularisation."""
    dim = embeddings.shape[1]
    covs = np.zeros((num_classes, dim, dim), dtype=np.float64)
    for c in range(num_classes):
        X = embeddings[labels == c]
        cov = np.cov(X, rowvar=False) if len(X) > 1 else np.eye(dim)
        covs[c] = cov + reg * np.eye(dim)
    return covs


def _mahalanobis_to_class(embeddings: np.ndarray, centroid: np.ndarray,
                          cov: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from every embedding to one class centroid."""
    inv = np.linalg.inv(cov)
    diff = embeddings - centroid
    return np.sqrt(np.einsum("nd,dd,nd->n", diff, inv, diff).clip(0))


def min_centroid_distances(embeddings: np.ndarray, centroids: np.ndarray,
                           covariances: np.ndarray = None) -> np.ndarray:
    """
    Mahalanobis distance to the nearest class centroid.
    Falls back to Euclidean if covariances is None (backwards compat).
    """
    if covariances is None:
        dists = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :], axis=2)
        return np.min(dists, axis=1)

    dists = np.stack([
        _mahalanobis_to_class(embeddings, centroids[c], covariances[c])
        for c in range(len(centroids))
    ], axis=1)
    return np.min(dists, axis=1)


def compute_distance_threshold(embeddings: np.ndarray, labels: np.ndarray,
                               centroids: np.ndarray,
                               covariances: np.ndarray = None) -> float:
    """Threshold at cfg.percentile_of_threshold of training distances."""
    if covariances is None:
        own_centroids = centroids[labels]
        distances = np.linalg.norm(embeddings - own_centroids, axis=1)
    else:
        distances = np.concatenate([
            _mahalanobis_to_class(embeddings[labels == c],
                                  centroids[c], covariances[c])
            for c in range(len(centroids))
            if (labels == c).any()
        ])
    return np.float64(np.percentile(distances, cfg.percentile_of_threshold))
