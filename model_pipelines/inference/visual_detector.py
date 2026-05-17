"""
Ready-to-use visual inference interface — loads all saved artifacts from disk
and scores individual images.
"""
import os
import numpy as np
import torch
from PIL import Image

from config import cfg, names
from data.visual_dataset import get_transforms
from models.visual_encoder import BirdEmbeddingModel


def load_model() -> BirdEmbeddingModel:
    """Load the best saved model weights from disk, with dim validation."""
    device = cfg.device()
    checkpoint = os.path.join(cfg.checkpoint_directory, "best_model.pt")
    state_dict = torch.load(checkpoint, map_location="cpu")

    saved_dim = state_dict["embedding.weight"].shape[0]
    if saved_dim != cfg.embedding_dim:
        raise ValueError(
            f"Checkpoint embedding_dim={saved_dim} does not match "
            f"cfg.embedding_dim={cfg.embedding_dim}. Either retrain the model "
            f"or update embedding_dim in config.yaml to match the checkpoint."
        )

    model = BirdEmbeddingModel(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class VisualAnomalyDetector:
    """
    Visual centroid-distance detector backed by saved training artifacts.

    Usage:
        detector = VisualAnomalyDetector()
        result   = detector.predict(r"D:\\path\\to\\image.jpg")
    """

    def __init__(self, checkpoint_dir: str = None):
        """
        Load model weights and centroid detector artifacts.

        Args:
            checkpoint_dir: Artifact directory. Defaults to `cfg.checkpoint_directory`.
        """
        if checkpoint_dir is None:
            checkpoint_dir = cfg.checkpoint_directory
        self.metric = cfg.distance_metric
        self.transform = get_transforms("val")
        self.device = cfg.device()
        self.classes = names
        self.model = load_model()
        self.centroids = np.load(
            os.path.join(checkpoint_dir, "centroids.npy"))
        self.centroid_threshold = float(np.load(
            os.path.join(checkpoint_dir, "centroid_threshold.npy")))
        if self.metric == "mahalanobis":
            self.covariances = np.load(
                os.path.join(checkpoint_dir, "covariances.npy"))
            self._inv_covs = np.stack([
                np.linalg.inv(self.covariances[c])
                for c in range(len(self.centroids))
            ])
        else:
            self.covariances = None
            self._inv_covs = None


    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Score one image against saved class centroids.

        Args:
            image_path: Path to an image file.

        Returns:
            Prediction fields, or `{"error": str}` if the path is missing.
        """
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb = self.model(tensor).cpu().numpy()[0]

        if self.metric == "mahalanobis":
            dists = np.array([
                float(np.sqrt((diff := emb - self.centroids[c]) @ self._inv_covs[c] @ diff))
                for c in range(len(self.centroids))
            ])
        else:
            # Euclidean
            dists = np.linalg.norm(self.centroids - emb, axis=1)

        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        return {
            "predicted_class": self.classes[min_idx],
            "distance": round(min_dist, 4),
            "threshold": round(self.centroid_threshold, 4),
            "is_outlier": min_dist > self.centroid_threshold,
        }


# Backwards-compat alias for the original class name.
BirdAnomalyDetector = VisualAnomalyDetector
