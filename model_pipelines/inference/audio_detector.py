"""
Audio inference detector — loads the trained prototypical probe and
scores individual audio files or folders for outlier detection.
"""
import os
import json
import numpy as np
import torch

try:
    from model_pipelines.config import AudioConfig
    from model_pipelines.utils.device_utils import get_device
    from model_pipelines.models.audio_encoder import (
        BirdMAEModel, PrototypicalProbe, precompute_window_features,
    )
    from model_pipelines.data.audio_dataset import sliding_window_split
except ImportError:
    from config import AudioConfig
    from utils.device_utils import get_device
    from models.audio_encoder import (
        BirdMAEModel, PrototypicalProbe, precompute_window_features,
    )
    from data.audio_dataset import sliding_window_split


def load_audio_model(checkpoint_dir: str = None):
    """Load the trained prototypical probe + frozen Bird-MAE backbone."""
    cfg = AudioConfig()
    if checkpoint_dir is None:
        checkpoint_dir = cfg.checkpoint_directory
    device = get_device()

    # Load class mapping to determine num_classes
    classes_path = os.path.join(checkpoint_dir, "classes.json")
    with open(classes_path) as f:
        classes = json.load(f)
    num_classes = len(classes)

    # Build models
    backbone = BirdMAEModel().to(device)
    backbone.eval()

    probe = PrototypicalProbe(
        feature_dim=cfg.birdmae_input_dim,
        num_classes=num_classes,
        num_prototypes=cfg.num_prototypes,
    ).to(device)

    # Load probe weights
    probe_path = os.path.join(checkpoint_dir, "best_prototypical_probe.pt")
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval()

    return backbone, probe, classes


class AudioAnomalyDetector:
    """
    Score one audio file or folder of clips.

    Usage:
        det = AudioAnomalyDetector()
        result = det.predict("path/to/clip.wav")
    """

    def __init__(self, checkpoint_dir: str = None):
        self.cfg = AudioConfig()
        if checkpoint_dir is None:
            checkpoint_dir = self.cfg.checkpoint_directory
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()

        self.backbone, self.probe, self.classes = load_audio_model(checkpoint_dir)

        threshold_path = os.path.join(checkpoint_dir, "threshold.npy")
        self.threshold = float(np.load(threshold_path))

    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        """Score a single audio file → outlier decision + predicted class."""
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}

        # Slide windows over the audio file
        windows = sliding_window_split(
            audio_path,
            sample_rate=self.cfg.sample_rate,
            window_duration=self.cfg.window_duration,
            step_duration=self.cfg.step_duration,
        )

        if not windows:
            return {"error": f"No valid windows extracted from {audio_path}"}

        # Extract features for every window
        features = []
        for window in windows:
            wav = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            feat = self.backbone(wav)
            features.append(feat)
        features = torch.cat(features, dim=0)

        # Run probe over all windows
        logits = self.probe(features)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

        # Aggregate windows: max-pool each class across windows then take max
        agg_probs = probs.max(axis=0)
        confidence = float(agg_probs.max())
        pred_idx = int(agg_probs.argmax())
        pred_class = self.classes[pred_idx]

        is_outlier = confidence < self.threshold

        return {
            "predicted_class": pred_class,
            "confidence":      round(confidence, 4),
            "threshold":       round(self.threshold, 4),
            "is_outlier":      bool(is_outlier),
            "n_windows":       len(windows),
        }
