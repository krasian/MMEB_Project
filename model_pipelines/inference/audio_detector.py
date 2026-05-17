"""
Audio inference detector — loads the trained prototypical probe and the
frozen Bird-MAE feature extractor, then scores individual audio files
or folders for outlier detection.
"""
import os
import json
import numpy as np
import torch
import librosa

try:
    from model_pipelines.config import AudioConfig
    from model_pipelines.utils.device_utils import get_device
    from model_pipelines.models.audio_encoder import (
        BirdMAEExtractor, PrototypicalProbe,
    )
except ImportError:
    from config import AudioConfig
    from utils.device_utils import get_device
    from models.audio_encoder import (
        BirdMAEExtractor, PrototypicalProbe,
    )


def sliding_window_split(audio_path: str,
                         sample_rate: int,
                         window_duration: float,
                         step_duration: float):
    """
    Split an audio file into overlapping fixed-duration windows.

    Loads `audio_path` at `sample_rate` (mono) and returns a list of
    float32 numpy arrays each of length `int(sample_rate * window_duration)`.
    The final window is zero-padded if the file's last segment is short.
    Files shorter than one window get a single zero-padded window.

    Args:
        audio_path:      path to a .wav / .mp3 / .flac / .ogg file
        sample_rate:     target sample rate (hz)
        window_duration: window length in seconds
        step_duration:   hop length between window starts in seconds

    Returns:
        list[np.ndarray]: one entry per window, shape (samples_per_window,)
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception:
        return []

    win_len  = int(sample_rate * window_duration)
    step_len = max(1, int(sample_rate * step_duration))

    # Short files: pad up to a single full window.
    if len(audio) <= win_len:
        pad = np.zeros(win_len, dtype=np.float32)
        pad[:len(audio)] = audio
        return [pad]

    windows = []
    for start in range(0, len(audio) - win_len + 1, step_len):
        windows.append(audio[start:start + win_len].astype(np.float32))

    # Capture any leftover tail as a final zero-padded window.
    last_start = windows.__len__() * step_len - step_len if windows else 0
    tail_start = last_start + step_len if windows else 0
    if tail_start < len(audio) and tail_start + win_len > len(audio):
        tail = np.zeros(win_len, dtype=np.float32)
        chunk = audio[tail_start:]
        tail[:len(chunk)] = chunk
        windows.append(tail)

    return windows


def load_audio_model(checkpoint_dir: str = None):
    """
    Load the frozen Bird-MAE extractor + the trained prototypical probe
    and return them along with the class mapping list.
    """
    cfg = AudioConfig()
    if checkpoint_dir is None:
        checkpoint_dir = cfg.checkpoint_directory
    device = get_device()

    # Load class mapping. classes.json is a JSON-serialised dict with
    # string keys ("0", "1", ...) -> species name. Sort numerically and
    # return a list so the model's argmax index can be used directly.
    classes_path = os.path.join(checkpoint_dir, "classes.json")
    with open(classes_path) as f:
        raw_classes = json.load(f)
    if isinstance(raw_classes, dict):
        classes = [raw_classes[str(i)] for i in sorted(int(k) for k in raw_classes)]
    else:
        classes = list(raw_classes)
    num_classes = len(classes)

    # Frozen Bird-MAE feature extractor (audio -> spatial features [H,W,D]).
    extractor = BirdMAEExtractor(cfg)

    # Learnable probe.
    probe = PrototypicalProbe(
        feature_dim=cfg.birdmae_input_dim,
        num_classes=num_classes,
        num_prototypes=cfg.num_prototypes,
    ).to(device)

    probe_path = os.path.join(checkpoint_dir, "best_prototypical_probe.pt")
    state = torch.load(probe_path, map_location=device)
    # Training saves a dict {model_state_dict: ...}; an older format saved
    # the state dict directly. Handle both.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Training wraps PrototypicalProbe inside BirdMAEModel, so keys are
    # saved as "probe.prototypes", "probe.final_weights", "probe.final_biases".
    # Strip that prefix so they load into a bare PrototypicalProbe.
    state = {
        (k[len("probe."):] if k.startswith("probe.") else k): v
        for k, v in state.items()
    }
    probe.load_state_dict(state)
    probe.eval()

    return extractor, probe, classes


class AudioAnomalyDetector:
    """
    Score one audio file or a folder of clips.

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

        self.extractor, self.probe, self.classes = load_audio_model(checkpoint_dir)

        threshold_path = os.path.join(checkpoint_dir, "threshold.npy")
        self.threshold = float(np.load(threshold_path))

    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        """Score a single audio file → outlier decision + predicted class."""
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}

        # Slide windows over the audio file.
        windows = sliding_window_split(
            audio_path,
            sample_rate=self.cfg.sample_rate,
            window_duration=self.cfg.window_duration,
            step_duration=self.cfg.step_duration,
        )
        if not windows:
            return {"error": f"No valid windows extracted from {audio_path}"}

        # For each window, get a Bird-MAE spatial feature map [H, W, D]
        # and stack into a batch [B, D, H, W] for the probe.
        feature_tensors = []
        for window in windows:
            spatial = self.extractor._extract_spatial_from_audio(window)  # [H, W, D]
            t = torch.from_numpy(spatial).permute(2, 0, 1).float()        # [D, H, W]
            feature_tensors.append(t)
        features = torch.stack(feature_tensors, dim=0).to(self.device)    # [B, D, H, W]

        # Run probe over all windows. Training uses sigmoid (multi-label),
        # so apply sigmoid here too — softmax would distort the threshold.
        logits = self.probe(features)
        probs  = torch.sigmoid(logits).cpu().numpy()                      # [B, C]

        # Aggregate windows: max-pool each class across windows.
        agg_probs   = probs.max(axis=0)                                   # [C]
        confidence  = float(agg_probs.max())
        pred_idx    = int(agg_probs.argmax())
        pred_class  = self.classes[pred_idx]

        is_outlier  = confidence < self.threshold

        return {
            "predicted_class": pred_class,
            "confidence":      round(confidence, 4),
            "threshold":       round(self.threshold, 4),
            "is_outlier":      bool(is_outlier),
            "n_windows":       len(windows),
        }
