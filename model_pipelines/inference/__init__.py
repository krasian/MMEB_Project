"""inference package — detector classes for visual + audio."""
from .visual_detector import VisualAnomalyDetector, BirdAnomalyDetector, load_model

# Audio exports — populated when audio_detector.py is implemented.
# from .audio_detector import ...

__all__ = [
    "VisualAnomalyDetector",
    "BirdAnomalyDetector",
    "load_model",
]
