"""
Configuration classes for the bird novelty detection pipelines.

This module defines a Config dataclass-like object whose values are
fallback defaults. In practice, they are overwritten at startup by
load_config.apply_yaml_config(), which reads config.yaml and patches
the instance in-place. The env var DATA_ROOT (or --data-root CLI flag)
always takes final priority.

The module-level `cfg` object is the single source of truth — every
file in the visual pipeline imports it from here.
"""
import os
import torch

try:
    import torch_directml
    _DIRECTML_AVAILABLE = True
except ImportError:
    _DIRECTML_AVAILABLE = False


class VisualConfig:
    """All visual-pipeline settings in one place."""

    def __init__(self):
        self.data_root = str(os.environ.get("DATA_ROOT", ""))
        self.known_csv = {
            "Common Blackbird":  "updated_blackbird_data.csv",
            "Eurasian Blue Tit": "updated_EurasianBlueTit_data.csv",
            "Great Tit":         "updated_GreatTit_data.csv",
            "House Sparrow":     "updated_HouseSparrow_data.csv",
        }
        self.outlier_csv = {"European Starling": "updated_EuropeanStarling.csv"}

        self.train_split_ratio = 0.70
        self.validation_split_ratio = 0.15
        self.number_of_classes = len(self.known_csv)

        self.embedding_dim = 512
        self.image_size = 224

        self.batch = 16
        self.epoches = 40
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.arcface_scaler = 64.0
        self.arcface_margin = 0.8

        self.percentile_of_threshold = 75
        self.distance_metric = "mahalanobis"  

        self.checkpoint_directory = "checkpoints"
        self.results_directory = "results"

        # Evaluation
        self.embeding_visulize_method = "tsne"
        self.result_dpi = 300

        # Inference
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.csv_out = "predictions.csv"

    def device(self) -> torch.device:
        if _DIRECTML_AVAILABLE:
            return torch.device(torch_directml.device())
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def make_dirs(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        os.makedirs(self.results_directory, exist_ok=True)


# Backwards-compat alias — kept around in case anything still imports `Config`.
Config = VisualConfig


# Module-level singleton. Every visual-pipeline file imports `cfg` from here.
cfg = VisualConfig()

# Class name lookups — populated from cfg.known_csv / cfg.outlier_csv.
# These are mutated in-place by load_config.apply_yaml_config().
names = list(cfg.known_csv.keys())
class_to_id = {name: i for i, name in enumerate(names)}
outlier_names = list(cfg.outlier_csv.keys())


# ---------------------------------------------------------------------------
# Audio config — placeholder for the teammate working on the audio pipeline.
# ---------------------------------------------------------------------------
class AudioConfig:
    """Placeholder — to be populated by the audio teammate."""
    pass


# Auto-load config.yaml if present, so importing `cfg` gives you up-to-date
# values without having to call apply_yaml_config() manually.
try:
    from load_config import apply_yaml_config
    apply_yaml_config(cfg=cfg)
except (FileNotFoundError, ImportError):
    pass
