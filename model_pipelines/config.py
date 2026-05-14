"""
Configuration classes for the bird novelty detection pipelines.

Two pipelines (visual + audio) share a BaseConfig. Each has its own
config class with modality-specific settings. The visual pipeline
exports a module-level `cfg` singleton which is patched in-place by
load_config.apply_yaml_config() at startup. The audio pipeline uses
AudioConfig directly (instantiated where needed).
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List


# ─────────────────────────────────────────────
# BASE CONFIG (shared)
# ─────────────────────────────────────────────

@dataclass
class BaseConfig:
    """Base configuration shared across all pipelines."""

    root_dir: str = r"D:\MMEB-Project\data\processed"
    embedding_dim: int = 512

    training_split_ratio: float = 0.70
    validation_split_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    arcface_scaler: float = 64.0
    arcface_margin: float = 0.8

    percentile_of_threshold: int = 95

    checkpoint_directory: str = r"D:\MMEB-Project\model_pipelines\checkpoints"
    results_directory: str = r"D:\MMEB-Project\model_pipelines\results"

    def device(self) -> torch.device:
        """DirectML > CUDA > CPU. Imported lazily to avoid circular imports."""
        try:
            from model_pipelines.utils.device_utils import get_device
        except ImportError:
            from utils.device_utils import get_device
        return get_device()

    def make_dirs(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        os.makedirs(self.results_directory, exist_ok=True)


# ─────────────────────────────────────────────
# AUDIO CONFIG
# ─────────────────────────────────────────────

@dataclass
class AudioConfig(BaseConfig):
    """Audio-specific configuration for Xeno-Canto data."""

    sample_rate: int = 32000
    duration: float = 5.0

    birdmae_input_dim: int = 768
    num_prototypes: int = 20

    window_duration: float = 5.0
    step_duration: float = 2.5

    filter_silent_windows: bool = True
    silence_threshold: float = 0.01

    aggregation_method: str = "max"

    use_orthogonality_loss: bool = True
    orthogonality_weight: float = 0.1

    @property
    def data_root(self) -> str:
        return os.path.join(self.root_dir, "xenocanto_data")

    native_folders: List[str] = field(default_factory=lambda: [
        "Turdus_merula", "Cyanistes_caeruleus", "Parus_major",
        "Passer_domesticus", "Corvus_corone", "Turdus_philomelos",
        "Erithacus_rubecula", "Anas_platyrhynchos",
    ])

    outlier_folders: List[str] = field(default_factory=lambda: [
        "Sturnus_vulgaris", "Phoenicopterus_roseus", "Ramphastos_sulfuratus",
    ])

    folder_to_species: Dict[str, str] = field(default_factory=lambda: {
        "Turdus_merula": "Eurasian Blackbird",
        "Cyanistes_caeruleus": "Eurasian Blue Tit",
        "Parus_major": "Great Tit",
        "Passer_domesticus": "House Sparrow",
        "Corvus_corone": "Carrion Crow",
        "Turdus_philomelos": "Song Thrush",
        "Erithacus_rubecula": "European Robin",
        "Anas_platyrhynchos": "Mallard",
        "Sturnus_vulgaris": "European Starling",
        "Phoenicopterus_roseus": "Greater Flamingo",
        "Ramphastos_sulfuratus": "Keel-billed Toucan",
    })

    short_names: Dict[str, str] = field(default_factory=lambda: {
        "Turdus_merula": "Blackbird", "Cyanistes_caeruleus": "Blue Tit",
        "Parus_major": "Great Tit", "Passer_domesticus": "Sparrow",
        "Corvus_corone": "Crow", "Turdus_philomelos": "Thrush",
        "Erithacus_rubecula": "Robin", "Anas_platyrhynchos": "Mallard",
        "Sturnus_vulgaris": "Starling", "Phoenicopterus_roseus": "Flamingo",
        "Ramphastos_sulfuratus": "Toucan",
    })

    def __post_init__(self):
        self.checkpoint_directory = "audio_checkpoints"
        self.results_directory = "audio_results"

    def get_active_native_folders(self) -> List[str]:
        return [f for f in self.native_folders
                if os.path.exists(os.path.join(self.data_root, f))]

    def get_active_outlier_folders(self) -> List[str]:
        return [f for f in self.outlier_folders
                if os.path.exists(os.path.join(self.data_root, f))]

    @property
    def number_of_classes(self) -> int:
        return len(self.get_active_native_folders())


# ─────────────────────────────────────────────
# VISUAL CONFIG
# ─────────────────────────────────────────────

class VisualConfig:
    """All visual-pipeline settings. Patched by load_config.apply_yaml_config()."""

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

        self.embeding_visulize_method = "tsne"
        self.result_dpi = 300

        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.csv_out = "predictions.csv"

    def device(self) -> torch.device:
        try:
            from model_pipelines.utils.device_utils import get_device
        except ImportError:
            from utils.device_utils import get_device
        return get_device()

    def make_dirs(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        os.makedirs(self.results_directory, exist_ok=True)


Config = VisualConfig


# Module-level singleton (visual). Patched by apply_yaml_config().
cfg = VisualConfig()
names = list(cfg.known_csv.keys())
class_to_id = {name: i for i, name in enumerate(names)}
outlier_names = list(cfg.outlier_csv.keys())


# Auto-load config.yaml if present.
try:
    try:
        from model_pipelines.load_config import apply_yaml_config
    except ImportError:
        from load_config import apply_yaml_config
    apply_yaml_config(cfg=cfg, verbose=False)
except (FileNotFoundError, ImportError):
    pass
