"""data package — visual + audio dataset utilities."""
from .visual_dataset import (
    BirdCSVDataset,
    birdcsvdata,
    load_csv_paths,
    split_samples,
    build_splits,
    get_transforms,
    make_loader,
)

# Audio exports — populated when audio_dataset.py is implemented.
# from .audio_dataset import ...

__all__ = [
    "BirdCSVDataset",
    "birdcsvdata",
    "load_csv_paths",
    "split_samples",
    "build_splits",
    "get_transforms",
    "make_loader",
]
