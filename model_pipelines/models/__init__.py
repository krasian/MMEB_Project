"""Models package exports for visual and audio components."""

from .arcface import ArcFaceLoss
from .visual_encoder import BirdEmbeddingModel

__all__ = [
    "BirdEmbeddingModel",
    "ArcFaceLoss",
    "BirdMAEExtractor",
    "BirdMAEModel",
    "PrototypicalProbe",
    "precompute_spatial_features",
    "precompute_window_features",
    "create_dataloader",
    "create_window_dataloader",
    "SpatialAudioDataset",
]

_AUDIO_EXPORTS = {
    "BirdMAEExtractor",
    "BirdMAEModel",
    "PrototypicalProbe",
    "precompute_spatial_features",
    "precompute_window_features",
    "create_dataloader",
    "create_window_dataloader",
    "SpatialAudioDataset",
}


def __getattr__(name):
    """Load audio symbols only when callers actually ask for them."""
    if name not in _AUDIO_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import audio_encoder

    value = getattr(audio_encoder, name)
    globals()[name] = value
    return value
