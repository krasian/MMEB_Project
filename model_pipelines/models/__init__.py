"""models package — visual + audio encoders and shared ArcFace loss."""
from .visual_encoder import BirdEmbeddingModel
from .arcface import ArcFaceLoss
from .audio_encoder import (
    BirdMAEExtractor,
    BirdMAEModel,
    PrototypicalProbe,
    precompute_spatial_features,
    precompute_window_features,
    create_dataloader,
    create_window_dataloader,
    SpatialAudioDataset
)


__all__ = [
    "BirdEmbeddingModel",
    "ArcFaceLoss",
    'BirdMAEExtractor',
    'BirdMAEModel',
    'PrototypicalProbe',
    'precompute_spatial_features',
    'precompute_window_features',
    'create_dataloader',
    'create_window_dataloader',
    'SpatialAudioDataset',
]
