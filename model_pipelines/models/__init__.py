"""models package — visual + audio encoders and shared ArcFace loss."""
from .visual_encoder import BirdEmbeddingModel
from .arcface import ArcFaceLoss



__all__ = [
    "BirdEmbeddingModel",
    "ArcFaceLoss",
]
