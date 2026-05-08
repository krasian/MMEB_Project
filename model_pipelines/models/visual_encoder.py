"""
Visual encoder: EfficientNet-B0 backbone with a learned embedding projection head.
"""
import torch.nn as nn
from torchvision import models

from config import cfg


class BirdEmbeddingModel(nn.Module):
    """
    Architecture:
        EfficientNetB0 features  (pretrained on ImageNet)
            ↓
        Global Average Pool  → 1280-dim vector
            ↓
        Linear projection    → embedding-dim vector
            ↓
        L2 normalization     → unit-length embedding
    """

    def __init__(self, embedding_dim: int = None, pretrained: bool = True):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = cfg.embedding_dim
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.backbone = base.features
        self.pool = base.avgpool
        self.embedding = nn.Linear(1280, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = self.extract_features(x)
        emb = self.embedding(emb)
        emb = nn.functional.normalize(emb, dim=1)
        return emb

    def extract_features(self, x):
        return self.pool(self.backbone(x)).flatten(1)
