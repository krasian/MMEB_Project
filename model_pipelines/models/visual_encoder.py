"""
Visual encoder: EfficientNet-B0 backbone with a learned embedding projection head.
"""
import torch.nn as nn
from torchvision import models

try:
    from model_pipelines.config import cfg
except ImportError:
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
        """
        Initialize the EfficientNet-B0 projection model.

        Args:
            embedding_dim: Output embedding width. Defaults to `cfg.embedding_dim`.
            pretrained: Whether to load ImageNet backbone weights.
        """
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
        """
        Return L2-normalized embeddings for an image batch.

        Args:
            x: Image tensor with shape `[batch_size, 3, height, width]`.

        Returns:
            Tensor with shape `[batch_size, embedding_dim]`.
        """
        emb = self.extract_features(x)
        emb = self.embedding(emb)
        emb = nn.functional.normalize(emb, dim=1)
        return emb

    def extract_features(self, x):
        """
        Return pooled backbone features before projection.

        Args:
            x: Image tensor with shape `[batch_size, 3, height, width]`.

        Returns:
            Tensor with shape `[batch_size, 1280]`.
        """
        return self.pool(self.backbone(x)).flatten(1)
