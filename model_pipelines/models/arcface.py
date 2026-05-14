"""
ArcFace (Additive Angular Margin) loss.

Adds an angular margin to the target class angle, forcing embeddings of
the same species to cluster much more tightly than plain cross-entropy.

Shared by both visual and audio pipelines.

NOTE ON OPTIMIZATION:
    The class-anchor weight matrix `self.weights` is a learnable
    nn.Parameter. The optimizer in the training script MUST include
    `criterion.parameters()` for the anchors to be updated. Without
    that, the anchors stay at their random initialization for the
    entire run, which yields looser clusters and worse outlier
    detection. Recommended: do NOT apply weight decay to these
    parameters (they get re-normalized to the unit sphere in forward).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5):
        """
        Args:
            embedding_dim: dimensionality of the input embeddings.
            num_classes:   number of known classes (size of the anchor matrix).
            s:             logit scale (typical: 30-64).
            m:             angular margin in radians. 0.5 is the standard
                           value from the paper; values above ~0.6 tend to
                           be unstable early in training.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weights)
        self.s = s
        self.m = m
        # Precompute the trig constants needed for the stable formulation.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, embeddings, labels):
        # Embeddings come in already L2-normalised from BirdEmbeddingModel,
        # but normalising again is cheap and defensive.
        emb = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weights, dim=1)

        # cos(theta) = <emb, anchor>   shape [B, C]
        cos_theta = torch.matmul(emb, weight.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Numerically stable cos(theta + m) using the angle-addition formula:
        #   cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        # This avoids the acos/cos round-trip, whose gradient blows up
        # near cos(theta) = +/-1.
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-7))
        target_logits = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Only apply the margin to the true class column.
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = one_hot * target_logits + (1 - one_hot) * cos_theta
        logits = logits * self.s

        return F.cross_entropy(logits, labels)
