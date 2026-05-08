"""
ArcFace (Additive Angular Margin) loss.

Adds an angular margin to the target class angle, forcing embeddings of
the same species to cluster much more tightly than plain cross-entropy.

Shared by both visual and audio pipelines.
"""
import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.7):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weights)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        weight = nn.functional.normalize(self.weights, dim=1)
        cos_theta = torch.matmul(embeddings, weight.t())
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = one_hot * target_logits + (1 - one_hot) * cos_theta
        logits *= self.s
        return nn.functional.cross_entropy(logits, labels)
