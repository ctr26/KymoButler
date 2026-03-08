"""Losses used for KymoButler segmentation training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Soft dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)

    @staticmethod
    def _foreground_prob(pred: Tensor) -> Tensor:
        if pred.ndim != 4:
            raise ValueError("prediction tensor must have shape (N, C, H, W)")
        if pred.shape[1] == 1:
            return pred
        if pred.shape[1] >= 2:
            return pred[:, 1:2, :, :]
        raise ValueError("prediction tensor has invalid channel dimension")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p = self._foreground_prob(pred)
        t = target.float()

        p = p.reshape(p.shape[0], -1)
        t = t.reshape(t.shape[0], -1)

        intersection = (p * t).sum(dim=1)
        denominator = p.sum(dim=1) + t.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of BCE and dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    @staticmethod
    def _foreground_prob(pred: Tensor) -> Tensor:
        if pred.shape[1] == 1:
            return pred
        return pred[:, 1:2, :, :]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p = self._foreground_prob(pred)
        t = target.float()

        bce_loss = self.bce(p, t)
        dice_loss = self.dice(pred, t)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
