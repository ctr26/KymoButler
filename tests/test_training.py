from __future__ import annotations

import torch

from kymobutler.data import KymographDataset, SyntheticKymographGenerator
from kymobutler.losses import CombinedLoss
from kymobutler.models import build_unet


def test_training_smoke_forward_and_loss() -> None:
    generator = SyntheticKymographGenerator(
        height=64,
        width=64,
        mode="bidirectional",
        min_tracks=2,
        max_tracks=4,
        seed=123,
    )
    dataset = KymographDataset(num_samples=2, generator=generator)

    x0, y0 = dataset[0]
    x1, y1 = dataset[1]

    x = torch.stack([x0, x1], dim=0)
    y = torch.stack([y0, y1], dim=0)

    model = build_unet(base_channels=8)
    criterion = CombinedLoss()

    pred = model(x)
    loss = criterion(pred, y)

    assert pred.shape == (2, 2, 64, 64)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
