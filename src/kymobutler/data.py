"""Synthetic kymograph data generation and PyTorch datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class SyntheticSample:
    """Container for a generated synthetic kymograph sample."""

    image: np.ndarray
    mask: np.ndarray | None = None
    ant_mask: np.ndarray | None = None
    ret_mask: np.ndarray | None = None


class SyntheticKymographGenerator:
    """Generate synthetic kymographs with track masks for segmentation training."""

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        mode: str = "bidirectional",
        min_tracks: int = 3,
        max_tracks: int = 10,
        velocity_range: tuple[float, float] = (0.2, 2.0),
        background_range: tuple[float, float] = (0.05, 0.25),
        noise_std_range: tuple[float, float] = (0.01, 0.07),
        blur_sigma_range: tuple[float, float] = (0.3, 1.0),
        seed: int | None = None,
    ) -> None:
        if mode not in {"bidirectional", "unidirectional"}:
            raise ValueError("mode must be 'bidirectional' or 'unidirectional'")
        if min_tracks < 1 or max_tracks < min_tracks:
            raise ValueError("track range is invalid")

        self.height = int(height)
        self.width = int(width)
        self.mode = mode
        self.min_tracks = int(min_tracks)
        self.max_tracks = int(max_tracks)
        self.velocity_range = velocity_range
        self.background_range = background_range
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.rng = np.random.default_rng(seed)

    def _draw_track(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        slope: float,
        intercept: float,
        thickness: int,
        amplitude: float,
    ) -> None:
        for t in range(self.height):
            y = slope * t + intercept
            if y < 0 or y >= self.width:
                continue
            yc = int(round(y))
            half = max(1, thickness // 2)
            y0 = max(0, yc - half)
            y1 = min(self.width, yc + half + 1)

            window = np.arange(y0, y1, dtype=np.float32)
            profile = np.exp(-0.5 * ((window - y) / max(0.7, thickness / 2.0)) ** 2)

            image[t, y0:y1] += amplitude * profile
            mask[t, y0:y1] = np.maximum(mask[t, y0:y1], profile)

    def _generate_background(self) -> np.ndarray:
        base = self.rng.uniform(*self.background_range, size=(self.height, self.width)).astype(np.float32)

        stripe_axis = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        drift = self.rng.uniform(-0.08, 0.08)
        base = base + drift * stripe_axis

        modulation = 1.0 + self.rng.normal(0.0, 0.08, size=(1, self.width)).astype(np.float32)
        base = base * modulation
        return np.clip(base, 0.0, 1.0)

    def generate(self) -> SyntheticSample:
        image = self._generate_background()
        ant_mask = np.zeros((self.height, self.width), dtype=np.float32)
        ret_mask = np.zeros((self.height, self.width), dtype=np.float32)

        n_tracks = int(self.rng.integers(self.min_tracks, self.max_tracks + 1))
        for _ in range(n_tracks):
            speed = self.rng.uniform(*self.velocity_range)
            direction = 1.0 if self.rng.random() > 0.5 else -1.0
            slope = direction * speed * self.rng.uniform(0.6, 1.4)

            t_anchor = self.rng.uniform(0, self.height)
            y_anchor = self.rng.uniform(0, self.width)
            intercept = y_anchor - slope * t_anchor

            thickness = int(self.rng.integers(1, 4))
            amplitude = float(self.rng.uniform(0.35, 1.0))

            if self.mode == "unidirectional":
                target_mask = ant_mask if direction > 0 else ret_mask
            else:
                target_mask = ant_mask

            self._draw_track(image, target_mask, slope, intercept, thickness, amplitude)

        image = gaussian_filter(image, sigma=float(self.rng.uniform(*self.blur_sigma_range)))

        shot_noise = self.rng.poisson(np.clip(image, 0.0, None) * 18.0).astype(np.float32) / 18.0
        gauss_noise = self.rng.normal(
            0.0,
            self.rng.uniform(*self.noise_std_range),
            size=image.shape,
        ).astype(np.float32)
        image = 0.6 * image + 0.4 * shot_noise + gauss_noise

        if self.mode == "bidirectional":
            mask = (ant_mask > 0.2).astype(np.float32)
            return SyntheticSample(image=np.clip(image, 0.0, 1.0), mask=mask)

        ant_mask = (ant_mask > 0.2).astype(np.float32)
        ret_mask = (ret_mask > 0.2).astype(np.float32)
        return SyntheticSample(image=np.clip(image, 0.0, 1.0), ant_mask=ant_mask, ret_mask=ret_mask)


class KymographDataset(Dataset[tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]]]):
    """PyTorch dataset backed by ``SyntheticKymographGenerator``."""

    def __init__(
        self,
        num_samples: int,
        generator: SyntheticKymographGenerator,
    ) -> None:
        self.num_samples = int(num_samples)
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]]:
        del index
        sample = self.generator.generate()

        x = torch.from_numpy(sample.image).unsqueeze(0).float()

        if self.generator.mode == "bidirectional":
            if sample.mask is None:
                raise RuntimeError("bidirectional sample did not include a mask")
            y = torch.from_numpy(sample.mask).unsqueeze(0).float()
            return x, y

        if sample.ant_mask is None or sample.ret_mask is None:
            raise RuntimeError("unidirectional sample did not include required masks")

        y = {
            "ant": torch.from_numpy(sample.ant_mask).unsqueeze(0).float(),
            "ret": torch.from_numpy(sample.ret_mask).unsqueeze(0).float(),
        }
        return x, y
