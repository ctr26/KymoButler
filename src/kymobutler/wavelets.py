"""Wavelet-based kymograph processing routines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt
from scipy import ndimage

from .utils import (
    binarize,
    chew_all_ends,
    component_centroids,
    connected_components,
    ensure_grayscale_array,
    hit_miss,
    image_adjust,
    is_negated,
    norm_lines,
    prune,
    select_components,
    smooth_bin,
    zhang_suen_thinning,
)


def stationary_wavelet_sum(
    image: np.ndarray,
    wavelet: str = "db1",
    level: int = 2,
) -> np.ndarray:
    """Compute summed SWT channels used by KymoButler wavelet segmentation."""
    x = ensure_grayscale_array(image)
    coeffs = pywt.swt2(x, wavelet=wavelet, level=level, start_level=0)
    accum = np.zeros_like(x, dtype=np.float32)
    for cA, (cH, cV, cD) in coeffs:
        accum += np.abs(cA).astype(np.float32)
        accum += np.abs(cH).astype(np.float32)
        accum += np.abs(cV).astype(np.float32)
        accum += np.abs(cD).astype(np.float32)
    return image_adjust(accum)


def _seed_mask(paths: np.ndarray) -> np.ndarray:
    """Extract seed pixels from skeleton endpoints."""
    endpoint_kernel = [np.array([[-1, -1, -1], [-1, 1, -1], [0, 0, 0]], dtype=np.int8)]
    return hit_miss(chew_all_ends(paths), endpoint_kernel)


def _extract_tracks_from_paths(paths: np.ndarray, min_time: int) -> list[list[tuple[int, int]]]:
    """Convert skeleton connected components into `(t, y)` tracks."""
    labels, comps = connected_components(paths)
    tracks: list[list[tuple[int, int]]] = []
    for comp in comps:
        coords = np.argwhere(labels == comp.label)
        if coords.size == 0:
            continue
        times = np.unique(coords[:, 0])
        trk: list[tuple[int, int]] = []
        for t in times:
            ys = coords[coords[:, 0] == t, 1]
            trk.append((int(t), int(round(float(ys.mean())))))
        if trk and (trk[-1][0] - trk[0][0]) >= min_time:
            tracks.append(trk)
    return tracks


def _make_overlay(base: np.ndarray, tracks: list[list[tuple[int, int]]]) -> np.ndarray:
    """Render color overlay for track visualization."""
    gray = ensure_grayscale_array(base)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rng = np.random.default_rng(42)
    for trk in tracks:
        color = rng.uniform(0.2, 1.0, size=3)
        for t, y in trk:
            if 0 <= t < rgb.shape[0] and 0 <= y < rgb.shape[1]:
                rgb[t, y, :] = color
    return np.clip(rgb, 0.0, 1.0)


def wavelet_segment(
    kymograph: np.ndarray,
    binthresh: float,
    min_size: int,
    min_time: int,
    wavelet: str = "db1",
    level: int = 2,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Segment bidirectional paths with stationary wavelets."""
    tmp = image_adjust(ensure_grayscale_array(kymograph))
    neg = is_negated(tmp)
    if neg:
        tmp = 1.0 - tmp
    tmp = norm_lines(tmp)

    wsum = stationary_wavelet_sum(tmp, wavelet=wavelet, level=level)
    out = binarize(wsum, binthresh)
    out = ndimage.binary_dilation(out, iterations=1)
    out = zhang_suen_thinning(out)
    out = prune(out, iterations=5)
    out = zhang_suen_thinning(out)
    paths = select_components(out, min_count=min_size, min_row_span=min_time, connectivity=2)
    return neg, tmp, wsum, paths


def analyse_kymograph_bi_wavelet(
    kymograph: np.ndarray,
    dim: tuple[int, int],
    binthresh: float,
    cnet: Any,
    vismod: Any,
    vthr: float,
    min_size: int,
    min_time: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]]] | str:
    """Python translation of `AnalyseKymographBIwavelet`.

    Notes:
    - `cnet`, `vismod`, and `vthr` are kept for API compatibility with Mathematica,
      but the wavelet path is model-free as in the original implementation.
    - Returns `(preprocessed, overlay, labeled_overlay, tracks)`.
    """
    del cnet, vismod, vthr

    if dim[0] > 5000 or dim[1] > 5000:
        return "Image too large! Try a smaller one or upgrade to paid version"

    _, tmp, _, paths = wavelet_segment(
        kymograph,
        binthresh=binthresh,
        min_size=min_size,
        min_time=min_time,
    )

    _ = _seed_mask(paths)
    paths = smooth_bin(paths)
    tracks = _extract_tracks_from_paths(paths, min_time=min_time)

    labels = component_centroids(paths.astype(np.int32))
    overlay = _make_overlay(tmp, tracks)
    labeled_overlay = overlay.copy()
    for idx, (_, (r, c)) in enumerate(labels, start=1):
        rr, cc = int(round(r)), int(round(c))
        if 0 <= rr < labeled_overlay.shape[0] and 0 <= cc < labeled_overlay.shape[1]:
            labeled_overlay[max(rr - 1, 0) : min(rr + 2, labeled_overlay.shape[0]), cc, :] = [1, 1, 1]
            if cc + 1 < labeled_overlay.shape[1]:
                labeled_overlay[rr, cc + 1, :] = [1, 1, 1]
            if cc + 2 < labeled_overlay.shape[1]:
                labeled_overlay[rr, cc + 2, :] = [1, 1, 1]
            if idx % 2 == 0:
                labeled_overlay[rr, cc, :] = [0, 0, 0]

    return tmp, overlay, labeled_overlay, tracks
