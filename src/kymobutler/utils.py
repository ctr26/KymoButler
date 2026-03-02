"""Shared utility functions for KymoButler Python modules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.spatial import cKDTree


ArrayLikeImage = Image.Image | np.ndarray
Coord = tuple[int, int]


@dataclass(slots=True)
class Component:
    """Connected component statistics for a binary image."""

    label: int
    count: int
    bbox: tuple[int, int, int, int]
    coords: np.ndarray


def ensure_grayscale_array(image: ArrayLikeImage) -> np.ndarray:
    """Convert a PIL image or ndarray to a float32 grayscale array in [0, 1]."""
    if isinstance(image, Image.Image):
        if image.mode in {"RGBA", "LA"}:
            image = remove_alpha_channel(image)
        gray = ImageOps.grayscale(image)
        arr = np.asarray(gray, dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.mean(axis=-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def remove_alpha_channel(image: Image.Image) -> Image.Image:
    """Drop alpha by compositing onto black, matching Mathematica semantics."""
    if image.mode in {"RGBA", "LA"}:
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        return Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")
    return image


def image_adjust(arr: np.ndarray) -> np.ndarray:
    """Contrast-normalize an image array to [0, 1]."""
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def resize_to_multiple_of_16(arr: np.ndarray) -> np.ndarray:
    """Resize image so each spatial dimension is nearest multiple of 16."""
    h, w = arr.shape[:2]
    nh = max(16, int(16 * round(h / 16)))
    nw = max(16, int(16 * round(w / 16)))
    if (nh, nw) == (h, w):
        return arr
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return ensure_grayscale_array(img)


def is_negated(arr: np.ndarray) -> bool:
    """Return True if image appears white-on-black and should be inverted."""
    x = ensure_grayscale_array(arr)
    n1 = int((x > 0.5).sum())
    n2 = int(((1.0 - x) > 0.5).sum())
    return n1 >= n2


def norm_lines(arr: np.ndarray) -> np.ndarray:
    """Normalize each row by its mean intensity and re-adjust to [0, 1]."""
    x = ensure_grayscale_array(arr)
    means = x.mean(axis=1, keepdims=True)
    safe = np.where(means > 0.0, x / np.maximum(means, 1e-8), x)
    return image_adjust(safe)


def binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize grayscale image with threshold in [0, 1]."""
    return ensure_grayscale_array(arr) >= float(threshold)


def hit_miss(binary: np.ndarray, kernels: Sequence[np.ndarray]) -> np.ndarray:
    """Apply hit-or-miss transform for a list of {-1,0,1} kernels."""
    b = np.asarray(binary, dtype=bool)
    out = np.zeros_like(b, dtype=bool)
    for k in kernels:
        k = np.asarray(k, dtype=np.int8)
        fg = k == 1
        bg = k == -1
        fg_ok = ndimage.binary_erosion(b, structure=fg, border_value=0) if fg.any() else np.ones_like(b)
        bg_ok = (
            ndimage.binary_erosion(~b, structure=bg, border_value=0) if bg.any() else np.ones_like(b)
        )
        out |= fg_ok & bg_ok
    return out


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Skeletonize a binary image with Zhang-Suen thinning."""
    img = np.asarray(binary, dtype=np.uint8).copy()
    changed = True
    while changed:
        changed = False
        to_del: list[tuple[int, int]] = []
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] != 1:
                    continue
                p2, p3, p4 = img[i - 1, j], img[i - 1, j + 1], img[i, j + 1]
                p5, p6, p7 = img[i + 1, j + 1], img[i + 1, j], img[i + 1, j - 1]
                p8, p9 = img[i, j - 1], img[i - 1, j - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                n = sum(neighbors)
                if n < 2 or n > 6:
                    continue
                s = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
                if s != 1:
                    continue
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_del.append((i, j))
        if to_del:
            changed = True
            for i, j in to_del:
                img[i, j] = 0

        to_del = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] != 1:
                    continue
                p2, p3, p4 = img[i - 1, j], img[i - 1, j + 1], img[i, j + 1]
                p5, p6, p7 = img[i + 1, j + 1], img[i + 1, j], img[i + 1, j - 1]
                p8, p9 = img[i, j - 1], img[i - 1, j - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                n = sum(neighbors)
                if n < 2 or n > 6:
                    continue
                s = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
                if s != 1:
                    continue
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_del.append((i, j))
        if to_del:
            changed = True
            for i, j in to_del:
                img[i, j] = 0
    return img.astype(bool)


def _endpoint_mask(binary: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    n = ndimage.convolve(binary.astype(np.uint8), k, mode="constant", cval=0)
    return binary & (n == 2)


def prune(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Prune skeleton endpoints iteratively."""
    out = np.asarray(binary, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        out &= ~_endpoint_mask(out)
    return out


def smooth_bin_uni(binary: np.ndarray) -> np.ndarray:
    """Unidirectional smoothing operator from KymoButler.wl."""
    kernels = [
        np.array([[0, 1, 0], [0, -1, 1], [0, 1, 0]], dtype=np.int8),
        np.array([[0, 1, 0], [1, -1, 1], [0, 0, 0]], dtype=np.int8),
        np.array([[0, 1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0], [1, -1, 1], [0, 1, 0]], dtype=np.int8),
    ]
    out = np.asarray(binary, dtype=bool)
    return out | hit_miss(out, kernels)


def smooth_bin(binary: np.ndarray) -> np.ndarray:
    """Bidirectional smoothing operator from KymoButler.wl."""
    add_kernels = [
        np.array([[0, 1, 1], [0, -1, 1], [0, 1, 1]], dtype=np.int8),
        np.array([[1, 1, 1], [1, -1, 1], [0, 0, 0]], dtype=np.int8),
        np.array([[1, 1, 0], [1, -1, 0], [1, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0], [1, -1, 1], [1, 1, 1]], dtype=np.int8),
    ]
    del_kernels = [
        np.array([[0, -1, -1], [0, 1, -1], [0, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, -1], [-1, 1, -1], [0, 0, 0]], dtype=np.int8),
        np.array([[-1, -1, 0], [-1, 1, 0], [-1, -1, 0]], dtype=np.int8),
        np.array([[0, 0, 0], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8),
    ]
    out = np.asarray(binary, dtype=bool)
    return (out | hit_miss(out, add_kernels)) & ~hit_miss(out, del_kernels)


def chew_ends(binary: np.ndarray) -> np.ndarray:
    """Remove horizontal terminal endpoints."""
    kernels = [
        np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8),
    ]
    out = np.asarray(binary, dtype=bool)
    return out & ~hit_miss(out, kernels)


def chew_all_ends(binary: np.ndarray) -> np.ndarray:
    """Iteratively remove terminal endpoints until stable."""
    prev = np.asarray(binary, dtype=bool)
    cur = chew_ends(prev)
    while not np.array_equal(prev, cur):
        prev = cur
        cur = chew_ends(cur)
    return cur


def connected_components(binary: np.ndarray, connectivity: int = 2) -> tuple[np.ndarray, list[Component]]:
    """Label connected components and compute per-component statistics."""
    b = np.asarray(binary, dtype=bool)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(b, structure=structure)
    comps: list[Component] = []
    for lb in range(1, n + 1):
        coords = np.argwhere(labels == lb)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        comps.append(Component(label=lb, count=int(coords.shape[0]), bbox=(int(r0), int(c0), int(r1), int(c1)), coords=coords))
    return labels, comps


def select_components(
    binary: np.ndarray,
    min_count: int,
    min_row_span: int,
    connectivity: int = 2,
) -> np.ndarray:
    """Keep components with at least `min_count` pixels and `min_row_span` rows."""
    labels, comps = connected_components(binary, connectivity=connectivity)
    keep = np.zeros_like(labels, dtype=bool)
    for comp in comps:
        row_span = comp.bbox[2] - comp.bbox[0]
        if comp.count >= min_count and row_span >= min_row_span:
            keep |= labels == comp.label
    return keep


def component_centroids(binary_or_labels: np.ndarray) -> list[tuple[int, tuple[float, float]]]:
    """Return `(label, (row, col))` centroids for non-zero connected components."""
    arr = np.asarray(binary_or_labels)
    if arr.dtype != np.int32 and arr.dtype != np.int64:
        labels, _ = connected_components(arr.astype(bool))
    else:
        labels = arr
    out: list[tuple[int, tuple[float, float]]] = []
    for lb in sorted(int(x) for x in np.unique(labels) if x != 0):
        coords = np.argwhere(labels == lb)
        if coords.size:
            out.append((lb, tuple(coords.mean(axis=0).tolist())))
    return out


def pixel_positions(binary_or_labels: np.ndarray, value: int | bool = True) -> np.ndarray:
    """Return pixel coordinates `(row, col)` for a value in array."""
    arr = np.asarray(binary_or_labels)
    return np.argwhere(arr == value)


def gather_by_first_mean(track: np.ndarray) -> np.ndarray:
    """Group `[(t, y), ...]` by `t` and average y, then round."""
    if track.size == 0:
        return track.reshape(0, 2)
    tvals = np.unique(track[:, 0])
    out = []
    for t in tvals:
        ys = track[track[:, 0] == t, 1]
        out.append((int(round(float(t))), int(round(float(ys.mean())))))
    return np.asarray(out, dtype=np.int32)


def nearest_within(points: np.ndarray, query: Sequence[float], radius: float) -> np.ndarray:
    """Find all points within `radius` from `query`, sorted by row then col."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 2)
    tree = cKDTree(pts)
    idx = tree.query_ball_point(np.asarray(query, dtype=np.float32), r=float(radius))
    if not idx:
        return np.empty((0, 2), dtype=np.int32)
    out = pts[idx]
    order = np.lexsort((out[:, 1], out[:, 0]))
    return np.round(out[order]).astype(np.int32)


def sort_coords(coords: np.ndarray) -> np.ndarray:
    """Sort candidate coordinates into the longest connected order."""
    pts = np.asarray(coords, dtype=np.int32)
    if len(pts) <= 1:
        return pts

    def grow(seed: np.ndarray, pick_last: bool) -> np.ndarray:
        out = [seed]
        rem = pts[1:].copy()
        while len(rem) > 0:
            cur = out[-1]
            d = np.sqrt(((rem - cur) ** 2).sum(axis=1))
            idx = np.where(d <= 1.5)[0]
            if idx.size == 0:
                break
            candidates = rem[idx]
            sel = np.lexsort((candidates[:, 1], candidates[:, 0]))
            nxt = candidates[-1 if pick_last else 0]
            out.append(nxt)
            rem = rem[~np.all(rem == nxt, axis=1)]
        return np.asarray(out, dtype=np.int32)

    left = grow(pts[0], pick_last=False)
    right = grow(pts[0], pick_last=True)
    return right if len(right) >= len(left) else left


def shortest_path_in_binary(binary: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    """Find shortest 8-connected path inside `binary` from `start` to `goal`."""
    b = np.asarray(binary, dtype=bool)
    h, w = b.shape

    def in_bounds(p: Coord) -> bool:
        return 0 <= p[0] < h and 0 <= p[1] < w

    if not in_bounds(start) or not in_bounds(goal):
        return []
    if start == goal:
        return [start]

    q: deque[Coord] = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not in_bounds(nxt) or nxt in prev:
                continue
            if not b[nr, nc] and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if goal not in prev:
        return []

    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def track_to_set(track: Iterable[Sequence[int]]) -> set[tuple[int, int]]:
    """Convert track coordinates to a hashable set."""
    return {(int(t), int(y)) for t, y in track}
