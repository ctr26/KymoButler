"""Proper port of KymoButler iterative tracking algorithm from Mathematica.

This implements the REAL tracking algorithm, not the connected-components shortcut.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Protocol

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

Track = list[tuple[int, int]]
Coord = tuple[int, int]


class VisionModuleProtocol(Protocol):
    """Protocol for vision module network."""

    def __call__(
        self, tile: np.ndarray, track_mask: np.ndarray, candidates_mask: np.ndarray
    ) -> np.ndarray:
        """Run vision module on tile inputs.

        Args:
            tile: Grayscale image tile (dim x dim)
            track_mask: Binary mask of current track in tile
            candidates_mask: Binary mask of all candidate pixels in tile

        Returns:
            Probability map (dim x dim) for next candidate
        """
        ...


def find_short_path_image(
    binary: np.ndarray, start: Coord, goal: Coord
) -> list[Coord]:
    """Find shortest path through binary image using BFS.

    Port of Mathematica FindShortPathImage.

    Args:
        binary: Binary image where 1=passable
        start: Starting coordinate (row, col)
        goal: Goal coordinate (row, col)

    Returns:
        List of coordinates forming path, or empty if no path exists.
    """
    h, w = binary.shape

    def in_bounds(p: Coord) -> bool:
        return 0 <= p[0] < h and 0 <= p[1] < w

    if not in_bounds(start) or not in_bounds(goal):
        return []
    if start == goal:
        return [start]

    # BFS
    queue: deque[Coord] = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not in_bounds(nxt) or nxt in prev:
                continue
            if not binary[nr, nc] and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                queue.clear()
                break
            queue.append(nxt)

    if goal not in prev:
        return []

    # Reconstruct path
    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def sort_coords(coords: np.ndarray) -> np.ndarray:
    """Sort candidate coordinates into longest continuous path.

    Port of Mathematica SortCoords. Grows path in both directions from
    first point, keeping the longer result.

    Args:
        coords: Array of (row, col) coordinates

    Returns:
        Sorted coordinates as array
    """
    if len(coords) == 0:
        return coords
    if len(coords) == 1:
        return coords

    pts = [tuple(c) for c in coords]

    def grow(seed: Coord, remaining: list[Coord], pick_last: bool) -> list[Coord]:
        out = [seed]
        rem = list(remaining)
        while rem:
            cur = out[-1]
            # Find neighbors within 1.5 pixels
            nearby = [(p, np.sqrt((p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)) for p in rem]
            nearby = [(p, d) for p, d in nearby if d <= 1.5]
            if not nearby:
                break
            # Sort by column (second coord)
            nearby.sort(key=lambda x: x[0][1])
            chosen = nearby[-1][0] if pick_last else nearby[0][0]
            out.append(chosen)
            rem.remove(chosen)
        return out

    left = grow(pts[0], pts[1:], pick_last=False)
    right = grow(pts[0], pts[1:], pick_last=True)

    result = right if len(right) >= len(left) else left
    return np.array(result, dtype=np.int32)


def get_tile(
    kym: np.ndarray,
    track: list[Coord],
    all_candidates: list[Coord],
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Extract tile around track end for vision module.

    Port of Mathematica GetTile.

    Args:
        kym: Kymograph image
        track: Current track coordinates
        all_candidates: All candidate coordinates
        dim: Tile dimension (e.g., 48)

    Returns:
        (tile_image, track_mask, candidates_mask, window_slices)
    """
    h, w = kym.shape
    last = track[-1] if track else (h // 2, w // 2)

    # Compute window bounds
    r0 = int(round(last[0] - dim // 2))
    r1 = int(round(last[0] + dim // 2))
    c0 = int(round(last[1] - dim // 2))
    c1 = int(round(last[1] + dim // 2))

    # Shift if out of bounds
    if r0 < 0:
        r1 -= r0
        r0 = 0
    if r1 > h:
        r0 -= r1 - h
        r1 = h
    if c0 < 0:
        c1 -= c0
        c0 = 0
    if c1 > w:
        c0 -= c1 - w
        c1 = w

    # Clamp to valid range
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, c0), min(w, c1)

    window = (slice(r0, r1), slice(c0, c1))

    # Extract tile
    tile = kym[window].copy()

    # Create track mask in tile coordinates
    track_mask = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
    for t, y in track:
        tr, tc = t - r0, y - c0
        if 0 <= tr < track_mask.shape[0] and 0 <= tc < track_mask.shape[1]:
            track_mask[tr, tc] = 1.0

    # Create candidates mask in tile coordinates
    cand_mask = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
    for t, y in all_candidates:
        tr, tc = t - r0, y - c0
        if 0 <= tr < cand_mask.shape[0] and 0 <= tc < cand_mask.shape[1]:
            cand_mask[tr, tc] = 1.0

    return tile, track_mask, cand_mask, window


def get_cand_from_pmap(pmap: np.ndarray, threshold: float) -> list[Coord]:
    """Extract candidate coordinates from probability map.

    Port of Mathematica GetCandFromPmap.

    Args:
        pmap: Probability map from vision module
        threshold: Binarization threshold

    Returns:
        List of candidate coordinates from largest connected component
    """
    binary = (pmap > threshold).astype(np.int32)
    labels, n_components = label(binary)

    if n_components == 0:
        return []

    # Find largest component
    max_size = 0
    max_label = 0
    for lb in range(1, n_components + 1):
        size = np.sum(labels == lb)
        if size > max_size:
            max_size = size
            max_label = lb

    if max_label == 0:
        return []

    # Get coordinates of largest component
    coords = np.argwhere(labels == max_label)
    return [(int(r), int(c)) for r, c in coords]


def go_back(track: list[Coord]) -> list[Coord]:
    """Remove trailing backwards-in-time coordinates from track.

    Port of Mathematica GoBack with bounds checking fix.

    Args:
        track: Current track

    Returns:
        Track with backwards tail removed
    """
    if len(track) < 2:
        return track

    i = -1
    # Find where track stops going backwards
    while -i < len(track) and -i - 1 < len(track):
        if track[i][0] - track[i - 1][0] > 0:
            break
        i -= 1
        # Bounds check fix from audit
        if -i >= len(track) - 1:
            break

    if i == -1:
        return track
    return track[:i] if i < -1 else track


def get_cand(
    kym: np.ndarray,
    track: list[Coord],
    all_candidates: list[Coord],
    threshold: float,
    vision_module: VisionModuleProtocol | None,
    dim: int = 48,
) -> list[Coord]:
    """Get next candidate coordinates using vision module.

    Port of Mathematica GetCand.

    Args:
        kym: Padded kymograph
        track: Current track
        all_candidates: All candidate coordinates
        threshold: Probability threshold
        vision_module: Vision module network (or None for fallback)
        dim: Tile dimension

    Returns:
        List of new candidate coordinates
    """
    if not all_candidates or not track:
        return []

    last_track = track[-1]

    # Find nearby candidates
    if not all_candidates:
        return []

    cand_arr = np.array(all_candidates, dtype=np.float32)
    tree = cKDTree(cand_arr)
    nearby_idx = tree.query_ball_point(last_track, r=dim * 1.5)

    if not nearby_idx:
        return []

    nearby = [all_candidates[i] for i in nearby_idx]
    nearby.sort(key=lambda x: x[0])  # Sort by time

    # Get tile
    tile, track_mask, cand_mask, window = get_tile(kym, track[:-1], nearby, dim)

    # Run vision module if available
    if vision_module is not None:
        pmap = vision_module(tile, track_mask, cand_mask)
        cands = get_cand_from_pmap(pmap, threshold)
    else:
        # Fallback: use candidate mask directly
        cands = [(int(r), int(c)) for r, c in np.argwhere(cand_mask > 0)]

    # Remove coordinates already in track
    track_set = set(track)
    cands = [c for c in cands if c not in track_set]

    if len(cands) <= 2:
        return []

    # Sort by distance to last track point
    r0, c0 = window[0].start, window[1].start
    last_in_tile = (last_track[0] - r0, last_track[1] - c0)
    cands.sort(key=lambda c: np.sqrt((c[0] - last_in_tile[0]) ** 2 + (c[1] - last_in_tile[1]) ** 2))

    # Check if candidates are forward in time
    cand_arr = np.array(cands)
    if np.mean(cand_arr[:, 0]) - last_in_tile[0] < -1:
        return []

    # Sort into continuous path
    cands = sort_coords(np.array(cands)).tolist()

    # Pathfinding to fill gaps
    if cands and np.sqrt((cands[0][0] - last_in_tile[0]) ** 2 + (cands[0][1] - last_in_tile[1]) ** 2) > 1.5:
        # Build binary for pathfinding
        path_bin = np.zeros(tile.shape, dtype=bool)
        for r, c in nearby:
            tr, tc = r - r0, c - c0
            if 0 <= tr < path_bin.shape[0] and 0 <= tc < path_bin.shape[1]:
                path_bin[tr, tc] = True

        shortpath = find_short_path_image(path_bin, last_in_tile, tuple(cands[0]))
        if shortpath:
            cands = shortpath + cands

    # Limit to 24 coordinates
    cands = cands[:24]

    # Check again for forward progress
    if cands:
        cand_arr = np.array(cands)
        if np.mean(cand_arr[:, 0]) - last_in_tile[0] < -1:
            return []

    # Convert back to global coordinates
    result = [(c[0] + r0, c[1] + c0) for c in cands]
    return result


def get_next_coord(
    track: list[Coord],
    backwards_count: int,
    all_candidates: list[Coord],
    kym: np.ndarray,
    threshold: float,
    vision_module: VisionModuleProtocol | None,
) -> tuple[list[Coord], int]:
    """Get next coordinate for track extension.

    Port of Mathematica GetNextCoord.

    Args:
        track: Current track
        backwards_count: Count of backwards steps
        all_candidates: All candidate coordinates
        kym: Kymograph image
        threshold: Decision threshold
        vision_module: Vision module network

    Returns:
        (extended_track, new_backwards_count)
    """
    if not track:
        return [(-1, -1)], backwards_count

    last = track[-1]
    track_set = set(track)

    # Find candidates within 1.5 pixels
    if all_candidates:
        cand_arr = np.array(all_candidates, dtype=np.float32)
        tree = cKDTree(cand_arr)
        nearby_idx = tree.query_ball_point(last, r=1.5)
        cand = [all_candidates[i] for i in nearby_idx if all_candidates[i] not in track_set]
    else:
        cand = []

    # If not exactly one candidate, use vision module
    if len(cand) != 1:
        if len(track) > 2:
            cand = get_cand(kym, track, all_candidates, threshold, vision_module)
        else:
            cand = []
    else:
        cand = [cand[0]]

    if not cand:
        return track + [(-1, -1)], backwards_count

    # Check if going backwards in time
    new_last = cand[-1]
    if new_last[0] > last[0]:
        backwards_count = 0
    elif new_last[0] < last[0]:
        if backwards_count < 1:
            backwards_count += 1
        else:
            # Too many backwards steps, trim track
            track = go_back(track)
            return track + [(-1, -1)], backwards_count

    return track + cand, backwards_count


def make_track(
    kym: np.ndarray,
    all_candidates: list[Coord],
    threshold: float,
    seed: Coord,
    vision_module: VisionModuleProtocol | None,
) -> Track:
    """Build a complete track starting from a seed point.

    Port of Mathematica MakeTrack.

    Args:
        kym: Kymograph image
        all_candidates: All candidate coordinates
        threshold: Decision threshold
        seed: Starting seed coordinate
        vision_module: Vision module network

    Returns:
        Complete track as list of (time, position) coordinates
    """
    h, w = kym.shape
    backwards_count = 0

    # Find first neighbor
    if all_candidates:
        cand_arr = np.array(all_candidates, dtype=np.float32)
        tree = cKDTree(cand_arr)
        nearby_idx = tree.query_ball_point(seed, r=1.5)
        neighbors = [all_candidates[i] for i in nearby_idx if all_candidates[i] != seed]
    else:
        neighbors = []

    if len(neighbors) == 0:
        return [seed]

    if len(neighbors) > 1:
        # Pick the one furthest forward in time
        neighbors.sort(key=lambda x: x[0])
        neighbors = [neighbors[-1]]

    # Start track with seed and first neighbor
    track: list[Coord] = [seed, neighbors[0]]

    # Extend track iteratively
    while True:
        track, backwards_count = get_next_coord(
            track, backwards_count, all_candidates, kym, threshold, vision_module
        )
        if track[-1] == (-1, -1):
            track = track[:-1]  # Remove sentinel
            break

    # Remove out-of-bounds and invalid coordinates
    track = [
        (t, y)
        for t, y in track
        if 0 < t <= h and 0 < y <= w and t != 0 and y != 0
    ]

    return track


def extract_seeds(skeleton: np.ndarray) -> list[Coord]:
    """Extract seed points (endpoints) from skeleton.

    Args:
        skeleton: Binary skeleton image

    Returns:
        List of endpoint coordinates
    """
    # Endpoint kernel: pixel with exactly one neighbor
    h, w = skeleton.shape
    seeds: list[Coord] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if skeleton[r, c]:
                # Count 8-neighbors
                neighbors = (
                    skeleton[r - 1, c - 1]
                    + skeleton[r - 1, c]
                    + skeleton[r - 1, c + 1]
                    + skeleton[r, c - 1]
                    + skeleton[r, c + 1]
                    + skeleton[r + 1, c - 1]
                    + skeleton[r + 1, c]
                    + skeleton[r + 1, c + 1]
                )
                if neighbors == 1:  # Endpoint
                    seeds.append((r, c))

    return seeds


def bi_track(
    kym: np.ndarray,
    segmentation: np.ndarray,
    threshold: float,
    vision_module: VisionModuleProtocol | None,
    min_size: int = 10,
    min_frames: int = 5,
) -> list[Track]:
    """Run full bidirectional tracking algorithm.

    Args:
        kym: Preprocessed kymograph
        segmentation: Binary segmentation (skeleton)
        threshold: Decision threshold for vision module
        vision_module: Vision module network (or None for fallback)
        min_size: Minimum track length in pixels
        min_frames: Minimum track duration in frames

    Returns:
        List of extracted tracks
    """
    # Get all candidate positions from segmentation
    all_candidates = [(int(r), int(c)) for r, c in np.argwhere(segmentation > 0)]

    if not all_candidates:
        return []

    # Get seed points
    seeds = extract_seeds(segmentation)

    if not seeds:
        # Fallback: use endpoints of connected components
        labels, n = label(segmentation)
        for lb in range(1, n + 1):
            coords = np.argwhere(labels == lb)
            if len(coords) > 0:
                # Use first and last point as seeds
                coords = sorted(coords.tolist(), key=lambda x: x[0])
                seeds.append(tuple(coords[0]))
                if len(coords) > 1:
                    seeds.append(tuple(coords[-1]))

    # Build tracks from each seed
    tracks: list[Track] = []
    used_pixels: set[Coord] = set()

    for seed in seeds:
        if seed in used_pixels:
            continue

        track = make_track(kym, all_candidates, threshold, seed, vision_module)

        if len(track) >= min_size:
            duration = track[-1][0] - track[0][0] if track else 0
            if duration >= min_frames:
                tracks.append(track)
                used_pixels.update(track)

    # Remove subset tracks
    filtered: list[Track] = []
    for i, t1 in enumerate(tracks):
        is_subset = False
        s1 = set(t1)
        for j, t2 in enumerate(tracks):
            if i == j:
                continue
            s2 = set(t2)
            if len(s1 & s2) == len(s1) and len(s2) > len(s1):
                is_subset = True
                break
        if not is_subset:
            filtered.append(t1)

    return filtered
