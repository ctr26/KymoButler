"""Core KymoButler processing functions translated from KymoButler.wl."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image

from .utils import (
    binarize,
    component_centroids,
    connected_components,
    ensure_grayscale_array,
    gather_by_first_mean,
    image_adjust,
    is_negated,
    norm_lines,
    prune,
    resize_to_multiple_of_16,
    select_components,
    smooth_bin,
    smooth_bin_uni,
    zhang_suen_thinning,
)

Track = list[tuple[int, int]]


def _to_tensor(arr: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x.to(device)


def _resize_back(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return ensure_grayscale_array(img)


def _to_probmap(output: Any) -> np.ndarray:
    if isinstance(output, torch.Tensor):
        out = output.detach().cpu().numpy()
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3 and out.shape[0] >= 2:
            return out[1]
        if out.ndim == 2:
            return out
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return arr[1]
    return ensure_grayscale_array(arr)


def _extract_tracks(binary: np.ndarray, min_frames: int) -> list[Track]:
    labels, comps = connected_components(binary)
    tracks: list[Track] = []
    for comp in comps:
        coords = np.argwhere(labels == comp.label)
        if coords.size == 0:
            continue
        trk = gather_by_first_mean(coords[:, [0, 1]])
        if trk.size == 0:
            continue
        if int(trk[-1, 0] - trk[0, 0]) >= min_frames:
            tracks.append([(int(t), int(y)) for t, y in trk.tolist()])
    return tracks


def _tracks_to_overlay(base: np.ndarray, tracks: list[Track], seed: int = 1) -> np.ndarray:
    gray = ensure_grayscale_array(base)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rng = np.random.default_rng(seed)
    for trk in tracks:
        color = rng.uniform(0.25, 1.0, size=3)
        for t, y in trk:
            if 0 <= t < rgb.shape[0] and 0 <= y < rgb.shape[1]:
                rgb[t, y, :] = color
    return np.clip(rgb, 0.0, 1.0)


def _label_overlay(overlay: np.ndarray, labels: list[tuple[int, tuple[float, float]]], dark: bool) -> np.ndarray:
    out = overlay.copy()
    color = np.array([0.0, 0.0, 0.0]) if dark else np.array([1.0, 1.0, 1.0])
    for idx, (_, (r, c)) in enumerate(labels, start=1):
        rr, cc = int(round(r)), int(round(c))
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[max(rr - 1, 0) : min(rr + 2, out.shape[0]), cc, :] = color
            if cc + 1 < out.shape[1]:
                out[rr, cc + 1, :] = color
            if cc + 2 < out.shape[1]:
                out[rr, cc + 2, :] = color
            if idx % 2 == 0:
                out[rr, cc, :] = 1.0 - color
    return out


def _run_net(net: Callable[[torch.Tensor], Any], image: np.ndarray, target_device: str = "cpu") -> Any:
    with torch.no_grad():
        x = _to_tensor(image, device=target_device)
        return net(x)


def load_default_nets(directory: str | Path) -> dict[str, Path]:
    """Download default model files to `directory/models`.

    Returns local paths keyed by Mathematica names: `binet`, `classnet`, `uninet`, `decnet`.
    """
    base = Path(directory)
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    names = {
        "binet": "Bidirectional_Segmentation_Module_V1_1",
        "classnet": "Classification_Module_V1_0",
        "uninet": "Unidrectional_Segmentation_Module_V1_0",
        "decnet": "Decision_Module_V1_0",
    }
    cpath = "https://www.wolframcloud.com/objects/deepmirror/Projects/KymoButler/networks/"

    out: dict[str, Path] = {}
    for key, name in names.items():
        p = models_dir / name
        if not p.exists():
            urlretrieve(cpath + name, p)
        out[key] = p
    return out


def uni_kymobutler_segment(
    kym: np.ndarray,
    net: Callable[[torch.Tensor], Any],
    target_device: str = "cpu",
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Unidirectional segmentation stage."""
    raw = image_adjust(ensure_grayscale_array(kym))
    bool_neg = is_negated(raw)
    neg = 1.0 - raw if bool_neg else raw
    neg = norm_lines(neg)

    resized = resize_to_multiple_of_16(neg)
    out = _run_net(net, resized, target_device=target_device)

    if not isinstance(out, dict):
        raise TypeError("Unidirectional net must return a dict with 'ant' and 'ret'.")

    ant = _to_probmap(out["ant"])
    ret = _to_probmap(out["ret"])
    ant = _resize_back(ant, raw.shape)
    ret = _resize_back(ret, raw.shape)

    return bool_neg, raw, neg, {"ant": ant, "ret": ret}


def uni_kymobutler_track(
    bool_neg: bool,
    tmpkym: np.ndarray,
    out: dict[str, np.ndarray],
    binthresh: float,
    min_size: int = 3,
    min_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Track], list[Track]]:
    """Unidirectional tracking stage."""
    ant = binarize(out["ant"], binthresh)
    ret = binarize(out["ret"], binthresh)

    ant = zhang_suen_thinning(ant)
    ant = smooth_bin_uni(smooth_bin_uni(smooth_bin_uni(ant)))
    ant = prune(zhang_suen_thinning(ant), iterations=2)
    ant = select_components(ant, min_count=min_size, min_row_span=min_frames)

    ret = zhang_suen_thinning(ret)
    ret = smooth_bin_uni(smooth_bin_uni(smooth_bin_uni(ret)))
    ret = prune(zhang_suen_thinning(ret), iterations=2)
    ret = select_components(ret, min_count=min_size, min_row_span=min_frames)

    antrks = _extract_tracks(ant, min_frames=min_frames)
    retrks = _extract_tracks(ret, min_frames=min_frames)

    if antrks or retrks:
        tracks = antrks + retrks
        colored = _tracks_to_overlay(np.zeros_like(tmpkym), tracks, seed=7)
        overlay = _tracks_to_overlay(tmpkym, tracks, seed=7)

        ant_labeled = np.zeros_like(ant, dtype=np.int32)
        for i, trk in enumerate(antrks, start=1):
            for t, y in trk:
                ant_labeled[t, y] = i
        ret_labeled = np.zeros_like(ret, dtype=np.int32)
        for i, trk in enumerate(retrks, start=1 + len(antrks)):
            for t, y in trk:
                ret_labeled[t, y] = i
        labels = component_centroids(ant_labeled + ret_labeled)
        overlay_labeled = _label_overlay(overlay, labels, dark=bool_neg)
    else:
        colored = np.zeros((*tmpkym.shape, 3), dtype=np.float32)
        overlay = np.stack([tmpkym, tmpkym, tmpkym], axis=-1)
        overlay_labeled = overlay.copy()

    return tmpkym, colored, overlay, overlay_labeled, antrks, retrks


def uni_kymobutler(
    kym: np.ndarray,
    binthresh: float,
    target_device: str,
    net: Callable[[torch.Tensor], Any],
    min_size: int,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Track], list[Track]]:
    """Run full unidirectional KymoButler pipeline."""
    bool_neg, raw, _, out = uni_kymobutler_segment(kym, net, target_device=target_device)
    return uni_kymobutler_track(bool_neg, raw, out, binthresh, min_size=min_size, min_frames=min_frames)


def bi_kymobutler_segment(
    kym: np.ndarray,
    net: Callable[[torch.Tensor], Any],
    target_device: str = "cpu",
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Bidirectional segmentation stage."""
    kympre = image_adjust(ensure_grayscale_array(kym))
    bool_neg = is_negated(kympre)
    if bool_neg:
        kympre = 1.0 - kympre
    kympre = norm_lines(kympre)

    resized = resize_to_multiple_of_16(kympre)
    pred = _run_net(net, resized, target_device=target_device)
    pmap = _resize_back(_to_probmap(pred), kympre.shape)
    return bool_neg, ensure_grayscale_array(kym), kympre, pmap


def _remove_subset_tracks(tracks: list[Track], min_size: int) -> list[Track]:
    keep = [True] * len(tracks)
    sets = [set(trk) for trk in tracks]
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i == j:
                continue
            inter = len(sets[i].intersection(sets[j]))
            if abs(inter - len(tracks[j])) <= min_size and len(tracks[j]) < len(tracks[i]):
                keep[j] = False
    return [t for t, k in zip(tracks, keep) if k]


def _resolve_overlaps(tracks: list[Track], probs: list[float]) -> list[Track]:
    out = [list(t) for t in tracks]
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            ov = set(out[i]).intersection(out[j])
            if len(ov) > 10:
                if probs[i] >= probs[j]:
                    out[j] = [p for p in out[j] if p not in ov]
                else:
                    out[i] = [p for p in out[i] if p not in ov]
    return [t for t in out if t]


def bi_kymobutler_track(
    pred: np.ndarray,
    kym: np.ndarray,
    kympreproc: np.ndarray,
    bool_neg: bool,
    binthresh: float,
    vthr: float,
    vismod: Any,
    min_size: int,
    min_frames: int,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Track]]:
    """Bidirectional tracking stage.

    Uses the full iterative tracking algorithm from the original KymoButler,
    not the connected-components shortcut.

    Args:
        pred: Probability map from segmentation network
        kym: Raw kymograph
        kympreproc: Preprocessed kymograph
        bool_neg: Whether image was negated
        binthresh: Binarization threshold
        vthr: Vision module threshold
        vismod: Vision module network (can be None for fallback mode)
        min_size: Minimum track size in pixels
        min_frames: Minimum track duration in frames
        debug: Enable debug output
    """
    del debug  # Not used yet

    from .tracking import bi_track

    # Preprocess segmentation
    out = binarize(pred, binthresh)
    out = smooth_bin(smooth_bin(out))
    paths = select_components(
        prune(zhang_suen_thinning(out), iterations=3),
        min_count=min_size,
        min_row_span=min_frames,
    )

    # Run proper iterative tracking algorithm
    trks = bi_track(
        kym=kympreproc,
        segmentation=paths,
        threshold=vthr,
        vision_module=vismod,
        min_size=min_size,
        min_frames=min_frames,
    )

    # Post-process tracks
    probs = [1.0 for _ in trks]
    trks = _remove_subset_tracks(trks, min_size=min_size)
    trks = _resolve_overlaps(trks, probs=probs[: len(trks)])
    trks = [t for t in trks if len(t) > 1 and (t[-1][0] - t[0][0]) >= min_frames]

    if trks:
        colored = _tracks_to_overlay(np.zeros_like(kympreproc), trks, seed=11)
        overlay = _tracks_to_overlay(kym, trks, seed=11)
        labels_map = np.zeros_like(kympreproc, dtype=np.int32)
        for i, trk in enumerate(trks, start=1):
            for t, y in trk:
                labels_map[t, y] = i
        labels = component_centroids(labels_map)
        overlay_labeled = _label_overlay(overlay, labels, dark=bool_neg)
        return kym, colored, overlay, overlay_labeled, trks

    blank = np.zeros((*kym.shape, 3), dtype=np.float32)
    rgb = np.stack([kym, kym, kym], axis=-1)
    return kym, blank, rgb, rgb.copy(), []


def bi_kymobutler(
    kym: np.ndarray,
    binthresh: float,
    vthr: float,
    target_device: str,
    cnet: Callable[[torch.Tensor], Any],
    vismod: Any,
    min_size: int,
    min_frames: int,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Track]]:
    """Run full bidirectional KymoButler pipeline."""
    bool_neg, raw, kympre, pred = bi_kymobutler_segment(kym, cnet, target_device=target_device)
    return bi_kymobutler_track(
        pred,
        raw,
        kympre,
        bool_neg,
        binthresh,
        vthr,
        vismod,
        min_size,
        min_frames,
        debug=debug,
    )


def _find_close_track(nf: Callable[[tuple[int, int]], np.ndarray], trk: Track) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in trk:
        near = nf(p)
        out.extend((int(a), int(b)) for a, b in near)
    uniq = list(dict.fromkeys(out))
    return uniq


def _rec(nf_pred: list[Callable[[tuple[int, int]], np.ndarray]], syntrk: Track) -> float:
    tmp = sorted((_find_close_track(nf, syntrk) for nf in nf_pred), key=len)
    if tmp and len(tmp[-1]) > 0:
        return 1.0 - abs(len(tmp[-1]) - len(syntrk)) / max(len(syntrk), 1)
    return 0.0


def _prec(nf_gt: list[Callable[[tuple[int, int]], np.ndarray]], predtrk: Track) -> float:
    tmp = sorted((_find_close_track(nf, predtrk) for nf in nf_gt), key=len)
    if tmp and len(tmp[-1]) > 0:
        return 1.0 - abs(len(tmp[-1]) - len(predtrk)) / max(len(predtrk), 1)
    return 0.0


def benchmark_prediction(kym: np.ndarray, ptrks: list[Track], trks: list[Track]) -> dict[str, Any]:
    """Compute precision/recall/F1 for predicted tracks vs ground truth."""

    def _nearest_fn(pts: Track) -> Callable[[tuple[int, int]], np.ndarray]:
        arr = np.asarray(pts, dtype=np.float32)

        def f(p: tuple[int, int]) -> np.ndarray:
            if arr.size == 0:
                return np.empty((0, 2), dtype=np.int32)
            d = np.sqrt(((arr - np.asarray(p, dtype=np.float32)) ** 2).sum(axis=1))
            return arr[d <= 3.2].astype(np.int32)

        return f

    p_nonempty = [t for t in ptrks if t]
    t_nonempty = [t for t in trks if t]

    nf_pred = [_nearest_fn(t) for t in p_nonempty]
    nf_gt = [_nearest_fn(t) for t in t_nonempty]

    precision = float(np.mean([_prec(nf_gt, t) for t in p_nonempty])) if p_nonempty else 0.0
    recall = float(np.mean([_rec(nf_pred, t) for t in t_nonempty])) if t_nonempty else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "kym": ensure_grayscale_array(kym),
        "tracks": ptrks,
        "recall": recall,
        "precision": precision,
        "F1": f1,
    }
