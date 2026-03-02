"""Post-processing for KymoButler tracks."""

from __future__ import annotations

from typing import Any

import numpy as np


Track = list[tuple[int, int]]


def get_derived_quantities(track: Track) -> dict[str, float | int] | None:
    """Compute per-track summary metrics.

    Returns None for tracks shorter than 2 points.
    """
    if len(track) <= 1:
        return None

    arr = np.asarray(track, dtype=np.float32)
    diff = np.diff(arr, axis=0)
    dt = np.abs(diff[:, 0])
    dy = np.abs(diff[:, 1])

    valid = dt > 0
    v = float(np.mean(dy[valid] / dt[valid])) if np.any(valid) else 0.0
    direct = int(np.sign(arr[-1, 1] - arr[0, 1]))
    dist = float(np.sum(np.abs(np.diff(arr[:, 1]))))
    t = float(abs(arr[-1, 0] - arr[0, 0] + 1.0))

    return {
        "direct": direct,
        "v": v,
        "dist": dist,
        "pauseT": 0.0,
        "T": t,
        "reversals": 0,
    }


def _hist(values: np.ndarray, bins: int = 20) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins) if values.size else (np.array([]), np.array([]))
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def _summary_rows(
    quant: list[dict[str, float | int]],
    tsz: float,
    xsz: float,
) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for q in quant:
        vel = float(xsz / tsz * float(q["v"]))
        dur = float(tsz * float(q["T"]))
        dist = float(xsz * float(q["dist"]))
        se_vel = float(dist / dur) if dur > 0 else 0.0
        rows.append([
            int(q["direct"]),
            round(vel, 4),
            round(dur, 4),
            round(dist, 4),
            round(se_vel, 4),
        ])
    return rows


def pproc_local(
    tracks: list[Track],
    tsz: float,
    xsz: float,
) -> dict[str, Any]:
    """Local post-processing equivalent of Mathematica `pprocLocal`."""
    quant = [q for q in (get_derived_quantities(t) for t in tracks) if q is not None]

    vvals = np.asarray([xsz / tsz * float(q["v"]) for q in quant], dtype=np.float32)
    tvals = np.asarray([tsz * float(q["T"]) for q in quant], dtype=np.float32)
    dvals = np.asarray([xsz * float(q["dist"]) for q in quant], dtype=np.float32)

    return {
        "histograms": {
            "velocity_um_per_sec": _hist(vvals),
            "duration_sec": _hist(tvals),
            "distance_um": _hist(dvals),
        },
        "metadata": [f"pixelsize time= {tsz} sec", f"pixelsize space= {xsz} um"],
        "columns": [
            "Direction",
            "Av frame2frame velocity [um/sec]",
            "track duration [sec]",
            "track total displacement [um]",
            "Start2end velocity [um/sec]",
        ],
        "rows": _summary_rows(quant, tsz=tsz, xsz=xsz),
    }


def pproc(
    tracks: list[Track],
    tsz: float,
    xsz: float,
    min_t: int,
    min_sz: int,
    thr: float,
    cls: str,
    version: str,
) -> dict[str, Any]:
    """Full post-processing equivalent of Mathematica `pproc`."""
    del cls

    quant = [q for q in (get_derived_quantities(t) for t in tracks) if q is not None]

    vvals = np.asarray([xsz / tsz * float(q["v"]) for q in quant], dtype=np.float32)
    tvals = np.asarray([tsz * float(q["T"]) for q in quant], dtype=np.float32)
    dvals = np.asarray([xsz * float(q["dist"]) for q in quant], dtype=np.float32)

    return {
        "histograms": {
            "velocity_um_per_sec": _hist(vvals),
            "duration_sec": _hist(tvals),
            "distance_um": _hist(dvals),
        },
        "metadata": [
            f"KymoButler Version {version} Summary",
            f"pixelsize time= {tsz} sec",
            f"pixelsize space= {xsz} um",
            f"minimum frames= {min_t}",
            f"minimum obj size= {min_sz}",
            f"threshold= {thr}",
        ],
        "columns": [
            "Direction",
            "Av frame2frame velocity [um/sec]",
            "track duration [sec]",
            "track total displacement [um]",
            "Start2end velocity [um/sec]",
        ],
        "rows": _summary_rows(quant, tsz=tsz, xsz=xsz),
    }
