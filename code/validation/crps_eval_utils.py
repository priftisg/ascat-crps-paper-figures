"""Shared helpers for CRPS exceedance post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from lib.postprocess import extract_cold_pool_objects


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def stack_path(infer_out: str | Path, orbit: str | int) -> Path:
    orbit = f"{int(orbit):05d}" if str(orbit).isdigit() else str(orbit)
    return Path(infer_out) / orbit[:4] / f"{orbit}_exceedance.npz"


def load_stack(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def list_stack_files(infer_out: str | Path) -> List[Path]:
    root = Path(infer_out)
    return sorted(root.glob("*/*_exceedance.npz"))


def probability_map(stack: Dict[str, np.ndarray], tau_idx: int) -> np.ndarray:
    """Return float probability field in [0, 1] for one tau level."""
    p = stack["p"]
    if p.dtype == np.uint8:
        return p[tau_idx].astype(np.float32) / 255.0
    return p[tau_idx].astype(np.float32)


def tau_index(taus: np.ndarray, tau: float, rtol: float = 1e-3) -> int:
    taus = np.asarray(taus, dtype=float)
    i = int(np.argmin(np.abs(taus - float(tau))))
    if abs(taus[i] - float(tau)) > rtol * max(float(tau), 1e-12) + 1e-12:
        raise ValueError(f"tau={tau} not in stack taus={taus}")
    return i


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km). Inputs in degrees; broadcasts."""
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    lon1, lat1, lon2, lat2 = (np.deg2rad(a.astype(float)) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def extract_objects_from_stack(
    stack: Dict[str, np.ndarray],
    tau_idx: int,
    p_thresh: float,
    min_area: int = 4,
) -> Dict:
    pmap = probability_map(stack, tau_idx)
    valid = stack.get("valid")
    if valid is None:
        valid = np.ones(pmap.shape, dtype=np.float32)
    return extract_cold_pool_objects(pmap, valid, p_thresh, min_area)


def max_prob_near_point(
    pmap: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    valid: np.ndarray,
    buoy_lon: float,
    buoy_lat: float,
    radius_km: float,
) -> Tuple[float, float]:
    """
    Max exceedance probability within radius_km of a buoy.

    Returns (p_max, min_dist_km). If no valid pixel in radius, p_max=0 and
    min_dist_km is the nearest valid pixel distance.
    """
    dist = haversine_km(buoy_lon, buoy_lat, lon, lat)
    good = (valid > 0) & np.isfinite(dist) & np.isfinite(pmap)
    if not np.any(good):
        return 0.0, float("nan")
    d = dist[good]
    p = pmap[good]
    min_d = float(d.min())
    in_r = d <= float(radius_km)
    if not np.any(in_r):
        return 0.0, min_d
    return float(p[in_r].max()), min_d


def _bbox_mask(lon: np.ndarray, lat: np.ndarray, buoy_lon: float, buoy_lat: float, radius_km: float):
    """Cheap lat/lon window before haversine (~1.1 km per 0.01 deg lat)."""
    dlat = float(radius_km) / 111.0 + 0.05
    coslat = max(np.cos(np.deg2rad(buoy_lat)), 0.2)
    dlon = float(radius_km) / (111.0 * coslat) + 0.05
    return (
        (lat >= buoy_lat - dlat)
        & (lat <= buoy_lat + dlat)
        & (lon >= buoy_lon - dlon)
        & (lon <= buoy_lon + dlon)
    )


def score_buoy_on_swath(
    pmap: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    valid: np.ndarray,
    obj_mask: np.ndarray,
    buoy_lon: float,
    buoy_lat: float,
    radii_km: List[float],
    point_radius_km: float = 6.25,
) -> Dict[str, float]:
    """
    Exact-radius buoy scores on one swath.

    For each radius r in radii_km (r<=0 uses point_radius_km = ~½ ASCAT FOV):
      - p_max[r]: max P within exact r
      - det_obj[r]: 1 if any object pixel within r, else 0
      - p_obj[r]: max P on object pixels within r (0 if none)

    Also returns min_dist_km to nearest valid pixel and nearest object pixel.
    """
    radii = [float(point_radius_km if r <= 0 else r) for r in radii_km]
    r_max = max(radii) if radii else point_radius_km

    valid_f = (valid > 0) & np.isfinite(pmap) & np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid_f):
        out = {"min_dist_km": float("nan"), "min_obj_dist_km": float("nan")}
        for r0, r in zip(radii_km, radii):
            key = f"{float(r0):g}"
            out[f"p_max::{key}"] = 0.0
            out[f"det_obj::{key}"] = 0.0
            out[f"p_obj::{key}"] = 0.0
        return out

    # Restrict to bbox of largest radius for speed
    box = _bbox_mask(lon, lat, buoy_lon, buoy_lat, r_max + 20.0) & valid_f
    if not np.any(box):
        # fall back to full valid field for nearest-pixel distance only
        dist_all = haversine_km(buoy_lon, buoy_lat, lon[valid_f], lat[valid_f])
        min_d = float(dist_all.min())
        out = {"min_dist_km": min_d, "min_obj_dist_km": float("nan")}
        for r0 in radii_km:
            key = f"{float(r0):g}"
            out[f"p_max::{key}"] = 0.0
            out[f"det_obj::{key}"] = 0.0
            out[f"p_obj::{key}"] = 0.0
        return out

    lon_b, lat_b, p_b = lon[box], lat[box], pmap[box]
    obj_b = obj_mask[box] > 0
    dist = haversine_km(buoy_lon, buoy_lat, lon_b, lat_b)
    min_d = float(dist.min())
    if np.any(obj_b):
        min_obj = float(dist[obj_b].min())
    else:
        min_obj = float("nan")

    out = {"min_dist_km": min_d, "min_obj_dist_km": min_obj}
    for r0, r in zip(radii_km, radii):
        key = f"{float(r0):g}"
        in_r = dist <= r
        if np.any(in_r):
            out[f"p_max::{key}"] = float(p_b[in_r].max())
        else:
            out[f"p_max::{key}"] = 0.0
        in_obj = in_r & obj_b
        if np.any(in_obj):
            out[f"det_obj::{key}"] = 1.0
            out[f"p_obj::{key}"] = float(p_b[in_obj].max())
        else:
            out[f"det_obj::{key}"] = 0.0
            out[f"p_obj::{key}"] = 0.0
    return out
