#!/usr/bin/env python3
"""
Cross-platform monthly stats for CRPS density-matched catalogs (ASCAT-A/B/C).

Uses monthly GeoPackages + exceedance-stack valid masks (no re-inference).
Default period: 2019-01 .. 2021-12 (triple overlap; ASCAT-C from mid-2019).

Example
-------
  python validation/compute_crps_cross_platform_monthly.py
  python validation/compute_crps_cross_platform_monthly.py --workers 8 --resume
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from paths_config import ensure_sys_path, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, get, research_root  # type: ignore
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)


def load_config(path: Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def stack_path(infer_out: Path, orbit: int) -> Path:
    o = f"{int(orbit):05d}"
    return Path(infer_out) / o[:4] / f"{o}_exceedance.npz"


SERIES = ("ascat_a", "ascat_b", "ascat_c")
SERIES_CONFIG = {
    "ascat_a": "config_ascat_a_crps.json",
    "ascat_b": "config_ascat_b_crps.json",
    "ascat_c": "config_ascat_c_crps.json",
}


def gpkg_path(gpkg_root: Path, series: str, yyyymm: str) -> Path:
    return gpkg_root / series / "gpkg" / f"{series}_fullswath_crps_coldpools_{yyyymm}.gpkg"


def month_keys(start: str, end: str) -> List[str]:
    """Inclusive YYYY-MM keys."""
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    y, m = ys, ms
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def yyyymm(month_key: str) -> str:
    return month_key.replace("-", "")


def read_gpkg_object_stats(
    path: Path, lat_min: float, lat_max: float
) -> Tuple[int, int, np.ndarray]:
    """Return (n_objects, n_coldpool_pixels, areas_km2) for tropical objects."""
    if not path.exists():
        return 0, 0, np.asarray([], dtype=np.float64)
    con = sqlite3.connect(str(path))
    try:
        q = (
            "SELECT n_pixels, area_km2 FROM cold_pools "
            "WHERE centroid_lat BETWEEN ? AND ?"
        )
        rows = con.execute(q, (lat_min, lat_max)).fetchall()
    finally:
        con.close()
    if not rows:
        return 0, 0, np.asarray([], dtype=np.float64)
    n_pix = np.asarray([r[0] for r in rows], dtype=np.float64)
    areas = np.asarray([r[1] for r in rows], dtype=np.float64)
    return int(len(rows)), int(n_pix.sum()), areas


def _valid_count_one(args: Tuple[str, str, str]) -> Tuple[str, int]:
    """Worker: (orbit, infer_out, _) -> (orbit, n_valid)."""
    orbit, infer_out, _ = args
    try:
        sp = stack_path(Path(infer_out), int(orbit))
        if not sp.exists():
            return orbit, 0
        with np.load(sp) as z:
            valid = np.asarray(z["valid"], dtype=np.float32)
        return orbit, int((valid > 0).sum())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN valid fail orbit={orbit}: {exc}", flush=True)
        return orbit, 0


def accumulate_valid_for_month(
    orbits: List[str],
    infer_out: Path,
    cache_dir: Path,
    workers: int,
) -> int:
    """Sum valid pixels over orbits; cache per-orbit counts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    need = []
    total = 0
    for orbit in orbits:
        key = f"{int(orbit):05d}"
        cf = cache_dir / f"{key}.json"
        if cf.exists():
            total += int(json.loads(cf.read_text())["n_valid"])
        else:
            need.append((str(int(orbit)), str(infer_out), ""))

    if not need:
        return total

    results = []
    if workers <= 1:
        results = [_valid_count_one(a) for a in need]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_valid_count_one, a) for a in need]
            for fut in as_completed(futs):
                results.append(fut.result())

    for orbit, n_valid in results:
        key = f"{int(orbit):05d}"
        (cache_dir / f"{key}.json").write_text(
            json.dumps({"orbit": int(orbit), "n_valid": int(n_valid)}) + "\n"
        )
        total += int(n_valid)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2019-01")
    p.add_argument("--end", default="2021-12")
    p.add_argument("--lat-min", type=float, default=-30.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument(
        "--gpkg-root",
        default=str(ROOT / "evaluation/fullswath/crps"),
    )
    p.add_argument(
        "--outdir",
        default=str(ROOT / "evaluation/cross_platform/crps/triple_overlap"),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--skip-valid",
        action="store_true",
        help="Skip stack valid counts (object/area metrics only; frequency NaN)",
    )
    args = p.parse_args()

    gpkg_root = Path(args.gpkg_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    valid_cache = outdir / "valid_cache"
    months = month_keys(args.start, args.end)

    # Load configs / month indices
    cfgs = {}
    month_idx = {}
    for series in SERIES:
        cfg_path = ROOT / SERIES_CONFIG[series]
        cfgs[series] = load_config(cfg_path)
        idx = pd.read_csv(
            gpkg_root / series / f"{series}_swath_months.csv",
            dtype={"orbit": str},
        )
        month_idx[series] = idx

    op = json.loads(
        (ROOT / "evaluation/crps_operating_point/density_match_operating_point.json")
        .read_text()
    )
    meta = {
        "method": "crps_density_matched",
        "label": "CRPS probabilistic catalog (density-matched)",
        "tau": float(op["tau"]),
        "p_star": float(op["p_star"]),
        "start": args.start,
        "end": args.end,
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "note_valid": (
            "n_valid = sum of exceedance-stack valid pixels (full swath; "
            "objects lat-gated to [lat_min, lat_max])"
        ),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    rows = []
    for series in SERIES:
        infer_out = Path(cfgs[series]["infer_out"])
        idx = month_idx[series]
        for mk in months:
            ym = yyyymm(mk)
            gp = gpkg_path(gpkg_root, series, ym)
            n_obj, n_cp, areas = read_gpkg_object_stats(
                gp, args.lat_min, args.lat_max
            )
            orbits = idx.loc[idx["month"] == mk, "orbit"].astype(str).tolist()
            n_swaths = len(orbits)

            if args.skip_valid or n_swaths == 0:
                n_valid = 0
            else:
                print(
                    f"{series} {mk}: objects={n_obj} swaths={n_swaths} "
                    f"gpkg={'yes' if gp.exists() else 'NO'}",
                    flush=True,
                )
                n_valid = accumulate_valid_for_month(
                    orbits,
                    infer_out,
                    valid_cache / series,
                    workers=max(1, int(args.workers)),
                )

            freq = (n_cp / n_valid) if n_valid > 0 else np.nan
            dens = (n_obj / (n_valid / 1e6)) if n_valid > 0 else np.nan
            rows.append(
                {
                    "month": mk,
                    "satellite": series,
                    "method": "crps",
                    "n_swaths": n_swaths,
                    "n_valid": int(n_valid),
                    "n_coldpool": int(n_cp),
                    "n_objects": int(n_obj),
                    "coldpool_frequency": float(freq) if n_valid else np.nan,
                    "object_density_per_million": float(dens) if n_valid else np.nan,
                    "mean_object_area_km2": float(np.mean(areas)) if len(areas) else np.nan,
                    "median_object_area_km2": float(np.median(areas)) if len(areas) else np.nan,
                    "p95_object_area_km2": float(np.percentile(areas, 95))
                    if len(areas)
                    else np.nan,
                    "gpkg_exists": bool(gp.exists()),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = outdir / "monthly_stats.csv"
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv)
    print(df.groupby("satellite")[["n_objects", "n_swaths"]].sum())


if __name__ == "__main__":
    main()
