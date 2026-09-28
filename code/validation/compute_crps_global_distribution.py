#!/usr/bin/env python3
"""
Global CRPS cold-pool distribution diagnostics (normalized by ASCAT overpasses).

Builds a compact publication figure from density-matched CRPS monthly GeoPackages:
  (a) Detection frequency (%) = object centroids / ASCAT overpasses × 100
  (b) Mean object area (km²)
  (c) Object-area PDF (log-x)

Overpass denominators are taken from the existing REMSS-based overpass grids
(same ASCAT sampling footprint as the multiamb occurrence climatology). Object
numerators/areas come from CRPS GPKGs.

Example
-------
  env -u PYTHONPATH -u LD_LIBRARY_PATH \\
    /rhome/gpriftis/miniconda3/envs/impact/bin/python \\
    validation/compute_crps_global_distribution.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

try:
    from paths_config import ensure_sys_path, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, get, research_root  # type: ignore
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

# Grid matches Garg Fig. 13 / multiamb occurrence climatology
LON_MIN, LON_MAX = -180.0, 180.0
GRID_RES = 0.5
LAT_MIN, LAT_MAX = -30.0, 30.0
NLON = int(round((LON_MAX - LON_MIN) / GRID_RES))
NLAT = int(round((LAT_MAX - LAT_MIN) / GRID_RES))

SERIES = ("ascat_a", "ascat_b", "ascat_c")
DEFAULT_GPKG_ROOT = Path(get("GPKG_ROOT") or str(ROOT / "evaluation/fullswath/crps"))
DEFAULT_OVERPASS = Path(get("OVERPASS_CLIM") or str(ROOT / "evaluation/figures/occurrence_climatology"))
DEFAULT_OUT = ROOT / "evaluation/global_distribution/crps"
AREA_MAX_KM2 = 5.0e5


def lon_to_j(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.floor((lon - LON_MIN) / GRID_RES).astype(np.int64)


def lat_to_i(lat: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    return np.floor((lat - LAT_MIN) / GRID_RES).astype(np.int64)


def bin_centroids(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.zeros((NLAT, NLON), dtype=np.int64)
    vsum = np.zeros((NLAT, NLON), dtype=np.float64)
    j = lon_to_j(lon)
    i = lat_to_i(lat)
    ok = (
        (i >= 0) & (i < NLAT) & (j >= 0) & (j < NLON)
        & np.isfinite(lon) & np.isfinite(lat)
    )
    if not np.any(ok):
        return count, vsum, count.copy()
    i = i[ok]
    j = j[ok]
    np.add.at(count, (i, j), 1)
    if values is not None:
        v = np.asarray(values, dtype=float)[ok]
        good = np.isfinite(v)
        if np.any(good):
            np.add.at(vsum, (i[good], j[good]), v[good])
    return count, vsum, count.copy()


def season_of_month(month: int) -> str | None:
    if month in (12, 1, 2):
        return "DJF"
    if month in (6, 7, 8):
        return "JJA"
    return None


def aggregate_crps_gpkgs(
    gpkg_root: Path,
) -> Dict[str, np.ndarray]:
    """Stream all A/B/C monthly GPKGs into combined grids + size sample."""
    object_count = np.zeros((NLAT, NLON), dtype=np.int64)
    area_sum = np.zeros((NLAT, NLON), dtype=np.float64)
    area_n = np.zeros((NLAT, NLON), dtype=np.int64)
    season_count = {
        "DJF": np.zeros((NLAT, NLON), dtype=np.int64),
        "JJA": np.zeros((NLAT, NLON), dtype=np.int64),
    }
    areas_sample: List[float] = []
    n_objects = 0
    n_files = 0
    year_min, year_max = 10**9, -1

    for series in SERIES:
        gdir = gpkg_root / series / "gpkg"
        paths = sorted(gdir.glob(f"{series}_fullswath_crps_coldpools_*.gpkg"))
        print(f"{series}: {len(paths)} gpkgs", flush=True)
        for path in paths:
            n_files += 1
            # YYYYMM from filename
            stem = path.stem
            yyyymm = stem.rsplit("_", 1)[-1]
            year = int(yyyymm[:4])
            month = int(yyyymm[4:6])
            year_min = min(year_min, year)
            year_max = max(year_max, year)
            season = season_of_month(month)

            con = sqlite3.connect(str(path))
            try:
                rows = con.execute(
                    "SELECT centroid_lon, centroid_lat, "
                    "COALESCE(area_grid_km2, area_km2) AS area "
                    "FROM cold_pools "
                    "WHERE centroid_lat BETWEEN ? AND ?",
                    (LAT_MIN, LAT_MAX),
                ).fetchall()
            finally:
                con.close()
            if not rows:
                continue
            lon = np.asarray([r[0] for r in rows], dtype=float)
            lat = np.asarray([r[1] for r in rows], dtype=float)
            area = np.asarray([r[2] for r in rows], dtype=float)

            c, _, _ = bin_centroids(lon, lat)
            object_count += c
            n_objects += int(c.sum())
            if season is not None:
                season_count[season] += c

            area_ok = np.isfinite(area) & (area > 0) & (area < AREA_MAX_KM2)
            if np.any(area_ok):
                _, a_sum, a_cnt = bin_centroids(lon[area_ok], lat[area_ok], area[area_ok])
                # a_cnt from bin_centroids is counts of points with values
                area_sum += a_sum
                # recount with area_ok only
                c_a, _, _ = bin_centroids(lon[area_ok], lat[area_ok])
                area_n += c_a
                # reservoir-ish sample for PDF (cap memory)
                if len(areas_sample) < 2_000_000:
                    areas_sample.extend(area[area_ok].tolist())

            if n_files % 50 == 0:
                print(f"  … {n_files} files, objects={n_objects:,}", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        area_mean = np.where(area_n > 0, area_sum / np.maximum(area_n, 1), np.nan)

    return {
        "object_count": object_count,
        "area_mean": area_mean.astype(np.float32),
        "area_n": area_n,
        "djf_count": season_count["DJF"],
        "jja_count": season_count["JJA"],
        "areas_sample": np.asarray(areas_sample, dtype=np.float32),
        "n_objects": np.asarray(n_objects),
        "n_files": np.asarray(n_files),
        "year_start": np.asarray(year_min),
        "year_end": np.asarray(year_max),
    }


def load_combined_overpass(overpass_dir: Path) -> Tuple[np.ndarray, dict]:
    """Sum A/B/C overpass grids (ASCAT sampling denominator)."""
    total = np.zeros((NLAT, NLON), dtype=np.float64)
    meta = {"sources": []}
    n_swaths = 0
    for series in SERIES:
        path = overpass_dir / f"occurrence_grids_{series}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        z = np.load(path)
        total += z["overpass_count"].astype(np.float64)
        n_swaths += int(z["n_overpass_swaths"])
        meta["sources"].append(
            {
                "series": series,
                "year_start": int(z["year_start"]),
                "year_end": int(z["year_end"]),
                "n_overpass_swaths": int(z["n_overpass_swaths"]),
            }
        )
    return total.astype(np.float64), {
        **meta,
        "n_overpass_swaths": n_swaths,
    }


def frequency_pct(object_count: np.ndarray, overpass: np.ndarray, min_overpasses: int = 20) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(
            overpass >= min_overpasses,
            100.0 * object_count / overpass,
            np.nan,
        )
    return freq.astype(np.float32)


def roll_lon0_left(grid: np.ndarray) -> np.ndarray:
    """Shift so plot x runs 0..360 with 0 at left (Atlantic-centered maps optional)."""
    # Keep -180..180 native; plotting uses lon edges directly.
    return np.asarray(grid, dtype=float)


def plot_figure(
    freq: np.ndarray,
    area_mean: np.ndarray,
    areas: np.ndarray,
    djf: np.ndarray,
    jja: np.ndarray,
    overpass: np.ndarray,
    out_png: Path,
    *,
    year_start: int,
    year_end: int,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Row = map + colorbar column so colorbar height matches the map axes.
    # right must leave room for colorbar labels (esp. Area km^2) without
    # bbox_inches='tight'; keep it tight so the margin is not huge.
    fig = plt.figure(figsize=(11.0, 7.8))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 0.028],
        wspace=0.05,
        hspace=0.18,
        left=0.06,
        right=0.905,
        top=0.98,
        bottom=0.05,
    )
    map_cax_pairs = []

    def _map(row, label, data, vmin, vmax, cmap, cbar_label, *, extend="max"):
        ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
        cax = fig.add_subplot(gs[row, 1])
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_aspect("auto")
        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#d9d9d9",
            edgecolor="none",
            zorder=0,
        )
        ax.coastlines("110m", linewidth=0.5, color="k", zorder=4)
        plot = np.ma.masked_invalid(np.asarray(data, dtype=float))
        im = ax.imshow(
            plot,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=2,
            aspect="auto",
        )
        cb = fig.colorbar(im, cax=cax, extend=extend, extendfrac=0.0)
        cb.set_label(cbar_label, fontsize=9, labelpad=4)
        # Same longitude over Africa; slightly lower than the map top edge.
        ax.text(
            20.0, 22.0, label,
            transform=ccrs.PlateCarree(),
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="0.25",
                linewidth=0.7,
                alpha=0.92,
            ),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--",
        )
        # Cartopy API differs by version; set both styles.
        for attr, val in (
            ("top_labels", False),
            ("right_labels", False),
            ("xlabels_top", False),
            ("ylabels_right", False),
            ("bottom_labels", True),
            ("left_labels", True),
            ("xlabels_bottom", True),
            ("ylabels_left", True),
        ):
            if hasattr(gl, attr):
                setattr(gl, attr, val)
        map_cax_pairs.append((ax, cax))

    f = freq.copy()
    finite = f[np.isfinite(f) & (overpass >= 20)]
    vmax_f = float(np.nanpercentile(finite, 98)) if finite.size else 5.0
    vmax_f = max(vmax_f, 1.0)
    _map(0, "(a)", f, 0.0, vmax_f, "YlOrRd", "Frequency (%)")

    a = area_mean.copy()
    a[overpass < 20] = np.nan
    finite_a = a[np.isfinite(a)]
    vmax_a = float(np.nanpercentile(finite_a, 95)) if finite_a.size else 2.0e4
    _map(1, "(b)", a, 0.0, vmax_a, "YlGnBu", r"Area (km$^2$)")

    with np.errstate(divide="ignore", invalid="ignore"):
        seas_over = overpass * (3.0 / 12.0)
        djf_f = np.where(seas_over >= 5, 100.0 * djf / seas_over, np.nan)
        jja_f = np.where(seas_over >= 5, 100.0 * jja / seas_over, np.nan)
        diff = djf_f - jja_f
    vmax_d = (
        float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95))
        if np.isfinite(diff).any()
        else 1.0
    )
    vmax_d = max(vmax_d, 0.2)
    _map(
        2, "(c)", diff, -vmax_d, vmax_d, "RdBu_r", r"$\Delta$ frequency (%)",
        extend="both",
    )

    # Exact colorbar height = map axes height (after Cartopy layout).
    fig.canvas.draw()
    for ax, cax in map_cax_pairs:
        pos = ax.get_position()
        cpos = cax.get_position()
        cax.set_position([cpos.x0, pos.y0, cpos.width, pos.height])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Sync cbar heights first, then crop to artists so right margin stays
    # tight without clipping colorbar labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer).expanded(1.015, 1.01)
    fig.savefig(out_png, dpi=220, facecolor="white", bbox_inches=bbox)
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches=bbox)
    plt.close(fig)
    print("wrote", out_png)


def write_snippet(out_path: Path, meta: dict) -> None:
    text = rf"""\subsection{{Global Distribution and Characteristics of Oceanic Cold Pools}}

The density-matched CRPS probabilistic catalog
($\tau=1.2\times10^{{-4}}$, $p^*=0.325$) provides a multi-platform record of oceanic
cold pools spanning ASCAT-A/B/C ({meta['year_start']}--{meta['year_end']};
{meta['n_objects']:,} tropical objects). Because the number of operating satellites
changes through the record, spatial diagnostics are shown as
\textbf{{overpass-normalized detection frequency}}
(object centroids divided by ASCAT overpass counts on a $0.5^\circ$ grid), rather than
raw object counts.

Detection frequency is highest in the tropical warm pools and monsoon-influenced basins,
with a clear seasonal preference revealed by the DJF$-$JJA contrast. Mean object areas
are typically a few thousand km$^2$ (catalog-wide median ${meta['area_median_km2']:.0f}$\,km$^2$,
mean ${meta['area_mean_km2']:.0f}$\,km$^2$). Together these maps summarize the global
spatial pattern, seasonal variability, and characteristic sizes of the probabilistic
ASCAT cold-pool catalog.

"""
    out_path.write_text(text)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpkg-root", default=str(DEFAULT_GPKG_ROOT))
    p.add_argument("--overpass-dir", default=str(DEFAULT_OVERPASS))
    p.add_argument("--outdir", default=str(DEFAULT_OUT))
    p.add_argument("--min-overpasses", type=int, default=20)
    p.add_argument(
        "--pack-dir",
        default=str(ROOT / "evaluation/paper_revision_pack_crps_ensemble/01_figures"),
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Aggregating CRPS GeoPackages …", flush=True)
    agg = aggregate_crps_gpkgs(Path(args.gpkg_root))
    overpass, op_meta = load_combined_overpass(Path(args.overpass_dir))
    freq = frequency_pct(agg["object_count"], overpass, min_overpasses=args.min_overpasses)

    np.savez_compressed(
        outdir / "crps_global_distribution_grids.npz",
        object_count=agg["object_count"],
        overpass_count=overpass,
        frequency_pct=freq,
        area_mean=agg["area_mean"],
        djf_count=agg["djf_count"],
        jja_count=agg["jja_count"],
        n_objects=agg["n_objects"],
        year_start=agg["year_start"],
        year_end=agg["year_end"],
        grid_res=GRID_RES,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
    )
    areas = agg["areas_sample"]
    meta = {
        "tau": 1.2e-4,
        "p_star": 0.325,
        "year_start": int(agg["year_start"]),
        "year_end": int(agg["year_end"]),
        "n_objects": int(agg["n_objects"]),
        "n_gpkg_files": int(agg["n_files"]),
        "area_median_km2": float(np.median(areas)) if areas.size else None,
        "area_mean_km2": float(np.mean(areas)) if areas.size else None,
        "normalization": "object_centroids / ASCAT_overpass_count * 100",
        "overpass_source": op_meta,
        "note": (
            "Overpass denominators from REMSS coverage grids "
            "(multiamb occurrence climatology checkpoints); "
            "numerators from CRPS density-matched GPKGs."
        ),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    fig_path = outdir / "fig_global_distribution.png"
    plot_figure(
        freq,
        agg["area_mean"],
        areas,
        agg["djf_count"],
        agg["jja_count"],
        overpass,
        fig_path,
        year_start=int(agg["year_start"]),
        year_end=int(agg["year_end"]),
    )
    write_snippet(outdir / "paper_snippet.tex", meta)

    pack = Path(args.pack_dir)
    pack.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        ("fig_global_distribution.png", "12_global_distribution.png"),
        ("fig_global_distribution.pdf", "12_global_distribution.pdf"),
        ("paper_snippet.tex", "12_global_distribution_paper_snippet.tex"),
        ("run_meta.json", "12_global_distribution_meta.json"),
    ]:
        s = outdir / src
        if s.exists():
            (pack / dst).write_bytes(s.read_bytes())
            print("copied", pack / dst)


if __name__ == "__main__":
    main()
