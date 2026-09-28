#!/usr/bin/env python3
"""
Three-way swath maps: classical Garg | CRPS baseline OP | CRPS OP C.

Makes the density-tiering visible so OP C does not look "sparse by bug"
when compared against the full classical catalog (~50 vs ~16 objs/swath).

Run with the impact env (Cartopy), clearing PYTHONPATH:

  env -u PYTHONPATH -u LD_LIBRARY_PATH /rhome/gpriftis/miniconda3/envs/impact/bin/python \\
      validation/plot_crps_tier_comparison.py \\
      --orbits 72010,73598,75001,72850,74200 \\
      --outdir evaluation/figures/crps_tier_comparison

Publication layout (no suptitle, shared colorbar, tight gaps):

  ... --orbits 73598 --publication --outdir evaluation/paper_revision_pack_crps_ensemble/01_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from paths_config import ensure_sys_path, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, get, research_root  # type: ignore
from typing import Dict, List, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

from lib.gradient_labels import labels_for_full_swath  # noqa: E402
from lib.postprocess import extract_cold_pool_objects  # noqa: E402
from lib.swath_io import load_swath_mat, trim_swath_edges  # noqa: E402
from validation.plot_crps_vs_garg_maps import (  # noqa: E402
    build_orbit_index,
    load_config,
    load_stack,
    probability_map,
    split_segment_masks,
    stack_path,
    tau_index,
)


def _binary_style():
    cmap = ListedColormap(["#d9d9d9", "#d7301f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N, clip=True)
    return cmap, norm


def make_tier_figures(
    orbit: int,
    lon: np.ndarray,
    lat: np.ndarray,
    valid: np.ndarray,
    panels: List[Tuple[str, np.ndarray]],
    out_png_root: Path,
    title_note: str,
    *,
    show_suptitle: bool = True,
    wspace: float = 0.28,
    save_pdf: bool = False,
    shared_cbar: bool = False,
    lon_min_fixed: float = None,
    lon_max_fixed: float = None,
    panel_gap: float = None,
):
    """One row of binary detection panels per tropical segment."""
    ref = np.asarray(panels[0][1], dtype=np.float32)
    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(ref) & (valid > 0)
    mask &= (lat >= -30.0) & (lat <= 30.0)
    seg_masks = split_segment_masks(mask, max_segments=2)

    n = len(panels)
    cmap, norm = _binary_style()
    # Shared colorbar removes the large per-panel cbar gutters between maps.
    use_shared = bool(shared_cbar or wspace <= 0.12)

    for seg_idx, seg_mask in enumerate(seg_masks, start=1):
        if use_shared:
            # Absolute axes positions: modest gap; one shared cbar.
            fig_w = 4.05 * n + 0.55
            fig_h = 6.4
            fig = plt.figure(figsize=(fig_w, fig_h))
            left0 = 0.055
            right_maps = 0.945
            bottom = 0.075
            top = 0.88 if not show_suptitle else 0.84
            gap = 0.04 if panel_gap is None else float(panel_gap)
            usable = right_maps - left0
            panel_w = (usable - gap * (n - 1)) / float(n)
            panel_h = top - bottom
            cbar_left = 0.958
            cbar_w = 0.016
            fig.set_size_inches(4.5 * n + 0.7, 6.2, forward=True)
        else:
            fig = plt.figure(figsize=(5.8 * n, 7.2))
            gs = fig.add_gridspec(
                1, n, left=0.10, right=0.97, bottom=0.12, top=0.86, wspace=wspace,
            )

        lon_seg = lon[seg_mask]
        lat_seg = lat[seg_mask]
        # Prefer [-180, 180] for fixed crop / labels.
        lon_seg_plot = np.where(lon_seg > 180.0, lon_seg - 360.0, lon_seg)
        lon_min = float(np.nanmin(lon_seg_plot))
        lon_max = float(np.nanmax(lon_seg_plot))
        lat_min = max(-30.0, float(np.nanmin(lat_seg)) - 2.0)
        lat_max = min(30.0, float(np.nanmax(lat_seg)) + 2.0)
        if lon_min_fixed is not None and lon_max_fixed is not None:
            extent = [
                float(lon_min_fixed),
                float(lon_max_fixed),
                lat_min,
                lat_max,
            ]
        else:
            lon_pad = max(2.0, 0.07 * (lon_max - lon_min))
            extent = [lon_min - lon_pad, lon_max + lon_pad, lat_min, lat_max]

        last_sc = None
        for i, (title, field) in enumerate(panels):
            if use_shared:
                left = left0 + i * (panel_w + gap)
                ax = fig.add_axes(
                    [left, bottom, panel_w, panel_h],
                    projection=ccrs.PlateCarree(),
                )
            else:
                ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            # Fill the axes box (default Cartopy aspect leaves side gutters
            # that look like large gaps between panels).
            ax.set_aspect("auto")
            ax.coastlines("110m", linewidth=1.0, color="black")
            ax.tick_params(
                labeltop=False, labelright=False, labelbottom=False, labelleft=False,
                top=False, right=False, bottom=False, left=False,
            )
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=1.2,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            lon0, lon1, lat0, lat1 = extent

            def _fmt_lon(lon_val):
                x = ((float(lon_val) + 180.0) % 360.0) - 180.0
                return LONGITUDE_FORMATTER(x)

            if i == 0:
                for lat_t in np.arange(-30.0, 31.0, 10.0):
                    if lat0 < lat_t < lat1:
                        ax.text(
                            lon0 - 0.04 * (lon1 - lon0),
                            lat_t,
                            LATITUDE_FORMATTER(lat_t),
                            transform=ccrs.PlateCarree(),
                            ha="right",
                            va="center",
                            fontsize=9,
                            clip_on=False,
                        )
            # Lon ticks every 6°. Drop labels on the shared panel edges so
            # neighboring 102°W / 78°W text does not fill the inter-panel gap.
            step = 6.0
            lon_start = np.ceil((lon0 - 1e-6) / step) * step
            for lon_t in np.arange(lon_start, lon1 + 1e-6, step):
                if lon_t < lon0 - 0.25 or lon_t > lon1 + 0.25:
                    continue
                if use_shared and n > 1:
                    if i < n - 1 and abs(lon_t - lon1) < 0.6:
                        continue  # right edge (shared with next panel)
                    if i > 0 and abs(lon_t - lon0) < 0.6:
                        continue  # left edge (shared with previous panel)
                ax.text(
                    lon_t,
                    lat0 - 0.03 * (lat1 - lat0),
                    _fmt_lon(lon_t),
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="top",
                    fontsize=9,
                    clip_on=False,
                )

            this_data = np.asarray(field, dtype=np.float32)
            last_sc = ax.scatter(
                lon_seg_plot,
                lat_seg,
                c=this_data[seg_mask],
                s=4,
                cmap=cmap,
                norm=norm,
                linewidths=0,
                transform=ccrs.PlateCarree(),
            )
            if not use_shared:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(
                    "right", size="4.0%", pad=0.06, axes_class=plt.Axes
                )
                cbar = fig.colorbar(last_sc, cax=cax)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["0", "1"])
            ax.set_title(title, fontsize=11, pad=8)

        if use_shared and last_sc is not None:
            cax = fig.add_axes([cbar_left, bottom, cbar_w, panel_h])
            cbar = fig.colorbar(last_sc, cax=cax)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["0", "1"])

        if show_suptitle:
            fig.suptitle(
                "Swath {0} — Segment {1}  ({2})".format(orbit, seg_idx, title_note),
                y=0.96,
            )
        out_png = out_png_root.with_name(
            "{0}_segment{1}.png".format(out_png_root.stem, seg_idx)
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, pad_inches=0.15, bbox_inches="tight")
        if save_pdf:
            out_pdf = out_png.with_suffix(".pdf")
            fig.savefig(out_pdf, pad_inches=0.15, bbox_inches="tight")
            print("  wrote", out_pdf)
        plt.close(fig)
        print("  wrote", out_png)


def load_op(path: Path) -> Dict[str, float]:
    op = json.loads(path.read_text())
    return {"tau": float(op["tau"]), "p_star": float(op["p_star"])}


def crps_binary_at(
    stack: Dict[str, np.ndarray],
    valid: np.ndarray,
    tau: float,
    p_star: float,
    min_area: int,
) -> np.ndarray:
    taus = np.asarray(stack["taus"], dtype=float)
    ti = tau_index(taus, tau)
    pmap = probability_map(stack, ti)
    objs = extract_cold_pool_objects(pmap, valid, p_star, min_area)
    return objs["binary"].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config_ascat_a_crps.json")
    p.add_argument("--infer-out", help="CRPS exceedance root (default: config infer_out)")
    p.add_argument(
        "--op-baseline",
        default="evaluation/crps_operating_point/density_match_operating_point.json",
    )
    p.add_argument(
        "--op-c",
        default="evaluation/crps_operating_point_tau2e4/density_match_operating_point.json",
    )
    p.add_argument("--orbits", default="72010,73598,75001,72850,74200")
    p.add_argument("--outdir", default="evaluation/figures/crps_tier_comparison")
    p.add_argument(
        "--publication",
        action="store_true",
        help="Publication panel titles; no suptitle; shared colorbar; tight gaps",
    )
    p.add_argument("--no-suptitle", action="store_true")
    p.add_argument("--wspace", type=float, default=None)
    p.add_argument("--panel-gap", type=float, default=None,
                   help="Figure-fraction gap between map panels (publication layout)")
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--save-pdf", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)

    def _abs(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else ROOT / path

    op_b = load_op(_abs(args.op_baseline))
    op_c = load_op(_abs(args.op_c))

    infer_out = Path(args.infer_out or cfg["infer_out"])
    outdir = _abs(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    orbit_index = build_orbit_index(cfg["processed_location"], cfg["series_prefix"])
    edge_trim = int(cfg.get("edge_trim", 10))
    min_area = int(cfg.get("min_object_area", cfg.get("min_cold_pool_pixels", 4)))
    orbits = [int(x.strip()) for x in args.orbits.split(",") if x.strip()]

    title_note = (
        "classical | CRPS baseline (τ={0:.1e}, p*={1:.3f}) | "
        "CRPS OP C (τ={2:.1e}, p*={3:.3f})"
    ).format(op_b["tau"], op_b["p_star"], op_c["tau"], op_c["p_star"])
    show_suptitle = not (args.publication or args.no_suptitle)
    wspace = (
        float(args.wspace)
        if args.wspace is not None
        else (0.02 if args.publication else 0.28)
    )
    # Publication default crop for this eastern-Pacific example swath.
    lon_min_fixed = args.lon_min
    lon_max_fixed = args.lon_max
    if args.publication and lon_min_fixed is None and lon_max_fixed is None:
        lon_min_fixed = -102.0
        lon_max_fixed = -78.0
    panel_gap = args.panel_gap
    if args.publication and panel_gap is None:
        panel_gap = 0.04
    print(title_note)
    print("CRPS infer_out:", infer_out)
    print("outdir:", outdir)
    print("layout: suptitle={0} wspace={1} shared_cbar={2} lon=[{3},{4}] gap={5}".format(
        show_suptitle, wspace, args.publication or wspace <= 0.12,
        lon_min_fixed, lon_max_fixed, panel_gap,
    ))

    meta = {
        "op_baseline": op_b,
        "op_c": op_c,
        "orbits": orbits,
        "infer_out": str(infer_out),
        "panels": [
            "classical_Garg",
            "CRPS_baseline",
            "CRPS_OP_C_high_precision",
        ],
    }
    (outdir / "map_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for i, orbit in enumerate(orbits, 1):
        sp = stack_path(infer_out, orbit)
        if not sp.exists():
            print("[{0}/{1}] skip {2}: missing {3}".format(i, len(orbits), orbit, sp))
            continue
        mat = orbit_index.get(orbit)
        if mat is None:
            print("[{0}/{1}] skip {2}: no .mat".format(i, len(orbits), orbit))
            continue

        print("[{0}/{1}] orbit {2}".format(i, len(orbits), orbit))
        swath = trim_swath_edges(load_swath_mat(mat), edge_trim)
        stack = load_stack(sp)
        if "valid" in stack:
            valid = np.asarray(stack["valid"], dtype=np.float32)
        elif "valid_mask" in swath:
            valid = np.asarray(swath["valid_mask"], dtype=np.float32)
        else:
            valid = np.isfinite(swath["u"]).astype(np.float32)
        lon = np.asarray(swath["lon"], dtype=np.float32)
        lat = np.asarray(swath["lat"], dtype=np.float32)

        swath_for_garg = dict(swath)
        swath_for_garg["valid_mask"] = valid
        garg = labels_for_full_swath(
            swath_for_garg,
            thresh1=float(cfg["gradient_thresh1"]),
            thresh2=float(cfg["gradient_thresh2"]),
            min_pixels=int(cfg.get("min_cold_pool_pixels", 4)),
            res=float(cfg.get("fov_m", 12500.0)),
        )
        classical = (np.asarray(garg["label"], dtype=np.float32) > 0).astype(np.float32)
        crps_b = crps_binary_at(
            stack, valid, op_b["tau"], op_b["p_star"], min_area
        )
        crps_c = crps_binary_at(
            stack, valid, op_c["tau"], op_c["p_star"], min_area
        )

        if args.publication:
            panels = [
                ("(a) Classical GF\n ", classical),
                (
                    "(b) Density-matched probabilistic catalog\n"
                    + r"$P(S>1.2\times10^{-4}\,\mathrm{s}^{-1})$, "
                    + r"$p^*={0:.3f}$".format(op_b["p_star"]),
                    crps_b,
                ),
                (
                    "(c) High-precision probabilistic catalog\n"
                    + r"$P(S>2.0\times10^{-4}\,\mathrm{s}^{-1})$, "
                    + r"$p^*={0:.2f}$".format(op_c["p_star"]),
                    crps_c,
                ),
            ]
        else:
            panels = [
                (
                    "classical Garg\n(~label-rate / full catalog)",
                    classical,
                ),
                (
                    "CRPS baseline\nτ={0:.1e}, p*={1:.2f}".format(
                        op_b["tau"], op_b["p_star"]
                    ),
                    crps_b,
                ),
                (
                    "CRPS OP C (high-precision)\nτ={0:.1e}, p*={1:.2f}".format(
                        op_c["tau"], op_c["p_star"]
                    ),
                    crps_c,
                ),
            ]
        make_tier_figures(
            orbit=orbit,
            lon=lon,
            lat=lat,
            valid=valid,
            panels=panels,
            out_png_root=outdir / "{0}_tier_comparison.png".format(orbit),
            title_note=title_note,
            show_suptitle=show_suptitle,
            wspace=wspace,
            save_pdf=bool(args.save_pdf or args.publication),
            shared_cbar=bool(args.publication),
            lon_min_fixed=lon_min_fixed,
            lon_max_fixed=lon_max_fixed,
            panel_gap=panel_gap,
        )


if __name__ == "__main__":
    main()
