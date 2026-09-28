#!/usr/bin/env python3
"""
Replot IG explainability panels from cached geo NPZs with Cartopy coastlines.

Does not need torch — run in the impact env after IG caches exist:

  env -u PYTHONPATH -u LD_LIBRARY_PATH \\
    /rhome/gpriftis/miniconda3/envs/impact/bin/python \\
    validation/replot_ig_geo_cartopy.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from paths_config import ensure_sys_path, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, get, research_root  # type: ignore

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

from validation.ig_geo_plot import (  # noqa: E402
    extent_from_seg,
    geo_scatter_field,
    tropical_segment_mask,
)


def plot_si_cases(outdir: Path, si_dir: Path, orbits) -> None:
    n = len(orbits)
    fig = plt.figure(figsize=(16.5, 4.4 * n))
    gs = fig.add_gridspec(
        n, 4, left=0.07, right=0.97, bottom=0.06, top=0.94,
        wspace=0.22, hspace=0.28,
    )
    for row, orbit in enumerate(orbits):
        zpath = outdir / f"13_ig_geo_orbit{orbit}.npz"
        if not zpath.exists():
            raise FileNotFoundError(zpath)
        z = np.load(zpath)
        lon, lat, valid = z["lon"], z["lat"], z["valid"]
        p_full, abs_full = z["p_full"], z["abs_full"]
        roi_full, grad_full, amb_full = z["roi_full"], z["grad_full"], z["amb_full"]
        seg = tropical_segment_mask(lon, lat, valid, p_full, prefer_largest=True)
        extent = extent_from_seg(lon, lat, seg)
        roi_mask = np.isfinite(abs_full) & (roi_full > 0.5) & seg
        vmax = np.nanpercentile(abs_full[roi_mask], 99) if roi_mask.any() else 1.0
        vmax_a = np.nanpercentile(amb_full[roi_mask], 99) if roi_mask.any() else 1.0
        g_finite = grad_full[seg & np.isfinite(grad_full)]
        g_vmax = float(np.nanpercentile(g_finite, 99)) if g_finite.size else 1.0
        panels = [
            (p_full, "YlOrRd", 0.0, 1.0, f"orbit {orbit}  $P(S>\\tau^*)$", roi_full),
            (abs_full, "magma", 0.0, max(float(vmax), 1e-12), r"|IG| (all ASCAT channels)", None),
            (grad_full, "viridis", 0.0, max(g_vmax, 1e-12), r"$|\nabla U|_{\mathrm{smooth}}$ input", None),
            (amb_full, "inferno", 0.0, max(float(vmax_a), 1e-12), "|IG| ambiguity winds/SOS", None),
        ]
        for col, (field, cmap, vmin, vmax_c, title, roi) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            sc, _ = geo_scatter_field(
                ax, lon, lat, field, seg,
                cmap=cmap, vmin=vmin, vmax=vmax_c, s=4.0, roi=roi, extent=extent,
                draw_left_labels=(col == 0),
                draw_bottom_labels=(row == n - 1),
            )
            ax.set_title(title, fontsize=10, pad=6)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

    png = outdir / "13_integrated_gradients_cases.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for ext in (".png", ".pdf"):
        shutil.copy2(png.with_suffix(ext), si_dir / f"SI_explain_ig_cases{ext}")
    print("wrote", png)


def plot_main_figure(outdir: Path, pack_fig: Path, orbit: int) -> None:
    zpath = outdir / f"13_ig_main_orbit{orbit}_geo.npz"
    if not zpath.exists():
        zpath = outdir / f"13_ig_geo_orbit{orbit}.npz"
    z = np.load(zpath)
    lon, lat, valid = z["lon"], z["lat"], z["valid"]
    p_full, abs_full, roi_full = z["p_full"], z["abs_full"], z["roi_full"]

    latent = outdir / "13_latent_embeddings.npz"
    perm = outdir / "13_predictor_group_permutation.csv"
    if not latent.exists() or not perm.exists():
        raise FileNotFoundError("need latent npz + permutation csv for full main figure")

    zz = np.load(latent)
    Z = zz["Z"]
    var12 = float(zz["pca_explained_variance_ratio"][:2].sum())
    df = pd.read_csv(perm)
    plot_df = df[df["group"] != "baseline"].copy()
    labels = {
        "smooth_context": r"smooth $U$, $|\nabla U|$",
        "rain_qc": "rain / QC",
        "numamb": r"$N_{\mathrm{amb}}$",
        "ambiguity_winds": "ambiguity winds / SOS",
    }
    plot_df = plot_df.sort_values("delta_afCRPS", ascending=True)
    ylabels = [labels.get(g, g) for g in plot_df["group"]]
    xlim = (-5.0, 15.0)
    ylim = (-5.0, 15.0)

    fig = plt.figure(figsize=(12.8, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    sc = ax_a.scatter(Z[:, 0], Z[:, 1], c=zz["P"], s=3, cmap="YlOrRd", alpha=0.5, linewidths=0)
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(ylim)
    ax_a.set_xlabel("PC1")
    ax_a.set_ylabel("PC2")
    ax_a.set_title(rf"$P(S>\tau^*)$  (PC1–2: {100*var12:.0f}% var.)", fontsize=10)
    fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.04)
    ax_a.text(-0.08, 1.05, "(a)", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")

    ax_b = fig.add_subplot(gs[0, 1])
    sc = ax_b.scatter(Z[:, 0], Z[:, 1], c=zz["N_amb"], s=3, cmap="coolwarm", alpha=0.5, linewidths=0)
    ax_b.set_xlim(xlim)
    ax_b.set_ylim(ylim)
    ax_b.set_xlabel("PC1")
    ax_b.set_ylabel("PC2")
    ax_b.set_title(r"$N_{\mathrm{amb}}$", fontsize=10)
    fig.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.04)
    ax_b.text(-0.08, 1.05, "(b)", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(plot_df))
    hbar = 0.38
    ax.barh(y + hbar / 2, plot_df["delta_afCRPS"], height=hbar, color="#1a5fa8",
            label=r"$\Delta$ afCRPS", edgecolor="white")
    ax.barh(y - hbar / 2, plot_df["delta_Brier"], height=hbar, color="#c23b22",
            label=r"$\Delta$ Brier", edgecolor="white")
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("permute − baseline")
    ax.set_title("Predictor-group importance", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(-0.08, 1.05, "(c)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")

    gs_d = gs[1, 1].subgridspec(1, 5, width_ratios=[1.0, 0.055, 0.18, 1.0, 0.055], wspace=0.05)
    ax0 = fig.add_subplot(gs_d[0, 0], projection=ccrs.PlateCarree())
    cax0 = fig.add_subplot(gs_d[0, 1])
    ax1 = fig.add_subplot(gs_d[0, 3], projection=ccrs.PlateCarree())
    cax1 = fig.add_subplot(gs_d[0, 4])

    seg = tropical_segment_mask(lon, lat, valid, p_full, prefer_largest=True)
    extent = extent_from_seg(lon, lat, seg)
    sc0, _ = geo_scatter_field(
        ax0, lon, lat, p_full, seg,
        cmap="YlOrRd", vmin=0.0, vmax=1.0, s=4.0, roi=roi_full, extent=extent,
        draw_left_labels=True, draw_bottom_labels=True,
    )
    ax0.set_title(rf"orbit {orbit}  $P(S>\tau^*)$", fontsize=9)
    fig.colorbar(sc0, cax=cax0).ax.tick_params(labelsize=7)

    roi_mask = np.isfinite(abs_full) & (roi_full > 0.5) & seg
    vmax = np.nanpercentile(abs_full[roi_mask], 99) if roi_mask.any() else 1.0
    sc1, _ = geo_scatter_field(
        ax1, lon, lat, abs_full, seg,
        cmap="magma", vmin=0.0, vmax=max(float(vmax), 1e-12), s=4.0,
        extent=extent, draw_left_labels=False, draw_bottom_labels=True,
    )
    ax1.set_title(r"|IG| (all channels)", fontsize=9)
    fig.colorbar(sc1, cax=cax1).ax.tick_params(labelsize=7)
    ax0.text(-0.08, 1.05, "(d)", transform=ax0.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")

    out_png = outdir / "13_explainability_main.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for ext in (".png", ".pdf"):
        shutil.copy2(out_png.with_suffix(ext), pack_fig / out_png.with_suffix(ext).name)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "evaluation" / "explainability" / "crps"))
    ap.add_argument(
        "--pack-dir",
        default=str(ROOT / "evaluation" / "paper_revision_pack_crps_ensemble" / "01_figures"),
    )
    ap.add_argument(
        "--si-dir",
        default=str(ROOT / "evaluation" / "paper_revision_pack_crps_ensemble" / "07_explainability_si"),
    )
    ap.add_argument("--ig-orbit-main", type=int, default=73598)
    ap.add_argument("--ig-orbits", default="73598,72850,75001")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pack_fig = Path(args.pack_dir)
    si_dir = Path(args.si_dir)
    pack_fig.mkdir(parents=True, exist_ok=True)
    si_dir.mkdir(parents=True, exist_ok=True)

    orbits = [int(x) for x in args.ig_orbits.split(",") if x.strip()]
    plot_si_cases(outdir, si_dir, orbits)
    plot_main_figure(outdir, pack_fig, args.ig_orbit_main)


if __name__ == "__main__":
    main()
