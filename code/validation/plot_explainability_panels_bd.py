#!/usr/bin/env python3
"""
Publication panels (b) and (d) from Garg XAI share_pack_v1 numerical results.

  (b) Predictor-group permutation with error bars (exceedance Brier Δscore)
  (d) Ensemble coherence: along- vs cross-track correlation length of
      member deviations (noise-scale jitter diagnosis)

Optionally rebuilds a revised 4-panel main figure:
  (a) bottleneck PCA × P(S>τ*)
  (b) permutation with uncertainty
  (c) representative IG map (from existing geo cache)
  (d) coherence summary

Example
-------
  python code/validation/plot_explainability_panels_bd.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from paths_config import ensure_sys_path, figures_repo_root, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, figures_repo_root, get, research_root  # type: ignore
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

_share = get("SHARE_PACK")
SHARE = (
    Path(_share) / "results"
    if _share
    else ROOT / "evaluation/Garg_explainability/share_pack_v1/results"
)
_explain = get("EXPLAIN_CACHE")
OUTDIR = Path(_explain) if _explain else ROOT / "evaluation/explainability/crps"
_FIG = figures_repo_root()
PACK = Path(get("FIGURES_MAIN") or str(_FIG / "figures" / "main"))
SI = Path(get("FIGURES_SI") or str(_FIG / "figures" / "si"))

PERM_ORDER = ["numamb", "ambiguity_winds", "smooth_context", "rain_qc"]
PERM_LABELS = {
    "numamb": r"$N_{\mathrm{amb}}$",
    "ambiguity_winds": "Ambiguity winds\n/ SOS",
    "smooth_context": r"Smooth $U$, $|\nabla U|$",
    "rain_qc": "Rain / QC",
}
PERM_LABELS_COMPACT = {
    "numamb": r"$N_{\mathrm{amb}}$",
    "ambiguity_winds": "Ambiguity winds\n/ SOS",
    "smooth_context": r"Smooth $U$, $|\nabla U|$",
    "rain_qc": "Rain / QC",
}
PERM_COLORS = {
    "numamb": "#b2182b",
    "ambiguity_winds": "#ef8a62",
    "smooth_context": "#67a9cf",
    "rain_qc": "#2166ac",
}


def _panel_letter(ax, letter: str, *, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="bottom", ha="right",
    )


def load_perm(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.set_index("group").reindex(PERM_ORDER).reset_index()
    if df["mean"].isna().any():
        raise SystemExit(f"missing groups in {csv_path}")
    return df


def plot_panel_b(
    ax, df: pd.DataFrame, *, letter: Optional[str] = "b", compact: bool = False,
) -> None:
    y = np.arange(len(df))[::-1]  # highest importance on top
    means = df["mean"].to_numpy(dtype=float)
    stds = df["std"].to_numpy(dtype=float)
    colors = [PERM_COLORS[g] for g in df["group"]]
    labmap = PERM_LABELS_COMPACT if compact else PERM_LABELS
    labels = [labmap[g] for g in df["group"]]

    # Plot in units of 10^{-2} so ticks stay simple and the scale sits in the xlabel
    # (avoids matplotlib's scientific offset overlapping the axis title).
    scale = 1.0e-2
    ax.barh(
        y, means / scale, xerr=stds / scale, height=0.62,
        color=colors, edgecolor="white", linewidth=0.8,
        error_kw=dict(ecolor="0.15", capsize=3.5, lw=1.1, capthick=1.1),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(
        r"$\Delta$ exceedance Brier score  (permute $-$ baseline)  "
        r"[$\times 10^{-2}$]",
        fontsize=10,
        labelpad=8,
    )
    xmax = max(1.8, float((means.max() + stds.max()) / scale) * 1.12)
    ax.set_xlim(0.0, xmax)
    ax.axvline(0.0, color="0.35", lw=0.7)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if letter:
        _panel_letter(ax, letter)


def plot_panel_d(ax, coh: pd.DataFrame, *, letter: Optional[str] = "d") -> None:
    """Compact coherence summary: along- vs cross-track correlation lengths."""
    along = coh["L_along"].to_numpy(dtype=float)
    cross = coh["L_cross"].to_numpy(dtype=float)
    data = [along, cross]
    labels = ["Along-track", "Cross-track"]
    colors = ["#2c7fb8", "#7fcdbb"]

    # Violin + box for discrete-ish L values.
    parts = ax.violinplot(
        data, positions=[0, 1], widths=0.7,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("0.25")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    bp = ax.boxplot(
        data, positions=[0, 1], widths=0.22, patch_artist=True,
        showfliers=False, medianprops=dict(color="0.1", lw=1.4),
        whiskerprops=dict(color="0.25", lw=1.0),
        capprops=dict(color="0.25", lw=1.0),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.95)
        patch.set_edgecolor("0.2")

    # Light jitter of orbit-mean points (one point per orbit×half).
    g = coh.groupby(["orbit", "half"], as_index=False)[["L_along", "L_cross"]].mean()
    rng = np.random.default_rng(0)
    for i, col in enumerate(["L_along", "L_cross"]):
        vals = g[col].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.04, size=vals.size)
        ax.scatter(
            np.full(vals.size, i) + jitter, vals,
            s=10, color="0.2", alpha=0.35, zorder=5, linewidths=0,
        )

    # Reference: latent noise grid spacing (8 px). Member L sits at/below that
    # scale → local jitter, not long front-scale modes (which need L ≫ 8).
    ax.axhline(8.0, color="#b2182b", ls="--", lw=1.0, zorder=1)
    ax.annotate(
        "Latent-grid spacing (8 pixels)",
        xy=(1.35, 8.0), xytext=(1.35, 9.15),
        fontsize=7.5, color="#b2182b", ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color="#b2182b", lw=0.8),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Member-difference correlation length (pixels)", fontsize=10)
    ax.set_ylim(3.5, 10.2)
    ax.set_xlim(-0.55, 1.55)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if letter:
        _panel_letter(ax, letter)


def save_standalone_panels(
    perm_df: pd.DataFrame,
    coh: pd.DataFrame,
    outdir: Path,
) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    fig_b, ax_b = plt.subplots(figsize=(5.2, 3.4))
    plot_panel_b(ax_b, perm_df, letter=None)
    fig_b.tight_layout()
    path_b = outdir / "13_panel_b_perm_errorbars.png"
    fig_b.savefig(path_b, dpi=300, bbox_inches="tight", facecolor="white")
    fig_b.savefig(path_b.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig_b)

    fig_d, ax_d = plt.subplots(figsize=(5.0, 3.8))
    plot_panel_d(ax_d, coh, letter=None)
    fig_d.tight_layout()
    path_d = outdir / "13_panel_d_member_coherence.png"
    fig_d.savefig(path_d, dpi=300, bbox_inches="tight", facecolor="white")
    fig_d.savefig(path_d.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig_d)

    print("wrote", path_b)
    print("wrote", path_d)
    return path_b, path_d


def try_plot_revised_main(
    perm_df: pd.DataFrame,
    coh: pd.DataFrame,
    outdir: Path,
    pack: Path,
    *,
    ig_orbit: int = 73598,
) -> Optional[Path]:
    """Five-panel main figure:

    (a) bottleneck PCA × P(S>τ*)
    (b) bottleneck PCA × N_amb
    (c) permutation importance with error bars (exceedance Brier)
    (d) representative IG map
    (e) member-difference correlation length

    Old dual-bar afCRPS+Brier permutation is omitted from the main figure
    (retain in SI / text).
    """
    latent = outdir / "13_latent_embeddings.npz"
    ig_cache = outdir / f"13_ig_main_orbit{ig_orbit}_geo.npz"
    if not latent.exists() or not ig_cache.exists():
        print(
            f"skip revised main figure (need {latent.name} and {ig_cache.name})",
            flush=True,
        )
        return None

    try:
        from validation.ig_geo_plot import (
            extent_from_seg,
            geo_scatter_field,
            tropical_segment_mask,
        )
    except Exception as exc:  # pragma: no cover
        print(f"skip revised main (ig_geo_plot): {exc}", flush=True)
        return None

    try:
        import cartopy.crs as ccrs
        has_cartopy = True
    except Exception:
        ccrs = None  # type: ignore
        has_cartopy = False

    z = np.load(latent)
    Z = z["Z"]
    igz = np.load(ig_cache)
    p_full = igz["p_full"]
    abs_full = igz["abs_full"]
    roi_full = igz["roi_full"]
    lon = igz["lon"]
    lat = igz["lat"]
    valid = igz["valid"]

    xlim = (-5.0, 15.0)
    ylim = (-5.0, 15.0)

    fig = plt.figure(figsize=(14.2, 9.8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 1.15],
        height_ratios=[1.0, 1.12],
        hspace=0.48,
        wspace=0.42,
        left=0.055,
        right=0.985,
        top=0.96,
        bottom=0.06,
    )

    # (a) PCA × P — horizontal colorbar under plot
    ax_a = fig.add_subplot(gs[0, 0])
    sc_a = ax_a.scatter(
        Z[:, 0], Z[:, 1], c=z["P"], s=3, cmap="YlOrRd",
        alpha=0.5, linewidths=0,
    )
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(ylim)
    ax_a.set_xlabel("PC1")
    ax_a.set_ylabel("PC2")
    cb_a = fig.colorbar(
        sc_a, ax=ax_a, orientation="horizontal",
        fraction=0.046, pad=0.20, aspect=28,
    )
    cb_a.ax.tick_params(labelsize=8)
    _panel_letter(ax_a, "a", x=-0.08, y=1.03)

    # (b) PCA × N_amb
    ax_b = fig.add_subplot(gs[0, 1])
    sc_b = ax_b.scatter(
        Z[:, 0], Z[:, 1], c=z["N_amb"], s=3, cmap="coolwarm",
        alpha=0.5, linewidths=0,
    )
    ax_b.set_xlim(xlim)
    ax_b.set_ylim(ylim)
    ax_b.set_xlabel("PC1")
    ax_b.set_ylabel("PC2")
    cb_b = fig.colorbar(
        sc_b, ax=ax_b, orientation="horizontal",
        fraction=0.046, pad=0.20, aspect=28,
    )
    cb_b.ax.tick_params(labelsize=8)
    _panel_letter(ax_b, "b", x=-0.08, y=1.03)

    # (c) new permutation with error bars
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_b(ax_c, perm_df, letter=None, compact=True)
    _panel_letter(ax_c, "c", x=-0.16, y=1.03)

    # (d) IG spatial — spans first two columns of bottom row
    gs_d = gs[1, :2].subgridspec(
        1, 5, width_ratios=[1.0, 0.055, 0.16, 1.0, 0.055], wspace=0.08,
    )
    if has_cartopy:
        ax0 = fig.add_subplot(gs_d[0, 0], projection=ccrs.PlateCarree())
        ax1 = fig.add_subplot(gs_d[0, 3], projection=ccrs.PlateCarree())
    else:
        ax0 = fig.add_subplot(gs_d[0, 0])
        ax1 = fig.add_subplot(gs_d[0, 3])
    cax0 = fig.add_subplot(gs_d[0, 1])
    cax1 = fig.add_subplot(gs_d[0, 4])
    seg = tropical_segment_mask(lon, lat, valid, p_full, prefer_largest=True)
    extent = extent_from_seg(lon, lat, seg)
    sc0, _ = geo_scatter_field(
        ax0, lon, lat, p_full, seg,
        cmap="YlOrRd", vmin=0.0, vmax=1.0, s=4.0, roi=roi_full, extent=extent,
        draw_left_labels=True, draw_bottom_labels=True,
    )
    cb0 = fig.colorbar(sc0, cax=cax0)
    cb0.ax.tick_params(labelsize=7)
    roi_mask = np.isfinite(abs_full) & (roi_full > 0.5) & seg
    vmax = np.nanpercentile(abs_full[roi_mask], 99) if roi_mask.any() else 1.0
    sc1, _ = geo_scatter_field(
        ax1, lon, lat, abs_full, seg,
        cmap="magma", vmin=0.0, vmax=max(float(vmax), 1e-12), s=4.0,
        extent=extent, draw_left_labels=False, draw_bottom_labels=True,
    )
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.ax.tick_params(labelsize=7)
    _panel_letter(ax0, "d", x=-0.08, y=1.03)

    # (e) coherence
    ax_e = fig.add_subplot(gs[1, 2])
    plot_panel_d(ax_e, coh, letter=None)
    _panel_letter(ax_e, "e", x=-0.14, y=1.03)

    out_png = outdir / "13_explainability_main_v2.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    plt.close(fig)
    print("wrote", out_png)

    # Also overwrite pack main figure with v2 (keep old as backup once).
    backup = pack / "13_explainability_main_v1_backup.png"
    legacy = pack / "13_explainability_main.png"
    if legacy.exists() and not backup.exists():
        shutil.copy2(legacy, backup)
        pdf_legacy = legacy.with_suffix(".pdf")
        if pdf_legacy.exists():
            shutil.copy2(pdf_legacy, backup.with_suffix(".pdf"))
    for ext in (".png", ".pdf"):
        shutil.copy2(out_png.with_suffix(ext), pack / f"13_explainability_main{ext}")
        shutil.copy2(out_png.with_suffix(ext), pack / f"13_explainability_main_v2{ext}")
        shutil.copy2(out_png.with_suffix(ext), outdir / f"13_explainability_main{ext}")
    return out_png


def write_caption(pack: Path) -> None:
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{13_explainability_main.pdf}
\caption{Learned representation, attribution, and ensemble coherence for the
CRPS ensemble U-Net.
(a)~Bottleneck PCA colored by soft exceedance probability $P(S>\tau^*)$.
(b)~Same embedding colored by ambiguity count $N_{\mathrm{amb}}$.
(c)~Predictor-group along-track permutation importance for the exceedance
Brier score at $\tau^*=1.2\times10^{-4}$ (mean $\Delta$score $\pm$ SD over
latent seeds $\times$ permutation draws). Ambiguity predictors exceed smooth
context; uncertainties are small relative to group differences.
(d)~Integrated gradients on a representative ASCAT swath (lon/lat scatter):
$P(S>\tau^*)$ and $|\mathrm{IG}|$ summed over input channels (cyan: high-$P$ ROI).
(e)~Member-difference correlation length of ensemble deviations from the
member mean: along-track and cross-track $L$ concentrate near 5--6~pixels.
The dashed line marks the latent-grid spacing (8~pixels) of the stochastic
noise input; member diversity therefore sits at the noise-grid scale rather
than at long front-scale modes.
Corresponding afCRPS permutation numbers and latent $z$-traversal cases appear
in the Supporting Information.}
\label{fig:explainability}
\end{figure}
"""
    path = pack / "13_explainability_main_caption.tex"
    path.write_text(text)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--perm-csv",
        default=str(SHARE / "perm_errorbars.csv"),
    )
    ap.add_argument(
        "--coherence-csv",
        default=str(SHARE / "member_coherence.csv"),
    )
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--pack-dir", default=str(PACK))
    ap.add_argument("--si-dir", default=str(SI))
    ap.add_argument("--ig-orbit", type=int, default=73598)
    ap.add_argument("--skip-main", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pack = Path(args.pack_dir)
    si = Path(args.si_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pack.mkdir(parents=True, exist_ok=True)
    si.mkdir(parents=True, exist_ok=True)

    perm_df = load_perm(Path(args.perm_csv))
    coh = pd.read_csv(args.coherence_csv)

    path_b, path_d = save_standalone_panels(perm_df, coh, outdir)
    for src in (path_b, path_d):
        shutil.copy2(src, pack / src.name)
        shutil.copy2(src.with_suffix(".pdf"), pack / src.with_suffix(".pdf").name)
        shutil.copy2(src, si / src.name)
        shutil.copy2(src.with_suffix(".pdf"), si / src.with_suffix(".pdf").name)

    # Side-by-side preview of the two new panels only.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    plot_panel_b(axes[0], perm_df, letter="b")
    plot_panel_d(axes[1], coh, letter="d")
    fig.tight_layout()
    preview = outdir / "13_panels_bd_preview.png"
    fig.savefig(preview, dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(preview.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(preview, pack / preview.name)
    print("wrote", preview)

    if not args.skip_main:
        try_plot_revised_main(
            perm_df, coh, outdir, pack, ig_orbit=args.ig_orbit,
        )
        write_caption(pack)

    print("done", flush=True)


if __name__ == "__main__":
    main()
