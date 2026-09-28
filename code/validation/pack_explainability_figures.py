#!/usr/bin/env python3
"""
Pack explainability into one main-text figure + Supporting Information panels.

Main text (2×2) — revised layout:
  (a) bottleneck PCA colored by P(S>τ*)
  (b) predictor-group permutation with error bars (share_pack_v1)
  (c) one IG spatial example (orbit 73598) as lon/lat scatter
  (d) member-difference correlation length (share_pack_v1)

The main figure is built by validation/plot_explainability_panels_bd.py
(which supersedes the older dual-PCA + dual-bar permutation layout).

SI:
  SI_explain_pca_extra — remaining PCA colorings (S, |∇U|, U_smooth, rain)
  SI_explain_ig_cases — all three IG orbits (lon/lat scatter, not index imshow)
  SI_explain_ig_group_share — group-share bars

Example
-------
  source slurm_env.sh && setup_pytorch_env
  "$PYTHON" validation/pack_explainability_figures.py --force-ig

For coastlines on geo panels (Cartopy), prefer the impact env after IG caches exist:
  env -u PYTHONPATH -u LD_LIBRARY_PATH \\
    /rhome/gpriftis/miniconda3/envs/impact/bin/python \\
    validation/pack_explainability_figures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from paths_config import ensure_sys_path, get, research_root
except ImportError:
    from code.paths_config import ensure_sys_path, get, research_root  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

from validation.explain_crps_representation import (  # noqa: E402
    TAU_STAR,
    _HAS_CARTOPY,
    _ig_half,
    copy_to_pack,
    find_orbit_path,
    load_cfg,
    load_swath_lonlat,
    resolve_groups,
    run_integrated_gradients,
    run_latent_analysis,
    select_files,
)
from validation.ig_geo_plot import (  # noqa: E402
    extent_from_seg,
    geo_scatter_field,
    stitch_halves_geo as _stitch_halves_geo,
    tropical_segment_mask,
)
from lib.crps_dataset import split_files  # noqa: E402
from lib.swath_halves import HALF_NAMES  # noqa: E402

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover
    ccrs = None  # type: ignore


def _label(ax, letter: str) -> None:
    ax.text(
        -0.08, 1.05, f"({letter})",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="bottom", ha="right",
    )


def ensure_latent(outdir: Path, args) -> Path:
    npz = outdir / "13_latent_embeddings.npz"
    if npz.exists() and not args.force_latent:
        print(f"reuse {npz}", flush=True)
        return npz
    print("=== regenerating latent embeddings ===", flush=True)
    model, cfg, device, m, alpha, peak = load_cfg(args)
    h_soft = float(cfg.get("exceedance_softness", 0.05))
    files = select_files(cfg, args.limit, args.stride)
    run_latent_analysis(
        model, cfg, device, m, files, outdir,
        seed=args.seed, max_pixels=args.max_latent_pixels,
        tau=args.tau, h_soft=h_soft,
    )
    return npz


def plot_main_figure(
    outdir: Path,
    pack_fig: Path,
    si_dir: Path,
    *,
    perm_csv: Path,
    ig_orbit: int,
    args,
) -> None:
    z = np.load(outdir / "13_latent_embeddings.npz")
    Z = z["Z"]
    var12 = float(z["pca_explained_variance_ratio"][:2].sum())
    df = pd.read_csv(perm_csv)
    plot_df = df[df["group"] != "baseline"].copy()
    labels = {
        "smooth_context": r"smooth $U$, $|\nabla U|$",
        "rain_qc": "rain / QC",
        "numamb": r"$N_{\mathrm{amb}}$",
        "ambiguity_winds": "ambiguity winds / SOS",
    }
    # Sort by ΔafCRPS for a consistent story
    plot_df = plot_df.sort_values("delta_afCRPS", ascending=True)
    ylabels = [labels.get(g, g) for g in plot_df["group"]]

    # Shared PCA axis limits for (a) and (b)
    xlim = (-5.0, 15.0)
    ylim = (-5.0, 15.0)

    # IG for one orbit (both halves) — cache maps for quick figure tweaks
    # Cache v2: full-swath geo stitch (no mid-swath gap) + lon/lat
    ig_cache = outdir / f"13_ig_main_orbit{ig_orbit}_geo.npz"
    legacy_cache = outdir / f"13_ig_main_orbit{ig_orbit}.npz"
    if ig_cache.exists() and not getattr(args, "force_ig", False):
        print(f"reuse IG cache {ig_cache}", flush=True)
        igz = np.load(ig_cache)
        p_full = igz["p_full"]
        abs_full = igz["abs_full"]
        roi_full = igz["roi_full"]
        lon = igz["lon"]
        lat = igz["lat"]
        valid = igz["valid"]
        path = str(igz["path"].item()) if "path" in igz.files else None
    else:
        print(f"=== IG example orbit {ig_orbit} for panel (d) ===", flush=True)
        model, cfg, device, m, alpha, peak = load_cfg(args)
        h_soft = float(cfg.get("exceedance_softness", 0.05))
        channels = list(cfg["input_channels"])
        groups = resolve_groups(channels)
        scale = float(cfg.get("gradient_scale", 5.0e-4))
        all_val = split_files(cfg, "val")
        path = find_orbit_path(all_val, ig_orbit)
        if path is None:
            raise FileNotFoundError(f"orbit {ig_orbit} not in val split")
        halves = {}
        for half in HALF_NAMES:
            halves[half] = _ig_half(
                model, cfg, device, path, half,
                channels=channels, groups=groups, scale=scale,
                tau=args.tau, h_soft=h_soft, seed=args.seed,
                orbit=ig_orbit, steps=args.ig_steps,
            )
        L, R = halves["left"], halves["right"]
        p_full = _stitch_halves_geo(L["p"], R["p"])
        abs_full = _stitch_halves_geo(L["abs_map"], R["abs_map"])
        roi_full = _stitch_halves_geo(L["roi"].astype(float), R["roi"].astype(float))
        lon, lat, valid = load_swath_lonlat(path, cfg)
        h = min(lon.shape[0], p_full.shape[0])
        w = min(lon.shape[1], p_full.shape[1])
        lon, lat, valid = lon[:h, :w], lat[:h, :w], valid[:h, :w]
        p_full = p_full[:h, :w]
        abs_full = abs_full[:h, :w]
        roi_full = roi_full[:h, :w]
        np.savez_compressed(
            ig_cache,
            p_full=p_full.astype(np.float32),
            abs_full=abs_full.astype(np.float32),
            roi_full=roi_full.astype(np.float32),
            lon=lon.astype(np.float32),
            lat=lat.astype(np.float32),
            valid=valid.astype(np.float32),
            path=np.asarray(path),
        )
        if legacy_cache.exists():
            print(f"(legacy index-space cache still present: {legacy_cache})", flush=True)
        print("wrote", ig_cache, flush=True)

    if not _HAS_CARTOPY:
        print(
            "NOTE: cartopy not available — panel (d) lon/lat scatter without coastlines",
            flush=True,
        )

    fig = plt.figure(figsize=(12.8, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)

    # (a) PCA × P
    ax_a = fig.add_subplot(gs[0, 0])
    sc = ax_a.scatter(Z[:, 0], Z[:, 1], c=z["P"], s=3, cmap="YlOrRd", alpha=0.5, linewidths=0)
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(ylim)
    ax_a.set_xlabel("PC1")
    ax_a.set_ylabel("PC2")
    ax_a.set_title(rf"$P(S>\tau^*)$  (PC1–2: {100*var12:.0f}% var.)", fontsize=10)
    fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.04)
    _label(ax_a, "a")

    # (b) PCA × N_amb — identical axis range
    ax_b = fig.add_subplot(gs[0, 1])
    sc = ax_b.scatter(Z[:, 0], Z[:, 1], c=z["N_amb"], s=3, cmap="coolwarm", alpha=0.5, linewidths=0)
    ax_b.set_xlim(xlim)
    ax_b.set_ylim(ylim)
    ax_b.set_xlabel("PC1")
    ax_b.set_ylabel("PC2")
    ax_b.set_title(r"$N_{\mathrm{amb}}$", fontsize=10)
    fig.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.04)
    _label(ax_b, "b")

    # (c) permutation dual bars
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
    _label(ax, "c")

    # (d) IG spatial example — lon/lat scatter (segment-map style)
    gs_d = gs[1, 1].subgridspec(
        1, 5, width_ratios=[1.0, 0.055, 0.18, 1.0, 0.055], wspace=0.05,
    )
    if _HAS_CARTOPY:
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
    ax0.set_title(rf"orbit {ig_orbit}  $P(S>\tau^*)$", fontsize=9)
    cb0 = fig.colorbar(sc0, cax=cax0)
    cb0.ax.tick_params(labelsize=7)

    roi_mask = np.isfinite(abs_full) & (roi_full > 0.5) & seg
    vmax = np.nanpercentile(abs_full[roi_mask], 99) if roi_mask.any() else 1.0
    sc1, _ = geo_scatter_field(
        ax1, lon, lat, abs_full, seg,
        cmap="magma", vmin=0.0, vmax=max(float(vmax), 1e-12), s=4.0,
        extent=extent, draw_left_labels=False, draw_bottom_labels=True,
    )
    ax1.set_title(r"|IG| (all channels)", fontsize=9)
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.ax.tick_params(labelsize=7)
    _label(ax0, "d")

    out_png = outdir / "13_explainability_main.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_png, flush=True)

    for ext in (".png", ".pdf"):
        shutil.copy2(out_png.with_suffix(ext), pack_fig / out_png.with_suffix(ext).name)


def plot_si_pca_extra(outdir: Path, si_dir: Path) -> None:
    z = np.load(outdir / "13_latent_embeddings.npz")
    Z = z["Z"]
    var12 = float(z["pca_explained_variance_ratio"][:2].sum())
    xpad = 0.03 * (Z[:, 0].max() - Z[:, 0].min() + 1e-6)
    ypad = 0.03 * (Z[:, 1].max() - Z[:, 1].min() + 1e-6)
    xlim = (-5.0, 15.0)
    ylim = (-5.0, 15.0)
    panels = [
        ("S", r"Target $S$", "magma"),
        ("gradU_smooth", r"$|\nabla U|_{\mathrm{smooth}}$", "viridis"),
        ("U_smooth", r"$U_{\mathrm{smooth}}$", "plasma"),
        ("rain_flag", "rain flag", "Greys"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2))
    letters = "abcd"
    for ax, (key, title, cmap), let in zip(axes.ravel(), panels, letters):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=z[key], s=3, cmap=cmap, alpha=0.5, linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"({let})  {title}", loc="left", fontsize=11, fontweight="bold")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png = si_dir / "SI_explain_pca_extra.png"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # also keep a copy under explainability/
    shutil.copy2(png, outdir / png.name)
    shutil.copy2(png.with_suffix(".pdf"), outdir / png.with_suffix(".pdf").name)
    print("wrote", png, flush=True)


def ensure_ig_si(outdir: Path, si_dir: Path, args) -> None:
    """Ensure full 3-orbit IG cases + group-share exist; copy to SI folder."""
    cases = outdir / "13_integrated_gradients_cases.png"
    share = outdir / "13_integrated_gradients_group_share.png"
    if args.force_ig or not cases.exists() or not share.exists():
        print("=== regenerating full IG SI panels ===", flush=True)
        model, cfg, device, m, alpha, peak = load_cfg(args)
        h_soft = float(cfg.get("exceedance_softness", 0.05))
        files = select_files(cfg, args.limit, args.stride)
        orbits = [int(x) for x in args.ig_orbits.split(",") if x.strip()]
        run_integrated_gradients(
            model, cfg, device, files, outdir,
            orbits=orbits, seed=args.seed, tau=args.tau,
            h_soft=h_soft, steps=args.ig_steps,
        )

    mapping = {
        "13_integrated_gradients_cases.png": "SI_explain_ig_cases.png",
        "13_integrated_gradients_cases.pdf": "SI_explain_ig_cases.pdf",
        "13_integrated_gradients_group_share.png": "SI_explain_ig_group_share.png",
        "13_integrated_gradients_group_share.pdf": "SI_explain_ig_group_share.pdf",
        "13_integrated_gradients_group_share.csv": "SI_explain_ig_group_share.csv",
    }
    for src_name, dst_name in mapping.items():
        src = outdir / src_name
        if src.exists():
            shutil.copy2(src, si_dir / dst_name)
            print(f"SI copy {dst_name}", flush=True)


def write_captions(si_dir: Path, pack_fig: Path) -> None:
    # Main-text caption is owned by plot_explainability_panels_bd.py
    from validation.plot_explainability_panels_bd import write_caption
    write_caption(pack_fig)

    si_cap = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{SI_explain_pca_extra.pdf}
\caption{Supporting Information: additional bottleneck PCA colorings
(target $S$, $|\nabla U|_{\mathrm{smooth}}$, $U_{\mathrm{smooth}}$, rain flag)
and $N_{\mathrm{amb}}$ (also in the expanded all-colorings panel).
Main-text Fig.~\ref{fig:explainability} shows $P(S>\tau^*)$ in panel~(a).}
\label{fig:si-explain-pca}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{SI_explain_ig_cases.pdf}
\caption{Supporting Information: integrated-gradients spatial attribution for
three validation orbits, plotted as lon/lat scatter on the ASCAT swath
footprint (largest tropical segment). Columns: $P(S>\tau^*)$,
$|\mathrm{IG}|$ (all ASCAT channels), $|\nabla U|_{\mathrm{smooth}}$ input, and
$|\mathrm{IG}|$ from ambiguity winds/SOS.}
\label{fig:si-explain-ig-cases}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{SI_explain_ig_group_share.pdf}
\caption{Supporting Information: fraction of ROI $|\mathrm{IG}|$ attributed to
each predictor group, by orbit and swath half.}
\label{fig:si-explain-ig-share}
\end{figure}
"""
    (si_dir / "SI_explainability_captions.tex").write_text(si_cap)
    print("wrote captions", flush=True)


def update_snippet(outdir: Path, pack_fig: Path) -> None:
    src = outdir / "13_explainability_paper_snippet.tex"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    shutil.copy2(src, pack_fig / src.name)
    print("wrote", pack_fig / src.name, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config_ascat_a_crps.json"))
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "models" / "unet_crps" / "crps_ensemble_best.pt"),
    )
    ap.add_argument("--outdir", default=str(ROOT / "evaluation" / "explainability" / "crps"))
    ap.add_argument(
        "--pack-dir",
        default=str(ROOT / "evaluation" / "paper_revision_pack_crps_ensemble" / "01_figures"),
    )
    ap.add_argument(
        "--si-dir",
        default=str(
            ROOT / "evaluation" / "paper_revision_pack_crps_ensemble" / "05_member_sensitivity"
        ),
        help="Default SI landing zone; explainability SI files go to 07_explainability_si",
    )
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=TAU_STAR)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-latent-pixels", type=int, default=20000)
    ap.add_argument("--ig-orbit-main", type=int, default=73598)
    ap.add_argument("--ig-orbits", default="73598,72850,75001")
    ap.add_argument("--ig-steps", type=int, default=24)
    ap.add_argument("--force-latent", action="store_true")
    ap.add_argument("--force-ig", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    pack_fig = Path(args.pack_dir)
    if not pack_fig.is_absolute():
        pack_fig = ROOT / pack_fig
    pack_fig.mkdir(parents=True, exist_ok=True)
    si_dir = ROOT / "evaluation" / "paper_revision_pack_crps_ensemble" / "07_explainability_si"
    si_dir.mkdir(parents=True, exist_ok=True)

    ensure_latent(outdir, args)

    # Main figure: revised 4-panel layout (PCA × P | perm±err | IG | coherence).
    # Delegates to plot_explainability_panels_bd.py so the old dual-PCA /
    # dual-bar permutation layout is not regenerated.
    import validation.plot_explainability_panels_bd as panels_bd
    perm_df = panels_bd.load_perm(panels_bd.SHARE / "perm_errorbars.csv")
    coh = pd.read_csv(panels_bd.SHARE / "member_coherence.csv")
    panels_bd.save_standalone_panels(perm_df, coh, outdir)
    main_png = panels_bd.try_plot_revised_main(
        perm_df, coh, outdir, pack_fig, ig_orbit=args.ig_orbit_main,
    )
    if main_png is None:
        raise SystemExit(
            "failed to build revised main figure; need latent + IG geo caches"
        )
    # Canonical name used by the manuscript.
    for ext in (".png", ".pdf"):
        shutil.copy2(
            outdir / f"13_explainability_main_v2{ext}",
            outdir / f"13_explainability_main{ext}",
        )
        shutil.copy2(
            outdir / f"13_explainability_main{ext}",
            pack_fig / f"13_explainability_main{ext}",
        )

    plot_si_pca_extra(outdir, si_dir)
    ensure_ig_si(outdir, si_dir, args)
    write_captions(si_dir, pack_fig)
    update_snippet(outdir, pack_fig)

    # Keep legacy multi-panel PCA in SI as well (renamed)
    legacy = outdir / "13_latent_bottleneck_pca.png"
    if legacy.exists():
        shutil.copy2(legacy, si_dir / "SI_explain_pca_all_colorings.png")
        pdf = legacy.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, si_dir / "SI_explain_pca_all_colorings.pdf")

    readme = si_dir / "README.md"
    readme.write_text(
        """# Explainability — Supporting Information

Main-text combined figure: `../01_figures/13_explainability_main.{png,pdf}`

SI panels in this folder:

| File | Content |
|------|---------|
| `SI_explain_pca_extra` | Extra PCA colorings (S, \\|∇U\\|, U_smooth, rain) |
| `SI_explain_ig_cases` | All three IG orbits (left\\|right) |
| `SI_explain_ig_group_share` | IG group-share bars + CSV |
| `SI_explainability_captions.tex` | Caption drafts |

Working outputs also under `evaluation/explainability/crps/`.
"""
    )
    print(f"done → main in {pack_fig}, SI in {si_dir}", flush=True)


if __name__ == "__main__":
    main()
