#!/usr/bin/env python3
"""
Field-level ensemble calibration on the satellite val split (PDF / Garg §).

Computes the three certificates that belong on the *field*, not the buoy board:

  1. Held-out almost-fair CRPS (same loss as training)
  2. Spread–skill ratio (ensemble std / RMSE of member-mean)
  3. Rank histogram of the observation among M sorted members

Requires a live forward pass (exceedance NPZs do not store members).

Example
-------
  # smoke
  python validation/score_ensemble_field.py --limit 10 --outdir evaluation/crps_field_calibration

  # full val (SLURM GPU)
  python validation/score_ensemble_field.py --outdir evaluation/crps_field_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from infer_crps import load_crps_model  # noqa: E402
from lib.crps_dataset import build_crps_half, orbit_from_path, split_files  # noqa: E402
from lib.losses import almost_fair_crps, spread_skill_ratio  # noqa: E402
from lib.model import ensemble_forward  # noqa: E402
from lib.swath_halves import HALF_NAMES  # noqa: E402


def rank_histogram_update(
    members: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    counts: np.ndarray,
    max_pixels: int,
    rng: np.random.Generator,
) -> int:
    """
    Accumulate ranks of y among M sorted members on valid pixels.

    members: (M, H, W), y/valid: (H, W)
    Rank r in 0..M means y falls in slot r of the ordered members
    (ties broken by mid-rank randomisation via uniform noise).
    """
    m = members.shape[0]
    mask = valid > 0.5
    if not np.any(mask):
        return 0
    yy = y[mask]
    mm = members[:, mask]  # (M, N)
    n = yy.shape[0]
    if n > max_pixels:
        sel = rng.choice(n, size=max_pixels, replace=False)
        yy = yy[sel]
        mm = mm[:, sel]
        n = max_pixels
    # Add tiny noise so ties don't pile into one bin
    noise = rng.uniform(-1e-6, 1e-6, size=mm.shape).astype(np.float32)
    order = np.sort(mm + noise, axis=0)  # (M, n)
    # rank = number of members strictly < y  (0 .. M)
    ranks = (order < yy[None, :]).sum(axis=0).astype(np.int64)
    for r in range(m + 1):
        counts[r] += int(np.sum(ranks == r))
    return int(n)


def plot_rank_histogram(counts: np.ndarray, out_png: Path, title: str) -> None:
    m = len(counts) - 1
    total = counts.sum()
    freq = counts / max(total, 1)
    expected = 1.0 / (m + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(m + 1), freq, width=0.85, color="#1a5fa8", edgecolor="white")
    ax.axhline(expected, color="0.3", ls="--", lw=1.0, label=f"flat = {expected:.3f}")
    ax.set_xlabel("rank (0 = obs below all members, M = obs above all)")
    ax.set_ylabel("relative frequency")
    ax.set_title(title)
    ax.set_xticks(np.arange(m + 1))
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"n_pix={total:,}\nU-shape → underdispersed\nhump → overdispersed",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spread_skill_histogram(per_orbit: pd.DataFrame, out_png: Path, m: int) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(per_orbit["spread_skill"], bins=40, color="#1a5fa8", edgecolor="white")
    ax.axvline(1.0, color="0.2", ls="--", lw=1.2, label="perfect dispersion")
    med = float(per_orbit["spread_skill"].median())
    ax.axvline(med, color="#c23b22", lw=1.4, label=f"median={med:.2f}")
    ax.set_xlabel("per-orbit spread–skill")
    ax.set_ylabel("count")
    ax.set_title(f"Spread–skill distribution (M={m})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_panel(
    counts: np.ndarray,
    per_orbit: pd.DataFrame,
    out_png: Path,
    m: int,
    n_orb: int,
) -> None:
    """Two-panel figure matching the reviewer field-calibration certificate."""
    total = counts.sum()
    freq = counts / max(total, 1)
    expected = 1.0 / (m + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.bar(np.arange(m + 1), freq, width=0.85, color="#1a5fa8", edgecolor="white")
    ax.axhline(expected, color="0.3", ls="--", lw=1.0, label=f"flat = {expected:.3f}")
    ax.set_xlabel("rank (0 = obs below all members, M = obs above all)")
    ax.set_ylabel("relative frequency")
    ax.set_title(f"Rank histogram — val field ensemble (M={m}, n_orb={n_orb})")
    # Avoid overcrowded ticks for large M
    if m <= 12:
        ax.set_xticks(np.arange(m + 1))
    else:
        ax.set_xticks(np.arange(0, m + 1, 2))
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"n_pix={total:,}\nU-shape → underdispersed\nhump → overdispersed",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="0.35",
    )

    ax = axes[1]
    ax.hist(per_orbit["spread_skill"], bins=40, color="#1a5fa8", edgecolor="white")
    ax.axvline(1.0, color="0.2", ls="--", lw=1.2, label="perfect dispersion")
    med = float(per_orbit["spread_skill"].median())
    ax.axvline(med, color="#c23b22", lw=1.4, label=f"median={med:.2f}")
    ax.set_xlabel("per-orbit spread–skill")
    ax.set_ylabel("count")
    ax.set_title("Spread–skill distribution")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary_card(summary: dict, out_png: Path) -> None:
    m = int(summary["members"])
    rows = [
        ("Held-out afCRPS", f"{summary['crps']:.6f}"),
        ("Spread–skill ratio", f"{summary['spread_skill']:.4f}"),
        ("  mean spread", f"{summary['spread']:.6f}"),
        ("  mean RMSE", f"{summary['rmse']:.6f}"),
        ("Rank hist χ² / flat", f"{summary['rank_chi2']:.3f}"),
        ("n halves", f"{summary['n_halves']}"),
        ("n orbits", f"{summary['n_orbits']}"),
        ("n valid pixels (rank)", f"{summary['n_rank_pixels']:,}"),
        ("M members", f"{m}"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axis("off")
    ax.set_title(
        f"Field ensemble calibration (val orbits, M={m})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    table = ax.table(
        cellText=[[a, b] for a, b in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#1a5fa8")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6fa")
    note = (
        "Target spread–skill ≈ 1.  Flat rank histogram = calibrated.\n"
        "These are satellite-field scores (Sobel-WGM), not buoy Brier/BSS."
    )
    ax.text(0.5, 0.06, note, ha="center", va="bottom", fontsize=8.5, color="0.35",
            transform=ax.transAxes)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config_ascat_a_crps.json"))
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "models" / "unet_crps" / "crps_ensemble_best.pt"),
    )
    ap.add_argument("--outdir", default=str(ROOT / "evaluation" / "crps_field_calibration"))
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Max val orbits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-rank-pixels-per-half",
        type=int,
        default=20000,
        help="Subsample valid pixels per half for rank hist (memory/speed)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth val orbit (1 = all)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open() as f:
        file_cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model, ckpt_cfg, rung = load_crps_model(args.checkpoint, device)
    if rung != "ensemble":
        raise SystemExit(f"need ensemble checkpoint, got rung={rung}")
    cfg = {
        **file_cfg,
        **{
            k: ckpt_cfg[k]
            for k in (
                "input_channels", "num_channels", "filters", "num_layers",
                "pool_along_track", "pool_cross_track", "ensemble",
                "gradient_scale", "fov_m", "gradient_peak_weight",
            )
            if k in ckpt_cfg
        },
    }
    ens = cfg.get("ensemble", {})
    m = int(args.members or ens.get("members", 8))
    alpha = float(ens.get("alpha", 0.95))
    peak = float(cfg.get("gradient_peak_weight", 4.0))

    files = split_files(cfg, "val")
    if args.stride > 1:
        files = files[:: int(args.stride)]
    if args.limit:
        files = files[: int(args.limit)]
    print(
        f"val orbits={len(files)}  halves={len(files) * 2}  M={m}  "
        f"afCRPS alpha={alpha}",
        flush=True,
    )

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rank_counts = np.zeros(m + 1, dtype=np.int64)
    n_rank_pix = 0
    crps_num = 0.0
    crps_den = 0.0
    spread_acc = 0.0
    rmse_acc = 0.0
    weight_acc = 0.0
    per_orbit = []
    t0 = time.time()
    n_halves = 0

    model.eval()
    with torch.no_grad():
        for i, path in enumerate(files, 1):
            orbit = orbit_from_path(path)
            orb_crps = []
            orb_ssr = []
            for half in HALF_NAMES:
                x_np, y_np = build_crps_half(path, half, cfg)
                x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device)
                y = torch.from_numpy(y_np).permute(2, 0, 1).unsqueeze(0).to(device)
                members = ensemble_forward(model, x, m=m)  # (1,M,H,W)
                y_t, valid = y[:, 0:1], y[:, 1:2]
                crps = almost_fair_crps(
                    members, y_t, valid, alpha=alpha, peak=peak,
                )
                ssr, sp, rm = spread_skill_ratio(members, y_t, valid)
                # Weighted running means by valid pixel count
                w = float(valid.sum().item())
                if w <= 0:
                    continue
                crps_num += float(crps.item()) * w
                crps_den += w
                spread_acc += sp * w
                rmse_acc += rm * w
                weight_acc += w
                orb_crps.append(float(crps.item()))
                orb_ssr.append(ssr)

                mem_np = members[0].detach().cpu().numpy()
                y_np2 = y_t[0, 0].detach().cpu().numpy()
                v_np = valid[0, 0].detach().cpu().numpy()
                n_rank_pix += rank_histogram_update(
                    mem_np, y_np2, v_np, rank_counts,
                    max_pixels=int(args.max_rank_pixels_per_half),
                    rng=rng,
                )
                n_halves += 1

            if orb_crps:
                per_orbit.append(
                    dict(
                        orbit=orbit,
                        crps=float(np.mean(orb_crps)),
                        spread_skill=float(np.mean(orb_ssr)),
                    )
                )
            if i % 25 == 0 or i == len(files):
                dt = time.time() - t0
                cur = crps_num / max(crps_den, 1.0)
                print(
                    f"[{i}/{len(files)}] halves={n_halves} "
                    f"crps≈{cur:.6f}  ({dt / 60:.1f} min)",
                    flush=True,
                )

    crps = crps_num / max(crps_den, 1.0)
    spread = spread_acc / max(weight_acc, 1.0)
    rmse = rmse_acc / max(weight_acc, 1.0)
    ssr = spread / max(rmse, 1e-12)
    expected = n_rank_pix / max(len(rank_counts), 1)
    chi2 = float(
        ((rank_counts.astype(float) - expected) ** 2 / max(expected, 1.0)).sum()
    ) if n_rank_pix else float("nan")

    summary = dict(
        crps=crps,
        spread_skill=ssr,
        spread=spread,
        rmse=rmse,
        rank_chi2=chi2,
        rank_counts=rank_counts.tolist(),
        rank_freq=(rank_counts / max(n_rank_pix, 1)).tolist(),
        n_rank_pixels=int(n_rank_pix),
        n_halves=n_halves,
        n_orbits=len(per_orbit),
        members=m,
        alpha=alpha,
        peak=peak,
        checkpoint=str(args.checkpoint),
        val_orbit_start=int(cfg["val_orbit_start"]),
        val_orbit_end=int(cfg["val_orbit_end"]),
        limit=args.limit,
        stride=args.stride,
        elapsed_sec=time.time() - t0,
        note=(
            "Field scores on Sobel-WGM target (normalized). "
            "spread_skill≈1 and flat rank hist = calibrated ensemble."
        ),
    )
    (outdir / "field_calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    pd.DataFrame(per_orbit).to_csv(outdir / "per_orbit_scores.csv", index=False)
    pd.DataFrame(
        {
            "rank": np.arange(m + 1),
            "count": rank_counts,
            "freq": rank_counts / max(n_rank_pix, 1),
        }
    ).to_csv(outdir / "rank_histogram.csv", index=False)

    per_orbit_df = pd.DataFrame(per_orbit)
    plot_rank_histogram(
        rank_counts,
        outdir / "rank_histogram.png",
        title=f"Rank histogram — val field ensemble (M={m}, n_orb={len(per_orbit)})",
    )
    if len(per_orbit_df):
        plot_spread_skill_histogram(
            per_orbit_df, outdir / "spread_skill_histogram.png", m=m,
        )
        plot_calibration_panel(
            rank_counts,
            per_orbit_df,
            outdir / "field_calibration_panel.png",
            m=m,
            n_orb=len(per_orbit),
        )
    plot_summary_card(summary, outdir / "field_calibration_summary.png")

    print(json.dumps({k: summary[k] for k in (
        "crps", "spread_skill", "spread", "rmse", "rank_chi2",
        "n_halves", "n_orbits", "n_rank_pixels", "elapsed_sec", "members",
    )}, indent=2), flush=True)
    print(f"wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
