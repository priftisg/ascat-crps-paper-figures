#!/usr/bin/env python3
"""
Garg follow-up analytics on existing score / opportunity files.

Implements the computable next pieces that do not need a retrain:

  (1) p_max_exact Murphy surface (same scorer as classical / member-fraction)
  (2) calibrated-skill ceiling = resolution / uncertainty for every flavor
  (3) common opportunity set (normalize orbit IDs) + paired bootstrap
  (4) spread–skill vs scene activity (95th-pct WGM proxy from per-orbit file
      if available, else CRPS as activity proxy until WGM is joined)

Does NOT yet do year-blocked isotonic (step 2 in Garg's list) — that is a
separate script once the common set + p_max surface exist.

Example
-------
  env -u PYTHONPATH /rhome/gpriftis/miniconda3/bin/python \\
      validation/garg_followup_surfaces.py \\
      --outdir evaluation/garg_followup
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = research_root(Path(__file__))
ensure_sys_path(ROOT)

from validation.buoy_prob_scores import (  # noqa: E402
    paired_bootstrap,
    plot_cells,
    reliability_table,
    score_surface,
)


KEY = ["orbit", "buoy", "W_hr", "d0_km"]


def norm_orbit(s) -> str:
    return str(int(float(s)))


def load_memberfrac_forecast(paths: list[Path]) -> pd.DataFrame:
    """Rebuild member-fraction p from per-member opportunities (dist predicate)."""
    from validation.buoy_prob_scores import build_forecast

    return build_forecast([str(p) for p in paths], "dist")


def load_classical_forecast(path: Path) -> pd.DataFrame:
    from validation.buoy_prob_scores import build_forecast

    return build_forecast([str(path)], "dist")


def load_pmax_forecast(scored_csv: Path) -> pd.DataFrame:
    """
    Convert probabilistic_buoy_verify_v2 scored rows into the scorer schema.

    p_max in that file IS p_max_exact (max P within exact d0).
    """
    df = pd.read_csv(scored_csv)
    need = {"orbit", "buoy", "W_hr", "d0_km", "o", "p_max"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{scored_csv}: missing {sorted(missing)}")
    out = pd.DataFrame({
        "orbit": df["orbit"].map(norm_orbit),
        "buoy": df["buoy"].astype(str),
        "W_hr": df["W_hr"].astype(float),
        "d0_km": df["d0_km"].astype(float),
        "p": df["p_max"].astype(float).clip(0.0, 1.0),
        "o": df["o"].astype(float),
        "M": 1,
    })
    return out


def normalize_fc(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["orbit"] = fc["orbit"].map(norm_orbit)
    fc["buoy"] = fc["buoy"].astype(str)
    return fc


def common_index(*fcs: pd.DataFrame) -> pd.Index:
    idx = None
    for fc in fcs:
        ix = fc.set_index(KEY).index
        idx = ix if idx is None else idx.intersection(ix)
    return idx


def restrict(fc: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    return fc.set_index(KEY).loc[idx].reset_index()


def ceiling_table(scores: pd.DataFrame, label: str) -> pd.DataFrame:
    s = scores.copy()
    s["ceiling"] = s["resolution"] / s["uncertainty"].replace(0, np.nan)
    s["flavor"] = label
    return s


def plot_ceiling_vs_d0(frames: dict[str, pd.DataFrame], out_png: Path) -> None:
    """Left panel style: ceiling vs d0, one curve per W (viridis)."""
    ws = sorted({float(w) for sc in frames.values() for w in sc["W_hr"].unique()})
    cmap = plt.cm.viridis
    colors = {w: cmap(i / max(len(ws) - 1, 1)) for i, w in enumerate(ws)}
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True, squeeze=False)
    for ax, (name, sc) in zip(axes[0], frames.items()):
        for w in ws:
            g = sc[np.isclose(sc["W_hr"], w)].sort_values("d0_km")
            if g.empty:
                continue
            ax.plot(
                g["d0_km"], g["ceiling"], "-o", ms=3.5,
                color=colors[w], label=f"W=±{w:g} h",
            )
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"$d_0$ (km)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(r"calibrated-skill ceiling (resolution / uncertainty)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Murphy calibrated-skill ceiling (in-sample bin estimate)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_spread_skill_diagnostics(
    per_orbit: Path, out_dir: Path
) -> None:
    df = pd.read_csv(per_orbit)
    if "spread_skill" not in df.columns or "crps" not in df.columns:
        raise SystemExit(f"{per_orbit}: need crps and spread_skill")
    df["orbit_num"] = df["orbit"].map(lambda x: int(float(x)))
    df = df.sort_values("orbit_num")

    # Middle-style: CRPS vs orbit + rolling median
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.scatter(df["orbit_num"], df["crps"], s=4, alpha=0.25, color="0.45", linewidths=0)
    roll = df["crps"].rolling(200, center=True, min_periods=50).median()
    ax.plot(df["orbit_num"], roll, color="#c23b22", lw=1.6, label="200-orbit median")
    ax.set_xlabel("orbit number")
    ax.set_ylabel("per-orbit afCRPS")
    ax.set_title("Field CRPS vs orbit (no-drift check)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "field_crps_vs_orbit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Right-style: SSR histogram
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(df["spread_skill"], bins=40, color="#1a5fa8", edgecolor="white")
    ax.axvline(1.0, color="0.2", ls="--", lw=1.2, label="perfect dispersion")
    med = float(df["spread_skill"].median())
    ax.axvline(med, color="#c23b22", lw=1.4, label=f"median={med:.2f}")
    ax.set_xlabel("per-orbit spread–skill")
    ax.set_ylabel("count")
    ax.set_title("Spread–skill distribution")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "spread_skill_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Step (4): SSR vs scene activity proxy = per-orbit CRPS
    # (higher CRPS ≈ harder/more active scenes until WGM p95 is joined)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(df["crps"], df["spread_skill"], s=8, alpha=0.3, color="#1a5fa8", linewidths=0)
    # binned means
    edges = np.quantile(df["crps"], np.linspace(0, 1, 11))
    edges = np.unique(edges)
    df["bin"] = pd.cut(df["crps"], edges, include_lowest=True)
    b = df.groupby("bin", observed=True).agg(
        crps=("crps", "mean"), ssr=("spread_skill", "mean"), n=("crps", "size")
    ).dropna()
    ax.plot(b["crps"], b["ssr"], "-o", color="#c23b22", ms=5, label="decile mean")
    ax.axhline(1.0, color="0.3", ls="--", lw=0.9)
    corr = float(df["crps"].corr(df["spread_skill"]))
    ax.set_xlabel("per-orbit afCRPS (scene-difficulty proxy)")
    ax.set_ylabel("spread–skill")
    ax.set_title(f"Conditional dispersion  (corr={corr:.2f})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "spread_skill_vs_crps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_orbits": int(len(df)),
        "corr_orbit_crps": float(df["orbit_num"].corr(df["crps"])),
        "corr_crps_spread_skill": corr,
        "ssr_median": med,
        "ssr_mean": float(df["spread_skill"].mean()),
        "ssr_std": float(df["spread_skill"].std()),
        "ssr_frac_in_0p8_1p3": float(
            ((df["spread_skill"] >= 0.8) & (df["spread_skill"] <= 1.3)).mean()
        ),
        "suggested_spread_rescale": float(1.0 / med) if med > 0 else None,
    }
    (out_dir / "field_orbit_diagnostics.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print("field diagnostics:", json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--memberfrac-opps",
        default=str(
            ROOT / "evaluation/crps_memberfrac_tau1p2e4/collocations/m*/opportunities.csv"
        ),
    )
    ap.add_argument(
        "--classical-opps",
        default=str(ROOT / "evaluation/buoy_collocation_v2/opportunities.csv"),
    )
    ap.add_argument(
        "--pmax-scored",
        default=str(
            ROOT / "evaluation/crps_buoy_probabilistic_v2/scored_opportunities_v2.csv"
        ),
        help="Baseline OP scored rows (p_max = p_max_exact)",
    )
    ap.add_argument(
        "--per-orbit",
        default=str(ROOT / "evaluation/crps_field_calibration/per_orbit_scores.csv"),
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "evaluation/garg_followup"),
    )
    ap.add_argument("--cells", default="3:50,3:100,6:100")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load forecasts ---
    import glob

    mpaths = sorted(glob.glob(args.memberfrac_opps))
    if not mpaths:
        raise SystemExit(f"no memberfrac files matched {args.memberfrac_opps}")
    print(f"memberfrac files: {len(mpaths)}", flush=True)
    fc_m = normalize_fc(load_memberfrac_forecast([Path(p) for p in mpaths]))
    fc_c = normalize_fc(load_classical_forecast(Path(args.classical_opps)))
    fc_p = normalize_fc(load_pmax_forecast(Path(args.pmax_scored)))

    print(
        f"rows: memberfrac={len(fc_m):,} classical={len(fc_c):,} p_max={len(fc_p):,}",
        flush=True,
    )

    idx = common_index(fc_m, fc_c, fc_p)
    print(f"common opportunities: {len(idx):,}", flush=True)
    fc_m = restrict(fc_m, idx)
    fc_c = restrict(fc_c, idx)
    fc_p = restrict(fc_p, idx)

    # save aligned forecasts for isotonic later
    fc_m.to_csv(outdir / "forecast_memberfrac_common.csv", index=False)
    fc_c.to_csv(outdir / "forecast_classical_common.csv", index=False)
    fc_p.to_csv(outdir / "forecast_pmax_exact_common.csv", index=False)

    # --- Murphy surfaces ---
    surfaces = {
        "classical": score_surface(fc_c),
        "memberfrac": score_surface(fc_m),
        "p_max_exact": score_surface(fc_p),
    }
    ceilings = {}
    for name, sc in surfaces.items():
        sc.to_csv(outdir / f"{name}_scores_common.csv", index=False)
        ceil = ceiling_table(sc, name)
        ceil.to_csv(outdir / f"{name}_ceiling_common.csv", index=False)
        ceilings[name] = ceil
        # peak cell
        i = ceil["ceiling"].idxmax()
        r = ceil.loc[i]
        print(
            f"{name:12s} peak ceiling={r['ceiling']:.4f} "
            f"at W={r['W_hr']:g} d0={r['d0_km']:g}  "
            f"AUC={r['auc']:.3f} BSS={r['bss']:.3f}",
            flush=True,
        )

    plot_ceiling_vs_d0(ceilings, outdir / "ceiling_vs_d0.png")

    # compare ceilings cell-by-cell
    merge = ceilings["memberfrac"][["W_hr", "d0_km", "ceiling", "auc", "bss"]].rename(
        columns={"ceiling": "ceil_mf", "auc": "auc_mf", "bss": "bss_mf"}
    )
    merge = merge.merge(
        ceilings["classical"][["W_hr", "d0_km", "ceiling", "auc", "bss"]].rename(
            columns={"ceiling": "ceil_cl", "auc": "auc_cl", "bss": "bss_cl"}
        ),
        on=["W_hr", "d0_km"],
    )
    merge = merge.merge(
        ceilings["p_max_exact"][["W_hr", "d0_km", "ceiling", "auc", "bss"]].rename(
            columns={"ceiling": "ceil_px", "auc": "auc_px", "bss": "bss_px"}
        ),
        on=["W_hr", "d0_km"],
    )
    merge["mf_beats_classical"] = merge["ceil_mf"] > merge["ceil_cl"]
    merge["px_beats_mf"] = merge["ceil_px"] > merge["ceil_mf"]
    merge.to_csv(outdir / "ceiling_comparison.csv", index=False)
    print(
        f"memberfrac > classical ceiling in "
        f"{int(merge['mf_beats_classical'].sum())}/{len(merge)} cells",
        flush=True,
    )
    print(
        f"p_max_exact > memberfrac ceiling in "
        f"{int(merge['px_beats_mf'].sum())}/{len(merge)} cells",
        flush=True,
    )

    # --- paired bootstraps on common set ---
    cells = [tuple(float(x) for x in c.split(":")) for c in args.cells.split(",")]
    paired_mc = paired_bootstrap(fc_m, fc_c, cells, n_boot=args.n_boot)
    paired_mc.to_csv(outdir / "paired_memberfrac_vs_classical.csv", index=False)
    paired_pm = paired_bootstrap(fc_p, fc_m, cells, n_boot=args.n_boot)
    paired_pm.to_csv(outdir / "paired_pmax_vs_memberfrac.csv", index=False)
    paired_pc = paired_bootstrap(fc_p, fc_c, cells, n_boot=args.n_boot)
    paired_pc.to_csv(outdir / "paired_pmax_vs_classical.csv", index=False)
    print("\npaired memberfrac vs classical (A better if d_brier<0):")
    print(paired_mc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\npaired p_max_exact vs memberfrac:")
    print(paired_pm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # reliability plots for p_max on focus cells
    rel = pd.concat(
        [
            reliability_table(
                fc_p[(fc_p.W_hr == W) & (fc_p.d0_km == d0)]
            ).assign(W_hr=W, d0_km=d0)
            for (W, d0) in cells
            if len(fc_p[(fc_p.W_hr == W) & (fc_p.d0_km == d0)])
        ],
        ignore_index=True,
    )
    rel.to_csv(outdir / "pmax_exact_reliability.csv", index=False)
    plot_cells(
        fc_p, cells, "p_max_exact",
        str(outdir / "pmax_exact_reliability.png"),
    )

    # --- field diagnostics ---
    plot_spread_skill_diagnostics(Path(args.per_orbit), outdir)

    # headline table PNG
    focus = merge[
        (np.isclose(merge.W_hr, 3) & np.isclose(merge.d0_km, 100))
        | (np.isclose(merge.W_hr, 6) & np.isclose(merge.d0_km, 100))
    ].copy()
    rows = []
    for _, r in focus.iterrows():
        rows.append([
            f"W={r.W_hr:g}, d0={r.d0_km:g}",
            f"{r.ceil_cl:.3f}", f"{r.ceil_mf:.3f}", f"{r.ceil_px:.3f}",
            f"{r.auc_cl:.3f}", f"{r.auc_mf:.3f}", f"{r.auc_px:.3f}",
        ])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    ax.set_title(
        "Calibrated-skill ceiling (res/unc) on common opportunities",
        fontsize=12, fontweight="bold", pad=8,
    )
    tbl = ax.table(
        cellText=rows,
        colLabels=[
            "cell", "ceil classical", "ceil memberfrac", "ceil p_max_exact",
            "AUC cl", "AUC mf", "AUC px",
        ],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a5fa8")
            cell.set_text_props(color="white", weight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "ceiling_focus_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "n_common": int(len(idx)),
        "n_memberfrac_raw": "see logs",
        "cells": args.cells,
        "pmax_source": str(args.pmax_scored),
        "note": (
            "Ceilings are in-sample Murphy bin estimates; bankable BSS needs "
            "year-blocked out-of-sample isotonic (next script)."
        ),
    }
    (outdir / "followup_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
