#!/usr/bin/env python3
"""
Out-of-sample isotonic calibration (Garg follow-up step 2).

- Two-fold blocked by YEAR (not random rows)
- Report only out-of-sample BSS / Brier / AUC
- Cross-basin transfer: fit Pacific, verify Atlantic + Indian
- Target plateau cells W in {3,4,6}, d0 in {75,100,150} plus full surface

Uses the common-opportunity forecasts from garg_followup_surfaces.py.

Example
-------
  python validation/isotonic_oos_calibration.py \\
      --outdir evaluation/garg_followup/isotonic_oos
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

from validation.buoy_prob_scores import auc_mannwhitney  # noqa: E402


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA) — no sklearn dependency
# ---------------------------------------------------------------------------

def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit increasing isotonic regression of y on x via PAVA.

    Returns (x_unique_sorted, y_fitted_on_unique) for piecewise-constant
    left-continuous prediction via searchsorted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    # Collapse duplicate x to mean y (weights = counts)
    xu, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    yu = np.zeros(len(xu), dtype=float)
    for i, yi in zip(inv, y):
        yu[i] += yi
    yu /= counts

    # PAVA on (yu, weights=counts)
    n = len(yu)
    level = yu.astype(float).copy()
    weight = counts.astype(float).copy()
    # block starts
    block_val = []
    block_wt = []
    block_end = []
    for i in range(n):
        block_val.append(level[i])
        block_wt.append(weight[i])
        block_end.append(i)
        while len(block_val) >= 2 and block_val[-2] > block_val[-1]:
            w1, w2 = block_wt[-2], block_wt[-1]
            v = (block_val[-2] * w1 + block_val[-1] * w2) / (w1 + w2)
            block_val[-2] = v
            block_wt[-2] = w1 + w2
            block_end[-2] = block_end[-1]
            block_val.pop()
            block_wt.pop()
            block_end.pop()

    fitted = np.empty(n, dtype=float)
    start = 0
    for v, end in zip(block_val, block_end):
        fitted[start : end + 1] = v
        start = end + 1
    return xu, fitted


def isotonic_predict(xu: np.ndarray, yu: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Piecewise-constant prediction from fitted unique knots (increasing)."""
    x = np.asarray(x, dtype=float)
    # map each x to nearest left knot
    idx = np.searchsorted(xu, x, side="right") - 1
    idx = np.clip(idx, 0, len(xu) - 1)
    return yu[idx]


def brier(p, o):
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    return float(np.mean((p - o) ** 2))


def bss(p, o):
    o = np.asarray(o, float)
    unc = float(o.mean() * (1.0 - o.mean()))
    if unc <= 0:
        return float("nan")
    return float(1.0 - brier(p, o) / unc)


def summarize(p, o) -> dict:
    o = np.asarray(o, float)
    p = np.asarray(p, float)
    return dict(
        n=int(len(p)),
        event_rate=float(o.mean()) if len(o) else float("nan"),
        mean_p=float(p.mean()) if len(p) else float("nan"),
        brier=brier(p, o) if len(p) else float("nan"),
        bss=bss(p, o) if len(p) else float("nan"),
        auc=auc_mannwhitney(p, o) if len(p) else float("nan"),
    )


# ---------------------------------------------------------------------------
# Metadata: year + basin
# ---------------------------------------------------------------------------

def basin_from_lon(lon: float) -> str:
    """Tropical buoy basins in 0–360 lon."""
    lon = float(lon) % 360.0
    # Indian Ocean roughly 20E–120E
    if 20.0 <= lon < 120.0:
        return "indian"
    # Pacific 120E–290E (70W)
    if 120.0 <= lon <= 290.0:
        return "pacific"
    # Atlantic: remainder (west of Africa / east of Americas)
    return "atlantic"


def attach_meta(fc: pd.DataFrame, opps_path: Path) -> pd.DataFrame:
    """Join year and basin from classical opportunities (t0, buoy_lon)."""
    meta = pd.read_csv(
        opps_path,
        usecols=["orbit", "buoy", "t0", "buoy_lon"],
    )
    meta["orbit"] = meta["orbit"].map(lambda x: str(int(float(x))))
    meta["buoy"] = meta["buoy"].astype(str)
    meta["year"] = pd.to_datetime(meta["t0"]).dt.year
    meta["basin"] = meta["buoy_lon"].map(basin_from_lon)
    # one meta row per (orbit, buoy)
    meta = (
        meta.groupby(["orbit", "buoy"], as_index=False)
        .agg(year=("year", "first"), basin=("basin", "first"), buoy_lon=("buoy_lon", "first"))
    )
    out = fc.copy()
    out["orbit"] = out["orbit"].map(lambda x: str(int(float(x))))
    out["buoy"] = out["buoy"].astype(str)
    out = out.merge(meta, on=["orbit", "buoy"], how="left")
    if out["year"].isna().any():
        nmiss = int(out["year"].isna().sum())
        raise SystemExit(f"missing year/basin meta for {nmiss} rows")
    out["year"] = out["year"].astype(int)
    return out


def year_folds(years: np.ndarray, seed: int = 0) -> tuple[set, set]:
    """Split unique years into two folds (sorted, then alternate after shuffle)."""
    rng = np.random.default_rng(seed)
    uy = np.array(sorted(set(int(y) for y in years)))
    rng.shuffle(uy)
    mid = len(uy) // 2
    # ensure both non-empty
    a = set(uy[: max(mid, 1)].tolist())
    b = set(uy[max(mid, 1) :].tolist()) or set(uy[:1].tolist())
    if not a:
        a = set(uy[:1].tolist())
    return a, b


def oos_isotonic_by_year(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Two-fold year-blocked OOS isotonic; returns df with p_cal column."""
    fold_a, fold_b = year_folds(df["year"].to_numpy(), seed=seed)
    p_cal = np.full(len(df), np.nan)
    for fit_years, eval_years in ((fold_a, fold_b), (fold_b, fold_a)):
        fit = df["year"].isin(fit_years)
        ev = df["year"].isin(eval_years)
        if fit.sum() < 50 or ev.sum() < 50:
            continue
        xu, yu = isotonic_fit(df.loc[fit, "p"].to_numpy(), df.loc[fit, "o"].to_numpy())
        p_cal[ev.to_numpy()] = isotonic_predict(xu, yu, df.loc[ev, "p"].to_numpy())
    out = df.copy()
    out["p_cal"] = p_cal
    out.attrs["fold_a_years"] = sorted(fold_a)
    out.attrs["fold_b_years"] = sorted(fold_b)
    return out


def cross_basin_transfer(df: pd.DataFrame) -> dict:
    """Fit isotonic on Pacific, evaluate on Atlantic and Indian (and Pacific in-sample for ref)."""
    pac = df["basin"] == "pacific"
    if pac.sum() < 100:
        return {"error": "too few Pacific rows"}
    xu, yu = isotonic_fit(df.loc[pac, "p"].to_numpy(), df.loc[pac, "o"].to_numpy())
    rows = []
    for basin in ("pacific", "atlantic", "indian"):
        g = df[df["basin"] == basin]
        if len(g) < 20:
            continue
        p_hat = isotonic_predict(xu, yu, g["p"].to_numpy())
        raw = summarize(g["p"].to_numpy(), g["o"].to_numpy())
        cal = summarize(p_hat, g["o"].to_numpy())
        rows.append(dict(
            basin=basin,
            n=cal["n"],
            raw_bss=raw["bss"],
            cal_bss=cal["bss"],
            raw_brier=raw["brier"],
            cal_brier=cal["brier"],
            raw_auc=raw["auc"],
            cal_auc=cal["auc"],
            raw_mean_p=raw["mean_p"],
            cal_mean_p=cal["mean_p"],
            event_rate=cal["event_rate"],
            transfer=("in_sample" if basin == "pacific" else "oos_transfer"),
        ))
    return rows


def score_cells(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    rows = []
    for (w, d0), g in df.groupby(["W_hr", "d0_km"]):
        g = g.dropna(subset=[pcol])
        if not len(g):
            continue
        s = summarize(g[pcol].to_numpy(), g["o"].to_numpy())
        rows.append(dict(W_hr=float(w), d0_km=float(d0), **s))
    return pd.DataFrame(rows).sort_values(["W_hr", "d0_km"])


def plot_reliability(p, o, title, out_png, n_bins=10):
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        bins = p
    else:
        bins = pd.cut(p, np.unique(np.r_[-1e-9, edges[1:-1], 1 + 1e-9]))
    t = pd.DataFrame({"p": p, "o": o, "bin": bins}).groupby("bin", observed=True).agg(
        n=("o", "size"), mean_p=("p", "mean"), obs=("o", "mean")
    )
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(t["mean_p"], t["obs"], "-o", color="#1a5fa8", ms=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_flavor(
    name: str,
    fc_path: Path,
    opps_path: Path,
    outdir: Path,
    plateau: list[tuple[float, float]],
    seed: int,
) -> dict:
    print(f"\n=== {name} ===", flush=True)
    fc = pd.read_csv(fc_path)
    fc = attach_meta(fc, opps_path)
    print(
        f"  rows={len(fc):,} years={sorted(fc.year.unique().tolist())} "
        f"basins={fc.basin.value_counts().to_dict()}",
        flush=True,
    )

    flav_dir = outdir / name
    flav_dir.mkdir(parents=True, exist_ok=True)

    # Year-blocked OOS on full surface (per cell independently — important:
    # calibration curve can depend on (W,d0) because base rates differ)
    oos_rows = []
    raw_rows = []
    for (w, d0), g in fc.groupby(["W_hr", "d0_km"]):
        g = g.copy()
        cal = oos_isotonic_by_year(g, seed=seed)
        ok = cal["p_cal"].notna()
        raw_rows.append(dict(W_hr=float(w), d0_km=float(d0), **summarize(
            g["p"].to_numpy(), g["o"].to_numpy()
        )))
        if ok.sum() < 50:
            continue
        s = summarize(cal.loc[ok, "p_cal"].to_numpy(), cal.loc[ok, "o"].to_numpy())
        oos_rows.append(dict(W_hr=float(w), d0_km=float(d0), **s))

        # reliability for plateau cells
        if any(np.isclose(w, ww) and np.isclose(d0, dd) for ww, dd in plateau):
            plot_reliability(
                cal.loc[ok, "p_cal"].to_numpy(),
                cal.loc[ok, "o"].to_numpy(),
                f"{name} OOS isotonic  W={w:g} d0={d0:g}\n"
                f"BSS={s['bss']:.3f} AUC={s['auc']:.3f}",
                flav_dir / f"reliability_oos_W{w:g}_d0{d0:g}.png",
            )

    raw = pd.DataFrame(raw_rows).sort_values(["W_hr", "d0_km"])
    oos = pd.DataFrame(oos_rows).sort_values(["W_hr", "d0_km"])
    raw.to_csv(flav_dir / "scores_raw.csv", index=False)
    oos.to_csv(flav_dir / "scores_oos_isotonic.csv", index=False)

    # Plateau summary
    plat = []
    for w, d0 in plateau:
        rr = raw[np.isclose(raw.W_hr, w) & np.isclose(raw.d0_km, d0)]
        oo = oos[np.isclose(oos.W_hr, w) & np.isclose(oos.d0_km, d0)]
        if len(rr) and len(oo):
            plat.append(dict(
                W_hr=w, d0_km=d0,
                raw_bss=float(rr.iloc[0].bss),
                oos_bss=float(oo.iloc[0].bss),
                raw_auc=float(rr.iloc[0].auc),
                oos_auc=float(oo.iloc[0].auc),
                raw_brier=float(rr.iloc[0].brier),
                oos_brier=float(oo.iloc[0].brier),
                oos_mean_p=float(oo.iloc[0].mean_p),
                event_rate=float(oo.iloc[0].event_rate),
                n=int(oo.iloc[0].n),
            ))
    plat_df = pd.DataFrame(plat)
    plat_df.to_csv(flav_dir / "plateau_oos.csv", index=False)
    print(plat_df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

    # Cross-basin on plateau cells pooled, and on focus cells
    basin_rows = []
    plat_mask = np.zeros(len(fc), dtype=bool)
    for w, d0 in plateau:
        plat_mask |= np.isclose(fc["W_hr"].to_numpy(), w) & np.isclose(
            fc["d0_km"].to_numpy(), d0
        )
    for label, mask in (
        ("plateau_pooled", plat_mask),
        ("W3_d0100", np.isclose(fc["W_hr"], 3) & np.isclose(fc["d0_km"], 100)),
        ("W6_d0100", np.isclose(fc["W_hr"], 6) & np.isclose(fc["d0_km"], 100)),
    ):
        sub = fc.loc[mask]
        for row in cross_basin_transfer(sub):
            basin_rows.append(dict(scope=label, **row))
    basin_df = pd.DataFrame(basin_rows)
    basin_df.to_csv(flav_dir / "cross_basin_transfer.csv", index=False)
    print("cross-basin:", flush=True)
    print(basin_df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

    # Peak OOS BSS cell
    if len(oos):
        i = oos["bss"].idxmax()
        peak = oos.loc[i]
        print(
            f"  peak OOS BSS={peak.bss:.4f} at W={peak.W_hr:g} d0={peak.d0_km:g} "
            f"AUC={peak.auc:.3f}",
            flush=True,
        )
    else:
        peak = None

    fold_a, fold_b = year_folds(fc["year"].to_numpy(), seed=seed)
    return dict(
        flavor=name,
        fold_a_years=sorted(fold_a),
        fold_b_years=sorted(fold_b),
        plateau=plat_df.to_dict(orient="records"),
        peak_oos_bss=(
            None if peak is None else dict(
                W_hr=float(peak.W_hr), d0_km=float(peak.d0_km),
                bss=float(peak.bss), auc=float(peak.auc),
            )
        ),
        n_positive_oos_bss=int((oos["bss"] > 0).sum()) if len(oos) else 0,
        n_cells=int(len(oos)),
    )


def plot_comparison_table(summaries: list[dict], out_png: Path, plateau):
    # Build rows for W=3/100 and W=6/100 across flavors
    rows = []
    for cell in ((3.0, 100.0), (6.0, 100.0), (6.0, 75.0)):
        for s in summaries:
            hits = [
                p for p in s["plateau"]
                if np.isclose(p["W_hr"], cell[0]) and np.isclose(p["d0_km"], cell[1])
            ]
            if not hits and cell == (6.0, 75.0):
                # may not be in plateau list — skip
                continue
            if not hits:
                continue
            p = hits[0]
            rows.append([
                s["flavor"],
                f"W={cell[0]:g}, d0={cell[1]:g}",
                f"{p['raw_bss']:.3f}",
                f"{p['oos_bss']:.3f}",
                f"{p['oos_auc']:.3f}",
                f"{p['n']}",
            ])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 0.6 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title(
        "Year-blocked OOS isotonic BSS (bankable)",
        fontsize=12, fontweight="bold", pad=8,
    )
    tbl = ax.table(
        cellText=rows,
        colLabels=["flavor", "cell", "raw BSS", "OOS BSS", "OOS AUC", "n"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a5fa8")
            cell.set_text_props(color="white", weight="bold")
        else:
            # highlight positive OOS BSS
            try:
                if c == 3 and float(rows[r - 1][3]) > 0:
                    cell.set_facecolor("#d9f2e3")
            except Exception:
                pass
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--forecast-dir",
        default=str(ROOT / "evaluation/garg_followup"),
    )
    ap.add_argument(
        "--opportunities",
        default=str(ROOT / "evaluation/buoy_collocation_v2/opportunities.csv"),
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "evaluation/garg_followup/isotonic_oos"),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fdir = Path(args.forecast_dir)
    if not fdir.is_absolute():
        fdir = ROOT / fdir
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    opps = Path(args.opportunities)
    if not opps.is_absolute():
        opps = ROOT / opps

    plateau = [
        (3.0, 75.0), (3.0, 100.0), (3.0, 150.0),
        (4.0, 75.0), (4.0, 100.0), (4.0, 150.0),
        (6.0, 75.0), (6.0, 100.0), (6.0, 150.0),
    ]

    flavors = [
        ("classical", fdir / "forecast_classical_common.csv"),
        ("memberfrac", fdir / "forecast_memberfrac_common.csv"),
        ("p_max_exact", fdir / "forecast_pmax_exact_common.csv"),
    ]

    summaries = []
    for name, path in flavors:
        if not path.exists():
            print(f"SKIP missing {path}", flush=True)
            continue
        summaries.append(
            run_flavor(name, path, opps, outdir, plateau, seed=args.seed)
        )

    (outdir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot_comparison_table(summaries, outdir / "oos_bss_focus_table.png", plateau)

    # Combined plateau CSV
    rows = []
    for s in summaries:
        for p in s["plateau"]:
            rows.append(dict(flavor=s["flavor"], **p))
    pd.DataFrame(rows).to_csv(outdir / "plateau_all_flavors.csv", index=False)

    print(f"\nwrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
