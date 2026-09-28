#!/usr/bin/env python3
"""
buoy_prob_scores.py

Probabilistic buoy verification for GF/cold-pool products, on top of the
corrected collocation's opportunities.csv files.

Three jobs in one tool:

1. CLASSICAL BASELINE ROW (the row missing from the summary tables):
   score the classical catalog as the binary forecast it is, under the
   identical Brier/AUC metric.
       python buoy_prob_scores.py --opps results_v1qc/opportunities_sym.csv \
           --label classical -o probscores/

2. MEMBER-FRACTION ENSEMBLE SCORING (design doc section 4.7): run the
   collocation once per member catalog (cheap off the GF cache), then
       python buoy_prob_scores.py --opps results_m*/opportunities.csv \
           --label crps_memberfrac -o probscores/
   Forecast probability per opportunity = fraction of members with a
   qualifying detection; scored with Brier / BSS / AUC + Murphy
   reliability-resolution decomposition + reliability diagrams.

3. PAIRED COMPARISON with orbit-block bootstrap CIs:
       python buoy_prob_scores.py --opps results_m*/opportunities.csv \
           --compare results_v1qc/opportunities_sym.csv \
           --label crps_vs_classical -o probscores/

Forecast definition (important): by default the detection predicate is the
truth-independent spatial one, min_gf_dist_km <= d0  (a detection exists
within d0 of the buoy, regardless of event timing), so the "forecast" never
peeks at the verifying observation. --forecast-from outcome falls back to
outcome in {hit, false_alarm} for files without min_gf_dist_km.

Outputs in -o DIR:
  <label>_scores.csv        per (W, d0): n, event_rate, mean_p, brier, bss,
                            auc, reliability, resolution, uncertainty
  <label>_reliability.csv   binned reliability table for the focus cells
  <label>_reliability.png   reliability diagrams + ROC for the focus cells
  <label>_paired.csv        (with --compare) dBrier, dAUC + 95% bootstrap CI

Only numpy/pandas/matplotlib required.  Run --selftest first.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

KEY = ['orbit', 'buoy', 'W_hr', 'd0_km']


# ----------------------------------------------------------------------
# loading and forecast construction
# ----------------------------------------------------------------------

def _load_one(path: str, forecast_from: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'orbit': str})
    need = set(KEY + ['outcome', 'n_events'])
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f'{path}: missing columns {sorted(missing)}')
    df = df[df['outcome'] != 'invalid'].copy()
    if forecast_from == 'dist' and 'min_gf_dist_km' in df.columns:
        df['det'] = (df['min_gf_dist_km'] <= df['d0_km']).fillna(False)
    else:
        df['det'] = df['outcome'].isin(['hit', 'false_alarm'])
    df['event'] = df['n_events'] > 0
    return df.set_index(KEY)[['det', 'event']]


def build_forecast(paths: list, forecast_from: str) -> pd.DataFrame:
    """Align member files on (orbit, buoy, W, d0); p = member fraction with
    a detection; truth from the first member (verified consistent)."""
    mems = [_load_one(p, forecast_from) for p in paths]
    idx = mems[0].index
    for m in mems[1:]:
        idx = idx.intersection(m.index)
    if len(idx) < len(mems[0]):
        print(f'[align] common opportunities: {len(idx)} '
              f'(largest member file: {max(len(m) for m in mems)})',
              file=sys.stderr)
    det = pd.concat([m.loc[idx, 'det'] for m in mems], axis=1)
    ev = pd.concat([m.loc[idx, 'event'] for m in mems], axis=1)
    if not ev.nunique(axis=1).eq(1).all():
        raise SystemExit('event truth differs between member files -- these '
                         'are not collocations of the same buoy record')
    out = pd.DataFrame({'p': det.mean(axis=1),
                        'o': ev.iloc[:, 0].astype(float)}, index=idx)
    out['M'] = len(mems)
    return out.reset_index()


# ----------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------

def auc_mannwhitney(p: np.ndarray, o: np.ndarray) -> float:
    """ROC AUC via the rank-sum statistic (tie-aware midranks)."""
    n1 = int(o.sum())
    n0 = len(o) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(p).rank(method='average').to_numpy()
    return float((ranks[o > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def murphy_decomposition(p, o, n_bins=10):
    """Brier = reliability - resolution + uncertainty (binned)."""
    df = pd.DataFrame({'p': p, 'o': o})
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:                       # binary / few distinct values
        df['bin'] = df['p']
    else:
        df['bin'] = pd.cut(df['p'], np.unique(np.r_[-1e-9, edges[1:-1],
                                                    1 + 1e-9]))
    obar = o.mean()
    rel = res = 0.0
    for _b, g in df.groupby('bin', observed=True):
        if not len(g):
            continue
        w = len(g) / len(df)
        rel += w * (g['p'].mean() - g['o'].mean()) ** 2
        res += w * (g['o'].mean() - obar) ** 2
    return rel, res, obar * (1 - obar)


def cell_scores(g: pd.DataFrame) -> dict:
    p, o = g['p'].to_numpy(float), g['o'].to_numpy(float)
    brier = float(np.mean((p - o) ** 2))
    unc = o.mean() * (1 - o.mean())
    rel, res, _ = murphy_decomposition(p, o)
    return dict(n=len(g), event_rate=float(o.mean()),
                mean_p=float(p.mean()), brier=brier,
                bss=(1.0 - brier / unc) if unc > 0 else np.nan,
                auc=auc_mannwhitney(p, o),
                reliability=rel, resolution=res, uncertainty=unc)


def score_surface(fc: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(W_hr=W, d0_km=d, **cell_scores(g))
            for (W, d), g in fc.groupby(['W_hr', 'd0_km'])]
    return pd.DataFrame(rows).sort_values(['W_hr', 'd0_km'])


def reliability_table(g: pd.DataFrame, n_bins=10) -> pd.DataFrame:
    M = int(g['M'].iloc[0])
    if M > 1 and M <= n_bins:                # natural ensemble bins
        edges = (np.arange(M + 2) - 0.5) / M
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] += 1e-9
    b = pd.cut(g['p'], edges, include_lowest=True)
    t = g.groupby(b, observed=True).agg(n=('o', 'size'),
                                        mean_p=('p', 'mean'),
                                        obs_freq=('o', 'mean')).reset_index()
    return t.rename(columns={'p': 'bin'})


# ----------------------------------------------------------------------
# paired comparison with orbit-block bootstrap
# ----------------------------------------------------------------------

def paired_bootstrap(fc_a, fc_b, cells, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    m = fc_a.merge(fc_b, on=KEY, suffixes=('_a', '_b'))
    if not m['o_a'].eq(m['o_b']).all():
        raise SystemExit('paired products disagree on event truth')
    for (W, d0) in cells:
        g = m[(m['W_hr'] == W) & (m['d0_km'] == d0)]
        if not len(g):
            continue
        pa, pb = g['p_a'].to_numpy(), g['p_b'].to_numpy()
        o = g['o_a'].to_numpy(float)
        orbits = g['orbit'].to_numpy()
        uorb = np.unique(orbits)
        idx_by_orb = {u: np.flatnonzero(orbits == u) for u in uorb}
        d_brier = np.mean((pa - o) ** 2) - np.mean((pb - o) ** 2)
        d_auc = auc_mannwhitney(pa, o) - auc_mannwhitney(pb, o)
        db, da = [], []
        for _ in range(n_boot):
            sel = np.concatenate([idx_by_orb[u] for u in
                                  rng.choice(uorb, len(uorb), replace=True)])
            ob = o[sel]
            if ob.sum() in (0, len(ob)):
                continue
            db.append(np.mean((pa[sel] - ob) ** 2)
                      - np.mean((pb[sel] - ob) ** 2))
            da.append(auc_mannwhitney(pa[sel], ob)
                      - auc_mannwhitney(pb[sel], ob))
        rows.append(dict(
            W_hr=W, d0_km=d0, n=len(g),
            d_brier=d_brier,
            d_brier_lo=np.percentile(db, 2.5), d_brier_hi=np.percentile(db, 97.5),
            d_auc=d_auc,
            d_auc_lo=np.percentile(da, 2.5), d_auc_hi=np.percentile(da, 97.5),
            better_brier='A' if d_brier < 0 else 'B',
            significant=bool(np.percentile(db, 97.5) < 0
                             or np.percentile(db, 2.5) > 0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------

def plot_cells(fc, cells, label, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(cells), figsize=(3.6 * len(cells), 6.4),
                             squeeze=False)
    for j, (W, d0) in enumerate(cells):
        g = fc[(fc['W_hr'] == W) & (fc['d0_km'] == d0)]
        if not len(g):
            continue
        t = reliability_table(g)
        ax = axes[0][j]
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.axhline(g['o'].mean(), color='0.6', lw=0.8, ls=':')
        ax.plot(t['mean_p'], t['obs_freq'], '-o', ms=4, color='#1a5fa8')
        for _, r in t.iterrows():
            if np.isfinite(r['mean_p']):
                ax.annotate(f"{int(r['n'])}", (r['mean_p'], r['obs_freq']),
                            fontsize=6, xytext=(2, 3),
                            textcoords='offset points')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f'W=±{W:g} h, d0={d0:g} km', fontsize=9)
        ax.set_xlabel('forecast probability', fontsize=8)
        if j == 0:
            ax.set_ylabel('observed event frequency', fontsize=8)
        ax = axes[1][j]
        p, o = g['p'].to_numpy(), g['o'].to_numpy()
        order = np.argsort(-p)
        tpr = np.cumsum(o[order]) / max(o.sum(), 1)
        fpr = np.cumsum(1 - o[order]) / max((1 - o).sum(), 1)
        ax.plot(fpr, tpr, color='#c23b22', lw=1.4,
                label=f'AUC={auc_mannwhitney(p, o):.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlabel('false positive rate', fontsize=8)
        if j == 0:
            ax.set_ylabel('true positive rate', fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'{label}: reliability (top) and ROC (bottom)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


# ----------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------

def selftest() -> int:
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f'  [{"PASS" if cond else "FAIL"}] {name}')
        ok &= bool(cond)

    rng = np.random.default_rng(1)
    n_orb, M = 400, 4
    rows_m = [[] for _ in range(M)]
    rows_cls = []
    for k in range(n_orb):
        orbit, buoy = f'{k:05d}', '8n95w'
        event = rng.random() < 0.10
        # skillful ensemble: detections more likely when an event occurred
        base = 0.55 if event else 0.15
        dets = rng.random(M) < base
        cls_det = rng.random() < (0.50 if event else 0.20)
        for W in (3.0,):
            for d0 in (50.0, 100.0):
                for m in range(M):
                    rows_m[m].append((orbit, buoy, 0., 0., '2010', 5., 1.,
                                      60., W, d0,
                                      'hit' if (event and dets[m]) else
                                      'miss' if event else
                                      'false_alarm' if dets[m] else
                                      'correct_rejection',
                                      int(event), np.nan, np.nan,
                                      int(dets[m]),
                                      10.0 if dets[m] else np.nan))
                rows_cls.append((orbit, buoy, 0., 0., '2010', 5., 1., 60.,
                                 W, d0,
                                 'hit' if (event and cls_det) else
                                 'miss' if event else
                                 'false_alarm' if cls_det else
                                 'correct_rejection',
                                 int(event), np.nan, np.nan, int(cls_det),
                                 10.0 if cls_det else np.nan))
    cols = ['orbit', 'buoy', 'buoy_lon', 'buoy_lat', 't0', 'approach_km',
            'frac_valid_wind_50km', 'cadence_min', 'W_hr', 'd0_km',
            'outcome', 'n_events', 'min_dT', 'max_wind_jump', 'n_gfs_near',
            'min_gf_dist_km']
    with tempfile.TemporaryDirectory() as td:
        mpaths = []
        for m in range(M):
            p = os.path.join(td, f'opps_m{m}.csv')
            pd.DataFrame(rows_m[m], columns=cols).to_csv(p, index=False)
            mpaths.append(p)
        cpath = os.path.join(td, 'opps_classical.csv')
        pd.DataFrame(rows_cls, columns=cols).to_csv(cpath, index=False)

        fc = build_forecast(mpaths, 'dist')
        check('member fraction takes M+1 distinct values',
              set(np.round(fc['p'] * M).astype(int)) <= set(range(M + 1)))
        sc = score_surface(fc)
        r = sc[(sc.W_hr == 3.0) & (sc.d0_km == 50.0)].iloc[0]
        check('skillful ensemble: AUC > 0.65', r.auc > 0.65)
        check('Murphy decomposition consistent (brier = rel - res + unc)',
              abs(r.brier - (r.reliability - r.resolution + r.uncertainty))
              < 0.02)
        fcc = build_forecast([cpath], 'dist')
        check('single file -> binary forecast', set(fcc['p']) <= {0.0, 1.0})
        scc = score_surface(fcc)
        rc = scc[(scc.W_hr == 3.0) & (scc.d0_km == 50.0)].iloc[0]
        # manual Brier for the binary product
        g = fcc[(fcc.W_hr == 3.0) & (fcc.d0_km == 50.0)]
        check('binary Brier matches manual computation',
              abs(rc.brier - np.mean((g.p - g.o) ** 2)) < 1e-12)
        check('ensemble Brier beats binary product on this synthetic',
              r.brier < rc.brier)
        pt = paired_bootstrap(fc, fcc, [(3.0, 50.0)], n_boot=200)
        check('paired bootstrap runs and flags direction',
              len(pt) == 1 and pt.iloc[0].better_brier == 'A')
        out = os.path.join(td, 'rel.png')
        plot_cells(fc, [(3.0, 50.0), (3.0, 100.0)], 'selftest', out)
        check('reliability/ROC figure written', os.path.getsize(out) > 10000)
    print('selftest:', 'ALL PASSED' if ok else 'FAILURES ABOVE')
    return 0 if ok else 1


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opps', nargs='+',
                    help='opportunities.csv file(s); several = ensemble '
                         'members (globs ok)')
    ap.add_argument('--compare', default=None,
                    help='second product (single opportunities.csv or glob '
                         'of member files) for the paired comparison')
    ap.add_argument('--label', default='product')
    ap.add_argument('--forecast-from', choices=['dist', 'outcome'],
                    default='dist',
                    help='detection predicate: truth-independent spatial '
                         '(min_gf_dist_km <= d0; default) or outcome-based')
    ap.add_argument('--cells', default='3:50,3:100',
                    help='focus cells "W:d0,W:d0" for plots and bootstrap')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('-o', '--outdir', default='./probscores')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.opps:
        ap.error('provide --opps (or --selftest)')
    os.makedirs(args.outdir, exist_ok=True)

    paths = sorted(sum([glob.glob(p) for p in args.opps], []))
    if not paths:
        sys.exit('no files matched --opps')
    print(f'{len(paths)} member file(s) -> '
          f'{"member-fraction" if len(paths) > 1 else "binary"} forecast')
    fc = build_forecast(paths, args.forecast_from)
    sc = score_surface(fc)
    sc.to_csv(os.path.join(args.outdir, f'{args.label}_scores.csv'),
              index=False)
    print(sc.to_string(index=False, float_format=lambda x: f'{x:7.3f}'))

    cells = [tuple(float(x) for x in c.split(':'))
             for c in args.cells.split(',')]
    rel = pd.concat([reliability_table(
        fc[(fc.W_hr == W) & (fc.d0_km == d0)]).assign(W_hr=W, d0_km=d0)
        for (W, d0) in cells
        if len(fc[(fc.W_hr == W) & (fc.d0_km == d0)])], ignore_index=True)
    rel.to_csv(os.path.join(args.outdir, f'{args.label}_reliability.csv'),
               index=False)
    plot_cells(fc, cells, args.label,
               os.path.join(args.outdir, f'{args.label}_reliability.png'))

    if args.compare:
        cpaths = sorted(glob.glob(args.compare))
        fcb = build_forecast(cpaths, args.forecast_from)
        pt = paired_bootstrap(fc, fcb, cells, n_boot=args.n_boot)
        pt.to_csv(os.path.join(args.outdir, f'{args.label}_paired.csv'),
                  index=False)
        print('\npaired (A = --opps, B = --compare); negative d_brier = A '
              'better:')
        print(pt.to_string(index=False,
                           float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
