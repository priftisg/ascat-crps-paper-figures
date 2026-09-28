# Figure manifest

Asset IDs under `figures/main/` match the paper-revision pack numbering
(`01_`…`13_`). Regenerators often write **unprefixed** names; copy/rename into
`figures/main/` when packing (Figs 11–13 support `--pack-dir` directly).

| Asset | Script(s) | Key inputs | Default / unprefixed output |
|-------|-----------|------------|-----------------------------|
| `01_ceiling_vs_d0` | `code/validation/garg_followup_surfaces.py` | Buoy collocation + flavor score CSVs under research tree | `ceiling_vs_d0.png` |
| `02_ceiling_focus_table` | same | same | `ceiling_focus_table.png` |
| `03_field_crps_vs_orbit` | `code/validation/score_ensemble_field.py` | Checkpoint, val orbits, `lib/` | `field_crps_vs_orbit.png` |
| `04_spread_skill_histogram` | same | same | spread–skill histogram PNG |
| `05_spread_skill_vs_crps` | same | same | SSR vs CRPS scatter |
| `06_oos_bss_focus_table` | `code/validation/isotonic_oos_calibration.py` | `evaluation/garg_followup`, buoy opportunities CSV | `oos_bss_focus_table.png` |
| `07_pmax_reliability_W6_d075` | same | same | `reliability_oos_W6_d075.png` |
| `08_pmax_reliability_W3_d0100` | same | same | `reliability_oos_W3_d0100.png` |
| `09_rank_histogram` | `score_ensemble_field.py` | Checkpoint + val swaths | `rank_histogram.png` / `field_rank_histogram.png` |
| `10_tier_map_73598` | `code/validation/plot_crps_tier_comparison.py` | Orbit 73598 exceedance / catalogs; Cartopy | `73598_tier_comparison_segment1.png` (headline crop may be manual) |
| `11_cross_platform_monthly_timeseries` | `compute_crps_cross_platform_monthly.py` → `plot_crps_cross_platform.py` | Monthly GPKGs (`GPKG_ROOT`) | writes `11_*` via `--pack-dir` |
| `12_global_distribution` | `compute_crps_global_distribution.py` | `GPKG_ROOT`, `OVERPASS_CLIM` | writes `12_*` via `--pack-dir` |
| `13_explainability_main` | `explain_crps_representation.py` → `pack_explainability_figures.py` → `plot_explainability_panels_bd.py` (+ optional `replot_ig_geo_cartopy.py`) | Checkpoint, swaths, `SHARE_PACK`, IG/latent caches | writes `13_*` via `--pack-dir` |

## Supplementary (`figures/si/`)

| Asset | Notes |
|-------|-------|
| `SI_ensemble_size_sensitivity_3x2` | Assembled from M=4/8/16 field-calibration panels; summary in `metadata/comparison_summary.csv` |
| `SI_explain_*` | Built by `pack_explainability_figures.py` / Cartopy replot into SI dir |
| `ensemble_train_val_loss` | From training log `metadata/crps_ensemble_training.csv` (ad hoc plot; PNG/PDF shipped) |

## Small intermediates in `metadata/`

Useful when you want to inspect numbers without re-running GPU jobs:

- `field_calibration_summary.json`, `per_orbit_scores.csv`
- `11_ceiling_comparison.csv`, `isotonic_oos_summary.json`
- `17_cross_platform_*.csv`
- `density_match_operating_point.json`, `density_match_catalog_tau1p2e-4.json`
- `comparison_summary.csv` (member sensitivity)
- Explainability SI tables (`SI_tab_*.csv/.tex`)

## Operating point (catalog / grids)

τ = 1.2×10⁻⁴ s⁻¹, p* = 0.325 (density-matched). See metadata JSON files above.
