# Data needed to regenerate figures

This repository ships **rendered figures** and **code**. Regeneration needs
local access to the research data tree and/or the Zenodo extracts.

## Zenodo dataset (recommended public inputs)

After publication, set:

- `ZENODO_CRPS_EXTRACT` → extracted `ascat_crps_v1/` (GeoPackages + swath indices)
- `ZENODO_GRID_EXTRACT` → extracted `gridded_cold_pools_compressed/` (monthly NetCDF)

DOI: *to be filled after Zenodo publish*.

For scripts that still expect the live layout
`evaluation/fullswath/crps/{ascat_a,b,c}/gpkg/`, either:

- point `GPKG_ROOT` at a directory that mirrors that layout, or
- symlink Zenodo `ASCAT-A/gpkg` → `ascat_a/gpkg` (and likewise for B/C).

## Research checkout (`ASCAT_CRPS_ROOT`)

Set to your local ASCAT CRPS research checkout root if regenerating GPU figures:

| Path under root | Used by |
|-----------------|---------|
| `models/unet_crps/crps_ensemble_best.pt` | Field CRPS, explainability |
| `config_ascat_a_crps.json` | Dataset / channel config |
| `evaluation/fullswath/crps/*/gpkg/` | Figs 11–12 (or Zenodo) |
| `evaluation/garg_followup/` | Figs 01–02, 06–08 |
| `evaluation/buoy_collocation_v2/` | Buoy verification inputs |
| `evaluation/explainability/crps/` | Fig 13 caches (`EXPLAIN_CACHE`) |
| `evaluation/Garg_explainability/share_pack_v1/` | Permutation / coherence CSVs (`SHARE_PACK`) |
| Processed ASCAT swath `.mat` trees | Forward passes (paths inside config) |

## Overpass climatology (Fig. 12)

Directory of REMSS-based overpass grids used as the detection-frequency
denominator. Set `OVERPASS_CLIM` to that directory on your machine.

## What is intentionally excluded from git

- `*.gpkg`, `*.nc`, `*.pt`, `*.mat`, exceedance NPZ stacks  
- Raw ASCAT `Processed_Swaths`  
- Large IG/latent caches (regenerate with `pack_explainability_figures.py`)

See `.gitignore`.

## Minimal regeneration checks (smoke)

```bash
export ASCAT_CRPS_ROOT=...
export PYTHONPATH=$PWD/code:$PYTHONPATH

# Field metrics on a few orbits
python code/validation/score_ensemble_field.py --limit 2 \
  --outdir /tmp/crps_field_smoke

# Global distribution (needs GPKG + overpass)
python code/validation/compute_crps_global_distribution.py \
  --pack-dir figures/main
```

Full Fig 13 regeneration requires GPU time and the share-pack CSVs; prefer the
shipped `figures/main/13_*` unless you are actively revising explainability panels.
