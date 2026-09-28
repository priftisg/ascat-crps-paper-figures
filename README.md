# ASCAT CRPS paper figures

Rendered manuscript figures and Python scripts to regenerate them for the
**ASCAT CRPS ensemble U-Net** cold-pool identification study.

Large input data (GeoPackages, NetCDF grids, ASCAT swaths, model checkpoints)
are **not** stored here — see the Zenodo dataset and [docs/DATA_NEEDED.md](docs/DATA_NEEDED.md).

| Resource | Role |
|----------|------|
| This repository | Figures + regeneration code |
| Zenodo dataset | Monthly GPKGs + compressed grids |
| Training / inference archive | Separate software deposit (`CODE_ARCHIVE_crps_ascat_coldpools`) |

## Repository layout

```text
figures/main/     Manuscript assets 01_…13_ (PNG/PDF + captions where present)
figures/si/       Supplementary figures (ensemble-size, explainability SI, training curve)
code/validation/  Figure-making scripts
code/lib/         Minimal model/data helpers imported by those scripts
code/paths_config.py
metadata/         Small CSV/JSON intermediates used for tables / replot
configs/          Path and model config templates
docs/DATA_NEEDED.md
FIGURE_MANIFEST.md
```

## Quick start

1. Clone this repo.
2. Obtain the research data tree and/or Zenodo extracts (see `docs/DATA_NEEDED.md`).
3. Copy `configs/paths.example.json` → `configs/paths.json` and edit, **or** export:

```bash
export ASCAT_CRPS_ROOT=/path/to/full_swath_unet_CRPS
export GPKG_ROOT=$ASCAT_CRPS_ROOT/evaluation/fullswath/crps
export CHECKPOINT=$ASCAT_CRPS_ROOT/models/unet_crps/crps_ensemble_best.pt
export OVERPASS_CLIM=/path/to/occurrence_climatology
export PYTHONPATH=/path/to/ascat-crps-paper-figures/code:$PYTHONPATH
```

4. Install dependencies (see env matrix below).
5. Run a generator, e.g. Fig. 12:

```bash
python code/validation/compute_crps_global_distribution.py \
  --gpkg-root "$GPKG_ROOT" \
  --overpass-dir "$OVERPASS_CLIM" \
  --pack-dir figures/main
```

Scripts that write **unprefixed** filenames (Figs 01–10) can be renamed into
`figures/main/` using the mapping in [FIGURE_MANIFEST.md](FIGURE_MANIFEST.md)
or the stub `scripts_pack_to_figures.sh`.

## Environment matrix

| Figures | Environment | Notes |
|---------|-------------|-------|
| 03–05, 09, 13 (GPU steps) | PyTorch CUDA venv | Forward passes / IG / latent PCA |
| 10, 12 (maps) | Conda env with **Cartopy** | Often `impact` on our cluster |
| 11, GPKG I/O | GeoPandas / py312 | SQLite GeoPackage reads |
| 01–02, 06–08 | CPU scipy/sklearn + matplotlib | Buoy CSV intermediates |

CPU requirements: `requirements-cpu.txt`  
GPU stack: `requirements-gpu.txt` (install a matching `torch` CUDA build first).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the paper and the Zenodo dataset DOI
when using figures or regenerated products.

## License

- **Code** (`code/`): MIT — see [LICENSE](LICENSE)
- **Figures** (`figures/`): CC BY 4.0
