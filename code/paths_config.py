"""Resolve project roots and optional path overrides for figure scripts.

Prefer environment variables or configs/paths.json over hardcoded machine paths.

Environment / JSON keys
-----------------------
ASCAT_CRPS_ROOT   Full research checkout (contains lib/, validation/, models/, …)
GPKG_ROOT         Monthly GeoPackage root (…/fullswath/crps)
CHECKPOINT        Path to crps_ensemble_best.pt
OVERPASS_CLIM     Overpass climatology directory (Fig. 12)
ZENODO_CRPS_EXTRACT / ZENODO_GRID_EXTRACT
EXPLAIN_CACHE     evaluation/explainability/crps
SHARE_PACK        Garg explainability share pack
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]  # …/ascat_crps_paper_figures
_CODE_ROOT = Path(__file__).resolve().parent      # …/code


def _load_json_paths() -> Dict[str, Any]:
    candidates = [
        _REPO_ROOT / "configs" / "paths.json",
        Path.cwd() / "configs" / "paths.json",
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text())
    return {}


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in os.environ and os.environ[key].strip():
        return os.environ[key].strip()
    cfg = _load_json_paths()
    val = cfg.get(key)
    if val is None or str(val).strip() == "":
        return default
    return str(val)


def research_root(fallback_from: Optional[Path] = None) -> Path:
    """Return the full ASCAT CRPS research project root."""
    env = get("ASCAT_CRPS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # Walk upward from the calling script looking for a real project root.
    start = (fallback_from or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "config_ascat_a_crps.json").is_file() and (p / "lib").is_dir():
            return p
        if (p / "models" / "unet_crps").is_dir() and (p / "lib").is_dir():
            return p

    # Figures-repo layout: code/validation/script.py → parents[2] is repo root only
    # (not the research tree). Keep legacy parents[1] behavior when scripts are
    # vendored next to lib/ inside a full checkout.
    if fallback_from is not None:
        here = fallback_from.resolve()
        # code/validation → code (has lib/)
        if (here.parents[1] / "lib").is_dir() and (here.parents[1] / "lib" / "model.py").is_file():
            # Still may not be research root; prefer parents[2] if configs exist
            cand = here.parents[2]
            if (cand / "config_ascat_a_crps.json").is_file():
                return cand
            return here.parents[1]
        return here.parents[1]

    raise FileNotFoundError(
        "Cannot locate ASCAT_CRPS_ROOT. Set the environment variable or create "
        "configs/paths.json (see configs/paths.example.json)."
    )


def figures_repo_root() -> Path:
    return _REPO_ROOT


def ensure_sys_path(root: Path) -> None:
    import sys

    for p in (root, _CODE_ROOT, root / "lib", _CODE_ROOT / "lib"):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
