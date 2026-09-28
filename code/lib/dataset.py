"""Shared swath load / QC helpers for the CRPS pipeline."""

import json
import os
from typing import Dict

import numpy as np

from .remss_qc import label_pixel_mask, training_valid_mask, tropical_lat_mask
from .swath_io import load_swath_mat, trim_swath_edges


def prepare_swath(mat_path: str, config: Dict) -> Dict[str, np.ndarray]:
    """
    Load, edge-trim, and attach QC masks.

    - ``label_mask``: v1-equivalent QC used to *build* Sobel-WGM / GF labels
    - ``valid_mask``: strict REMSS QC (± tropical band) used as loss validity
    """
    swath = load_swath_mat(mat_path)
    swath = trim_swath_edges(swath, config.get("edge_trim", 10))

    label_mask = label_pixel_mask(
        swath,
        min_iclass=int(config.get("label_min_iclass", 1)),
        mask_scat_rain=bool(config.get("label_mask_scat_rain", True)),
    )
    if config.get("apply_train_lat_mask", False):
        lat_min = config.get("train_lat_min", config.get("garg_lat_min", -30.0))
        lat_max = config.get("train_lat_max", config.get("garg_lat_max", 31.0))
        label_mask = (label_mask * tropical_lat_mask(swath["lat"], lat_min, lat_max)).astype(
            np.float32
        )
    swath["label_mask"] = label_mask
    swath["valid_mask"] = training_valid_mask(swath, config)
    return swath


def load_config(path: str = None) -> Dict:
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config_ascat_a_crps.json",
        )
    with open(path) as f:
        return json.load(f)
