"""Object-based postprocessing on full-swath probability fields."""

from typing import Dict, List, Tuple

import numpy as np
from skimage.measure import label as sk_label, regionprops

from .swath_halves import FULL_CROSS_TRACK, cross_track_slice


def _label_binary_region(
    binary: np.ndarray,
    prob: np.ndarray,
    min_area: int,
    label_offset: int = 0,
) -> Tuple[np.ndarray, List[Dict], int]:
    """Connected-component labeling on one 2-D binary field."""
    labeled, _ = sk_label(binary.astype(np.uint8), return_num=True)

    kept: List[Dict] = []
    for region in regionprops(labeled, prob):
        if region.area < min_area:
            labeled[labeled == region.label] = 0
            continue
        kept.append({
            "label": int(region.label) + label_offset,
            "area": region.area,
            "centroid": region.centroid,
            "mean_prob": float(region.mean_intensity),
            "max_prob": float(region.max_intensity),
        })

    labeled, n_obj = sk_label(labeled > 0, return_num=True)
    if n_obj > 0 and label_offset > 0:
        labeled = np.where(labeled > 0, labeled + label_offset, 0).astype(np.int32)
        for obj in kept:
            obj["label"] += label_offset

    return labeled.astype(np.int32), kept, int(n_obj)


def _extract_dual_swath_objects(
    binary: np.ndarray,
    prob: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, List[Dict], int]:
    """
    Label left and right 41-column halves independently, then stitch labels.

    Prevents false merges across the ~750 km nadir gap (array cols 40/41).
    """
    labeled_full = np.zeros(binary.shape, dtype=np.int32)
    all_objects: List[Dict] = []
    label_offset = 0
    n_total = 0

    for half in ("left", "right"):
        sl = cross_track_slice(half)
        lab_half, objs_half, n_half = _label_binary_region(
            binary[:, sl], prob[:, sl], min_area, label_offset=label_offset,
        )
        labeled_full[:, sl] = lab_half
        all_objects.extend(objs_half)
        n_total += n_half
        if n_half > 0:
            label_offset = int(labeled_full.max())

    return labeled_full, all_objects, n_total


def extract_cold_pool_objects(
    prob: np.ndarray,
    valid_mask: np.ndarray,
    prob_threshold: float = 0.5,
    min_area: int = 3,
) -> Dict:
    """
    Connected-component labeling on the swath probability field.

    For native dual-swath grids (82 columns), left and right 41-column halves
    are labeled separately so objects cannot bridge the nadir gap in array space.
    """
    binary = ((prob >= prob_threshold) & (valid_mask > 0)).astype(np.uint8)

    if prob.ndim == 2 and prob.shape[1] == FULL_CROSS_TRACK:
        labeled, objects, n_objects = _extract_dual_swath_objects(
            binary, prob, min_area,
        )
    else:
        labeled, objects, n_objects = _label_binary_region(binary, prob, min_area)

    return {
        "binary": binary,
        "labeled": labeled,
        "objects": objects,
        "n_objects": n_objects,
    }
