"""
On-the-fly cold-pool labels from horizontal wind-gradient magnitude.

Uses the Garg two-threshold procedure (6.8e-5 s^-1 then 1.2e-4 s^-1)
with NaN-aware gradients so QC holes / coasts do not manufacture fronts.
"""

from typing import Dict

import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label
from skimage.filters import sobel
from skimage.measure import regionprops


def wind_gradient_magnitude(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    res: float = 12500.0,
) -> np.ndarray:
    """
    Horizontal wind-gradient magnitude (s^-1).

    Invalid pixels are set to NaN so neighbors are poisoned rather than
    creating a fake wind-to-zero step at QC holes and coasts.
    """
    valid = valid.astype(bool)
    u = np.where(valid, u, np.nan).astype(np.float64)
    v = np.where(valid, v, np.nan).astype(np.float64)
    dqu_dy, dqu_dx = np.gradient(u, res, res)
    dqv_dy, dqv_dx = np.gradient(v, res, res)
    return np.sqrt((dqu_dx + dqv_dx) ** 2 + (dqu_dy + dqv_dy) ** 2).astype(np.float32)


def sobel_wind_gradient_magnitude(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    res: float = 12500.0,
) -> np.ndarray:
    """Sobel of WGM with a dilated-rim guard against fake edges."""
    valid = valid.astype(bool)
    gradient = wind_gradient_magnitude(u, v, valid, res=res)
    finite = np.isfinite(gradient)
    speed_asc = sobel(np.nan_to_num(gradient, nan=0.0))
    speed_asc[binary_dilation(~finite)] = 0.0
    speed_asc = np.where(valid, speed_asc, 0.0)
    return speed_asc.astype(np.float32)


def cold_pool_mask_from_gradient(
    u: np.ndarray,
    v: np.ndarray,
    valid_mask: np.ndarray,
    thresh1: float = 6.8e-5,
    thresh2: float = 1.2e-4,
    min_pixels: int = 4,
    res: float = 12500.0,
) -> Dict[str, np.ndarray]:
    """Binary cold-pool label mask and Sobel-WGM for a swath half."""
    valid = valid_mask > 0
    speed_asc = sobel_wind_gradient_magnitude(u, v, valid, res=res)
    gradient = wind_gradient_magnitude(u, v, valid, res=res)
    gradient = np.nan_to_num(gradient, nan=0.0).astype(np.float32)
    gradient = np.where(valid, gradient, 0.0).astype(np.float32)

    gf = np.zeros(speed_asc.shape, dtype=np.float32)
    gf[speed_asc > thresh1] = 1.0

    gf_id, _ = nd_label(gf)
    for region_id in range(1, int(np.max(gf_id)) + 1):
        ingf = gf_id == region_id
        if np.max(speed_asc[ingf]) <= thresh2:
            gf[ingf] = 0.0

    gf_id1, _ = nd_label(gf)
    label = np.zeros_like(gf, dtype=np.float32)
    regions = regionprops(gf_id1, speed_asc)
    for region in regions:
        if region.area >= min_pixels:
            label[gf_id1 == region.label] = 1.0

    label *= valid.astype(np.float32)
    return {
        "label": label,
        "gradient": gradient,
        "sobel_wgm": speed_asc.astype(np.float32),
    }


def labels_for_full_swath(
    swath: Dict[str, np.ndarray],
    thresh1: float,
    thresh2: float,
    min_pixels: int,
    res: float,
) -> Dict[str, np.ndarray]:
    """Build labels on full swath by processing left/right halves separately."""
    valid = swath.get("valid_mask")
    if valid is None:
        raise KeyError("swath must contain 'valid_mask' before labeling")

    n_along, n_cross = swath["u"].shape
    label = np.zeros((n_along, n_cross), dtype=np.float32)
    gradient = np.zeros((n_along, n_cross), dtype=np.float32)
    sobel_wgm = np.zeros((n_along, n_cross), dtype=np.float32)

    mid = n_cross // 2
    halves = [(slice(None), slice(None, mid)), (slice(None), slice(mid, None))]

    for along_sl, cross_sl in halves:
        result = cold_pool_mask_from_gradient(
            swath["u"][along_sl, cross_sl],
            swath["v"][along_sl, cross_sl],
            valid[along_sl, cross_sl],
            thresh1=thresh1,
            thresh2=thresh2,
            min_pixels=min_pixels,
            res=res,
        )
        label[along_sl, cross_sl] = result["label"]
        gradient[along_sl, cross_sl] = result["gradient"]
        sobel_wgm[along_sl, cross_sl] = result["sobel_wgm"]

    return {"label": label, "gradient": gradient, "sobel_wgm": sobel_wgm}
