"""Split ASCAT swaths into left/right 41-column halves and stitch outputs."""

from typing import Dict, Tuple

import numpy as np

FULL_CROSS_TRACK = 82
HALF_CROSS_TRACK = 41
HALF_NAMES = ("left", "right")


def half_index(half: str) -> int:
    if half == "left":
        return 0
    if half == "right":
        return 1
    raise ValueError(f"half must be 'left' or 'right', got {half!r}")


def half_name(index: int) -> str:
    if index == 0:
        return "left"
    if index == 1:
        return "right"
    raise ValueError(f"half index must be 0 or 1, got {index}")


def cross_track_slice(half: str) -> slice:
    """Column slice for left (0:41) or right (41:82) on the native grid."""
    if half == "left":
        return slice(0, HALF_CROSS_TRACK)
    if half == "right":
        return slice(HALF_CROSS_TRACK, FULL_CROSS_TRACK)
    raise ValueError(f"half must be 'left' or 'right', got {half!r}")


def slice_swath(swath: Dict[str, np.ndarray], half: str) -> Dict[str, np.ndarray]:
    """Extract one along-track × 41 cross-track half from a trimmed swath dict."""
    sl = cross_track_slice(half)
    out = {}
    for key, arr in swath.items():
        if key in ("orbit", "path"):
            out[key] = arr
        elif key == "atime":
            out[key] = arr
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            out[key] = arr[:, sl]
        elif isinstance(arr, np.ndarray) and arr.ndim == 3:
            out[key] = arr[:, :, sl]
        else:
            out[key] = arr
    return out


def stitch_halves(
    left: np.ndarray,
    right: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Concatenate left and right half arrays along cross-track (axis=1)."""
    return np.concatenate([left, right], axis=axis)


def stitch_swath_fields(
    left: Dict[str, np.ndarray],
    right: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Merge two half swath dicts back into a full 82-column swath dict."""
    out = {}
    keys = set(left) | set(right)
    for key in keys:
        lv, rv = left.get(key), right.get(key)
        if key in ("orbit", "path"):
            out[key] = lv if lv is not None else rv
        elif key == "atime":
            out[key] = lv if lv is not None else rv
        elif isinstance(lv, np.ndarray) and lv.ndim == 2:
            out[key] = stitch_halves(lv, rv, axis=1)
        else:
            out[key] = lv if lv is not None else rv
    return out
