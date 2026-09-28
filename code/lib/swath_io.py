"""Load REMSS ASCAT .mat swaths and expose native geometry."""

import os
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio

from .remss_qc import extract_selected_sos, u_v_from_dir_ocean

MAX_AMBIGUITIES = 4


def list_swath_files(
    processed_dir: str,
    series_prefix: str,
    orbit_start: Optional[int] = None,
    orbit_end: Optional[int] = None,
) -> List[str]:
    """Return sorted list of .mat paths, optionally filtered by orbit number."""
    pattern = os.path.join(processed_dir, f"{series_prefix}_winvec_*.mat")
    files = sorted(glob(pattern))
    if orbit_start is None and orbit_end is None:
        return files

    selected = []
    for path in files:
        orbit = int(os.path.basename(path).split("_")[2])
        if orbit_start is not None and orbit < orbit_start:
            continue
        if orbit_end is not None and orbit > orbit_end:
            continue
        selected.append(path)
    return selected


def _parse_atimes(atime: np.ndarray) -> np.ndarray:
    """Parse REMSS atime character array into datetime objects (or None)."""
    rows = atime.T[:, np.array((0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13))]
    rows = rows.copy()
    rows[rows == " "] = "0"

    dtimes = []
    for row in rows:
        if "?" in row:
            dtimes.append(None)
        else:
            s = (
                f"{''.join(row[:4])}-{''.join(row[4:7])}T"
                f"{''.join(row[7:9])}:{''.join(row[9:11])}"
            )
            dtimes.append(datetime.strptime(s, "%Y-%jT%H:%M"))
    return np.array(dtimes, dtype=object)


def _transpose_along_cross(arr: np.ndarray) -> np.ndarray:
    """REMSS native layout (cross, along) → (along, cross)."""
    if arr.ndim == 2:
        return arr.T
    if arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1))
    return arr


def _u_v_all_from_wind_dir(
    wind_all: np.ndarray,
    dir_all: np.ndarray,
) -> tuple:
    """Oceanographic u/v for each ambiguity: shape (4, along, cross)."""
    u_all = np.empty_like(wind_all, dtype=np.float32)
    v_all = np.empty_like(wind_all, dtype=np.float32)
    for amb in range(wind_all.shape[0]):
        u, v = u_v_from_dir_ocean(wind_all[amb], dir_all[amb])
        u_all[amb] = u
        v_all[amb] = v
    return u_all, v_all


def _selected_wind_dir(
    wind_all: np.ndarray,
    dir_all: np.ndarray,
    selamb: np.ndarray,
) -> tuple:
    """Selected-ambiguity wind/dir; NaN where selamb < 1."""
    idx = np.clip(selamb.astype(np.int64) - 1, 0, MAX_AMBIGUITIES - 1)[None]
    wind_sel = np.take_along_axis(wind_all, idx, axis=0)[0]
    dir_sel = np.take_along_axis(dir_all, idx, axis=0)[0]
    good = selamb >= 1
    wind_sel = np.where(good, wind_sel, np.nan)
    dir_sel = np.where(good, dir_sel, np.nan)
    return wind_sel.astype(np.float32), dir_sel.astype(np.float32)


def load_swath_mat(path: str) -> Dict[str, np.ndarray]:
    """
    Load one ASCAT swath and return arrays in (along_track, cross_track) layout.

    Primary u/v are from the SELECTED ambiguity (Garg catalog convention).
    wind_smooth / dir_smooth are retained as context channels.
    """
    ds = sio.loadmat(path, chars_as_strings=False)

    wind_smooth = ds["wind_smooth"].astype(np.float32)
    dir_smooth = ds["dir_smooth"].astype(np.float32)

    sos_all = ds["sos_all"].astype(np.float32)
    wind_all = ds["wind_all"].astype(np.float32)
    dir_all = ds["dir_all"].astype(np.float32)
    selamb = ds["selamb"].astype(np.int64)
    sos_selected = extract_selected_sos(sos_all, selamb.astype(np.int16))
    u_all, v_all = _u_v_all_from_wind_dir(wind_all, dir_all)

    wind_sel, dir_sel = _selected_wind_dir(wind_all, dir_all, selamb)
    u, v = u_v_from_dir_ocean(wind_sel, dir_sel)
    u_smooth, v_smooth = u_v_from_dir_ocean(wind_smooth, dir_smooth)

    rad_rain = ds["rad_rain"].astype(np.float32)
    irain_scat = ds["irain_scat"].astype(np.float32)
    iclass = ds["iclass"].astype(np.float32)
    numamb = ds["numamb"].astype(np.float32)
    lat = ds["lat"].astype(np.float32)
    lon = ds["lon"].astype(np.float32)

    orbit = os.path.basename(path).split("_")[2]

    return {
        "u": _transpose_along_cross(u),
        "v": _transpose_along_cross(v),
        "u_smooth": _transpose_along_cross(u_smooth),
        "v_smooth": _transpose_along_cross(v_smooth),
        "wind_speed": _transpose_along_cross(wind_sel),
        "wind_speed_smooth": _transpose_along_cross(wind_smooth),
        "dir_ocean": _transpose_along_cross(dir_sel),
        "dir_smooth": _transpose_along_cross(dir_smooth),
        "rad_rain": _transpose_along_cross(rad_rain),
        "irain_scat": _transpose_along_cross(irain_scat),
        "iclass": _transpose_along_cross(iclass),
        "numamb": _transpose_along_cross(numamb),
        "selamb": _transpose_along_cross(selamb.astype(np.float32)),
        "sos": _transpose_along_cross(sos_selected),
        "u_all": _transpose_along_cross(u_all),
        "v_all": _transpose_along_cross(v_all),
        "sos_all": _transpose_along_cross(sos_all),
        "lat": _transpose_along_cross(lat),
        "lon": _transpose_along_cross(lon),
        "atime": _parse_atimes(ds["atime"]),
        "orbit": orbit,
        "path": path,
    }


def trim_swath_edges(swath: Dict[str, np.ndarray], edge_trim: int) -> Dict[str, np.ndarray]:
    """Trim along-track edges to match the legacy 82×40 windowing footprint."""
    if edge_trim <= 0:
        return swath

    sl = slice(edge_trim, -edge_trim)
    out = {}
    for key, arr in swath.items():
        if key in ("atime", "orbit", "path"):
            out[key] = arr[sl] if key == "atime" else arr
        elif isinstance(arr, np.ndarray) and arr.ndim == 2:
            out[key] = arr[sl, :]
        elif isinstance(arr, np.ndarray) and arr.ndim == 3:
            out[key] = arr[:, sl, :]
        else:
            out[key] = arr
    return out
