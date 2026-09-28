"""Assemble multi-channel model input from a loaded swath dict."""

from typing import Dict, List, Optional

import numpy as np

from .gradient_labels import sobel_wind_gradient_magnitude
from .remss_qc import (
    iclass_for_model,
    rain_contamination_flag,
    rad_rain_for_model,
    valid_pixel_mask,
)
from .swath_io import MAX_AMBIGUITIES

NORM = {
    "u": 30.0,
    "v": 30.0,
    "wind_speed": 70.0,
    "wind_speed_smooth": 70.0,
    "wind_gradient": 5.0e-4,
    "wind_gradient_smooth": 5.0e-4,
    "rad_rain": 25.4,
    "rain_flag": 1.0,
    "iclass": 1.0,
    "sos": 2.0,
    "valid_mask": 1.0,
    "numamb_norm": 1.0,
}
for _amb in range(1, MAX_AMBIGUITIES + 1):
    NORM[f"u_amb{_amb}"] = 30.0
    NORM[f"v_amb{_amb}"] = 30.0
    NORM[f"sos_amb{_amb}"] = 2.0


def _ambiguity_channel_sources(swath: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-ambiguity u, v, sos planes from u_all / v_all / sos_all."""
    max_sos = 2.0
    numamb = swath.get("numamb")
    sources = {}
    for amb in range(1, MAX_AMBIGUITIES + 1):
        idx = amb - 1
        u = swath["u_all"][idx].astype(np.float32)
        v = swath["v_all"][idx].astype(np.float32)
        sos = np.nan_to_num(swath["sos_all"][idx], nan=max_sos + 1.0).astype(np.float32)
        if numamb is not None:
            inactive = numamb < amb
            u = np.where(inactive, 0.0, u)
            v = np.where(inactive, 0.0, v)
            sos = np.where(inactive, max_sos + 1.0, sos)
        sources[f"u_amb{amb}"] = u
        sources[f"v_amb{amb}"] = v
        sources[f"sos_amb{amb}"] = sos
    if numamb is not None:
        sources["numamb_norm"] = np.clip(
            numamb.astype(np.float32) / MAX_AMBIGUITIES, 0.0, 1.0,
        )
    else:
        sources["numamb_norm"] = np.zeros_like(swath["wind_speed"], dtype=np.float32)
    return sources


def _smooth_gradient(swath: Dict[str, np.ndarray], res: float) -> np.ndarray:
    """Sobel-WGM from REMSS wind_smooth (context only; not the CRPS target)."""
    if "u_smooth" in swath and "v_smooth" in swath:
        u, v = swath["u_smooth"], swath["v_smooth"]
    else:
        u, v = swath["u"], swath["v"]
    # Use a loose finite mask so smooth context is available near QC holes.
    wind = swath.get("wind_speed_smooth", swath["wind_speed"])
    valid = np.isfinite(wind) & (wind > 0)
    return sobel_wind_gradient_magnitude(u, v, valid, res=res)


def build_feature_stack(
    swath: Dict[str, np.ndarray],
    gradient: Optional[np.ndarray],
    channel_names: List[str],
    rain_wind_threshold_ms: float = 5.0,
    min_iclass: int = 2,
    max_sos: float = 1.9,
    fill_invalid: float = 0.0,
    fov_m: float = 12500.0,
) -> np.ndarray:
    """Stack channels into (H, W, C) float32 array."""
    if "valid_mask" not in swath:
        swath["valid_mask"] = valid_pixel_mask(
            swath, min_iclass=min_iclass, max_sos=max_sos,
        )

    valid = swath["valid_mask"]
    wind_for_rain = swath.get("wind_speed_smooth", swath["wind_speed"])
    rain_flag = rain_contamination_flag(
        swath["irain_scat"],
        wind_for_rain,
        swath["rad_rain"],
        rain_wind_threshold_ms=rain_wind_threshold_ms,
    )

    sources = {
        "u": swath["u"],
        "v": swath["v"],
        "wind_speed": swath["wind_speed"],
        "wind_speed_smooth": swath.get("wind_speed_smooth", swath["wind_speed"]),
        "wind_gradient": (
            np.nan_to_num(gradient, nan=0.0)
            if gradient is not None
            else np.zeros_like(swath["wind_speed"], dtype=np.float32)
        ),
        "wind_gradient_smooth": _smooth_gradient(swath, fov_m),
        "rad_rain": rad_rain_for_model(swath["rad_rain"]),
        "rain_flag": rain_flag,
        "iclass": iclass_for_model(swath["iclass"]),
        "sos": np.nan_to_num(swath["sos"], nan=max_sos + 1.0),
        "valid_mask": valid,
    }
    if "u_all" in swath:
        sources.update(_ambiguity_channel_sources(swath))

    planes = []
    for name in channel_names:
        if name not in sources:
            raise KeyError(f"Unknown feature channel '{name}'")
        arr = sources[name].astype(np.float32)
        scale = NORM.get(name, 1.0)
        normed = arr / scale
        normed = np.where(valid > 0, normed, fill_invalid)
        planes.append(normed)

    return np.stack(planes, axis=-1)
