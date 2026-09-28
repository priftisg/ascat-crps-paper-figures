"""
REMSS ASCAT quality control and variable decoding.

Reference: REMSS wind vector .mat file documentation (wind_smooth, dir_smooth,
iclass, sos_all, selamb, irain_scat, rad_rain).
"""

from typing import Dict, Tuple

import numpy as np

RAD_RAIN_MISSING = -999.0
RAD_RAIN_NONE = 0.0
RAD_RAIN_POSSIBLE = 0.1
RAD_RAIN_DEFINITE_MIN = 0.2

DEFAULT_MIN_ICLASS = 2
DEFAULT_MAX_SOS = 1.9
DEFAULT_RAIN_WIND_MS = 5.0
DEFAULT_TRAIN_LAT_MIN = -30.0
DEFAULT_TRAIN_LAT_MAX = 30.0


def oceanographic_to_meteorological_deg(dir_ocean: np.ndarray) -> np.ndarray:
    """REMSS dir_smooth: oceanographic → meteorological wind direction."""
    return (dir_ocean + 180.0) % 360.0


def u_v_from_dir_ocean(
    speed: np.ndarray,
    dir_ocean_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    East/north components from oceanographic direction (blowing TOWARD).

        u = speed * sin(dir_ocean)
        v = speed * cos(dir_ocean)
    """
    rad = np.deg2rad(dir_ocean_deg)
    u = speed * np.sin(rad)
    v = speed * np.cos(rad)
    return u.astype(np.float32), v.astype(np.float32)


def u_v_from_wind_smooth(
    wind_smooth: np.ndarray,
    dir_smooth_ocean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive u/v from REMSS wind/dir using the oceanographic convention."""
    return u_v_from_dir_ocean(wind_smooth, dir_smooth_ocean)


def label_pixel_mask(
    swath: Dict[str, np.ndarray],
    min_iclass: int = 1,
    mask_scat_rain: bool = True,
) -> np.ndarray:
    """
    v1-equivalent QC for GF label generation (paper baseline).

    Strict REMSS QC (iclass/sos) remains only as the loss-validity channel.
    """
    wind = swath["wind_speed"]
    mask = (
        np.isfinite(wind)
        & (wind > 0.0)
        & (swath["iclass"] >= min_iclass)
        & np.isfinite(swath["lat"])
        & np.isfinite(swath["lon"])
    )
    if mask_scat_rain:
        mask &= ~(swath["irain_scat"] == 1)
    return mask.astype(np.float32)


def extract_selected_sos(sos_all: np.ndarray, selamb: np.ndarray) -> np.ndarray:
    """Extract SOS for the selected ambiguity at each wind cell."""
    sos_selected = np.full(selamb.shape, np.nan, dtype=np.float32)
    for amb in range(1, 5):
        mask = selamb == amb
        if mask.any():
            sos_selected[mask] = sos_all[amb - 1][mask]
    return sos_selected


def decode_rad_rain(rad_rain: np.ndarray) -> Dict[str, np.ndarray]:
    """Decode REMSS radiometer columnar rain rate (km·mm/hr)."""
    return {
        "missing": rad_rain <= RAD_RAIN_MISSING + 1.0,
        "none": rad_rain == RAD_RAIN_NONE,
        "possible": rad_rain == RAD_RAIN_POSSIBLE,
        "definite": rad_rain >= RAD_RAIN_DEFINITE_MIN,
        "available": rad_rain > RAD_RAIN_MISSING + 1.0,
    }


def rain_contamination_flag(
    irain_scat: np.ndarray,
    wind_speed: np.ndarray,
    rad_rain: np.ndarray,
    rain_wind_threshold_ms: float = DEFAULT_RAIN_WIND_MS,
) -> np.ndarray:
    """
    Combined rain flag: 0 = none, 0.5 = possible, 1 = definite or scat rain < 5 m/s.
    """
    rad = decode_rad_rain(rad_rain)
    flag = np.zeros(wind_speed.shape, dtype=np.float32)
    flag[rad["possible"]] = 0.5
    scat_rain = (irain_scat >= 1) & (wind_speed < rain_wind_threshold_ms)
    flag[rad["definite"] | scat_rain] = 1.0
    return flag


def rad_rain_for_model(rad_rain: np.ndarray) -> np.ndarray:
    """Map REMSS rad_rain to non-negative value; missing → 0."""
    out = rad_rain.copy().astype(np.float32)
    out[out <= RAD_RAIN_MISSING + 1.0] = 0.0
    return np.clip(out, 0.0, None)


def rad_rain_rates_over_mask(
    rad_rain: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Mean and max REMSS radiometer rain rate over ``mask`` pixels.

    ``rad_rain`` is columnar rain rate (km·mm/hr). Missing (-999) is excluded.
    The REMSS "possible" code (0.1) is treated as 0 (not a measured rate).
    Returns (0.0, 0.0) if no non-missing pixels.
    """
    vals = np.asarray(rad_rain, dtype=np.float64)[mask]
    available = vals > RAD_RAIN_MISSING + 1.0
    if not np.any(available):
        return 0.0, 0.0
    rates = vals[available].copy()
    rates[rates == RAD_RAIN_POSSIBLE] = 0.0
    rates = np.clip(rates, 0.0, None)
    return float(np.mean(rates)), float(np.max(rates))


def valid_pixel_mask(
    swath: Dict[str, np.ndarray],
    min_iclass: int = DEFAULT_MIN_ICLASS,
    max_sos: float = DEFAULT_MAX_SOS,
) -> np.ndarray:
    """REMSS-recommended validity mask for training / loss."""
    wind = swath["wind_speed"]
    iclass = swath["iclass"]
    sos = swath["sos"]
    lat = swath["lat"]
    lon = swath["lon"]

    mask = (
        np.isfinite(wind)
        & (wind > 0.0)
        & np.isfinite(iclass)
        & (iclass >= min_iclass)
        & np.isfinite(sos)
        & (sos <= max_sos)
        & np.isfinite(lat)
        & np.isfinite(lon)
    )
    return mask.astype(np.float32)


def tropical_lat_mask(
    lat: np.ndarray,
    lat_min: float = DEFAULT_TRAIN_LAT_MIN,
    lat_max: float = DEFAULT_TRAIN_LAT_MAX,
) -> np.ndarray:
    """Pixel mask for Garg-valid tropical band (inclusive lat bounds)."""
    return (
        np.isfinite(lat)
        & (lat >= lat_min)
        & (lat <= lat_max)
    ).astype(np.float32)


def training_valid_mask(
    swath: Dict[str, np.ndarray],
    config: Dict,
) -> np.ndarray:
    """
    REMSS QC mask, optionally restricted to the tropical training band.

    Loss and gradient labels use this mask; full swath is still fed as input.
    """
    valid = valid_pixel_mask(
        swath,
        min_iclass=config.get("min_iclass", DEFAULT_MIN_ICLASS),
        max_sos=config.get("max_sos", DEFAULT_MAX_SOS),
    )
    if not config.get("apply_train_lat_mask", False):
        return valid

    lat_min = config.get(
        "train_lat_min",
        config.get("garg_lat_min", DEFAULT_TRAIN_LAT_MIN),
    )
    lat_max = config.get(
        "train_lat_max",
        config.get("garg_lat_max", DEFAULT_TRAIN_LAT_MAX),
    )
    tropical = tropical_lat_mask(swath["lat"], lat_min, lat_max)
    return (valid * tropical).astype(np.float32)


def iclass_for_model(iclass: np.ndarray) -> np.ndarray:
    """Normalize iclass (0–3) to [0, 1]."""
    return np.clip(iclass.astype(np.float32) / 3.0, 0.0, 1.0)
