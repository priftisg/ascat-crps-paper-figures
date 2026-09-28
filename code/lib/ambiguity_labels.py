"""Garg GF cold-pool candidate labels and Sobel-WGM regression targets."""

from typing import Dict

import numpy as np

from .gradient_labels import cold_pool_mask_from_gradient


def _gf_kwargs(config: Dict) -> Dict:
    return {
        "thresh1": config["gradient_thresh1"],
        "thresh2": config["gradient_thresh2"],
        "min_pixels": config["min_cold_pool_pixels"],
        "res": config["fov_m"],
    }


def build_cold_pool_labels(
    half_swath: Dict[str, np.ndarray],
    config: Dict,
) -> Dict[str, np.ndarray]:
    """
    Labels / Sobel-WGM from selected-ambiguity wind under the label QC mask.

    The loss validity channel is attached separately from ``valid_mask``.
    """
    label_mask = half_swath.get("label_mask", half_swath["valid_mask"])
    return cold_pool_mask_from_gradient(
        half_swath["u"],
        half_swath["v"],
        label_mask,
        **_gf_kwargs(config),
    )


def build_sobel_target(
    half_swath: Dict[str, np.ndarray],
    config: Dict,
) -> Dict[str, np.ndarray]:
    """Normalized Sobel-WGM target for CRPS training, clipped to [0, 4]."""
    labels = build_cold_pool_labels(half_swath, config)
    scale = float(config.get("gradient_scale", 5.0e-4))
    clip_max = float(config.get("sobel_target_clip", 4.0))
    sobel = np.clip(labels["sobel_wgm"] / scale, 0.0, clip_max).astype(np.float32)
    return {
        "sobel": sobel,
        "label": labels["label"],
        "gradient": labels["gradient"],
        "sobel_wgm": labels["sobel_wgm"],
    }
