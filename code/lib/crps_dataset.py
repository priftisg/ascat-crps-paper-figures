"""CRPS training dataset: ambiguity-only inputs + Sobel-WGM target + crops."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ambiguity_labels import build_sobel_target
from .dataset import prepare_swath
from .features import build_feature_stack
from .swath_halves import HALF_NAMES, slice_swath
from .swath_io import list_swath_files


SELECTED_LEAKAGE = {"u", "v", "sos", "wind_speed", "wind_gradient", "selamb"}


def orbit_from_path(path: str) -> str:
    return Path(path).name.split("_")[2]


def split_files(config: Dict, split: str) -> List[str]:
    if split == "train":
        start, end = config["train_orbit_start"], config["train_orbit_end"]
    elif split == "val":
        start, end = config["val_orbit_start"], config["val_orbit_end"]
    else:
        raise ValueError("split must be 'train' or 'val'")
    files = list_swath_files(
        config["processed_location"], config["series_prefix"],
        orbit_start=start, orbit_end=end,
    )
    if not files:
        raise FileNotFoundError(f"No {split} swaths in {config['processed_location']}")
    return files


def assert_no_selected_leakage(channels: List[str]) -> None:
    leaked = sorted(SELECTED_LEAKAGE.intersection(channels))
    if leaked:
        raise ValueError(
            "CRPS inputs must withhold selected wind / selamb to keep ensemble "
            f"spread meaningful; remove: {leaked}"
        )


def build_crps_half(
    path: str, half: str, config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    swath = slice_swath(prepare_swath(path, config), half)
    channels = list(config["input_channels"])
    assert_no_selected_leakage(channels)
    x = build_feature_stack(
        swath, None, channels, fov_m=float(config.get("fov_m", 12500.0)),
    )
    targets = build_sobel_target(swath, config)
    # Loss validity = strict QC; target field built under label_mask.
    y = np.stack([targets["sobel"], swath["valid_mask"]], axis=-1)
    return x.astype(np.float32), y.astype(np.float32)


class CRPSDataset(Dataset):
    """
    Train: random along-track crops + optional flip.
    Val: full swath halves (batch_size should be 1 under GroupNorm).
    """

    def __init__(self, files: List[str], config: Dict, split: str):
        self.files = files
        self.config = config
        self.split = split
        self.crop_len = int(config.get("crop_along_track", 0) or 0)
        self.positive_tries = int(config.get("crop_positive_tries", 8))

    def __len__(self) -> int:
        return len(self.files) * len(HALF_NAMES)

    def _crop_train(self, x: torch.Tensor, y: torch.Tensor):
        if self.crop_len <= 0:
            return x, y
        _, h, _ = x.shape
        if h <= self.crop_len:
            return x, y
        # Prefer crops that contain high-gradient / GF signal.
        for _ in range(self.positive_tries):
            i0 = int(np.random.randint(0, h - self.crop_len + 1))
            y_crop = y[:, i0 : i0 + self.crop_len]
            if np.random.rand() < 0.5 or bool((y_crop[0] > 0.05).any()):
                x = x[:, i0 : i0 + self.crop_len]
                y = y_crop
                break
        else:
            i0 = int(np.random.randint(0, h - self.crop_len + 1))
            x = x[:, i0 : i0 + self.crop_len]
            y = y[:, i0 : i0 + self.crop_len]
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        return x, y

    def __getitem__(self, idx: int):
        path = self.files[idx // len(HALF_NAMES)]
        half = HALF_NAMES[idx % len(HALF_NAMES)]
        x_np, y_np = build_crps_half(path, half, self.config)
        x = torch.from_numpy(x_np).permute(2, 0, 1)
        y = torch.from_numpy(y_np).permute(2, 0, 1)
        if self.split == "train":
            x, y = self._crop_train(x, y)
        return x, y


def crps_loader(config: Dict, split: str) -> DataLoader:
    dataset = CRPSDataset(split_files(config, split), config, split)
    if split == "train":
        batch_size = int(config.get("batch_size", 16))
    else:
        batch_size = int(config.get("val_batch_size", 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )
