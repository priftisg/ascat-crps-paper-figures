"""Read one non-duplicated buoy-collocation experiment for weak supervision."""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np


def valid_buoy_ids(manifest_path):
    paths = np.load(manifest_path, allow_pickle=True)
    ids = set()
    for path in paths:
        match = re.fullmatch(r"met(.+?)_hr\.cdf", Path(str(path)).name)
        if match:
            ids.add(match.group(1).lower())
    if not ids:
        raise ValueError(f"No buoy IDs found in {manifest_path}")
    return ids


def load_buoy_opportunities(
    path, W_hr, d0_km, val_fraction=0.2, test_fraction=0.2,
    manifest_path=None,
):
    allowed = valid_buoy_ids(manifest_path) if manifest_path else None
    rows, seen = [], set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["W_hr"]) - float(W_hr)) > 1.0e-6:
                continue
            if abs(float(row["d0_km"]) - float(d0_km)) > 1.0e-6:
                continue
            if row["outcome"].strip().lower() == "invalid":
                continue
            if allowed is not None and row["buoy"].strip().lower() not in allowed:
                continue
            key = (row["orbit"], row["buoy"], row["t0"])
            if key in seen:
                raise ValueError(f"Duplicate opportunity after W/d0 filtering: {key}")
            seen.add(key)
            row["label"] = int(float(row["n_events"]) > 0)
            token = int(hashlib.sha1(row["buoy"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if token < test_fraction:
                row["split"] = "test"
            elif token < test_fraction + val_fraction:
                row["split"] = "val"
            else:
                row["split"] = "train"
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid opportunities for W_hr={W_hr}, d0_km={d0_km}")
    return rows
