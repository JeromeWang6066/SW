"""City grid loaders and windowed datasets for CrossTReS / STBaT splits.

Expected layout (see docs/DATA.md):
  data/processed/{ny,chi,dc}/...

No downloads and no fake metrics here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.data.splits import split_x_y, trim_few_shot


class GridFlowDataset:
    """Windowed ``(x, y)`` pairs already cut by ``split_x_y``.

    ``x`` is ``(N, lag, H, W)`` and ``y`` is ``(N, 1, H, W)``, float32.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) != len(y):
            raise ValueError(f"x/y length mismatch {len(x)} vs {len(y)}")
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.x[idx], self.y[idx]


def load_city_array(city: str, processed_dir: str | Path = "data/processed") -> np.ndarray:
    """Try common filenames under ``processed/{city}/``.

    Looks for ``flow.npy``, ``data.npz`` (key ``flow``), or CrossTReS-style
    ``Taxi{CITY}_pickup.npy`` under ``upstream/`` soft-link. Does not download.
    """
    city = city.lower()
    root = Path(processed_dir) / city
    npy = root / "flow.npy"
    npz = root / "data.npz"
    city_up = city.upper()
    upstream_candidates = [
        root / "upstream" / f"Taxi{city_up}_pickup.npy",
        root / f"Taxi{city_up}_pickup.npy",
    ]

    if npy.is_file():
        arr = np.load(npy)
        _assert_time_major(arr, city)
        return arr

    if npz.is_file():
        with np.load(npz) as z:
            if "flow" not in z:
                raise KeyError(f"{npz} missing key 'flow'. Keys={list(z.keys())}")
            arr = z["flow"]
        _assert_time_major(arr, city)
        return arr

    for cand in upstream_candidates:
        if cand.is_file():
            arr = np.load(cand)
            _assert_time_major(arr, city)
            return arr

    raise FileNotFoundError(
        f"No processed array for city={city!r} under {root}. "
        f"Expected {npy.name}, {npz.name}, or CrossTReS "
        f"Taxi{city_up}_pickup.npy (see docs/DATA.md for verified download URLs)."
    )


def min_max_normalize(
    data: np.ndarray,
    percentile: float = 0.999,
) -> Tuple[np.ndarray, float, float]:
    """CrossTReS / STBaT percentile min-max. Returns a float32 copy plus max, min.

    ``min`` is ``max(0, smallest value)``, matching ``utils.min_max_normalize``.
    """
    data = np.array(data, dtype=np.float32, copy=True)
    flat = np.sort(data.reshape(-1))
    max_val = float(flat[int(len(flat) * percentile)])
    min_val = float(max(0.0, flat[0]))
    np.minimum(data, max_val, out=data)
    denom = max_val - min_val
    if denom <= 0:
        raise ValueError("min-max denominator is 0; check the input array")
    data -= min_val
    data /= denom
    return data, max_val, min_val


def valid_region_mask(raw: np.ndarray) -> np.ndarray:
    """Upstream mask: a cell is valid when its sum over time is positive."""
    if raw.ndim != 3:
        raise ValueError(f"expected T×H×W, got {raw.shape}")
    return raw.sum(axis=0) > 0


def prepare_city_windows(
    city: str,
    processed_dir: str | Path = "data/processed",
    *,
    few_shot_days: int = 0,
    concatenate_all_for_train: bool = False,
) -> Dict[str, Any]:
    """Load one city, normalize, and build train/val/test windows.

    The mask is computed on the raw array, before normalization.
    ``concatenate_all_for_train`` follows CrossTReS source training, which
    concatenates the source train, val, and test windows. Val and test arrays
    are still returned separately so checkpoint selection can use the val cut.
    ``few_shot_days > 0`` trims only the train windows (target 3-day protocol).
    """
    raw = load_city_array(city, processed_dir)
    mask = valid_region_mask(raw)
    if int(mask.sum()) == 0:
        raise ValueError(f"{city}: spatial mask is empty")
    normalized, max_val, min_val = min_max_normalize(raw)
    train_x, train_y, val_x, val_y, test_x, test_y = split_x_y(normalized)
    if concatenate_all_for_train:
        train_x = np.concatenate([train_x, val_x, test_x], axis=0)
        train_y = np.concatenate([train_y, val_y, test_y], axis=0)
    elif few_shot_days:
        train_x, train_y = trim_few_shot(train_x, train_y, few_shot_days)
    return {
        "city": city.lower(),
        "train": GridFlowDataset(train_x, train_y),
        "val": GridFlowDataset(val_x, val_y),
        "test": GridFlowDataset(test_x, test_y),
        "mask": mask,
        "max_val": max_val,
        "min_val": min_val,
        "grid_hw": (int(raw.shape[1]), int(raw.shape[2])),
    }


def _assert_time_major(arr: np.ndarray, city: str) -> None:
    if arr.ndim < 2:
        raise ValueError(f"{city}: expected T×… array, got shape {arr.shape}")
    # Soft check only — H/W filled in configs after real data arrives.
    if arr.shape[0] < 24:
        raise ValueError(
            f"{city}: time dimension T={arr.shape[0]} looks too short for hourly data."
        )
