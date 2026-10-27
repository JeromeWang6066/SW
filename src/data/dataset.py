"""Dataset stubs for city grid traffic flow.

Expected layout (see docs/DATA.md):
  data/processed/{ny,chi,dc}/...

No downloads and no fake metrics here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


class GridFlowDataset:
    """Minimal placeholder dataset.

    Replace ``load_city_array`` + windowing with a real CrossTReS-compatible
    loader once data is on disk.
    """

    def __init__(
        self,
        city: str,
        split: str,
        processed_dir: str | Path = "data/processed",
        history_len: int = 6,
        pred_len: int = 1,
    ) -> None:
        self.city = city.lower()
        self.split = split
        self.processed_dir = Path(processed_dir)
        self.history_len = history_len
        self.pred_len = pred_len
        self._data: Optional[np.ndarray] = None

    def load(self) -> np.ndarray:
        """Load processed array or raise a clear FileNotFoundError."""
        self._data = load_city_array(self.city, self.processed_dir)
        return self._data

    def __len__(self) -> int:
        raise NotImplementedError(
            "Windowed __len__ is TODO. Implement after real tensors are available."
        )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        raise NotImplementedError(
            "Windowed __getitem__ is TODO. Do not invent samples."
        )


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


def _assert_time_major(arr: np.ndarray, city: str) -> None:
    if arr.ndim < 2:
        raise ValueError(f"{city}: expected T×… array, got shape {arr.shape}")
    # Soft check only — H/W filled in configs after real data arrives.
    if arr.shape[0] < 24:
        raise ValueError(
            f"{city}: time dimension T={arr.shape[0]} looks too short for hourly data."
        )
