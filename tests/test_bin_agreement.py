"""Bin edges and agreement. No city data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import assign_similarity_bin, bin_agreement


def test_bin_edges() -> None:
    rho = np.array([0.5, 0.5 + 1e-6, 0.8, 0.8 + 1e-6, np.nan])
    bins = assign_similarity_bin(rho)
    assert bins.tolist() == [0, 1, 1, 2, -1]


def test_agreement_uses_cells_defined_twice() -> None:
    protocol = np.array([0, 1, 2, -1, 1])
    control = np.array([0, 1, 0, 2, -1])
    stats = bin_agreement(protocol, control)
    assert stats["n_both_defined"] == 3
    assert stats["n_same"] == 2
    assert abs(stats["agreement"] - 2 / 3) < 1e-12
    assert stats["n_only_protocol"] == 1
    assert stats["n_only_control"] == 1
    assert stats["passes_gate"] is False
