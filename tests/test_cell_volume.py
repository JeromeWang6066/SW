"""Spatial pooling and contour helpers. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import max_source_match, paired_profile_pearson

_SPEC = importlib.util.spec_from_file_location(
    "cell_volume",
    ROOT / "scripts" / "14_cell_volume.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_block_sum_marks_partial_edges() -> None:
    flow = np.ones((1, 3, 3), dtype=np.float64)
    reduced, full = mod.block_sum(flow, 2)
    assert reduced.shape == (1, 2, 2)
    assert reduced[0, 0, 0] == 4
    assert reduced[0, 0, 1] == 2
    assert reduced[0, 1, 0] == 2
    assert reduced[0, 1, 1] == 1
    assert bool(full[0, 0])
    assert not bool(full[0, 1] or full[1, 0] or full[1, 1])


def test_box_sum_center_uses_all_neighbors_and_corner_is_partial() -> None:
    flow = np.ones((1, 3, 3), dtype=np.float64)
    out = mod.box_sum(flow, 3)
    assert out.shape == (1, 3, 3)
    assert out[0, 1, 1] == 9
    assert out[0, 0, 0] == 4
    assert out[0, 0, 1] == 6
    assert out[0, 1, 0] == 6
    assert np.array_equal(mod.box_sum(flow, 1), flow)


def test_hour_energy_of_one_spike_is_almost_all_the_shape() -> None:
    profile = np.zeros((1, 24), dtype=np.float64)
    profile[0, 5] = 10
    share = mod.hour_energy_share(profile)
    assert share[0] > 0.9
    flat = np.ones((1, 24), dtype=np.float64)
    assert np.isnan(mod.hour_energy_share(flat)[0])


def test_knockout_replaces_only_the_hottest_hour() -> None:
    profile = np.array([[0.0, 5.0, 1.0, 1.0]])
    knocked, hot = mod.knock_out_hottest(profile)
    assert int(hot[0]) == 1
    assert knocked[0, 0] == 0
    assert knocked[0, 2] == 1
    assert knocked[0, 3] == 1
    assert knocked[0, 1] == (0 + 1 + 1) / 3


def test_generic_match_agrees_with_the_24_hour_helper() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(5, 24))
    target = rng.normal(size=(4, 24))
    lib = max_source_match(source, target)
    local = mod.max_source_match_d(source, target)
    assert np.allclose(lib[0], local[0], equal_nan=True)
    assert np.array_equal(lib[1], local[1])
    assert np.array_equal(lib[2], local[2])
    rho, defined = mod.row_pearson(target, target)
    lib_rho, lib_defined = paired_profile_pearson(target, target)
    assert np.array_equal(defined, lib_defined)
    assert np.allclose(rho, lib_rho, equal_nan=True)


def test_spearman_of_the_same_series_is_one() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    assert abs(mod.spearman(values, values) - 1.0) < 1e-12
