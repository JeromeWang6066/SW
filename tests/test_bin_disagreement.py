"""Why a bin changes: same source cell, or a different one. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import max_source_match, max_source_pearson

_SPEC = importlib.util.spec_from_file_location(
    "bin_disagreement",
    ROOT / "scripts" / "12_bin_disagreement.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _tied_one(n_source: int, index: int) -> np.ndarray:
    tied = np.zeros(n_source, dtype=bool)
    tied[index] = True
    return tied


def test_small_correlation_gap_stays_with_the_same_source() -> None:
    short_rho = np.array([0.81, 0.95, 0.90])
    long_rho = np.array([0.79, 0.60, 0.91])
    short_argmax = np.array([0, 0, 1])
    long_argmax = np.array([0, 2, 1])
    short_tied = np.vstack([_tied_one(3, 0), _tied_one(3, 0), _tied_one(3, 1)])
    long_tied = np.vstack([_tied_one(3, 0), _tied_one(3, 2), _tied_one(3, 1)])
    self_rho = np.array([0.99, 0.40, 0.97])
    summary = mod.summarize_bin_changes(
        short_rho, long_rho, short_argmax, long_argmax, short_tied, long_tied, self_rho
    )
    assert summary["n_both_defined"] == 3
    assert summary["n_diff_bin"] == 2
    assert summary["n_same_bin"] == 1
    different = summary["different_bin"]
    assert different["same_source"]["n"] == 1
    assert different["different_source"]["n"] == 1
    assert different["same_source"]["delta"]["le_0.02"] == 1
    assert different["different_source"]["delta"]["gt_0.10"] == 1
    assert different["same_source"]["self_profile"]["gt_0.8"] == 1
    assert different["different_source"]["self_profile"]["le_0.5"] == 1
    assert summary["transitions_short_by_long"][2][1] == 2
    assert summary["transitions_short_by_long"][2][2] == 1


def test_tie_counts_as_the_same_source_cell() -> None:
    short_rho = np.array([0.90])
    long_rho = np.array([0.70])
    short_argmax = np.array([0])
    long_argmax = np.array([1])
    short_tied = np.array([[True, True, False]])
    long_tied = np.array([[False, True, False]])
    self_rho = np.array([0.95])
    summary = mod.summarize_bin_changes(
        short_rho, long_rho, short_argmax, long_argmax, short_tied, long_tied, self_rho
    )
    assert summary["n_diff_bin"] == 1
    assert summary["different_bin"]["same_source"]["n"] == 1
    assert summary["n_shared_source_but_argmax_differs"] == 1


def test_match_index_agrees_with_max_pearson() -> None:
    hours = np.arange(24, dtype=np.float64)
    sine = np.sin(2 * np.pi * hours / 24)
    cosine = np.cos(2 * np.pi * hours / 24)
    source = np.vstack([np.zeros(24), sine, cosine])
    short = sine + 0.1 * cosine
    long = sine + 1.2 * cosine
    target = np.vstack([short, long, np.full(24, 4.0)])
    rho, defined, argmax, tied = max_source_match(source, target)
    plain, plain_defined = max_source_pearson(source, target)
    assert defined.tolist() == plain_defined.tolist() == [True, True, False]
    assert np.allclose(rho[defined], plain[defined])
    assert argmax[:2].tolist() == [1, 2]
    assert tied[0].tolist() == [False, True, False]
    assert tied[2].tolist() == [False, False, False]
    assert int(tied[0].sum()) == 1
