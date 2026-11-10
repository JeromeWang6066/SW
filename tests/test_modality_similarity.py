"""City-level modality Pearson rules. No city arrays and no training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "modality_similarity",
    ROOT / "scripts" / "10_modality_similarity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_constant_pair_is_undefined_and_identical_pair_is_one() -> None:
    flat = np.ones(24, dtype=np.float64)
    wave = np.arange(24, dtype=np.float64)
    assert mod.finite_rho(flat, wave) is None
    assert mod.finite_rho(flat, flat) is None
    assert mod.finite_rho(wave, wave) == 1.0


def test_short_window_has_three_weekday_samples_per_hour() -> None:
    counts = mod.short_weekday_counts()
    assert counts.tolist() == [3] * 24


def test_candidate_requires_the_long_window() -> None:
    assert mod.classify_pair(0.70, 0.70) == "candidate"
    assert mod.classify_pair(0.90, 0.80) == "candidate"
    assert mod.classify_pair(0.70, 0.81) == "attenuated"
    assert mod.classify_pair(0.90, 0.90) == "both_high"
    assert mod.classify_pair(None, 0.40) == "undefined"


def test_pair_counts_are_directed_and_unordered() -> None:
    series = [f"{city}:{channel}" for city in mod.CITIES for channel in mod.CHANNELS]
    assert len(series) == 9
    directed = [(a, b) for a in series for b in series if a != b]
    unordered = {(a, b) for i, a in enumerate(series) for b in series[i + 1 :]}
    assert len(directed) == 72
    assert len(unordered) == 36
