"""Train-only typologies on gated cells. Edges are locked before any test load."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dual_kappa_pilot.protocol import N_TRAIN_HOURS


AXES = ("mean_flow", "cv", "zero_flow_ratio")
BINS = ("low", "mid", "high")
CV_MEAN_ZERO = "mean_zero"


def _col(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("typology feature must be a 1-D cell series")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite typology feature")
    return arr


def linear_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    """Linear quantiles at 1/3 and 2/3. Duplicate edges may leave a bin empty."""
    arr = _col(values)
    if arr.size == 0:
        raise ValueError("cannot compute tertiles on an empty set")
    q1 = float(np.quantile(arr, 1.0 / 3.0, method="linear"))
    q2 = float(np.quantile(arr, 2.0 / 3.0, method="linear"))
    if not (math_isfinite(q1) and math_isfinite(q2)):
        raise ValueError("non-finite tertile edges")
    return q1, q2


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def assign_tertile(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def cell_train_features(target: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if truth.ndim != 2 or keep.shape != (truth.shape[1],):
        raise ValueError("expected train target (hours, cells) and one flag per cell")
    if truth.shape[0] != N_TRAIN_HOURS:
        raise ValueError(f"train hours {truth.shape[0]} != {N_TRAIN_HOURS}")
    if int(keep.sum()) <= 0:
        raise ValueError("no gated cells for typologies")
    gated_index = np.flatnonzero(keep)
    mean_flow = truth[:, keep].mean(axis=0)
    zero_ratio = np.mean(truth[:, keep] == 0.0, axis=0)
    std = truth[:, keep].std(axis=0, ddof=0)
    cv = np.full(mean_flow.shape, np.nan, dtype=np.float64)
    mean_zero = mean_flow == 0.0
    nonzero = ~mean_zero
    if np.any(nonzero):
        cv[nonzero] = std[nonzero] / mean_flow[nonzero]
    return {
        "gated_index": gated_index.astype(np.int64),
        "mean_flow": mean_flow.astype(np.float64),
        "cv": cv,
        "cv_mean_zero": mean_zero.astype(bool),
        "zero_flow_ratio": zero_ratio.astype(np.float64),
    }


def lock_typology(target: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    """Define bins from train truth only. Never reads test or val."""
    feats = cell_train_features(target, eligible)
    mean_edges = linear_tertile_edges(feats["mean_flow"])
    zero_edges = linear_tertile_edges(feats["zero_flow_ratio"])
    cv_values = feats["cv"][~feats["cv_mean_zero"]]
    if cv_values.size == 0:
        cv_edges: tuple[float, float] | None = None
    else:
        cv_edges = linear_tertile_edges(cv_values)
    cells = []
    for i, cell in enumerate(feats["gated_index"]):
        mean_bin = assign_tertile(float(feats["mean_flow"][i]), *mean_edges)
        zero_bin = assign_tertile(float(feats["zero_flow_ratio"][i]), *zero_edges)
        if bool(feats["cv_mean_zero"][i]):
            cv_bin = CV_MEAN_ZERO
        else:
            assert cv_edges is not None
            cv_bin = assign_tertile(float(feats["cv"][i]), *cv_edges)
        cells.append(
            {
                "cell_index": int(cell),
                "mean_flow": float(feats["mean_flow"][i]),
                "cv": None if bool(feats["cv_mean_zero"][i]) else float(feats["cv"][i]),
                "zero_flow_ratio": float(feats["zero_flow_ratio"][i]),
                "bin_mean_flow": mean_bin,
                "bin_cv": cv_bin,
                "bin_zero_flow_ratio": zero_bin,
            }
        )
    return {
        "defined_on": "train_truth_gated_cells_only",
        "n_gated": int(feats["gated_index"].size),
        "edges": {
            "mean_flow": {"q1": mean_edges[0], "q2": mean_edges[1]},
            "zero_flow_ratio": {"q1": zero_edges[0], "q2": zero_edges[1]},
            "cv": None if cv_edges is None else {"q1": cv_edges[0], "q2": cv_edges[1]},
        },
        "rule": "low if x<=q1; mid if q1<x<=q2; high if x>q2; duplicate edges may leave a bin empty; CV mean==0 is a separate column, excluded from CV tertiles",
        "cells": cells,
    }


def typologies_must_agree(left: dict[str, Any], right: dict[str, Any], *, context: str) -> None:
    if left["n_gated"] != right["n_gated"] or left["edges"] != right["edges"]:
        raise ValueError(f"train typologies disagree ({context}): edges or gated count differ")
    if left["cells"] != right["cells"]:
        raise ValueError(f"train typologies disagree ({context}): cell assignments differ")
