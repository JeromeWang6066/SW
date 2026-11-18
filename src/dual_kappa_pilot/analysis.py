"""Same-pass per-cell analysis and the pre-locked STOP judgment.

RMSE cell scores are not additive. City MAE/MSE contributions are exact
spatial averages over the official city mask. Typologies come from the lock
(train-only) and are never redefined from test.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.dual_kappa_pilot.protocol import (
    PRELOCKED_SEEDS,
    SOURCES,
    STOP_MAE_REL_MIN,
    STOP_N_TASKS_LE_OLD_MIN,
    STOP_REFERENCES,
    STOP_RMSE_RATIO_MAX,
    equality_tolerance,
    mae_less_or_equal,
    relative_mae_improvement,
)
from src.dual_kappa_pilot.select import expected_task_keys


def negative_prediction_report(prediction: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    neg = pred < 0.0
    count = int(np.count_nonzero(neg))
    return {
        "count": count,
        "min": None if count == 0 else float(pred.min()),
        "clipped": False,
    }


def cell_error_table(
    new_pred: np.ndarray,
    ref_pred: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    new = np.asarray(new_pred, dtype=np.float64)
    ref = np.asarray(ref_pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if new.shape != ref.shape or new.shape != truth.shape or new.ndim != 2:
        raise ValueError("aligned 2-D arrays required for per-cell errors")
    if keep.shape != (new.shape[1],):
        raise ValueError("eligible mask must match the cell axis")
    hours, n_cells = new.shape
    err_new = new - truth
    err_ref = ref - truth
    mae_new = np.abs(err_new).mean(axis=0)
    mae_ref = np.abs(err_ref).mean(axis=0)
    mse_new = (err_new ** 2).mean(axis=0)
    mse_ref = (err_ref ** 2).mean(axis=0)
    delta_mae = mae_ref - mae_new  # positive = new is better on MAE
    delta_mse = mse_ref - mse_new  # positive = new has smaller MSE (helps city RMSE)
    # City MAE is the mean of per-cell MAEs when every cell has the same hour count.
    contrib_mae = delta_mae / float(n_cells)
    contrib_mse = delta_mse / float(n_cells)
    if not np.isclose(float(contrib_mae.sum()), float(delta_mae.mean()), rtol=0, atol=1e-15):
        raise ValueError("internal MAE contribution identity failed")
    cells = []
    for j in range(n_cells):
        cells.append(
            {
                "cell_index": int(j),
                "gated": bool(keep[j]),
                "mae_new": float(mae_new[j]),
                "mae_ref": float(mae_ref[j]),
                "delta_mae": float(delta_mae[j]),
                "city_mae_contribution": float(contrib_mae[j]),
                "mse_new": float(mse_new[j]),
                "mse_ref": float(mse_ref[j]),
                "delta_mse": float(delta_mse[j]),
                "city_mse_contribution": float(contrib_mse[j]),
                "rmse_new": float(np.sqrt(mse_new[j])),
                "rmse_ref": float(np.sqrt(mse_ref[j])),
                "rmse_not_additive": True,
            }
        )
    # Direction is signed city-mask Delta MAE_j = MAE_ref_j - MAE_new_j.
    # Improved: Delta MAE_j > 0. Degraded: Delta MAE_j < 0. Zero is excluded
    # and does not pad either list. At most 10 each; fewer if fewer cells qualify.
    improved_idx = [j for j in range(n_cells) if float(delta_mae[j]) > 0.0]
    degraded_idx = [j for j in range(n_cells) if float(delta_mae[j]) < 0.0]
    improved_idx.sort(key=lambda j: (-float(delta_mae[j]), j))
    degraded_idx.sort(key=lambda j: (float(delta_mae[j]), j))
    top_improved = [cells[j] for j in improved_idx[:10]]
    top_degraded = [cells[j] for j in degraded_idx[:10]]
    abs_city = float(np.abs(contrib_mae).sum())
    top10_abs_share = None
    if abs_city > 0:
        # Concentration uses the 10 largest |Delta MAE| cells (zeros do not contribute).
        by_abs = [j for j in range(n_cells) if float(delta_mae[j]) != 0.0]
        by_abs.sort(key=lambda j: (-abs(float(delta_mae[j])), j))
        top10_abs_share = float(np.abs(contrib_mae[by_abs[:10]]).sum() / abs_city)
    return {
        "hours": int(hours),
        "n_cells": int(n_cells),
        "city_delta_mae": float(delta_mae.mean()),
        "city_delta_mse": float(delta_mse.mean()),
        "cells": cells,
        "top10_improved_by_delta_mae": top_improved,
        "top10_degraded_by_delta_mae": top_degraded,
        "n_improved": int(len(improved_idx)),
        "n_degraded": int(len(degraded_idx)),
        "zero_delta_excluded": True,
        "concentration_abs_city_mae_top10": top10_abs_share,
        "note": (
            "RMSE_j is descriptive only and must not be summed; city RMSE comes from MSE. "
            "Top-10 improved cells have Delta MAE_j > 0 (new better); top-10 degraded cells "
            "have Delta MAE_j < 0. Zero Delta MAE is not counted and lists are not padded."
        ),
    }


def seed_consistency(task_tables: dict[str, dict[str, Any]], source: str, seeds=PRELOCKED_SEEDS) -> dict[str, Any]:
    keys = [f"{source}:{int(seed)}" for seed in seeds]
    missing = [k for k in keys if k not in task_tables]
    if missing:
        return {"source": source, "complete": False, "missing": missing}
    n_cells = task_tables[keys[0]]["n_cells"]
    signs = []
    for key in keys:
        deltas = np.array([c["delta_mae"] for c in task_tables[key]["cells"]], dtype=np.float64)
        if deltas.size != n_cells:
            raise ValueError(f"{key} cell count disagrees")
        signs.append(np.sign(deltas))
    stacked = np.stack(signs, axis=0)
    agree_improve = np.all(stacked > 0, axis=0)
    agree_degrade = np.all(stacked < 0, axis=0)
    return {
        "source": source,
        "complete": True,
        "n_cells": int(n_cells),
        "n_agree_improve_all_seeds": int(agree_improve.sum()),
        "n_agree_degrade_all_seeds": int(agree_degrade.sum()),
        "agree_improve_indices": [int(i) for i in np.flatnonzero(agree_improve)],
        "agree_degrade_indices": [int(i) for i in np.flatnonzero(agree_degrade)],
    }


def typology_aggregates(cell_table: dict[str, Any], typology: dict[str, Any]) -> dict[str, Any]:
    by_index = {int(c["cell_index"]): c for c in typology["cells"]}
    groups: dict[str, dict[str, list[float]]] = {}
    for typo in typology["cells"]:
        idx = int(typo["cell_index"])
        cell = cell_table["cells"][idx]
        for axis, bin_key in (
            ("mean_flow", typo["bin_mean_flow"]),
            ("cv", typo["bin_cv"]),
            ("zero_flow_ratio", typo["bin_zero_flow_ratio"]),
        ):
            name = f"{axis}:{bin_key}"
            groups.setdefault(name, {"delta_mae": [], "delta_mse": [], "indices": []})
            groups[name]["delta_mae"].append(cell["delta_mae"])
            groups[name]["delta_mse"].append(cell["delta_mse"])
            groups[name]["indices"].append(idx)
    out = {}
    for name, payload in groups.items():
        arr = np.asarray(payload["delta_mae"], dtype=np.float64)
        out[name] = {
            "n_cells": int(arr.size),
            "mean_delta_mae": float(arr.mean()) if arr.size else None,
            "mean_delta_mse": float(np.mean(payload["delta_mse"])) if payload["delta_mse"] else None,
            "indices": payload["indices"],
        }
    # Empty bins (possible when tertile edges coincide) are reported as present with n=0.
    for axis in ("mean_flow", "zero_flow_ratio"):
        for bin_name in ("low", "mid", "high"):
            out.setdefault(f"{axis}:{bin_name}", {"n_cells": 0, "mean_delta_mae": None, "mean_delta_mse": None, "indices": []})
    for bin_name in ("low", "mid", "high", "mean_zero"):
        out.setdefault(f"cv:{bin_name}", {"n_cells": 0, "mean_delta_mae": None, "mean_delta_mse": None, "indices": []})
    return out


STOP_REQUIRED_ARMS: tuple[str, ...] = ("DOUBLE_VAL30", "OLD_FIXED", "SINGLE_VAL10")


def _finite_number(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _require_finite_nonnegative(value: Any, label: str) -> float:
    if not _finite_number(value):
        raise ValueError(f"STOP rejected: {label} is not a finite number")
    v = float(value)
    if v < 0.0:
        raise ValueError(f"STOP rejected: {label} is negative")
    return v


def _finite_mean(values: list[float], label: str) -> float:
    if not values:
        raise ValueError(f"STOP rejected: cannot aggregate an empty set for {label}")
    acc = 0.0
    for item in values:
        acc = acc + float(item)
        if not math.isfinite(acc):
            raise ValueError(f"STOP rejected: aggregation overflow in {label}")
    mean = acc / float(len(values))
    if not math.isfinite(mean):
        raise ValueError(f"STOP rejected: non-finite mean for {label}")
    return mean


def validate_stop_metrics(task_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    """Reject invalid required metrics. This is not an ordinary STOP decision."""
    keys = expected_task_keys()
    missing = [k for k in keys if k not in task_metrics]
    if missing:
        raise ValueError(f"STOP rejected: missing required tasks {missing}")
    for key in keys:
        arms = task_metrics[key]
        for arm in STOP_REQUIRED_ARMS:
            if arm not in arms:
                raise ValueError(f"STOP rejected: {key} missing required arm {arm}")
            _require_finite_nonnegative(arms[arm].get("mae"), f"{key} {arm} MAE")
            _require_finite_nonnegative(arms[arm].get("rmse"), f"{key} {arm} RMSE")


def source_means(task_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """task_metrics[task][arm] -> {mae, rmse}."""
    out: dict[str, Any] = {}
    for source in SOURCES:
        keys = [f"{source}:{int(seed)}" for seed in PRELOCKED_SEEDS]
        if any(k not in task_metrics for k in keys):
            out[source] = {"complete": False, "missing": [k for k in keys if k not in task_metrics]}
            continue
        arms = sorted({arm for k in keys for arm in task_metrics[k]})
        rec: dict[str, Any] = {"complete": True, "arms": {}}
        for arm in arms:
            maes = [float(task_metrics[k][arm]["mae"]) for k in keys]
            rmses = [float(task_metrics[k][arm]["rmse"]) for k in keys]
            rec["arms"][arm] = {
                "mae_mean": float(np.mean(maes)),
                "rmse_mean": float(np.mean(rmses)),
                "mae_per_seed": maes,
                "rmse_per_seed": rmses,
            }
        out[source] = rec
    return out


def stop_judgment(task_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Apply the pre-locked STOP rule. Does not retune thresholds.

    Invalid metrics (non-finite, negative, non-positive reference means,
    non-finite aggregates or ratios) raise ValueError. That rejection is not
    a PASS and not an ordinary STOP.
    """
    validate_stop_metrics(task_metrics)
    keys = expected_task_keys()
    mae_comparisons = []
    rmse_comparisons = []
    mae_ok = True
    rmse_ok = True
    for source in SOURCES:
        keys_s = [f"{source}:{int(seed)}" for seed in PRELOCKED_SEEDS]
        new_mae_mean = _finite_mean(
            [float(task_metrics[k]["DOUBLE_VAL30"]["mae"]) for k in keys_s],
            f"{source} DOUBLE_VAL30 MAE",
        )
        new_rmse_mean = _finite_mean(
            [float(task_metrics[k]["DOUBLE_VAL30"]["rmse"]) for k in keys_s],
            f"{source} DOUBLE_VAL30 RMSE",
        )
        for ref_arm in STOP_REFERENCES:
            ref_mae_mean = _finite_mean(
                [float(task_metrics[k][ref_arm]["mae"]) for k in keys_s],
                f"{source} {ref_arm} MAE",
            )
            ref_rmse_mean = _finite_mean(
                [float(task_metrics[k][ref_arm]["rmse"]) for k in keys_s],
                f"{source} {ref_arm} RMSE",
            )
            if ref_mae_mean <= 0.0:
                raise ValueError(
                    f"STOP rejected: {source} {ref_arm} MAE reference mean is not positive"
                )
            if ref_rmse_mean <= 0.0:
                raise ValueError(
                    f"STOP rejected: {source} {ref_arm} RMSE reference mean is not positive"
                )
            rel = relative_mae_improvement(new_mae_mean, ref_mae_mean)
            rel_pass = rel >= STOP_MAE_REL_MIN
            mae_ok = mae_ok and rel_pass
            mae_comparisons.append(
                {
                    "source": source,
                    "reference": ref_arm,
                    "new_mae_mean": new_mae_mean,
                    "ref_mae_mean": ref_mae_mean,
                    "relative_improvement": rel,
                    "formula": "(refmean-dualmean)/refmean",
                    "threshold": STOP_MAE_REL_MIN,
                    "pass": rel_pass,
                }
            )
            ratio = new_rmse_mean / ref_rmse_mean
            if not math.isfinite(ratio):
                raise ValueError(f"STOP rejected: non-finite RMSE ratio for {source} vs {ref_arm}")
            rmse_pass = ratio <= STOP_RMSE_RATIO_MAX
            rmse_ok = rmse_ok and rmse_pass
            rmse_comparisons.append(
                {
                    "source": source,
                    "reference": ref_arm,
                    "new_rmse_mean": new_rmse_mean,
                    "ref_rmse_mean": ref_rmse_mean,
                    "ratio": ratio,
                    "threshold": STOP_RMSE_RATIO_MAX,
                    "pass": rmse_pass,
                }
            )
    task_rows = []
    n_le = 0
    for key in keys:
        new_mae = float(task_metrics[key]["DOUBLE_VAL30"]["mae"])
        old_mae = float(task_metrics[key]["OLD_FIXED"]["mae"])
        le = mae_less_or_equal(new_mae, old_mae)
        if le:
            n_le += 1
        task_rows.append(
            {
                "task": key,
                "double_mae": new_mae,
                "old_fixed_mae": old_mae,
                "less_or_equal": le,
                "tolerance": equality_tolerance(old_mae),
            }
        )
    task_ok = n_le >= STOP_N_TASKS_LE_OLD_MIN
    passed = bool(mae_ok and rmse_ok and task_ok)
    return {
        "decision": "WORTH_SUBSEQUENT_DISCUSSION" if passed else "STOP",
        "passed": passed,
        "mae_comparisons": mae_comparisons,
        "rmse_comparisons": rmse_comparisons,
        "task_vs_old_fixed": task_rows,
        "n_tasks_le_old_fixed": n_le,
        "n_tasks_required": STOP_N_TASKS_LE_OLD_MIN,
        "mae_all_pass": mae_ok,
        "rmse_all_pass": rmse_ok,
        "task_rule_pass": task_ok,
        "note": (
            "All three screens must pass together. Thresholds are pre-locked engineering "
            "filters, not significance tests, and must not be adjusted after seeing test. "
            "Invalid, negative, non-finite, or non-positive-reference metrics are rejected "
            "by exception and are neither PASS nor ordinary STOP."
        ),
    }
