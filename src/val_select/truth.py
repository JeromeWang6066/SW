"""Shared DC label and gate comparisons.

Comparisons use the actual time-aligned, column-aligned arrays. Matching
manifest hashes alone is not sufficient. The default is exact equality on
the same generation chain; observed differences are not used to loosen a
tolerance.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.val_select.schema import SchemaError, TASK_IDENTITY_FIELDS, values_equivalent


def assert_aligned_labels_equal(left: dict[str, Any], right: dict[str, Any], *, what: str) -> None:
    if not np.array_equal(np.asarray(left["time_index"]), np.asarray(right["time_index"])):
        raise SchemaError(f"{what}: time_index is not aligned")
    if not np.array_equal(np.asarray(left["cell_ids"]).astype(str), np.asarray(right["cell_ids"]).astype(str)):
        raise SchemaError(f"{what}: cell_ids / column order is not aligned")
    if not np.array_equal(np.asarray(left["target"], dtype=np.float64), np.asarray(right["target"], dtype=np.float64)):
        raise SchemaError(f"{what}: DC target labels are not identical on the shared chain")


def assert_aligned_gate_equal(left: dict[str, Any], right: dict[str, Any], *, what: str) -> None:
    if left.get("raw_train_counts") is None or right.get("raw_train_counts") is None:
        raise SchemaError(f"{what}: raw train gate counts are required for the shared DC comparison")
    if not np.array_equal(np.asarray(left["cell_ids"]).astype(str), np.asarray(right["cell_ids"]).astype(str)):
        raise SchemaError(f"{what}: cell_ids / column order is not aligned for gate stats")
    if not np.array_equal(
        np.asarray(left["raw_train_counts"], dtype=np.float64),
        np.asarray(right["raw_train_counts"], dtype=np.float64),
    ):
        raise SchemaError(f"{what}: raw 72h gate counts are not identical on the shared DC")
    if not np.array_equal(np.asarray(left["eligible"], dtype=bool), np.asarray(right["eligible"], dtype=bool)):
        raise SchemaError(f"{what}: eligible masks are not identical on the shared DC")


def assert_task_split_consistency(base: dict[str, Any], other: dict[str, Any]) -> None:
    """Same source×seed must share identity, unit, normalization, and checkpoint.

    Different seeds are not required to share a weight hash.
    """
    for field in TASK_IDENTITY_FIELDS:
        if field not in base or field not in other:
            raise SchemaError(f"missing {field} when comparing splits of the same task")
        if not values_equivalent(field, base[field], other[field]):
            raise SchemaError(
                f"same-task splits disagree on {field}: {base['split']} vs {other['split']} "
                f"for {base['source']}:{base['seed']}"
            )


def assert_shared_dc_split(records: Iterable[dict[str, Any]], *, split: str, compare_gate: bool = False) -> None:
    items = list(records)
    if not items:
        raise SchemaError(f"no {split} records to compare for a shared DC truth")
    reference = items[0]
    for item in items[1:]:
        assert_aligned_labels_equal(reference, item, what=f"shared DC {split}")
        if compare_gate:
            assert_aligned_gate_equal(reference, item, what=f"shared DC {split} gate")
