"""Explicit prediction-cache schema and provenance checks.

The on-disk filename is only a lookup key. Source, seed, and split must be
declared inside the payload or a sidecar manifest. Missing provenance is a
hard error: this module does not reconstruct hours from shape, does not infer
identity from the filename, and does not invent checkpoint or normalization
proof. A format-valid recorded manifest is only schema-checkable; it is not
empirical proof of historical provenance.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.val_select.protocol import (
    EXPECTED_TEST_HOURS,
    EXPECTED_TEST_START,
    EXPECTED_VAL_HOURS,
    EXPECTED_VAL_START,
    GATE,
    LABEL_START_TRAIN,
    NORM_FITTED_ON,
    NORM_PERCENTILE,
    N_GATE_EXPECTED,
    N_TRAIN_HOURS,
    N_VALID_CELLS_EXPECTED,
    UNIT_DC_CLIPPED_COUNT,
    VAL_TEST_RANGE_STATUS,
    INPUT_PROTOCOL_ID,
    PROVENANCE_DOCUMENTED,
    TIME_INDEX_ORIGIN_DOCUMENTED,
    DOCUMENTED_PREDICT_SCRIPT_REVISION,
    DOCUMENTED_PREDICT_SCRIPT_FORM,
    CHECKPOINT_HASH_STATUS,
    FINE_TUNE_COMMIT_STATUS,
)

ALLOWED_PROVENANCE = frozenset({"synthetic", "recorded", "documented_from_bundle"})
# recorded rules are unchanged. documented_from_bundle is a separate explicit opt-in.

REQUIRED_COMMON_FIELDS = (
    "source",
    "seed",
    "split",
    "unit",
    "cell_ids",
    "time_index",
    "prediction",
    "target",
    "value_scale",
    "min_val",
    "norm_percentile",
    "norm_fitted_on",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_id",
    "generator_commit",
    "provenance_kind",
)

REQUIRED_TRAIN_FIELDS = REQUIRED_COMMON_FIELDS + ("eligible", "raw_train_counts", "gate_threshold")

CRITICAL_FIELDS = frozenset(REQUIRED_TRAIN_FIELDS)

PLACEHOLDERS = frozenset({
    "",
    "unknown",
    "unspecified",
    "synthetic",
    "null",
    "none",
    "n/a",
    "na",
    "placeholder",
    "tbd",
})

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

TASK_IDENTITY_FIELDS = (
    "source",
    "seed",
    "unit",
    "value_scale",
    "min_val",
    "norm_percentile",
    "norm_fitted_on",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_id",
    "generator_commit",
    "provenance_kind",
)


class SchemaError(ValueError):
    """Cache payload does not satisfy the declared schema or provenance rules."""


def _unwrap_scalar(value: Any, field: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0]
        raise SchemaError(f"{field} must be a scalar, got shape {value.shape}")
    return value


def _scalar_text(value: Any, field: str) -> str:
    value = _unwrap_scalar(value, field)
    if value is None:
        raise SchemaError(f"{field} is null")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bool_):
        return str(bool(value))
    return str(value)


def _scalar_float(value: Any, field: str) -> float:
    value = _unwrap_scalar(value, field)
    if value is None or isinstance(value, (bool, np.bool_)):
        raise SchemaError(f"{field} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field} must be a finite number") from exc
    if not np.isfinite(out):
        raise SchemaError(f"{field} must be finite")
    return out


def require_finite_int(value: Any, field: str) -> int:
    """Reject non-integers such as 42.9 before any int() truncation."""
    value = _unwrap_scalar(value, field)
    if value is None or isinstance(value, (bool, np.bool_)):
        raise SchemaError(f"{field} must be a finite integer")
    if isinstance(value, (bytes, str)):
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        text = text.strip()
        if text.startswith(("+", "-")) and text[1:].isdigit() or text.isdigit():
            return int(text)
        raise SchemaError(f"{field} must be a finite integer, got {text!r}")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            raise SchemaError(f"{field} must be a finite integer")
        if float(value).is_integer():
            return int(value)
        raise SchemaError(f"{field} must be a finite integer, got {value}")
    raise SchemaError(f"{field} must be a finite integer, got {type(value).__name__}")


def canonical_percentile(value: Any) -> float:
    number = _scalar_float(value, "norm_percentile")
    return float(number)


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    try:
        text = _scalar_text(value, "placeholder").strip().lower()
    except SchemaError:
        return True
    return text in PLACEHOLDERS


def require_int_vector(value: Any, field: str) -> np.ndarray:
    """Validate a time-index-like vector before any int64 truncation."""
    raw = np.asarray(value)
    if raw.ndim > 1:
        raw = raw.reshape(-1)
    if raw.size == 0:
        raise SchemaError(f"{field} is empty")
    if raw.dtype.kind in {"U", "S", "O"}:
        out = np.empty(raw.size, dtype=np.int64)
        for i, item in enumerate(raw.tolist()):
            out[i] = require_finite_int(item, field)
        return out
    as_float = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(as_float).all():
        raise SchemaError(f"{field} must contain only finite integers")
    if np.any(np.mod(as_float, 1.0) != 0.0):
        raise SchemaError(f"{field} must contain only finite integers; fractional hours are refused")
    return as_float.astype(np.int64)


def require_eligible(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim > 1:
        raw = raw.reshape(-1)
    if raw.size == 0:
        raise SchemaError("eligible is empty")
    if raw.dtype == bool or raw.dtype == np.bool_:
        return raw.astype(bool)
    if raw.dtype.kind in {"U", "S", "O"}:
        raise SchemaError("eligible must be strict bool or 0/1, not text")
    as_float = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(as_float).all():
        raise SchemaError("eligible must be strict bool or 0/1")
    if not np.isin(as_float, [0.0, 1.0]).all():
        raise SchemaError("eligible must be strict bool or 0/1")
    return as_float.astype(bool)


def require_raw_counts(value: Any) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(raw).all():
        raise SchemaError("raw_train_counts must be finite and non-negative")
    if np.any(raw < 0.0):
        raise SchemaError("raw_train_counts must be finite and non-negative")
    return raw


def extract_declared(payload: dict[str, Any], field: str) -> Any:
    if field not in payload:
        raise SchemaError(f"missing required field {field!r}; supplementary provenance is required")
    return payload[field]


def canonical_critical(field: str, value: Any) -> Any:
    if field in {"source", "split", "unit", "norm_fitted_on", "checkpoint_path", "config_id", "provenance_kind"}:
        return _scalar_text(value, field).strip()
    if field in {"checkpoint_sha256", "generator_commit"}:
        return _scalar_text(value, field).strip().lower()
    if field == "seed":
        return require_finite_int(value, field)
    if field == "norm_percentile":
        return canonical_percentile(value)
    if field in {"value_scale", "min_val", "gate_threshold"}:
        return _scalar_float(value, field)
    if field == "eligible":
        return require_eligible(value)
    if field == "time_index":
        return require_int_vector(value, field)
    if field == "raw_train_counts":
        return require_raw_counts(value)
    if field in {"prediction", "target"}:
        return np.asarray(value, dtype=np.float64)
    if field == "cell_ids":
        return np.asarray(value).reshape(-1)
    return value


def values_equivalent(field: str, left: Any, right: Any) -> bool:
    a = canonical_critical(field, left)
    b = canonical_critical(field, right)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def merge_payload_and_sidecar(payload: dict[str, Any], sidecar: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize overlapping critical fields and refuse conflicts.

    Equivalent dual representations (for example np.array('ny') vs 'ny',
    int 42 vs JSON 42, '99.9' vs 99.9, eligible True vs 1) are accepted.
    """
    if not sidecar:
        return dict(payload)
    if not isinstance(sidecar, dict):
        raise SchemaError("sidecar is not a JSON object")
    merged = dict(payload)
    conflicts: list[str] = []
    for key, value in sidecar.items():
        if key.startswith("_"):
            continue
        if key in merged:
            if key in CRITICAL_FIELDS or key in merged:
                try:
                    same = values_equivalent(key, merged[key], value) if key in CRITICAL_FIELDS else merged[key] == value
                except SchemaError as exc:
                    raise SchemaError(f"payload/sidecar field {key} is invalid: {exc}") from exc
                if not same:
                    conflicts.append(key)
            continue
        merged[key] = value
    if conflicts:
        raise SchemaError(
            "payload/sidecar critical fields conflict after normalization: " + ", ".join(conflicts)
        )
    return merged


def expected_hours(split: str) -> int:
    if split == "train":
        return N_TRAIN_HOURS
    if split == "val":
        return EXPECTED_VAL_HOURS
    if split == "test":
        return EXPECTED_TEST_HOURS
    raise SchemaError(f"unknown split {split}")


def expected_time_index(split: str) -> np.ndarray:
    if split == "train":
        return np.arange(LABEL_START_TRAIN, LABEL_START_TRAIN + N_TRAIN_HOURS, dtype=np.int64)
    if split == "val":
        return np.arange(EXPECTED_VAL_START, EXPECTED_VAL_START + EXPECTED_VAL_HOURS, dtype=np.int64)
    if split == "test":
        return np.arange(EXPECTED_TEST_START, EXPECTED_TEST_START + EXPECTED_TEST_HOURS, dtype=np.int64)
    raise SchemaError(f"unknown split {split}")


def _validate_recorded_field(payload: dict[str, Any], field: str) -> str:
    raw = extract_declared(payload, field)
    if raw is None or is_placeholder(raw):
        raise SchemaError(f"recorded provenance rejects null/placeholder for {field}")
    text = _scalar_text(raw, field).strip()
    if field == "checkpoint_sha256":
        compact = text.lower()
        if not SHA256_RE.fullmatch(compact):
            raise SchemaError("recorded checkpoint_sha256 must be a 64-character lowercase hex SHA256")
        return compact
    if field == "generator_commit":
        compact = text.lower()
        if not COMMIT_RE.fullmatch(compact):
            raise SchemaError("recorded generator_commit must be a 40-character lowercase hex git SHA")
        return compact
    if field == "norm_fitted_on":
        if text != NORM_FITTED_ON:
            raise SchemaError(f"norm_fitted_on must be the protocol value {NORM_FITTED_ON!r}")
        return text
    return text


def validate_payload(
    payload: dict[str, Any],
    *,
    requested_source: str,
    requested_seed: int,
    requested_split: str,
    allow_synthetic: bool,
    allow_documented_from_bundle: bool = False,
    expected_cells: int | None = None,
    expected_cell_ids: np.ndarray | None = None,
    check_default_time_range: bool = True,
) -> dict[str, Any]:
    """Validate a decoded payload. Never fills missing time_index from shape."""
    required = REQUIRED_TRAIN_FIELDS if requested_split == "train" else REQUIRED_COMMON_FIELDS
    missing = [name for name in required if name not in payload]
    if missing:
        raise SchemaError(
            "legacy or incomplete cache missing required fields "
            f"{missing}; do not guess or reconstruct; supply a complete manifest"
        )

    source = _scalar_text(extract_declared(payload, "source"), "source").strip()
    seed = require_finite_int(extract_declared(payload, "seed"), "seed")
    split = _scalar_text(extract_declared(payload, "split"), "split").strip()
    unit = _scalar_text(extract_declared(payload, "unit"), "unit").strip()
    provenance = _scalar_text(extract_declared(payload, "provenance_kind"), "provenance_kind").strip()

    if source != requested_source:
        raise SchemaError(f"declared source {source!r} does not match request {requested_source!r}")
    if seed != int(requested_seed):
        raise SchemaError(f"declared seed {seed} does not match request {requested_seed}")
    if split != requested_split:
        raise SchemaError(f"declared split {split!r} does not match request {requested_split!r}")
    if unit != UNIT_DC_CLIPPED_COUNT:
        raise SchemaError(f"unit must be {UNIT_DC_CLIPPED_COUNT!r}, got {unit!r}")
    if provenance not in ALLOWED_PROVENANCE:
        raise SchemaError(f"unknown provenance_kind {provenance!r}")
    if provenance == "synthetic" and not allow_synthetic:
        raise SchemaError("synthetic provenance is not accepted without an explicit allow_synthetic flag")
    if provenance == PROVENANCE_DOCUMENTED and not allow_documented_from_bundle:
        raise SchemaError("documented_from_bundle requires explicit opt-in; no fallback")

    percentile = canonical_percentile(extract_declared(payload, "norm_percentile"))
    if percentile != float(NORM_PERCENTILE):
        raise SchemaError(f"norm_percentile must be the protocol value {NORM_PERCENTILE}, got {percentile}")
    fitted_on = _scalar_text(extract_declared(payload, "norm_fitted_on"), "norm_fitted_on").strip()

    checkpoint_path = _scalar_text(extract_declared(payload, "checkpoint_path"), "checkpoint_path").strip()
    checkpoint_sha256 = _scalar_text(extract_declared(payload, "checkpoint_sha256"), "checkpoint_sha256").strip()
    config_id = _scalar_text(extract_declared(payload, "config_id"), "config_id").strip()
    generator_commit = _scalar_text(extract_declared(payload, "generator_commit"), "generator_commit").strip()

    if provenance == "recorded":
        checkpoint_sha256 = _validate_recorded_field(payload, "checkpoint_sha256")
        generator_commit = _validate_recorded_field(payload, "generator_commit")
        fitted_on = _validate_recorded_field(payload, "norm_fitted_on")
        for field, text in (
            ("checkpoint_path", checkpoint_path),
            ("config_id", config_id),
        ):
            if is_placeholder(text):
                raise SchemaError(f"recorded provenance rejects null/placeholder for {field}")
        # Format validity is not empirical proof of historical origin.
        provenance_format_only = True
        time_index_origin = None
        input_protocol_id = None
    elif provenance == PROVENANCE_DOCUMENTED:
        input_protocol_id = _scalar_text(extract_declared(payload, "input_protocol_id"), "input_protocol_id").strip()
        if input_protocol_id != INPUT_PROTOCOL_ID:
            raise SchemaError(
                f"documented_from_bundle input_protocol_id must be {INPUT_PROTOCOL_ID!r}, got {input_protocol_id!r}"
            )
        time_index_origin = _scalar_text(extract_declared(payload, "time_index_origin"), "time_index_origin").strip()
        if time_index_origin != TIME_INDEX_ORIGIN_DOCUMENTED:
            raise SchemaError(
                f"documented time_index_origin must be {TIME_INDEX_ORIGIN_DOCUMENTED!r}, got {time_index_origin!r}"
            )
        form = _scalar_text(extract_declared(payload, "generator_commit_form"), "generator_commit_form").strip()
        if form != DOCUMENTED_PREDICT_SCRIPT_FORM:
            raise SchemaError(
                f"generator_commit_form must be {DOCUMENTED_PREDICT_SCRIPT_FORM!r}, not a padded 40-char commit"
            )
        if generator_commit != DOCUMENTED_PREDICT_SCRIPT_REVISION:
            raise SchemaError(
                f"documented generator_commit must be the short revision {DOCUMENTED_PREDICT_SCRIPT_REVISION!r}, "
                "not an invented full SHA"
            )
        if len(generator_commit) == 40 or COMMIT_RE.fullmatch(generator_commit.lower() or ""):
            raise SchemaError("documented short revision must not be padded or treated as a recorded 40-char commit")
        status = _scalar_text(extract_declared(payload, "checkpoint_sha256_status"), "checkpoint_sha256_status").strip()
        if status != CHECKPOINT_HASH_STATUS:
            raise SchemaError("documented checkpoint hashes are claimed, not rehashed")
        compact = checkpoint_sha256.lower()
        if not SHA256_RE.fullmatch(compact):
            raise SchemaError("claimed checkpoint_sha256 must still be a 64-character lowercase hex SHA256")
        checkpoint_sha256 = compact
        ft_status = _scalar_text(extract_declared(payload, "fine_tune_commit_status"), "fine_tune_commit_status").strip()
        if ft_status != FINE_TUNE_COMMIT_STATUS:
            raise SchemaError("fine_tune_commit must remain unknown; do not invent a recorded FT commit")
        if fitted_on != NORM_FITTED_ON:
            raise SchemaError(f"norm_fitted_on must be the protocol value {NORM_FITTED_ON!r}")
        # Documentation + claimed hashes: not empirical proof of FT weights or full-raw stats.
        provenance_format_only = True
    else:
        provenance_format_only = False
        time_index_origin = None
        input_protocol_id = None

    prediction = np.asarray(payload["prediction"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    time_index = require_int_vector(payload["time_index"], "time_index")
    cell_ids = np.asarray(payload["cell_ids"]).reshape(-1)
    if prediction.ndim != 2 or target.ndim != 2:
        raise SchemaError("prediction and target must be 2-D (hours, cells)")
    if prediction.shape != target.shape:
        raise SchemaError("prediction and target shapes differ")
    if time_index.size != prediction.shape[0]:
        raise SchemaError("time_index length must equal the hour axis; hours are not inferred from shape")
    if cell_ids.size != prediction.shape[1]:
        raise SchemaError("cell_ids length must equal the cell axis")
    if len(set(map(str, cell_ids.tolist()))) != cell_ids.size:
        raise SchemaError("cell_ids must be unique and ordered")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise SchemaError("non-finite prediction or target")

    expected_n = expected_hours(split)
    if prediction.shape[0] != expected_n or time_index.size != expected_n:
        raise SchemaError(
            f"{split} must have exactly {expected_n} labelled hours; "
            f"got T={prediction.shape[0]}. Hours are never padded from shape."
        )
    if expected_cells is not None and prediction.shape[1] != expected_cells:
        raise SchemaError(f"expected {expected_cells} cells, got {prediction.shape[1]}")
    if split in {"train", "val", "test"} and expected_cells is None:
        if prediction.shape[1] != N_VALID_CELLS_EXPECTED and not allow_synthetic:
            raise SchemaError(f"expected {N_VALID_CELLS_EXPECTED} valid DC cells, got {prediction.shape[1]}")

    if check_default_time_range:
        expected = expected_time_index(split)
        if not np.array_equal(time_index, expected):
            raise SchemaError(
                f"{split} time_index is not the protocol default {int(expected[0])}–{int(expected[-1])} "
                f"({VAL_TEST_RANGE_STATUS})"
            )

    if expected_cell_ids is not None and not np.array_equal(cell_ids, expected_cell_ids):
        raise SchemaError("cell_ids or column order does not match the shared DC label order")

    value_scale = _scalar_float(payload["value_scale"], "value_scale")
    min_val = _scalar_float(payload["min_val"], "min_val")
    if value_scale <= 0:
        raise SchemaError("value_scale must be positive")

    eligible = None
    raw_counts = None
    if split == "train":
        eligible = require_eligible(payload["eligible"])
        raw_counts = require_raw_counts(payload["raw_train_counts"])
        threshold = _scalar_float(payload["gate_threshold"], "gate_threshold")
        if threshold != float(GATE):
            raise SchemaError(f"gate_threshold must be {GATE}, got {threshold}")
        if eligible.size != prediction.shape[1] or raw_counts.size != prediction.shape[1]:
            raise SchemaError("eligible and raw_train_counts must match the cell axis")
        gate_recomputed = raw_counts >= float(GATE)
        if not np.array_equal(gate_recomputed, eligible):
            raise SchemaError("eligible mask does not match raw 72h counts recomputed at the frozen gate")
        if not allow_synthetic and int(eligible.sum()) != N_GATE_EXPECTED:
            raise SchemaError(f"expected {N_GATE_EXPECTED} gated cells, got {int(eligible.sum())}")

    manifest = {
        "source": source,
        "seed": seed,
        "split": split,
        "unit": unit,
        "provenance_kind": provenance,
        "provenance_format_valid_only": provenance_format_only,
        "n_hours": int(prediction.shape[0]),
        "n_cells": int(prediction.shape[1]),
        "cell_ids": [_as_cell(v) for v in cell_ids.tolist()],
        "time_index_start": int(time_index[0]),
        "time_index_end": int(time_index[-1]),
        "time_index_sha256": _hash_array(time_index),
        "time_range_status": VAL_TEST_RANGE_STATUS if split != "train" else "frozen_train_5832_5903",
        "value_scale": value_scale,
        "min_val": min_val,
        "norm_percentile": percentile,
        "norm_fitted_on": fitted_on,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256.lower() if provenance == "recorded" else checkpoint_sha256,
        "config_id": config_id,
        "generator_commit": generator_commit.lower() if provenance == "recorded" else generator_commit,
        "prediction_sha256": _hash_array(prediction),
        "target_sha256": _hash_array(target),
        "note": (
            "Recorded fields were format-checked only; this is not empirical proof of historical provenance."
            if provenance == "recorded"
            else (
                "documented_from_bundle: observed arrays/hashes, documented identity/time/unit/norm, "
                "and unknown/claimed FT commit and weight rehash. Not a recorded provenance proof."
                if provenance == PROVENANCE_DOCUMENTED
                else "synthetic fixture; not a real cache."
            )
        ),
    }
    if provenance == PROVENANCE_DOCUMENTED:
        manifest["input_protocol_id"] = input_protocol_id
        manifest["time_index_origin"] = time_index_origin
        manifest["generator_commit_form"] = DOCUMENTED_PREDICT_SCRIPT_FORM
        manifest["checkpoint_sha256_status"] = CHECKPOINT_HASH_STATUS
        manifest["fine_tune_commit_status"] = FINE_TUNE_COMMIT_STATUS
    if eligible is not None:
        manifest["eligible"] = eligible.astype(bool).tolist()
        manifest["n_gated"] = int(eligible.sum())
        manifest["raw_train_counts_sha256"] = _hash_array(raw_counts)
        manifest["gate_recomputed_ok"] = True
        manifest["gate_threshold"] = float(GATE)

    arrays = {
        "prediction": prediction,
        "target": target,
        "time_index": time_index,
        "cell_ids": cell_ids,
        "value_scale": value_scale,
        "min_val": min_val,
        "eligible": eligible,
        "raw_train_counts": raw_counts,
        "source": source,
        "seed": seed,
        "split": split,
        "unit": unit,
        "norm_percentile": percentile,
        "norm_fitted_on": fitted_on,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_id": config_id,
        "generator_commit": manifest["generator_commit"],
        "provenance_kind": provenance,
        "manifest": manifest,
    }
    return arrays


def _as_cell(value: Any) -> int | str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)


def _hash_array(values: np.ndarray) -> str:
    import hashlib

    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def splits_are_contiguous(train_end: int, val_start: int, val_end: int, test_start: int) -> bool:
    return train_end + 1 == val_start and val_end + 1 == test_start
