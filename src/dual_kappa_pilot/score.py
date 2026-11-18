"""Test scoring after a verified dual-kappa lock.

Order is fail-closed: rematerialize train/val and binding, rematerialize
bundle evidence when documented, compare declared test *file bytes* to
sha256, and only then decode NPZ / run schema / shared truth / official
tables. Selections are never changed. Outputs go to a new exclusive directory.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.dual_kappa_pilot.analysis import (
    cell_error_table,
    negative_prediction_report,
    seed_consistency,
    source_means,
    stop_judgment,
    typology_aggregates,
)
from src.dual_kappa_pilot.protocol import (
    ARMS,
    IDENTITY,
    INPUT_PROTOCOL_ID,
    LABEL_BUDGET,
    PRELOCKED_SEEDS,
    PROVENANCE_DOCUMENTED,
    SOURCES,
    is_identity,
    parse_candidate,
)
from src.dual_kappa_pilot.select import assert_lock_scorable, expected_task_keys
from src.dual_kappa_pilot.snapshot import binding_mismatch, binding_payload
from src.dual_kappa_pilot.solvers import (
    apply_affine,
    fit_cells_dual,
    fit_identity,
    fit_new_k1,
    fit_old_m1,
)
from src.val_select.bundle_adapter import (
    verify_declared_test_file_bytes,
    verify_lock_input_evidence,
    verify_staged_cache_against_evidence,
)
from src.val_select.io_cache import CacheAccess, cache_path, canonical_hash, sha256_file, sidecar_path, write_json
from src.val_select.metrics import gated_mae, score_arrays
from src.val_select.schema import SchemaError, splits_are_contiguous
from src.val_select.score_test import allocate_run_dir
from src.val_select.truth import assert_shared_dc_split, assert_task_split_consistency

SHANGHAI = ZoneInfo("Asia/Shanghai")


def verify_lock_hash(lock: dict[str, Any]) -> str:
    expected = lock.get("lock_hash")
    actual = canonical_hash({k: v for k, v in lock.items() if k != "lock_hash"})
    if expected != actual:
        raise ValueError("selection lock hash mismatch; refusing to score test")
    return actual


def rematerialize_train_val(
    lock: dict[str, Any],
    cache_dir,
    *,
    allow_synthetic: bool,
    allow_documented_from_bundle: bool = False,
    root=None,
) -> tuple[dict, list[str]]:
    errors = binding_mismatch(lock.get("binding") or {}, root=root)
    if "input_hashes" not in lock or not lock["input_hashes"]:
        errors.append("lock is missing train/val input hashes")
        return {}, errors
    cell_ids = None
    if lock["input_hashes"] and lock["input_hashes"][0].get("train_manifest", {}).get("cell_ids") is not None:
        import numpy as np

        cell_ids = np.asarray(lock["input_hashes"][0]["train_manifest"]["cell_ids"])
    refit = CacheAccess(
        cache_dir,
        role="refit",
        allow_synthetic=allow_synthetic,
        allow_documented_from_bundle=allow_documented_from_bundle,
        expected_cell_ids=cell_ids,
    )
    trains: dict = {}
    expected_items = {(item["source"], int(item["seed"])) for item in lock["input_hashes"]}
    required = {(source, int(seed)) for source in lock["sources"] for seed in lock["prelocked_seeds"]}
    if expected_items != required:
        errors.append("locked train/val manifest set does not cover the pre-locked tasks")
    for item in lock["input_hashes"]:
        source = item["source"]
        seed = int(item["seed"])
        train_path = cache_path(Path(cache_dir), source, seed, "train")
        val_path = cache_path(Path(cache_dir), source, seed, "val")
        if not train_path.is_file() or sha256_file(train_path) != item["train_sha256"]:
            errors.append(f"train file hash changed for {source}:{seed}")
            continue
        if not val_path.is_file() or sha256_file(val_path) != item["val_sha256"]:
            errors.append(f"val file hash changed for {source}:{seed}")
            continue
        side_t = sidecar_path(train_path)
        side_v = sidecar_path(val_path)
        now_ts = sha256_file(side_t) if side_t.is_file() else None
        now_vs = sha256_file(side_v) if side_v.is_file() else None
        if now_ts != item.get("train_sidecar_sha256"):
            errors.append(f"train sidecar hash changed for {source}:{seed}")
            continue
        if now_vs != item.get("val_sidecar_sha256"):
            errors.append(f"val sidecar hash changed for {source}:{seed}")
            continue
        try:
            train = refit.load(source, seed, "train")
            val = refit.load(source, seed, "val")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"failed to rematerialize train/val for {source}:{seed}: {exc}")
            continue
        if train["manifest"] != item["train_manifest"]:
            errors.append(f"train manifest rematerialize mismatch for {source}:{seed}")
            continue
        if val["manifest"] != item["val_manifest"]:
            errors.append(f"val manifest rematerialize mismatch for {source}:{seed}")
            continue
        trains[(source, seed)] = train
        trains[(source, seed, "val")] = val
        try:
            assert_task_split_consistency(train, val)
        except SchemaError as exc:
            errors.append(f"same-task train/val identity mismatch for {source}:{seed}: {exc}")
    if not errors:
        try:
            only_trains = [v for k, v in trains.items() if len(k) == 2]
            only_vals = [v for k, v in trains.items() if len(k) == 3]
            assert_shared_dc_split(only_trains, split="train", compare_gate=True)
            assert_shared_dc_split(only_vals, split="val", compare_gate=False)
        except SchemaError as exc:
            errors.append(f"shared DC train/val labels disagree: {exc}")
            trains = {}
    if errors:
        trains = {}
    return trains, errors


def verify_test_bytes_before_decode(lock: dict[str, Any], cache_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Compare test file bytes to lock declarations before any NPZ decode."""
    errors: list[str] = []
    audit: dict[str, Any] = {
        "order": ["file_byte_sha256", "npz_decode_not_started"],
        "note": "sha256_file reads raw bytes; it does not call np.load or decode_npz",
        "compared": [],
        "decoded": False,
    }
    # Documented B2 declarations, if present.
    if (lock.get("input_evidence") or {}).get("declared_test_checksums"):
        ev_errors, ev_audit = verify_declared_test_file_bytes(lock, cache_dir)
        errors.extend(ev_errors)
        audit["bundle_declared"] = ev_audit
        if ev_errors:
            return errors, audit
    # Inventory hashes recorded at lock time (never decoded then either).
    locked_rows = lock.get("test_file_sha256_at_lock") or []
    by_key = {(row["source"], int(row["seed"])): row for row in locked_rows}
    for source in lock.get("sources") or SOURCES:
        for seed in lock.get("prelocked_seeds") or PRELOCKED_SEEDS:
            path = cache_path(Path(cache_dir), source, int(seed), "test")
            rec = {
                "source": source,
                "seed": int(seed),
                "path": str(path),
                "present": path.is_file(),
                "decoded": False,
            }
            if not path.is_file():
                errors.append(f"test file missing for {source}:{seed}; not decoded")
                audit["compared"].append(rec)
                continue
            got = sha256_file(path)
            rec["byte_sha256"] = got
            locked = by_key.get((source, int(seed)))
            if locked and locked.get("sha256"):
                rec["declared_sha256"] = locked["sha256"]
                rec["match"] = got == locked["sha256"]
                if not rec["match"]:
                    errors.append(
                        f"test byte sha256 for {source}:{seed} is {got}, "
                        f"locked inventory {locked['sha256']}; refusing to decode NPZ"
                    )
            audit["compared"].append(rec)
    return errors, audit


def precheck_all_tests(access: CacheAccess, lock: dict[str, Any], trains: dict) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    keys = expected_task_keys(lock["sources"], lock["prelocked_seeds"])
    for key in keys:
        source, seed_s = key.split(":")
        seed = int(seed_s)
        test = access.load(source, seed, "test")
        train = trains[(source, seed)]
        val = trains[(source, seed, "val")]
        assert_task_split_consistency(train, test)
        assert_task_split_consistency(val, test)
        if not splits_are_contiguous(
            int(train["time_index"][-1]),
            int(val["time_index"][0]),
            int(val["time_index"][-1]),
            int(test["time_index"][0]),
        ):
            raise ValueError(f"train/val/test time_index is not a contiguous DC label chain for {key}")
        loaded[key] = test
    assert_shared_dc_split(loaded.values(), split="test", compare_gate=False)
    return loaded


def _fit_arm(arm: str, chosen: Any, train, eligible):
    pred = train["prediction"]
    truth = train["target"]
    if arm == "FT":
        return fit_identity(pred, truth, eligible)
    if arm == "OLD_FIXED":
        return fit_old_m1(pred, truth, eligible)
    if arm == "NEW_FIXED":
        return fit_new_k1(pred, truth, eligible)
    return fit_cells_dual(pred, truth, eligible, parse_candidate(chosen))


def score_test(
    cache_dir,
    lock_path,
    report_path=None,
    *,
    allow_synthetic: bool = False,
    allow_documented_from_bundle: bool = False,
    run_dir=None,
    command: str | None = None,
    root=None,
    test_access_factory: Callable[..., CacheAccess] | None = None,
    i_understand_this_is_exploratory: bool = False,
) -> dict[str, Any]:
    if not i_understand_this_is_exploratory:
        raise ValueError("refusing to score test without the exploratory acknowledgement")

    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = verify_lock_hash(lock)
    assert_lock_scorable(lock)

    trains, bind_errors = rematerialize_train_val(
        lock,
        cache_dir,
        allow_synthetic=allow_synthetic,
        allow_documented_from_bundle=allow_documented_from_bundle,
        root=root,
    )
    bind_errors = list(bind_errors)
    test_byte_audit = None
    if lock.get("provenance_kind") == PROVENANCE_DOCUMENTED or lock.get("input_evidence") or lock.get("input_protocol_id") == INPUT_PROTOCOL_ID:
        bind_errors.extend(
            verify_lock_input_evidence(
                lock,
                cache_dir=cache_dir,
                zip_path=(lock.get("input_evidence") or {}).get("zip_path"),
                root=root,
            )
        )
        if lock.get("input_evidence"):
            bind_errors.extend(verify_staged_cache_against_evidence(cache_dir, lock["input_evidence"]))
    byte_errors, test_byte_audit = verify_test_bytes_before_decode(lock, Path(cache_dir))
    bind_errors.extend(byte_errors)
    if bind_errors:
        raise ValueError(
            "plan/implementation/config or train/val/test-bytes failed precheck; "
            "refusing to decode test: " + "; ".join(bind_errors)
        )

    if report_path is not None and Path(report_path).exists():
        raise FileExistsError(f"refusing to overwrite existing report {report_path}")

    started = datetime.now(SHANGHAI)
    allocated = allocate_run_dir(Path(run_dir) if run_dir is not None else None, default_parent=lock_path.parent)
    factory = test_access_factory or CacheAccess
    access = factory(
        cache_dir,
        role="score_test",
        allow_synthetic=allow_synthetic,
        allow_documented_from_bundle=allow_documented_from_bundle,
        expected_cell_ids=trains[(lock["sources"][0], int(lock["prelocked_seeds"][0]))]["cell_ids"],
    )
    tests = precheck_all_tests(access, lock, trains)

    task_metrics: dict[str, dict[str, dict[str, float]]] = {}
    official: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    keys = expected_task_keys(lock["sources"], lock["prelocked_seeds"])
    for key in keys:
        source, seed_s = key.split(":")
        seed = int(seed_s)
        train = trains[(source, seed)]
        test = tests[key]
        eligible = train["eligible"]
        value_scale = float(test["value_scale"])
        applied_by_arm = {}
        arm_scores = {}
        for arm in ARMS:
            chosen = lock["tasks"][key][arm]["chosen"]
            scale, shift = _fit_arm(arm, chosen, train, eligible)
            applied = apply_affine(test["prediction"], scale, shift)
            applied_by_arm[arm] = applied
            metrics = score_arrays(applied, test["target"], value_scale)
            arm_scores[arm] = {
                "chosen": chosen,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "gated_mae": gated_mae(applied, test["target"], eligible),
                "n_points": metrics["n_points"],
                "negative_predictions": negative_prediction_report(applied),
            }
        official[key] = arm_scores
        task_metrics[key] = {arm: {"mae": arm_scores[arm]["mae"], "rmse": arm_scores[arm]["rmse"]} for arm in ARMS}
        per_ref = {}
        for ref_arm in ("OLD_FIXED", "SINGLE_VAL10"):
            table = cell_error_table(applied_by_arm["DOUBLE_VAL30"], applied_by_arm[ref_arm], test["target"], eligible)
            table["typology_aggregates"] = typology_aggregates(table, lock["typology"])
            per_ref[ref_arm] = table
        analyses[key] = {"versus": per_ref}

    consistency = {
        source: {
            ref_arm: seed_consistency({k: analyses[k]["versus"][ref_arm] for k in keys if k.startswith(source + ":")}, source)
            for ref_arm in ("OLD_FIXED", "SINGLE_VAL10")
        }
        for source in SOURCES
    }
    means = source_means(task_metrics)
    judgment = stop_judgment(task_metrics)

    ended = datetime.now(SHANGHAI)
    report = {
        "protocol_id": lock["protocol_id"],
        "lock_hash": lock_hash,
        "lock_path": str(lock_path),
        "run_dir": str(allocated),
        "label_budget": LABEL_BUDGET,
        "exploratory": True,
        "historical_test_exposed": True,
        "not_a_blind_test": True,
        "arms": list(ARMS),
        "tasks": official,
        "source_means": means,
        "stop_judgment": judgment,
        "per_cell_analysis": analyses,
        "seed_consistency": consistency,
        "test_byte_audit": test_byte_audit,
        "started_cst": started.isoformat(),
        "ended_cst": ended.isoformat(),
        "command": command,
        "no_clip_negative": True,
        "selections_unchanged": True,
    }
    out = Path(report_path) if report_path is not None else allocated / "test_report.json"
    write_json(out, report)
    write_json(
        allocated / "run_audit.json",
        {
            "command": command,
            "started_cst": started.isoformat(),
            "ended_cst": ended.isoformat(),
            "lock_hash": lock_hash,
            "binding": binding_payload(root),
            "test_load_count": access.test_load_count,
            "test_byte_audit": test_byte_audit,
        },
    )
    report["report_path"] = str(out)
    report["report_written"] = True
    return report
