"""Test scoring entry. May run only after a selection lock is fully verified.

This module is the only allowed reader of test caches. It must not change
selections. Stage reports are produced in order 1 -> 2 -> 3. Scoring writes
into a new exclusive run directory and never overwrites prior outputs or the
lock file. There is no persistent test_touched flag on the lock.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.val_select.bundle_adapter import (
    verify_declared_test_file_bytes,
    verify_lock_input_evidence,
    verify_staged_cache_against_evidence,
)
from src.val_select.io_cache import CacheAccess, cache_path, canonical_hash, sha256_file, sidecar_path, write_json
from src.val_select.metrics import gated_mae, score_arrays
from src.val_select.protocol import (
    ARM_FIXED_STRENGTH,
    ARM_METHOD,
    ARMS,
    IDENTITY,
    LABEL_BUDGET,
    PRELOCKED_SEEDS,
    SOURCES,
    TUNABLE_ARMS,
    canonical_strength,
    is_identity,
)
from src.val_select.schema import SchemaError, splits_are_contiguous
from src.val_select.select import assert_lock_scorable, expected_task_keys
from src.val_select.snapshot import binding_mismatch, binding_payload
from src.val_select.solvers import apply_affine, fit_cells
from src.val_select.truth import assert_shared_dc_split, assert_task_split_consistency

SHANGHAI = ZoneInfo("Asia/Shanghai")


def typed_strength(value: Any):
    if is_identity(value) or value == "identity":
        return IDENTITY
    return float(value)


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
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    """Re-hash and re-read train/val, compare to the lock. Never opens test."""
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
    trains: dict[tuple[str, int], dict[str, Any]] = {}
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
        locked_ts = item.get("train_sidecar_sha256")
        locked_vs = item.get("val_sidecar_sha256")
        now_ts = sha256_file(side_t) if side_t.is_file() else None
        now_vs = sha256_file(side_v) if side_v.is_file() else None
        if now_ts != locked_ts:
            errors.append(f"train sidecar hash changed for {source}:{seed}")
            continue
        if now_vs != locked_vs:
            errors.append(f"val sidecar hash changed for {source}:{seed}")
            continue
        try:
            train = refit.load(source, seed, "train")
            val = refit.load(source, seed, "val")
        except Exception as exc:  # noqa: BLE001 — any rematerialize failure must block test
            errors.append(f"failed to rematerialize train/val for {source}:{seed}: {exc}")
            continue
        if train["manifest"] != item["train_manifest"]:
            errors.append(f"train manifest rematerialize mismatch for {source}:{seed}")
            continue
        if val["manifest"] != item["val_manifest"]:
            errors.append(f"val manifest rematerialize mismatch for {source}:{seed}")
            continue
        trains[(source, seed)] = train
        trains[(source, seed, "val")] = val  # type: ignore[index]
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


def allocate_run_dir(run_dir: Path | None, *, default_parent: Path) -> Path:
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=False)
        return path
    stamp = datetime.now(SHANGHAI).strftime("%Y%m%d-%H%M%S")
    base = Path(default_parent) / "score-runs"
    base.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        path = base / f"run-{stamp}-{suffix:03d}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate an exclusive score run directory")


def write_run_log(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(Path(run_dir) / "run_audit.json", payload)


def precheck_all_tests(access: CacheAccess, lock: dict[str, Any], trains: dict) -> dict[str, Any]:
    """Load and compare every test cache before any official scoring or table write."""
    loaded: dict[str, Any] = {}
    keys = expected_task_keys(lock["sources"], lock["prelocked_seeds"])
    for key in keys:
        source, seed_s = key.split(":")
        seed = int(seed_s)
        test = access.load(source, seed, "test")
        train = trains[(source, seed)]
        val = trains[(source, seed, "val")]  # type: ignore[index]
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
    if lock.get("provenance_kind") == "documented_from_bundle" or lock.get("input_evidence") or lock.get("input_protocol_id"):
        # B1: rematerialize ZIP + current adapter + sidecar mapping. Do not
        # compare lock["input_evidence"] to itself.
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
        # B2: declared test checksums must already be complete (checked above).
        # Compare staged test *file bytes* to declared_sha256 before any NPZ decode.
        byte_errors, test_byte_audit = verify_declared_test_file_bytes(lock, cache_dir)
        bind_errors.extend(byte_errors)
    if bind_errors:
        raise ValueError(
            "plan/implementation/config or train/val changed since lock; "
            "refusing to decode test: " + "; ".join(bind_errors)
        )

    if report_path is not None and Path(report_path).exists():
        raise FileExistsError(f"refusing to overwrite existing report {report_path}")

    started = datetime.now(SHANGHAI)
    allocated = allocate_run_dir(Path(run_dir) if run_dir is not None else None, default_parent=lock_path.parent)
    binding = binding_payload(root)
    audit = {
        "command": command,
        "started_asia_shanghai": started.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "lock_path": str(lock_path),
        "lock_hash": lock_hash,
        "implementation_fingerprint": binding["implementation"]["fingerprint"],
        "plan_sha256": binding["plan"]["sha256"],
        "allow_synthetic": allow_synthetic,
        "note": (
            "This run directory is exclusive and does not overwrite prior reports or the lock. "
            "The lock file is not mutated; there is no persistent test_touched flag. "
            + (
                "allow_synthetic=True: caches are synthetic fixtures, not a real experiment."
                if allow_synthetic
                else "real caches; test results remain exploratory because this test interval has been seen."
            )
        ),
        "test_opened": False,
        "exit_code": None,
    }
    if test_byte_audit is not None:
        audit["test_verification_order"] = [
            "b1_evidence_rematerialize_from_zip",
            "declared_test_checksums_complete",
            "test_file_byte_sha256",
            "schema_and_shared_truth",
            "official_score",
        ]
        audit["test_byte_sha_audit"] = test_byte_audit
        audit["test_byte_sha_before_decode"] = True
    write_run_log(allocated, audit)

    access_ctor = test_access_factory or CacheAccess
    first_ids = None
    if trains:
        first_train = next(v for k, v in trains.items() if len(k) == 2)
        first_ids = first_train["cell_ids"]
    access: CacheAccess = access_ctor(
        cache_dir,
        role="score_test",
        allow_synthetic=allow_synthetic,
        allow_documented_from_bundle=allow_documented_from_bundle,
        expected_cell_ids=first_ids,
    )

    try:
        tests = precheck_all_tests(access, lock, trains)
        stage1_test: dict[str, Any] = {}
        for key, arms in lock["stage1"].items():
            source, seed_s = key.split(":")
            seed = int(seed_s)
            test = tests[key]
            train = trains[(source, seed)]
            eligible = train["eligible"]
            value_scale = float(test["value_scale"])
            arm_scores = {}
            for arm in ARMS:
                method = ARM_METHOD[arm]
                chosen = typed_strength(arms[arm]["chosen"])
                scale, shift = fit_cells(
                    train["prediction"], train["target"], eligible, method, chosen
                )
                applied = apply_affine(test["prediction"], scale, shift)
                metrics = score_arrays(applied, test["target"], value_scale)
                arm_scores[arm] = {
                    "chosen": canonical_strength(chosen),
                    "test": {
                        **metrics,
                        "gated_mae": gated_mae(applied, test["target"], eligible),
                    },
                    "test_sha256": test["_sha256"],
                }
            stage1_test[key] = arm_scores

        audit["test_opened"] = True
        audit["test_load_count"] = access.test_load_count

        stage2_test = {
            "same_as_stage1": True,
            "scores": stage1_test,
            "note": "Stage 2 final test uses the Stage 1 full-val choice and must match Stage 1.",
        }

        stage3_test: dict[str, Any] = {}
        for source in lock["sources"]:
            shared = lock["stage3"][source]
            per_seed = {}
            for seed in lock["prelocked_seeds"]:
                key = f"{source}:{int(seed)}"
                train = trains[(source, int(seed))]
                test = tests[key]
                eligible = train["eligible"]
                value_scale = float(test["value_scale"])
                arm_scores = {}
                for arm in TUNABLE_ARMS:
                    method = ARM_METHOD[arm]
                    chosen = typed_strength(shared["arms"][arm]["chosen"])
                    scale, shift = fit_cells(
                        train["prediction"], train["target"], eligible, method, chosen
                    )
                    applied = apply_affine(test["prediction"], scale, shift)
                    metrics = score_arrays(applied, test["target"], value_scale)
                    independent = stage1_test[key][arm]["test"]["mae"]
                    arm_scores[arm] = {
                        "shared_chosen": canonical_strength(chosen),
                        "independent_chosen": stage1_test[key][arm]["chosen"],
                        "test": {**metrics, "gated_mae": gated_mae(applied, test["target"], eligible)},
                        "paired_mae_delta_shared_minus_independent": metrics["mae"] - independent,
                    }
                per_seed[key] = arm_scores
            stage3_test[source] = {
                "seeds": list(lock["prelocked_seeds"]),
                "shared": {arm: shared["arms"][arm]["chosen"] for arm in TUNABLE_ARMS},
                "per_seed": per_seed,
                "note": "Do not pick the deployed route from these test paired differences.",
            }

        finished = datetime.now(SHANGHAI)
        report = {
            "protocol": "val-select-kappa",
            "label_budget": LABEL_BUDGET,
            "exploratory": True,
            "allow_synthetic": allow_synthetic,
            "lock_hash": lock_hash,
            "lock_path": str(lock_path),
            "lock_unchanged": True,
            "run_dir": str(allocated),
            "started_asia_shanghai": audit["started_asia_shanghai"],
            "finished_asia_shanghai": finished.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "implementation_fingerprint": binding["implementation"]["fingerprint"],
            "plan_sha256": binding["plan"]["sha256"],
            "command": command,
            "sources": list(SOURCES),
            "seeds": list(PRELOCKED_SEEDS),
            "stage1": stage1_test,
            "stage2": stage2_test,
            "stage3": stage3_test,
            "test_load_count": access.test_load_count,
            "note": LABEL_BUDGET,
        }
        out_path = Path(report_path) if report_path is not None else allocated / "test_report.json"
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite existing report {out_path}")
        write_json(out_path, report)
        write_json(allocated / "test_report.json", report) if out_path != allocated / "test_report.json" else None
        audit["report_path"] = str(out_path)
        audit["finished_asia_shanghai"] = report["finished_asia_shanghai"]
        audit["exit_code"] = 0
        audit["test_opened"] = True
        audit["test_load_count"] = access.test_load_count
        write_run_log(allocated, audit)
        report["run_dir"] = str(allocated)
        return report
    except Exception:
        audit["exit_code"] = 1
        audit["test_opened"] = access.test_load_count > 0
        audit["test_load_count"] = access.test_load_count
        audit["finished_asia_shanghai"] = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S %Z")
        write_run_log(allocated, audit)
        raise
