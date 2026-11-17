"""Plan, implementation, and protocol-config snapshots for the dual-kappa pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import src.dual_kappa_pilot.protocol as protocol
from src.val_select.io_cache import canonical_hash

ROOT = Path(__file__).resolve().parents[2]

PLAN_REL = "docs/DUAL_KAPPA_PILOT_PROTOCOL.md"

# Vendor snapshot (byte-identical frozen val_select I/O stack) plus pilot files.
# New files must be added here.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "docs/DUAL_KAPPA_PILOT_PROTOCOL.md",
    "docs/VAL_SELECT_KAPPA_PLAN.md",
    "scripts/20_val_select_kappa.py",
    "scripts/21_dual_kappa_pilot.py",
    "src/__init__.py",
    "src/dual_kappa_pilot/__init__.py",
    "src/dual_kappa_pilot/analysis.py",
    "src/dual_kappa_pilot/protocol.py",
    "src/dual_kappa_pilot/score.py",
    "src/dual_kappa_pilot/select.py",
    "src/dual_kappa_pilot/snapshot.py",
    "src/dual_kappa_pilot/solvers.py",
    "src/dual_kappa_pilot/typology.py",
    "src/val_select/__init__.py",
    "src/val_select/blocks.py",
    "src/val_select/bundle_adapter.py",
    "src/val_select/io_cache.py",
    "src/val_select/metrics.py",
    "src/val_select/protocol.py",
    "src/val_select/schema.py",
    "src/val_select/score_test.py",
    "src/val_select/select.py",
    "src/val_select/snapshot.py",
    "src/val_select/solvers.py",
    "src/val_select/truth.py",
)

TEST_FILES: tuple[str, ...] = ("tests/test_dual_kappa_pilot.py",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def plan_hash(root: Path | None = None) -> dict[str, str]:
    path = (Path(root) if root else ROOT) / PLAN_REL
    return {"path": PLAN_REL, "sha256": sha256_path(path)}


def _file_list(root: Path, rels: tuple[str, ...]) -> list[dict[str, str]]:
    files = []
    for rel in rels:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"implementation file missing: {path}")
        files.append({"path": rel, "sha256": sha256_path(path)})
    return files


def _fingerprint(files: list[dict[str, str]]) -> str:
    return sha256_bytes("\n".join(f"{item['path']} {item['sha256']}" for item in files).encode("utf-8"))


def implementation_snapshot(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root else ROOT
    files = _file_list(base, IMPLEMENTATION_FILES)
    return {"files": files, "fingerprint": _fingerprint(files)}


def test_snapshot(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root else ROOT
    files = _file_list(base, TEST_FILES)
    return {"files": files, "fingerprint": _fingerprint(files)}


def config_snapshot() -> dict[str, Any]:
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "input_protocol_id": protocol.INPUT_PROTOCOL_ID,
        "val_test_range_status": protocol.VAL_TEST_RANGE_STATUS,
        "unit": protocol.UNIT_DC_CLIPPED_COUNT,
        "documented_value_scale": protocol.DOCUMENTED_VALUE_SCALE,
        "time_index_origin_documented": protocol.TIME_INDEX_ORIGIN_DOCUMENTED,
        "sources": list(protocol.SOURCES),
        "prelocked_seeds": list(protocol.PRELOCKED_SEEDS),
        "arms": list(protocol.ARMS),
        "tunable_arms": list(protocol.TUNABLE_ARMS),
        "fixed_arms": list(protocol.FIXED_ARMS),
        "grid_G": [float(x) for x in protocol.G],
        "grid_H": [float(x) for x in protocol.H],
        "budget_nested10": [protocol.canonical_strength_pair(x) for x in protocol.budget_nested10()],
        "budget_expand30": [protocol.canonical_strength_pair(x) for x in protocol.budget_expand30()],
        "identity": protocol.IDENTITY,
        "gate": protocol.GATE,
        "n_train_hours": protocol.N_TRAIN_HOURS,
        "n_valid_cells_expected": protocol.N_VALID_CELLS_EXPECTED,
        "n_gate_expected": protocol.N_GATE_EXPECTED,
        "label_start_train": protocol.LABEL_START_TRAIN,
        "expected_val_start": protocol.EXPECTED_VAL_START,
        "expected_val_hours": protocol.EXPECTED_VAL_HOURS,
        "expected_test_start": protocol.EXPECTED_TEST_START,
        "expected_test_hours": protocol.EXPECTED_TEST_HOURS,
        "tie_eps": protocol.TIE_EPS,
        "near_constant_rel": protocol.NEAR_CONSTANT_REL,
        "stop_mae_rel_min": protocol.STOP_MAE_REL_MIN,
        "stop_rmse_ratio_max": protocol.STOP_RMSE_RATIO_MAX,
        "stop_n_tasks_le_old_min": protocol.STOP_N_TASKS_LE_OLD_MIN,
        "stop_eq_abs": protocol.STOP_EQ_ABS,
        "stop_eq_rel": protocol.STOP_EQ_REL,
        "stop_references": list(protocol.STOP_REFERENCES),
        "budget_note": protocol.BUDGET_NOTE,
        "no_clip_negative": True,
        "nongate_remains_ft": True,
    }


def binding_payload(root: Path | None = None) -> dict[str, Any]:
    return {
        "plan": plan_hash(root),
        "implementation": implementation_snapshot(root),
        "test": test_snapshot(root),
        "config": config_snapshot(),
        "config_hash": canonical_hash(config_snapshot()),
    }


def binding_mismatch(locked: dict[str, Any], current: dict[str, Any] | None = None, root: Path | None = None) -> list[str]:
    now = current or binding_payload(root)
    errors = []
    if locked.get("plan") != now["plan"]:
        errors.append("plan hash changed")
    if locked.get("implementation", {}).get("fingerprint") != now["implementation"]["fingerprint"]:
        errors.append("implementation snapshot changed")
    if locked.get("config_hash") != now["config_hash"] or locked.get("config") != now["config"]:
        errors.append("protocol config changed")
    # Test fingerprint is recorded for audit. A later test-file-only change must
    # not silently rewrite a lock, but scoring should still see the current test
    # snapshot and report a mismatch in the audit (not used to rescue parameters).
    return errors
