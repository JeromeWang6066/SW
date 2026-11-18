"""Plan, implementation, and protocol-config snapshots bound into a selection lock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import src.val_select.protocol as protocol
from src.val_select.io_cache import canonical_hash

ROOT = Path(__file__).resolve().parents[2]

PLAN_PATH = ROOT / "docs" / "VAL_SELECT_KAPPA_PLAN.md"

# Every module the lock/score path actually uses. New files must be added here.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "docs/VAL_SELECT_KAPPA_PLAN.md",
    "scripts/20_val_select_kappa.py",
    "src/__init__.py",
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def plan_hash(root: Path | None = None) -> dict[str, str]:
    path = (Path(root) if root else ROOT) / "docs" / "VAL_SELECT_KAPPA_PLAN.md"
    return {"path": "docs/VAL_SELECT_KAPPA_PLAN.md", "sha256": sha256_path(path)}


def implementation_snapshot(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root else ROOT
    files = []
    for rel in IMPLEMENTATION_FILES:
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(f"implementation file missing: {path}")
        files.append({"path": rel, "sha256": sha256_path(path)})
    fingerprint = sha256_bytes(
        "\n".join(f"{item['path']} {item['sha256']}" for item in files).encode("utf-8")
    )
    return {"files": files, "fingerprint": fingerprint}


def config_snapshot() -> dict[str, Any]:
    return {
        "protocol_id": protocol.PROTOCOL_ID,
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
        "candidate_strengths": [float(x) for x in protocol.CANDIDATE_STRENGTHS],
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
        "n_val_blocks": protocol.N_VAL_BLOCKS,
        "tie_eps": protocol.TIE_EPS,
        "near_constant_rel": protocol.NEAR_CONSTANT_REL,
    }


def binding_payload(root: Path | None = None) -> dict[str, Any]:
    return {
        "plan": plan_hash(root),
        "implementation": implementation_snapshot(root),
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
    return errors
