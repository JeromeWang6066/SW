"""YAML config loading with optional shallow ``_base`` merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config.

    If the file contains ``_base: other.yaml``, merge the base (same directory)
    first, then apply overrides from the child (child wins on key conflict at
    top-level and one nested level for known dict sections).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("_base", None)
    if base_name:
        base_path = path.parent / base_name
        base = load_config(base_path)
        return _merge(base, raw)
    return raw


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out
