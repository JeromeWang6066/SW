"""Optional calendar filtering and three-way metric rows.

Absent ``calendar`` leaves training on the full splits. Present ``calendar``
filters train labels and early-stopping hours, then records all / weekday /
weekend-plus-holiday metrics on the full validation and test windows.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.calendar_mask import apply_calendar, slice_keeps, subset_dataset


def with_calendar(prepared: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    return apply_calendar(prepared, cfg.get("calendar") or None)


def early_stop_dataset(prepared: Dict[str, Any]):
    return prepared.get("val_early", prepared["val"])


def append_slices(
    result: Dict[str, Any],
    net: torch.nn.Module,
    prepared: Dict[str, Any],
    mask: torch.Tensor,
    device: torch.device,
    scale: float,
    min_val: float,
    batch_size: int,
) -> Dict[str, Any]:
    hours = prepared.get("label_hours")
    if not hours:
        return result
    from src.train.train_teacher import evaluate
    for split in ("val", "test"):
        dataset = prepared[split]
        block: Dict[str, Any] = {}
        for name, keep in slice_keeps(hours[split]).items():
            subset = subset_dataset(dataset, keep)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            _, rmse, mae, mape = evaluate(net, loader, mask, device, scale, min_val)
            block[name] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "hours": int(np.asarray(keep).sum()),
            }
        result[f"{split}_slices"] = block
    result["calendar"] = prepared.get("calendar")
    result["train_windows"] = len(prepared["train"])
    result["early_stop_windows"] = len(early_stop_dataset(prepared))
    return result
