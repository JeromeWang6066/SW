"""Stitch a weekday model and a weekend model by the label hour's calendar class.

Labor Day is weekend class, so the weekend model scores it. The combined row
sums squared error, absolute error, and positive-flow percentage error. It is
not the average of the two RMSEs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.calendar_mask import is_weekend_or_holiday, label_hours, slice_keeps, subset_dataset
from src.data.dataset import prepare_city_windows
from src.models.st_net import STNet
from src.train.train_finetune import _load_source_weights
from src.train.train_teacher import add_totals, evaluate_totals, metrics_from_totals
from src.utils.config import load_config
from src.utils.seed import set_seed

PAIRS = {
    "ny": (
        "cal-stitch-ny",
        "checkpoints/calendar/finetune_ny_weekday.pt",
        "checkpoints/calendar/finetune_ny_weekend.pt",
    ),
    "chi": (
        "cal-stitch-chi",
        "checkpoints/calendar/finetune_chi_weekday.pt",
        "checkpoints/calendar/finetune_chi_weekend.pt",
    ),
}


def _net(mask: torch.Tensor, device: torch.device, model_cfg: dict) -> STNet:
    return STNet(
        in_channels=1,
        channels=int(model_cfg.get("channels", 64)),
        residual_blocks=int(model_cfg.get("residual_blocks", 3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 128)),
        pred_len=int(model_cfg.get("pred_len", 1)),
        spatial_mask=mask,
    ).to(device)


def _totals(net, dataset, mask, device, scale, min_val, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluate_totals(net, loader, mask, device, scale, min_val)


def _row(totals: dict, scale: float) -> dict:
    _, rmse, mae, mape = metrics_from_totals(totals, scale)
    return {"rmse": rmse, "mae": mae, "mape": mape, "hours": int(totals["hours"])}


def stitch_one(source: str, device: torch.device) -> dict:
    run_id, weekday_path, weekend_path = PAIRS[source]
    cfg = load_config("configs/calendar/finetune_ny_weekday.yaml")
    prepared = prepare_city_windows("dc", cfg["paths"].get("processed_dir", "data/processed"), few_shot_days=0)
    mask = torch.as_tensor(prepared["mask"], device=device)
    scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    weekday_net = _net(mask, device, cfg.get("model", {}))
    weekend_net = _net(mask, device, cfg.get("model", {}))
    _load_source_weights(weekday_net, _ROOT / weekday_path, device)
    _load_source_weights(weekend_net, _ROOT / weekend_path, device)
    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    result = {
        "run_id": run_id,
        "status": "complete",
        "city": "dc",
        "source": source,
        "seed": 42,
        "weekday_checkpoint": weekday_path,
        "weekend_checkpoint": weekend_path,
        "routing": "weekend model on Saturday, Sunday, and listed US federal holidays; weekday model otherwise",
    }
    for split in ("val", "test"):
        dataset = prepared[split]
        hours = label_hours(split)
        keeps = slice_keeps(hours)
        if len(dataset) != len(hours):
            raise RuntimeError(f"{split} windows {len(dataset)} != label hours {len(hours)}")
        weekday_totals = _totals(
            weekday_net,
            subset_dataset(dataset, keeps["weekday"]),
            mask,
            device,
            scale,
            min_val,
            batch_size,
        )
        weekend_totals = _totals(
            weekend_net,
            subset_dataset(dataset, keeps["weekend"]),
            mask,
            device,
            scale,
            min_val,
            batch_size,
        )
        if not is_weekend_or_holiday(hours[keeps["weekend"]]).all():
            raise RuntimeError("weekend slice contains a weekday hour")
        if is_weekend_or_holiday(hours[keeps["weekday"]]).any():
            raise RuntimeError("weekday slice contains a weekend or holiday hour")
        combined = add_totals(weekday_totals, weekend_totals)
        result[split] = _row(combined, scale)
        result[f"{split}_slices"] = {
            "all": _row(combined, scale),
            "weekday": _row(weekday_totals, scale),
            "weekend": _row(weekend_totals, scale),
        }
    return result


def main() -> int:
    set_seed(42)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name != "cuda":
        print("CUDA is not available; stitch eval needs the same device as training")
        return 2
    device = torch.device(device_name)
    log_dir = _ROOT / "logs" / "calendar"
    log_dir.mkdir(parents=True, exist_ok=True)
    for source in ("ny", "chi"):
        result = stitch_one(source, device)
        path = log_dir / f"{result['run_id']}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"{result['run_id']} val mae {result['val']['mae']:.4f} "
            f"weekend mae {result['val_slices']['weekend']['mae']:.4f} "
            f"test mae {result['test']['mae']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
