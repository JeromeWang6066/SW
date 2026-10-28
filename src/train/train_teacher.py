"""Train a source-city or target-only ST-Net (Group 1).

Loss and mask follow CrossTReS ``train_epoch`` for ``STNet`` (3-D output):
squared error on valid cells, then ``mean`` over the batch and ``sum`` over cells.
Metrics are computed on the normalized scale and multiplied by ``(max-min)``
when printed, matching the upstream evaluate printout. No invented numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow ``python -m src.train.train_teacher`` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import prepare_city_windows
from src.models.st_net import STNet
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("sw.train_teacher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ST-Net teacher (Group 1)")
    p.add_argument(
        "--config",
        type=str,
        default="configs/teacher_ny.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load config + set seed, then exit 0",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = full run). For wiring checks.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/<name>.latest.pt if that file exists",
    )
    return p.parse_args(argv)


def masked_mse(out: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CrossTReS STNet loss: ``((out - y_valid) ** 2).mean(0).sum()``."""
    valid = mask.reshape(-1).bool()
    y_valid = y.reshape(y.shape[0], 1, -1)[:, :, valid]
    return ((out - y_valid) ** 2).mean(0).sum()


@torch.no_grad()
def evaluate(
    net: STNet,
    loader: DataLoader,
    mask: torch.Tensor,
    device: torch.device,
    scale: float,
    min_val: float,
) -> tuple[float, float, float, float]:
    """Normalized RMSE, plus denormalized RMSE / MAE / MAPE on valid cells."""
    net.eval()
    se = 0.0
    ae_raw = 0.0
    ape = 0.0
    valid_points = 0.0
    n_valid = float(mask.sum().item())
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = net(x, spatial_mask=mask)
        y_valid = y.reshape(y.shape[0], 1, -1)[:, :, mask.reshape(-1)]
        se += float(((out - y_valid) ** 2).sum().item())
        y_raw = y_valid * scale + min_val
        p_raw = out * scale + min_val
        ae_raw += float((p_raw - y_raw).abs().sum().item())
        denom = torch.clamp(y_raw.abs(), min=1e-6)
        ape += float(((p_raw - y_raw).abs() / denom).sum().item())
        valid_points += x.shape[0] * n_valid
    if valid_points <= 0:
        raise RuntimeError("evaluation saw no valid points")
    rmse_norm = (se / valid_points) ** 0.5
    return rmse_norm, rmse_norm * scale, ae_raw / valid_points, ape / valid_points * 100.0


def _loaders(train_ds, val_ds, test_ds, batch_size: int):
    drop_last = len(train_ds) % batch_size == 1
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    exp = cfg.get("experiment", {})
    cities = cfg.get("cities", {})
    logger.info("Loaded config: %s", args.config)
    logger.info("experiment=%s cities=%s seed=%s", exp, cities, seed)
    logger.info("model: %s", cfg.get("model", {}))

    if args.dry_run:
        logger.info("Dry-run OK — config load + seed only. No training.")
        return 0

    role = exp.get("role")
    if role == "teacher":
        # Held-out val/test. CrossTReS concatenates all source splits; Group 1 does not.
        city = cities.get("source")
        few_shot_days = 0
        concatenate = False
    elif role == "target_only":
        city = cities.get("target")
        few_shot_days = int(cfg.get("splits", {}).get("few_shot_days", 3))
        concatenate = False
    else:
        logger.error("Unsupported experiment.role=%r (expected teacher or target_only)", role)
        return 2
    if not city:
        logger.error("Config is missing the city for role=%s", role)
        return 2

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    paths = cfg.get("paths", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    patience = int(train_cfg.get("early_stop_patience", 5))
    device_name = str(cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.error("Config requests CUDA but torch.cuda.is_available() is False")
        return 2
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    prepared = prepare_city_windows(
        city,
        paths.get("processed_dir", "data/processed"),
        few_shot_days=few_shot_days,
        concatenate_all_for_train=concatenate,
    )
    mask = torch.as_tensor(prepared["mask"], device=device)
    logger.info(
        "city=%s grid=%s valid=%d train=%d val=%d test=%d norm=[%.4f, %.4f] few_shot_days=%s source_concat=%s",
        city,
        prepared["grid_hw"],
        int(mask.sum().item()),
        len(prepared["train"]),
        len(prepared["val"]),
        len(prepared["test"]),
        prepared["min_val"],
        prepared["max_val"],
        few_shot_days,
        concatenate,
    )

    net = STNet(
        in_channels=1,
        channels=int(model_cfg.get("channels", 64)),
        residual_blocks=int(model_cfg.get("residual_blocks", 3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 128)),
        pred_len=int(model_cfg.get("pred_len", 1)),
        spatial_mask=mask,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader, test_loader = _loaders(
        prepared["train"], prepared["val"], prepared["test"], batch_size
    )

    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = paths.get("checkpoint_name", f"{exp.get('id', city)}_stnet.pt")
    ckpt_path = ckpt_dir / ckpt_name
    latest_path = ckpt_dir / f"{Path(ckpt_name).stem}.latest.pt"
    log_dir = Path(paths.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(exp.get("id", city))
    scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    start_epoch = 0
    training_done = False
    steps_done = 0
    wiring = args.max_steps > 0

    if args.resume and latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        training_done = bool(state.get("training_done", False))
        logger.info(
            "resume %s from epoch %d (best epoch %d, stale %d)",
            latest_path,
            start_epoch,
            best_epoch,
            stale,
        )

    def save_latest(epoch: int, done: bool) -> None:
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "stale": stale,
                "training_done": done,
                "max_val": prepared["max_val"],
                "min_val": min_val,
                "mask": prepared["mask"],
                "city": city,
                "config": args.config,
                "seed": seed,
                "run_id": run_id,
            },
            latest_path,
        )

    for epoch in range(start_epoch, epochs):
        if training_done or (not wiring and stale >= patience and best_epoch >= 0):
            training_done = True
            break
        net.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = net(x, spatial_mask=mask)
            loss = masked_mse(out, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            opt.step()
            losses.append(float(loss.item()))
            steps_done += 1
            if wiring and steps_done >= args.max_steps:
                break
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        rmse_n, rmse_d, mae_d, mape_d = evaluate(
            net, val_loader, mask, device, scale, min_val
        )
        logger.info(
            "epoch %d/%d loss %.6f val_rmse %.4f val_mae %.4f val_mape %.4f",
            epoch,
            epochs,
            mean_loss,
            rmse_d,
            mae_d,
            mape_d,
        )
        if wiring:
            logger.info(
                "max-steps=%d wiring check finished after %d steps. Not a Group 1 result.",
                args.max_steps,
                steps_done,
            )
            return 0
        if rmse_n < best_val:
            best_val = rmse_n
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": net.state_dict(),
                    "epoch": epoch,
                    "val_rmse": rmse_d,
                    "val_mae": mae_d,
                    "val_mape": mape_d,
                    "max_val": prepared["max_val"],
                    "min_val": min_val,
                    "mask": prepared["mask"],
                    "city": city,
                    "config": args.config,
                    "seed": seed,
                    "run_id": run_id,
                },
                ckpt_path,
            )
            logger.info("saved best %s", ckpt_path)
        else:
            stale += 1
        save_latest(epoch, done=False)
        if stale >= patience:
            logger.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
            training_done = True
            save_latest(epoch, done=True)
            break
    else:
        if not wiring and start_epoch < epochs:
            save_latest(epochs - 1, done=True)

    if best_epoch < 0 or not ckpt_path.is_file():
        logger.error("no checkpoint written")
        return 2
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    val_rmse_n, val_rmse, val_mae, val_mape = evaluate(
        net, val_loader, mask, device, scale, min_val
    )
    _, test_rmse, test_mae, test_mape = evaluate(
        net, test_loader, mask, device, scale, min_val
    )
    del val_rmse_n
    logger.info(
        "best epoch %d val rmse %.4f mae %.4f mape %.4f | test rmse %.4f mae %.4f mape %.4f",
        best_epoch,
        val_rmse,
        val_mae,
        val_mape,
        test_rmse,
        test_mae,
        test_mape,
    )
    result = {
        "run_id": run_id,
        "status": "complete",
        "city": city,
        "config": args.config,
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "source_held_out": role == "teacher",
        "few_shot_days": few_shot_days,
        "val": {"rmse": val_rmse, "mae": val_mae, "mape": val_mape},
        "test": {"rmse": test_rmse, "mae": test_mae, "mape": test_mape},
    }
    result_path = log_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", result_path)
    logger.info("checkpoint %s", ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
