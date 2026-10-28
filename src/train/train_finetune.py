"""G1-C: copy a source ST-Net onto DC and fine-tune with task loss only.

No frozen teacher and no KD term. The source mask is dropped; the DC valid-cell
mask is used. Hyperparameters match the 3-day DC baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import prepare_city_windows
from src.models.st_net import STNet
from src.train.train_teacher import _loaders, evaluate, masked_mse
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("sw.train_finetune")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a source ST-Net on DC (G1-C)")
    p.add_argument("--config", type=str, default="configs/finetune_ny2dc.yaml")
    p.add_argument("--dry-run", action="store_true", help="Load config and exit 0")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/<name>.latest.pt if that file exists",
    )
    return p.parse_args(argv)


def _load_source_weights(net: STNet, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"source checkpoint missing: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    state = {k: v for k, v in state.items() if k != "spatial_mask"}
    missing, unexpected = net.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if k != "spatial_mask"]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={bad_missing} unexpected={unexpected}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    exp = cfg.get("experiment", {})
    cities = cfg.get("cities", {})
    kd = cfg.get("kd", {})
    logger.info("Loaded config: %s", args.config)
    logger.info("experiment=%s cities=%s seed=%s", exp, cities, seed)
    if kd.get("enabled") or kd.get("pearson_multiplier"):
        logger.error("G1-C is task-loss fine-tune; KD and Pearson multiplier stay off")
        return 2
    if args.dry_run:
        logger.info("Dry-run OK — config load + seed only. No training.")
        return 0
    if exp.get("role") != "finetune":
        logger.error("expected role=finetune, got %s", exp.get("role"))
        return 2

    target = cities.get("target")
    if not target:
        logger.error("config is missing the target city")
        return 2

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    paths = cfg.get("paths", {})
    splits = cfg.get("splits", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    patience = int(train_cfg.get("early_stop_patience", 5))
    few_shot_days = int(splits.get("few_shot_days", 3))
    device_name = str(cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.error("Config requests CUDA but torch.cuda.is_available() is False")
        return 2
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    prepared = prepare_city_windows(
        target,
        paths.get("processed_dir", "data/processed"),
        few_shot_days=few_shot_days,
        concatenate_all_for_train=False,
    )
    mask = torch.as_tensor(prepared["mask"], device=device)
    scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    logger.info(
        "target=%s grid=%s valid=%d train=%d val=%d test=%d norm=[%.4f, %.4f] few_shot_days=%s",
        target,
        prepared["grid_hw"],
        int(mask.sum().item()),
        len(prepared["train"]),
        len(prepared["val"]),
        len(prepared["test"]),
        min_val,
        prepared["max_val"],
        few_shot_days,
    )

    net = STNet(
        in_channels=1,
        channels=int(model_cfg.get("channels", 64)),
        residual_blocks=int(model_cfg.get("residual_blocks", 3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 128)),
        pred_len=int(model_cfg.get("pred_len", 1)),
        spatial_mask=mask,
    ).to(device)
    src_path = Path(kd.get("teacher_checkpoint", ""))
    if not src_path.is_absolute():
        src_path = _ROOT / src_path
    _load_source_weights(net, src_path, device)
    logger.info("initialized from %s; all parameters trainable", src_path)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader, test_loader = _loaders(
        prepared["train"], prepared["val"], prepared["test"], batch_size
    )

    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = paths.get("checkpoint_name", f"{exp.get('id', target)}_finetune.pt")
    ckpt_path = ckpt_dir / ckpt_name
    latest_path = ckpt_dir / f"{Path(ckpt_name).stem}.latest.pt"
    log_dir = Path(paths.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(exp.get("id", target))

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    start_epoch = 0
    training_done = False

    if args.resume and latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        training_done = bool(state.get("training_done", False))
        logger.info("resume %s from epoch %d (best epoch %d)", latest_path, start_epoch, best_epoch)

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
                "city": target,
                "source": cities.get("source"),
                "config": args.config,
                "seed": seed,
                "run_id": run_id,
                "init_checkpoint": str(src_path),
            },
            latest_path,
        )

    for epoch in range(start_epoch, epochs):
        if training_done or (stale >= patience and best_epoch >= 0):
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
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        rmse_n, rmse_d, mae_d, mape_d = evaluate(net, val_loader, mask, device, scale, min_val)
        logger.info(
            "epoch %d/%d loss %.6f val_rmse %.4f val_mae %.4f val_mape %.4f",
            epoch,
            epochs,
            mean_loss,
            rmse_d,
            mae_d,
            mape_d,
        )
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
                    "city": target,
                    "source": cities.get("source"),
                    "config": args.config,
                    "seed": seed,
                    "run_id": run_id,
                    "init_checkpoint": str(src_path),
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
        if start_epoch < epochs:
            save_latest(epochs - 1, done=True)

    if best_epoch < 0 or not ckpt_path.is_file():
        logger.error("no checkpoint written")
        return 2
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    _, val_rmse, val_mae, val_mape = evaluate(net, val_loader, mask, device, scale, min_val)
    _, test_rmse, test_mae, test_mape = evaluate(net, test_loader, mask, device, scale, min_val)
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
        "city": target,
        "source": cities.get("source"),
        "config": args.config,
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(src_path),
        "few_shot_days": few_shot_days,
        "kd": False,
        "val": {"rmse": val_rmse, "mae": val_mae, "mape": val_mape},
        "test": {"rmse": test_rmse, "mae": test_mae, "mape": test_mape},
    }
    result_path = log_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
