"""Few-shot DC student with a frozen source-city ST-Net teacher (Group 2).

The teacher is applied to the target grid directly: conv / LSTM / linear weights
are shared across cells, so no region matching is used. Both networks see the
DC-normalized windows. KD uses the same reduction as the task loss.
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
from src.models.kd import KDWrapper
from src.models.st_net import STNet
from src.train.train_teacher import _loaders, evaluate, masked_mse
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("sw.train_student_kd")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 3-day DC student with frozen-teacher KD")
    p.add_argument("--config", type=str, default="configs/student_kd_ny2dc.yaml")
    p.add_argument("--dry-run", action="store_true", help="Load config and exit 0")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/<name>.latest.pt if that file exists",
    )
    return p.parse_args(argv)


def _load_teacher(ckpt_path: Path, device: torch.device) -> STNet:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint missing: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    state = {k: v for k, v in state.items() if k != "spatial_mask"}
    teacher = STNet(
        in_channels=1,
        channels=64,
        residual_blocks=3,
        lstm_hidden=128,
        pred_len=1,
        spatial_mask=None,
    ).to(device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    return teacher


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    kd = cfg.get("kd", {})
    splits = cfg.get("splits", {})
    exp = cfg.get("experiment", {})
    cities = cfg.get("cities", {})
    logger.info("Loaded config: %s", args.config)
    logger.info("experiment=%s cities=%s seed=%s", exp, cities, seed)
    logger.info(
        "KD: enabled=%s lambda_kd=%s freeze_teacher=%s pearson_multiplier=%s",
        kd.get("enabled"),
        kd.get("lambda_kd"),
        kd.get("freeze_teacher", True),
        kd.get("pearson_multiplier", False),
    )
    logger.info("few_shot_days=%s", splits.get("few_shot_days", 3))

    if kd.get("pearson_multiplier"):
        logger.error("pearson_multiplier=true is Group 3 and is not implemented")
        return 2
    if args.dry_run:
        logger.info("Dry-run OK — config load + seed only. No training.")
        return 0
    if not kd.get("enabled", False) or not kd.get("freeze_teacher", True):
        logger.error("Group 2 requires kd.enabled and freeze_teacher")
        return 2

    role = exp.get("role")
    target = cities.get("target")
    if role != "student_kd" or not target:
        logger.error("expected role=student_kd and a target city, got %s / %s", role, target)
        return 2

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    paths = cfg.get("paths", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    patience = int(train_cfg.get("early_stop_patience", 5))
    lambda_kd = float(kd.get("lambda_kd", 1.0))
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
        "target=%s grid=%s valid=%d train=%d val=%d test=%d norm=[%.4f, %.4f]",
        target,
        prepared["grid_hw"],
        int(mask.sum().item()),
        len(prepared["train"]),
        len(prepared["val"]),
        len(prepared["test"]),
        min_val,
        prepared["max_val"],
    )

    student = STNet(
        in_channels=1,
        channels=int(model_cfg.get("channels", 64)),
        residual_blocks=int(model_cfg.get("residual_blocks", 3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 128)),
        pred_len=int(model_cfg.get("pred_len", 1)),
        spatial_mask=mask,
    ).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    ckpt_file = Path(kd.get("teacher_checkpoint", ""))
    if not ckpt_file.is_absolute():
        ckpt_file = _ROOT / ckpt_file
    teacher = _load_teacher(ckpt_file, device)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    wrap = KDWrapper(student, teacher, lambda_kd=lambda_kd).to(device)
    train_loader, val_loader, test_loader = _loaders(
        prepared["train"], prepared["val"], prepared["test"], batch_size
    )

    z_rmse_n, z_rmse, z_mae, z_mape = evaluate(teacher, val_loader, mask, device, scale, min_val)
    del z_rmse_n
    logger.info(
        "teacher zero-shot val rmse %.4f mae %.4f mape %.4f (denormalized, not a Group 2 result)",
        z_rmse,
        z_mae,
        z_mape,
    )

    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = paths.get("checkpoint_name", f"{exp.get('id', target)}_student.pt")
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
        student.load_state_dict(state["model"])
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
                "model": student.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "stale": stale,
                "training_done": done,
                "city": target,
                "config": args.config,
                "seed": seed,
                "run_id": run_id,
                "lambda_kd": lambda_kd,
                "teacher_checkpoint": str(ckpt_file),
            },
            latest_path,
        )

    for epoch in range(start_epoch, epochs):
        if training_done or (stale >= patience and best_epoch >= 0):
            training_done = True
            break
        wrap.train()
        losses = []
        task_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            student_out, teacher_out = wrap(x, spatial_mask=mask)
            task = masked_mse(student_out, y, mask)
            loss = wrap.total_loss(task, student_out, teacher_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            opt.step()
            losses.append(float(loss.item()))
            task_losses.append(float(task.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_task = float(np.mean(task_losses)) if task_losses else float("nan")
        rmse_n, rmse_d, mae_d, mape_d = evaluate(student, val_loader, mask, device, scale, min_val)
        logger.info(
            "epoch %d/%d loss %.6f task %.6f val_rmse %.4f val_mae %.4f val_mape %.4f",
            epoch,
            epochs,
            mean_loss,
            mean_task,
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
                    "model": student.state_dict(),
                    "epoch": epoch,
                    "val_rmse": rmse_d,
                    "val_mae": mae_d,
                    "val_mape": mape_d,
                    "max_val": prepared["max_val"],
                    "min_val": min_val,
                    "mask": prepared["mask"],
                    "city": target,
                    "config": args.config,
                    "seed": seed,
                    "run_id": run_id,
                    "lambda_kd": lambda_kd,
                    "teacher_checkpoint": str(ckpt_file),
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
    student.load_state_dict(state["model"])
    _, val_rmse, val_mae, val_mape = evaluate(student, val_loader, mask, device, scale, min_val)
    _, test_rmse, test_mae, test_mape = evaluate(student, test_loader, mask, device, scale, min_val)
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
        "lambda_kd": lambda_kd,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "teacher_checkpoint": str(ckpt_file),
        "few_shot_days": few_shot_days,
        "teacher_zeroshot_val": {"rmse": z_rmse, "mae": z_mae, "mape": z_mape},
        "val": {"rmse": val_rmse, "mae": val_mae, "mape": val_mape},
        "test": {"rmse": test_rmse, "mae": test_mae, "mape": test_mape},
    }
    result_path = log_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
