#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cloudSeg.data import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_MINMAX_16,
    DEFAULT_MINMAX_17,
    CloudSegmentationDataset,
    make_transforms,
)
from scripts.cloudSeg.losses import segmentation_loss
from scripts.cloudSeg.model import DinoCloudSegModel, load_adapted_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning entrypoint for the DINOv2 cloud segmentation baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "scripts/cloudSeg/configs/full_finetune_vitg16.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--resume-path", type=Path, default=None, help="Resume from a specific checkpoint path.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto-resume from output_dir/ckpt.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_minmax(path: Path | None) -> Dict[str, object] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_normalization(config: Dict[str, Any]) -> Dict[str, object]:
    mode = config["normalization_mode"]
    if mode != "dataset_minmax":
        return {"mode": mode}

    in_chans = int(config["in_chans"])
    if in_chans == 16:
        return {
            "mode": "dataset_minmax",
            "dataset_min": DEFAULT_MINMAX_16["min"],
            "dataset_max": DEFAULT_MINMAX_16["max"],
        }
    if in_chans == 17:
        return {
            "mode": "dataset_minmax",
            "dataset_min": DEFAULT_MINMAX_17["min"],
            "dataset_max": DEFAULT_MINMAX_17["max"],
        }
    raise ValueError("dataset_minmax normalization requires dataset_minmax_json for this channel count")


def build_dataloader(root: str, *, train: bool, config: Dict[str, Any]) -> DataLoader:
    dataset = CloudSegmentationDataset(
        root=root,
        num_classes=int(config["num_classes"]),
        ignore_index=int(config["ignore_index"]),
        class_names=list(DEFAULT_CLASS_NAMES),
        normalization=build_normalization(config),
        transforms=make_transforms(
            train=train,
            pad_to_size=int(config["pad_to_size"]) if config.get("pad_to_size") is not None else None,
            ignore_index=int(config["ignore_index"]),
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=train,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def get_amp_dtype(amp: str):
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    if not valid.any():
        return 0.0

    ious = []
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx) & valid
        target_mask = (targets == class_idx) & valid
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & target_mask).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


def update_confusion_matrix(
    conf_mat: np.ndarray,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    predictions = logits.argmax(dim=1)
    pred_np = predictions.detach().cpu().numpy().reshape(-1)
    target_np = targets.detach().cpu().numpy().reshape(-1)
    valid = target_np != ignore_index
    pred_np = pred_np[valid]
    target_np = target_np[valid]
    bincount = np.bincount(num_classes * target_np + pred_np, minlength=num_classes * num_classes)
    conf_mat += bincount.reshape(num_classes, num_classes)
    return conf_mat


def summarize_confusion_matrix(conf_mat: np.ndarray, class_names: list[str]) -> Dict[str, Any]:
    total = conf_mat.sum()
    diag = np.diag(conf_mat).astype(np.float64)
    row_sum = conf_mat.sum(axis=1).astype(np.float64)
    col_sum = conf_mat.sum(axis=0).astype(np.float64)

    pixel_acc = float(diag.sum() / total) if total > 0 else 0.0
    class_acc = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    iou = np.divide(diag, row_sum + col_sum - diag, out=np.zeros_like(diag), where=(row_sum + col_sum - diag) > 0)
    f1 = np.divide(2 * diag, row_sum + col_sum, out=np.zeros_like(diag), where=(row_sum + col_sum) > 0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "acc": float(class_acc[idx]),
            "iou": float(iou[idx]),
            "f1": float(f1[idx]),
            "tp": int(diag[idx]),
            "gt_pixels": int(row_sum[idx]),
            "pred_pixels": int(col_sum[idx]),
        }

    return {
        "pixel_acc": pixel_acc,
        "mIoU": float(iou.mean()) if len(iou) > 0 else 0.0,
        "F1score": float(f1.mean()) if len(f1) > 0 else 0.0,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
    }


def maybe_init_wandb(config: Dict[str, Any], output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but not installed in the current environment") from exc

    wandb_cfg = dict(config.get("wandb", {}))
    project = wandb_cfg.get("project")
    name = wandb_cfg.get("name")
    if not project or not name:
        raise ValueError("wandb.project and wandb.name must be set in the JSON config when --wandb is used")

    run = wandb.init(
        project=project,
        name=name,
        config=config,
        dir=str(output_dir),
    )
    return run


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    scheduler_cfg = dict(config.get("scheduler", {}))
    scheduler_name = scheduler_cfg.get("name", "constant")
    total_epochs = int(config["epochs"])
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    base_lr = float(config["lr"])
    warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))
    warmup_epochs = int(round(total_epochs * warmup_ratio))

    if "warmup_epochs" in scheduler_cfg:
        warmup_epochs = int(scheduler_cfg["warmup_epochs"])

    if scheduler_name == "constant":
        return None

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def lr_lambda(current_epoch: int) -> float:
        if total_epochs <= 0:
            return 1.0
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)

        cosine_total = max(total_epochs - warmup_epochs, 1)
        cosine_epoch = max(current_epoch - warmup_epochs, 0)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        return min_ratio + (1.0 - min_ratio) * cosine_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    amp_dtype = get_amp_dtype(config["amp"])
    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    stats = {"loss": 0.0}
    step_count = 0

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype is not None):
            logits = model(inputs)
            loss, _ = segmentation_loss(
                logits,
                labels,
                num_classes=int(config["num_classes"]),
                ignore_index=int(config["ignore_index"]),
                ce_weight=float(config["ce_weight"]),
                dice_weight=float(config["dice_weight"]),
                lambda_uncertainty=float(config["lambda_uncertainty"]),
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            conf_mat = update_confusion_matrix(conf_mat, logits, labels, num_classes, ignore_index)
            running_metrics = summarize_confusion_matrix(conf_mat, list(DEFAULT_CLASS_NAMES))

        stats["loss"] += float(loss.detach())
        step_count += 1
        progress.set_postfix(
            loss=f"{stats['loss'] / step_count:.4f}",
            miou=f"{running_metrics['mIoU']:.4f}",
        )

    metrics = summarize_confusion_matrix(conf_mat, list(DEFAULT_CLASS_NAMES))
    metrics["loss"] = stats["loss"] / max(step_count, 1)
    return metrics


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    amp_dtype = get_amp_dtype(config["amp"])
    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    stats = {"loss": 0.0}
    step_count = 0

    progress = tqdm(loader, desc=f"Eval {epoch}", leave=False)
    for batch in progress:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype is not None):
            logits = model(inputs)
            loss, _ = segmentation_loss(
                logits,
                labels,
                num_classes=int(config["num_classes"]),
                ignore_index=int(config["ignore_index"]),
                ce_weight=float(config["ce_weight"]),
                dice_weight=float(config["dice_weight"]),
                lambda_uncertainty=float(config["lambda_uncertainty"]),
            )

        stats["loss"] += float(loss.detach())
        conf_mat = update_confusion_matrix(conf_mat, logits, labels, num_classes, ignore_index)
        running_metrics = summarize_confusion_matrix(conf_mat, list(DEFAULT_CLASS_NAMES))
        step_count += 1
        progress.set_postfix(
            loss=f"{stats['loss'] / step_count:.4f}",
            miou=f"{running_metrics['mIoU']:.4f}",
        )

    metrics = summarize_confusion_matrix(conf_mat, list(DEFAULT_CLASS_NAMES))
    metrics["loss"] = stats["loss"] / max(step_count, 1)
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler,
    epoch: int,
    ckpt_dir: Path,
    filename: str,
    best_metric: float,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_miou": best_metric,
    }
    torch.save(checkpoint, ckpt_dir / filename)


def save_metrics(metrics_dir: Path, filename: str, payload: Dict[str, Any]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler,
    resume_path: Path | None,
    auto_resume: bool,
    output_dir: Path,
) -> tuple[int, float]:
    ckpt_dir = output_dir / "ckpt"

    candidate = None
    if resume_path:
        candidate = Path(resume_path)
    elif auto_resume and ckpt_dir.exists():
        candidates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if candidates:
            candidate = candidates[-1]

    if candidate is None or not candidate.exists():
        return 1, -math.inf

    checkpoint = torch.load(candidate, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = float(checkpoint.get("best_miou", -math.inf))
    print(f"Resumed from {candidate} at epoch {checkpoint['epoch']}")
    return start_epoch, best_metric


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    metrics_dir = output_dir / "metrics"

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if int(config["image_size"]) % int(config["patch_size"]) != 0:
        raise ValueError("image_size must be divisible by patch_size")
    if config.get("pad_to_size") is not None and int(config["pad_to_size"]) % int(config["patch_size"]) != 0:
        raise ValueError("pad_to_size must be divisible by patch_size")

    train_loader = build_dataloader(config["train_root"], train=True, config=config)
    val_loader = build_dataloader(config["val_root"], train=False, config=config) if config.get("val_root") else None

    model = DinoCloudSegModel(
        arch_name=config["arch"],
        image_size=int(config["pad_to_size"]),
        patch_size=int(config["patch_size"]),
        in_chans=int(config["in_chans"]),
        num_classes=int(config["num_classes"]),
        num_register_tokens=int(config["num_register_tokens"]),
    )
    load_info = load_adapted_pretrained(
        model.backbone,
        str(config["weights"]),
        new_in_chans=int(config["in_chans"]),
        new_patch_size=int(config["patch_size"]),
        image_size=int(config["pad_to_size"]),
        num_register_tokens=int(config["num_register_tokens"]),
    )
    print(
        f"Loaded pretrained weights from {config['weights']}\n"
        f"missing_keys={len(load_info['missing_keys'])} unexpected_keys={len(load_info['unexpected_keys'])}"
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"] == "fp16")
    scheduler = build_scheduler(optimizer, config)
    wandb_run = maybe_init_wandb(config, output_dir, args.wandb)
    start_epoch, best_metric = maybe_resume(
        model,
        optimizer,
        scaler,
        scheduler,
        args.resume_path,
        not args.no_auto_resume,
        output_dir,
    )

    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    save_every_epochs = int(config["save_every_epochs"])
    eval_every_epochs = int(config["eval_every_epochs"])

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[train] epoch={epoch} lr={current_lr:.8f} loss={train_stats['loss']:.4f} "
            f"pixel_acc={train_stats['pixel_acc']:.4f} mIoU={train_stats['mIoU']:.4f} F1={train_stats['F1score']:.4f}"
        )
        save_metrics(
            metrics_dir,
            f"train_epoch_{epoch:03d}.json",
            {"epoch": epoch, "split": "train", "lr": current_lr, **train_stats},
        )

        val_stats = None
        metric = train_stats["mIoU"]
        if val_loader is not None and (epoch % eval_every_epochs == 0 or epoch == int(config["epochs"])):
            val_stats = evaluate(model, val_loader, device, config, epoch)
            metric = val_stats["mIoU"]
            print(
                f"[val]   epoch={epoch} loss={val_stats['loss']:.4f} "
                f"pixel_acc={val_stats['pixel_acc']:.4f} mIoU={val_stats['mIoU']:.4f} F1={val_stats['F1score']:.4f}"
            )
            save_metrics(
                metrics_dir,
                f"val_epoch_{epoch:03d}.json",
                {"epoch": epoch, "split": "val", **val_stats},
            )

        if epoch % save_every_epochs == 0 or epoch == int(config["epochs"]):
            save_checkpoint(
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                ckpt_dir,
                f"checkpoint_epoch_{epoch:03d}.pth",
                best_metric,
            )

        if metric > best_metric:
            best_metric = metric
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, ckpt_dir, "best_model.pth", best_metric)
            best_payload = {
                "epoch": epoch,
                "metric_name": "mIoU",
                "metric_value": best_metric,
                "split": "val" if val_stats is not None else "train",
                "lr": current_lr,
                **(val_stats if val_stats is not None else train_stats),
            }
            save_metrics(metrics_dir, "best.json", best_payload)

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "lr": current_lr,
                "train/loss": train_stats["loss"],
                "train/pixel_acc": train_stats["pixel_acc"],
                "train/mIoU": train_stats["mIoU"],
                "train/F1score": train_stats["F1score"],
                "best/mIoU": best_metric,
            }
            if val_stats is not None:
                payload["val/loss"] = val_stats["loss"]
                payload["val/pixel_acc"] = val_stats["pixel_acc"]
                payload["val/mIoU"] = val_stats["mIoU"]
                payload["val/F1score"] = val_stats["F1score"]
                for class_name, class_metrics in val_stats["per_class"].items():
                    payload[f"val/{class_name}_acc"] = class_metrics["acc"]
                    payload[f"val/{class_name}_iou"] = class_metrics["iou"]
                    payload[f"val/{class_name}_f1"] = class_metrics["f1"]
            wandb_run.log(payload)

        if scheduler is not None:
            scheduler.step()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
