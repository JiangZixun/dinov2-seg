#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cloudSeg.full_finetune_train import (
    build_dataloader,
    build_scheduler,
    evaluate,
    load_config,
    maybe_init_wandb,
    maybe_resume,
    save_checkpoint,
    save_metrics,
    set_seed,
    train_one_epoch,
)
from scripts.cloudSeg.lora import apply_lora, mark_only_lora_trainable, trainable_parameter_summary
from scripts.cloudSeg.model import DinoCloudSegModel, load_adapted_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning entrypoint for the DINOv2 cloud segmentation baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/mnt/data1/dinov2-seg/scripts/cloudSeg/configs/lora_vitg16.json"),
        help="Path to the JSON config file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--resume-path", type=Path, default=None, help="Resume from a specific checkpoint path.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto-resume from output_dir/ckpt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    metrics_dir = output_dir / "metrics"

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

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

    lora_cfg = dict(config["lora"])
    replaced = apply_lora(
        model.backbone,
        target_suffixes=lora_cfg["target_modules"],
        rank=int(lora_cfg["rank"]),
        alpha=float(lora_cfg["alpha"]),
        dropout=float(lora_cfg["dropout"]),
    )
    mark_only_lora_trainable(
        model,
        train_decode_head=bool(lora_cfg.get("train_decode_head", True)),
        train_patch_embed=bool(lora_cfg.get("train_patch_embed", False)),
        train_norm=bool(lora_cfg.get("train_norm", False)),
    )
    summary = trainable_parameter_summary(model)
    print(
        f"Applied LoRA to {len(replaced)} linear layers\n"
        f"trainable={summary['trainable']:,} frozen={summary['frozen']:,} total={summary['total']:,}"
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
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

    with open(output_dir / "lora_layers.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "replaced_modules": replaced,
                "parameter_summary": summary,
            },
            f,
            indent=2,
        )

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
