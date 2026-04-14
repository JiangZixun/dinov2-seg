#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a local DINOv2 checkpoint.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the local dinov2 repository.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pretrained" / "dinov2_vitg14_reg4_pretrain.pth",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--entry",
        default="dinov2_vitg14_reg",
        help="torch.hub entry name matching the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="Dummy input size. 518 matches the default DINOv2 hub setup.",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "fp16", "bf16"),
        default="bf16" if torch.cuda.is_available() else "off",
        help="Autocast mode to use during inference.",
    )
    return parser


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_autocast_dtype(amp: str) -> torch.dtype | None:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


def main() -> None:
    args = build_parser().parse_args()
    repo = args.repo.resolve()
    weights = args.weights.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"repo       : {repo}")
    print(f"weights    : {weights}")
    print(f"entry      : {args.entry}")
    print(f"device     : {device}")
    if device.type == "cuda":
        print(f"gpu        : {torch.cuda.get_device_name(device)}")
    print(f"torch      : {torch.__version__}")
    print(f"amp        : {args.amp}")

    model = torch.hub.load(str(repo), args.entry, source="local", pretrained=True, weights=str(weights))
    model.eval().to(device)

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    autocast_dtype = get_autocast_dtype(args.amp)
    use_autocast = device.type == "cuda" and autocast_dtype is not None
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            cls_token = model(dummy)
            features = model.forward_features(dummy)

    patch_tokens = features["x_norm_patchtokens"]
    reg_tokens = features["x_norm_regtokens"]

    print("\nModel Summary")
    print(f"class      : {model.__class__.__name__}")
    print(f"params     : {count_parameters(model):,}")
    print(f"patch size : {model.patch_size}")
    print(f"embed dim  : {model.embed_dim}")
    print(f"depth      : {model.n_blocks}")
    print(f"heads      : {model.num_heads}")
    print(f"registers  : {model.num_register_tokens}")

    print("\nOutput Shapes")
    print(f"cls token  : {tuple(cls_token.shape)}")
    print(f"reg tokens : {tuple(reg_tokens.shape)}")
    print(f"patch toks : {tuple(patch_tokens.shape)}")

    grid = args.image_size // model.patch_size
    print("\nExpected Token Layout")
    print(f"image/grid : {args.image_size} -> {grid} x {grid}")
    print(f"patch count: {grid * grid}")
    print(f"token seq  : 1 cls + {model.num_register_tokens} registers + {grid * grid} patches")

    print("\nTop-level Modules")
    print(f"patch_embed: {model.patch_embed.__class__.__name__}")
    print(f"blocks     : {len(model.blocks)} container(s)")
    first_block = model.blocks[0][-1] if getattr(model, 'chunked_blocks', False) else model.blocks[0]
    print(f"block[0]   : {first_block.__class__.__name__}")
    print(f"norm       : {model.norm.__class__.__name__}")
    print(f"head       : {model.head.__class__.__name__}")


if __name__ == "__main__":
    main()
