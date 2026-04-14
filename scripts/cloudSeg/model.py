from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.models import vision_transformer as vits


ARCH_CONFIGS = {
    "vit_small": {"embed_dim": 384, "depth": 12, "heads": 6, "ffn_layer": "mlp"},
    "vit_base": {"embed_dim": 768, "depth": 12, "heads": 12, "ffn_layer": "mlp"},
    "vit_large": {"embed_dim": 1024, "depth": 24, "heads": 16, "ffn_layer": "mlp"},
    "vit_giant2": {"embed_dim": 1536, "depth": 40, "heads": 24, "ffn_layer": "swiglufused"},
}


def _num_patches(image_size: int, patch_size: int) -> int:
    if image_size % patch_size != 0:
        raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
    grid = image_size // patch_size
    return grid * grid


def build_dinov2_backbone(
    *,
    arch_name: str,
    image_size: int,
    patch_size: int,
    in_chans: int,
    num_register_tokens: int,
):
    kwargs = dict(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_register_tokens=num_register_tokens,
    )
    if arch_name == "vit_giant2":
        kwargs["ffn_layer"] = "swiglufused"
    return vits.__dict__[arch_name](**kwargs)


def _resize_patch_embed_weight(weight: torch.Tensor, new_in_chans: int, new_patch_size: int) -> torch.Tensor:
    out_chans, old_in_chans, old_h, old_w = weight.shape
    if old_h != new_patch_size or old_w != new_patch_size:
        weight = weight.reshape(out_chans * old_in_chans, 1, old_h, old_w)
        weight = F.interpolate(weight, size=(new_patch_size, new_patch_size), mode="bicubic", align_corners=False)
        weight = weight.reshape(out_chans, old_in_chans, new_patch_size, new_patch_size)

    if old_in_chans == new_in_chans:
        return weight

    if new_in_chans < old_in_chans:
        return weight[:, :new_in_chans]

    expanded = weight.new_empty(out_chans, new_in_chans, new_patch_size, new_patch_size)
    expanded[:, :old_in_chans] = weight
    channel_mean = weight.mean(dim=1, keepdim=True)
    expanded[:, old_in_chans:] = channel_mean.expand(-1, new_in_chans - old_in_chans, -1, -1)
    return expanded


def _resize_pos_embed(
    pos_embed: torch.Tensor,
    new_patch_count: int,
) -> torch.Tensor:
    embed_dim = pos_embed.shape[-1]
    cls_token = pos_embed[:, :1]
    patch_tokens = pos_embed[:, 1:]

    old_patch_count = patch_tokens.shape[1]
    old_size = int(math.sqrt(old_patch_count))
    new_size = int(math.sqrt(new_patch_count))
    if old_size * old_size != old_patch_count or new_size * new_size != new_patch_count:
        raise ValueError("Patch token counts must form square grids for interpolation")

    patch_tokens = patch_tokens.reshape(1, old_size, old_size, embed_dim).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(patch_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, new_patch_count, embed_dim)
    return torch.cat([cls_token, patch_tokens], dim=1)


def load_adapted_pretrained(
    model: nn.Module,
    weights_path: str,
    *,
    new_in_chans: int,
    new_patch_size: int,
    image_size: int,
    num_register_tokens: int,
) -> Dict[str, object]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}

    if "patch_embed.proj.weight" in checkpoint:
        checkpoint["patch_embed.proj.weight"] = _resize_patch_embed_weight(
            checkpoint["patch_embed.proj.weight"],
            new_in_chans=new_in_chans,
            new_patch_size=new_patch_size,
        )

    if "pos_embed" in checkpoint:
        checkpoint["pos_embed"] = _resize_pos_embed(
            checkpoint["pos_embed"],
            new_patch_count=_num_patches(image_size, new_patch_size),
        )

    msg = model.load_state_dict(checkpoint, strict=False)
    return {
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.classifier(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DinoCloudSegModel(nn.Module):
    def __init__(
        self,
        *,
        arch_name: str,
        image_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        num_register_tokens: int,
    ):
        super().__init__()
        self.backbone = build_dinov2_backbone(
            arch_name=arch_name,
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_register_tokens=num_register_tokens,
        )
        self.decode_head = LinearSegmentationHead(self.backbone.embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(pixel_values)
        patch_tokens = features["x_norm_patchtokens"]
        batch_size, patch_count, channels = patch_tokens.shape
        grid_size = int(math.sqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Patch token count {patch_count} is not a square grid")
        patch_tokens = patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        return self.decode_head(patch_tokens, output_size=pixel_values.shape[-2:])
