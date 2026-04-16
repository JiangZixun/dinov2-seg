from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

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


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class FPNSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        out_channels: int = 256,
        fusion_channels: int = 256,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
    ):
        super().__init__()
        if len(scale_factors) < 2:
            raise ValueError("FPN decoder requires at least two feature levels")

        self.scale_factors = tuple(float(scale) for scale in scale_factors)
        self.lateral_convs = nn.ModuleList(nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in self.scale_factors)
        self.smooth_convs = nn.ModuleList(ConvNormAct(out_channels, out_channels, kernel_size=3) for _ in self.scale_factors)
        self.fusion = ConvNormAct(out_channels * len(self.scale_factors), fusion_channels, kernel_size=3)
        self.classifier = nn.Conv2d(fusion_channels, num_classes, kernel_size=1)

    def _rescale_feature(self, feature: torch.Tensor, scale_factor: float) -> torch.Tensor:
        if scale_factor == 1.0:
            return feature
        if scale_factor > 1.0:
            return F.interpolate(feature, scale_factor=scale_factor, mode="bilinear", align_corners=False)

        stride = int(round(1.0 / scale_factor))
        if not math.isclose(scale_factor, 1.0 / stride):
            raise ValueError(f"Unsupported FPN downsample factor: {scale_factor}")
        return F.max_pool2d(feature, kernel_size=stride, stride=stride)

    def forward(self, features: Sequence[torch.Tensor], output_size: Tuple[int, int]) -> torch.Tensor:
        if len(features) != len(self.scale_factors):
            raise ValueError(f"Expected {len(self.scale_factors)} feature levels, got {len(features)}")

        pyramid_inputs = [self._rescale_feature(feat, scale) for feat, scale in zip(features, self.scale_factors)]
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, pyramid_inputs)]

        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level],
                size=laterals[level - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = [conv(feat) for conv, feat in zip(self.smooth_convs, laterals)]
        highest_resolution = outputs[0].shape[-2:]
        fused = torch.cat(
            [
                feat
                if feat.shape[-2:] == highest_resolution
                else F.interpolate(feat, size=highest_resolution, mode="bilinear", align_corners=False)
                for feat in outputs
            ],
            dim=1,
        )
        logits = self.classifier(self.fusion(fused))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def _default_fpn_block_indices(arch_name: str) -> Tuple[int, int, int, int]:
    depth = int(ARCH_CONFIGS[arch_name]["depth"])
    quarter = depth // 4
    return (quarter - 1, 2 * quarter - 1, 3 * quarter - 1, depth - 1)


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
        decoder_type: str = "linear",
        decoder_cfg: Dict[str, object] | None = None,
    ):
        super().__init__()
        self.backbone = build_dinov2_backbone(
            arch_name=arch_name,
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_register_tokens=num_register_tokens,
        )
        self.decoder_type = str(decoder_type).lower()
        decoder_cfg = dict(decoder_cfg or {})

        if self.decoder_type == "linear":
            self.decode_head = LinearSegmentationHead(self.backbone.embed_dim, num_classes)
            self.fpn_block_indices = None
        elif self.decoder_type == "fpn":
            self.fpn_block_indices = tuple(
                int(index) for index in decoder_cfg.get("block_indices", _default_fpn_block_indices(arch_name))
            )
            scale_factors = tuple(float(scale) for scale in decoder_cfg.get("scale_factors", (4.0, 2.0, 1.0, 0.5)))
            if len(self.fpn_block_indices) != len(scale_factors):
                raise ValueError("decoder.block_indices and decoder.scale_factors must have the same length")
            self.decode_head = FPNSegmentationHead(
                self.backbone.embed_dim,
                num_classes,
                out_channels=int(decoder_cfg.get("out_channels", 256)),
                fusion_channels=int(decoder_cfg.get("fusion_channels", 256)),
                scale_factors=scale_factors,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.decoder_type == "linear":
            features = self.backbone.forward_features(pixel_values)
            patch_tokens = features["x_norm_patchtokens"]
            batch_size, patch_count, channels = patch_tokens.shape
            grid_size = int(math.sqrt(patch_count))
            if grid_size * grid_size != patch_count:
                raise ValueError(f"Patch token count {patch_count} is not a square grid")
            patch_tokens = patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
            return self.decode_head(patch_tokens, output_size=pixel_values.shape[-2:])

        pyramid_features = self.backbone.get_intermediate_layers(
            pixel_values,
            n=self.fpn_block_indices,
            reshape=True,
        )
        return self.decode_head(pyramid_features, output_size=pixel_values.shape[-2:])
