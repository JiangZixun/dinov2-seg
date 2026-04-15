from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    target_suffixes = tuple(target_suffixes)
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short_name = module_name.split(".")[-1]
        if short_name not in target_suffixes:
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    return replaced


def mark_only_lora_trainable(
    model: nn.Module,
    *,
    train_decode_head: bool = True,
    train_patch_embed: bool = False,
    train_norm: bool = False,
) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    if train_decode_head:
        for param in model.decode_head.parameters():
            param.requires_grad = True

    if train_patch_embed:
        for param in model.backbone.patch_embed.parameters():
            param.requires_grad = True

    if train_norm:
        for name, param in model.backbone.named_parameters():
            if ".norm" in name or name.startswith("norm"):
                param.requires_grad = True


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
