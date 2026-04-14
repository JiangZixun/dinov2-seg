from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, target: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for class_idx in range(self.num_classes):
            tensor_list.append((target == class_idx).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    @staticmethod
    def _dice_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[Sequence[float]] = None,
        softmax: bool = True,
    ) -> torch.Tensor:
        inputs = torch.softmax(logits, dim=1) if softmax else logits

        mask = target != self.ignore_index
        safe_target = target * mask
        one_hot_target = self._one_hot_encoder(safe_target)
        inputs = inputs * mask.unsqueeze(1)

        if weight is None:
            weight = [1.0] * self.num_classes

        if inputs.size() != one_hot_target.size():
            raise ValueError(f"predict {inputs.size()} & target {one_hot_target.size()} shape do not match")

        loss = 0.0
        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue
            dice = self._dice_loss(inputs[:, class_idx], one_hot_target[:, class_idx])
            loss += dice * weight[class_idx]

        return loss / self.num_classes


class AttentionWeightedCELoss(nn.Module):
    def __init__(self, num_classes: int, lambda_uncertainty: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_uncertainty = lambda_uncertainty
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)

        valid_mask = targets != self.ignore_index
        valid_count = valid_mask.sum().item()
        if valid_count == 0:
            return logits.new_zeros(())

        weight_base = torch.zeros(self.num_classes, device=logits.device)
        entropy_mean = torch.zeros(self.num_classes, device=logits.device)

        for class_idx in range(self.num_classes):
            class_mask = (targets == class_idx) & valid_mask
            class_count = class_mask.sum().item()
            if class_count > 0:
                weight_base[class_idx] = (valid_count - class_count) / valid_count
                entropy_mean[class_idx] = entropy[class_mask].mean()

        weight_combined = weight_base * (1 + self.lambda_uncertainty * entropy_mean)
        weight_map = weight_combined[targets.clamp(0, self.num_classes - 1)] * valid_mask

        ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        return (ce_loss * weight_map).sum() / (valid_mask.sum() + 1e-6)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    lambda_uncertainty: float = 0.5,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce_criterion = AttentionWeightedCELoss(
        num_classes=num_classes,
        lambda_uncertainty=lambda_uncertainty,
        ignore_index=ignore_index,
    )
    dice_criterion = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    ce_loss = ce_criterion(logits, targets)
    dice_loss = dice_criterion(logits, targets, softmax=True)
    total_loss = ce_weight * ce_loss + dice_weight * dice_loss

    return total_loss, {
        "loss_ce": float(ce_loss.detach()),
        "loss_dice": float(dice_loss.detach()),
        "loss_total": float(total_loss.detach()),
    }
