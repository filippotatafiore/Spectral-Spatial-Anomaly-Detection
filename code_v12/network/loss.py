import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
        Dice Loss: difference between a predicted mask and a target mask
        Dice Loss = 1 - (2 * intersection + smooth) / (union + smooth)

        Computes Dice Loss per sample and averages across the batch.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  # (B, H, W)

        # Skip normal samples (empty mask), keep only samples and targets with anomalies
        has_anomaly = targets.sum(dim=(1, 2)) > 0
        if not has_anomaly.any():  # all samples are normal
            return probs.sum() * 0.0
        probs = probs[has_anomaly]
        targets = targets[has_anomaly]

        intersection = (probs * targets).sum(dim=(1, 2))
        cardinality = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)


class DeviationLoss(nn.Module):
    """
        Deviation loss on the token grid:
            normal tokens    (no anomalous pixel)      -> deviation pulled to 0 (close to a normal prototype)
            anomalous tokens (>= 50% anomalous pixels) -> deviation pushed above the margin
        Boundary tokens (0 < anomalous fraction < 0.5) are ignored.
    """

    def __init__(self, margin=0.5):
        super(DeviationLoss, self).__init__()
        self.margin = margin

    def forward(self, dev_map, seg_targets):
        # dev_map: (B, h, w), seg_targets: (B, 120, 120)
        dev = dev_map.float()
        frac = F.adaptive_avg_pool2d(seg_targets.unsqueeze(1).float(), dev.shape[-2:]).squeeze(1)  # anomalous fraction per token
        normal, anomal = frac == 0, frac >= 0.5

        loss_pull = dev[normal].mean() if normal.any() else dev.sum() * 0.0
        loss_push = F.relu(self.margin - dev[anomal]).mean() if anomal.any() else dev.sum() * 0.0
        return loss_pull + loss_push


class MultiTaskLoss(nn.Module):
    """
        Total Loss = alpha * Classification BCE + beta * Segmentation BCE + gamma * Segmentation Dice
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, seg_pos_weight=8.0, delta=0.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha                      # Weight for Classification BCE loss
        self.beta = beta                        # Weight for Segmentation BCE loss
        self.gamma = gamma                      # Weight for Segmentation Dice loss
        self.seg_pos_weight = seg_pos_weight    # up-weights rare anomalous pixels
        self.delta = delta                      # weight of the anomaly-type CE
        
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()
        self.ce_type = nn.CrossEntropyLoss(ignore_index=-1)   # -1 = normal image (no type)

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets, type_logits=None, type_targets=None):
        loss_cls = self.bce_cls(cls_logits, cls_targets)
        
        pos_weight = torch.tensor(self.seg_pos_weight, device=seg_logits.device)
        loss_seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, pos_weight=pos_weight)
        loss_seg_dice = self.dice_seg(seg_logits, seg_targets)

        loss_type = cls_logits.sum() * 0.0
        if type_logits is not None and (type_targets >= 0).any():
            loss_type = self.ce_type(type_logits.float(), type_targets)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_seg_bce) + (self.gamma * loss_seg_dice) + (self.delta * loss_type)
        return total_loss, loss_cls, loss_seg_bce, loss_seg_dice, loss_type