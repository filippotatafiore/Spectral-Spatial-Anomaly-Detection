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



class FocalTverskyLoss(nn.Module):
    """
        Focal Tversky Loss: missed anomalous pixels (FN) cost more than false alarms (FP),
        and every anomalous image counts equally regardless of anomaly size (targets small anomalies).
        TI = TP / (TP + alpha * FP + beta * FN),   Loss = (1 - TI) ^ gamma

        Computed per sample (only samples with anomalies) and averaged across the batch.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float())  # (B, H, W)

        # Skip normal samples (empty mask)
        has_anomaly = targets.sum(dim=(1, 2)) > 0
        if not has_anomaly.any():
            return probs.sum() * 0.0
        probs = probs[has_anomaly]
        targets = targets[has_anomaly]

        tp = (probs * targets).sum(dim=(1, 2))
        fp = (probs * (1.0 - targets)).sum(dim=(1, 2))
        fn = ((1.0 - probs) * targets).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.mean((1.0 - tversky).clamp(min=1e-6) ** self.gamma)



class MultiTaskLoss(nn.Module):
    """
        Total Loss = alpha * Classification BCE + beta * Segmentation BCE + gamma * Segmentation Dice
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, seg_pos_weight=8.0, tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=0.75):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha                      # Weight for Classification BCE loss
        self.beta = beta                        # Weight for Segmentation BCE loss
        self.gamma = gamma                      # Weight for Segmentation Dice loss
        self.seg_pos_weight = seg_pos_weight    # up-weights rare anomalous pixels

        self.bce_cls = nn.BCEWithLogitsLoss()
        self.dice_seg = FocalTverskyLoss(tversky_alpha, tversky_beta, tversky_gamma)

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        loss_cls = self.bce_cls(cls_logits, cls_targets)
        pos_weight = torch.tensor(self.seg_pos_weight, device=seg_logits.device)
        loss_seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, pos_weight=pos_weight)
        loss_seg_dice = self.dice_seg(seg_logits, seg_targets)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_seg_bce) + (self.gamma * loss_seg_dice)
        return total_loss, loss_cls, loss_seg_bce, loss_seg_dice