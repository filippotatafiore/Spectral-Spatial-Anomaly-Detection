import torch
import torch.nn as nn


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
        intersection = (probs * targets).sum(dim=(1, 2))
        cardinality = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)



class MultiTaskLoss(nn.Module):
    """
        Total Loss = alpha * Classification BCE + beta * Segmentation BCE + gamma * Segmentation Dice
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha      # Weight for Classification BCE loss
        self.beta = beta        # Weight for Segmentation BCE loss
        self.gamma = gamma      # Weight for Segmentation Dice loss

        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        loss_cls = self.bce_cls(cls_logits, cls_targets)
        loss_seg_bce = self.bce_seg(seg_logits, seg_targets)
        loss_seg_dice = self.dice_seg(seg_logits, seg_targets)
        
        total_loss = (self.alpha * loss_cls) + (self.beta * loss_seg_bce) + (self.gamma * loss_seg_dice)
        return total_loss, loss_cls, loss_seg_bce, loss_seg_dice