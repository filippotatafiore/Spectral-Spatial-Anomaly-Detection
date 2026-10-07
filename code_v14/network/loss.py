import torch
import torch.nn as nn
import torch.nn.functional as F




def seg_pixel_weights(images, seg_targets, border=0, ignore_nodata=True):
    """
        Per-pixel weights for the segmentation / deviation losses (coarse-label-aware supervision):
            0 on no-data pixels (all bands = 0) and within `border` px of an anomaly-mask boundary (coarse/offset polygons), 1 elsewhere.
        images: raw (B, C, H, W) reflectances, seg_targets: (B, H, W)
    """
    w = (images.abs().sum(dim=1) > 0).float() if ignore_nodata else torch.ones_like(seg_targets, dtype=torch.float32)
    if border > 0:
        t = seg_targets.unsqueeze(1).float()
        k = 2 * border + 1
        dilated = F.max_pool2d(t, k, stride=1, padding=border)
        eroded = -F.max_pool2d(-t, k, stride=1, padding=border)
        w = w * (1.0 - (dilated - eroded).squeeze(1))
    return w


class DiceLoss(nn.Module):
    """
        Dice Loss: difference between a predicted mask and a target mask
        Dice Loss = 1 - (2 * intersection + smooth) / (union + smooth)

        Computes Dice Loss per sample and averages across the batch.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights=None):
        probs = torch.sigmoid(logits)  # (B, H, W)
        if weights is not None:        # ignored pixels (no-data, boundary band) leave both terms
            probs, targets = probs * weights, targets * weights

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

    def forward(self, dev_map, seg_targets, pixel_weights=None):
        # dev_map: (B, h, w), seg_targets: (B, 120, 120), pixel_weights: (B, 120, 120) or None
        dev = dev_map.float()
        frac = F.adaptive_avg_pool2d(seg_targets.unsqueeze(1).float(), dev.shape[-2:]).squeeze(1)  # anomalous fraction per token
        normal, anomal = frac == 0, frac >= 0.5
        if pixel_weights is not None:  # ignore tokens that are mostly no-data / boundary band
            keep = F.adaptive_avg_pool2d(pixel_weights.unsqueeze(1).float(), dev.shape[-2:]).squeeze(1) >= 0.5
            normal, anomal = normal & keep, anomal & keep

        loss_pull = dev[normal].mean() if normal.any() else dev.sum() * 0.0
        loss_push = F.relu(self.margin - dev[anomal]).mean() if anomal.any() else dev.sum() * 0.0
        return loss_pull + loss_push


class MultiTaskLoss(nn.Module):
    """
        Total Loss = alpha * Classification BCE + beta * Segmentation BCE + gamma * Segmentation Dice + delta * Type Loss
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

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets, type_logits=None, type_targets=None, pixel_weights=None):
        loss_cls = self.bce_cls(cls_logits, cls_targets)
        
        pos_weight = torch.tensor(self.seg_pos_weight, device=seg_logits.device)
        if pixel_weights is None:
            loss_seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, pos_weight=pos_weight)
        else:  # mean over the non-ignored pixels only
            bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, pos_weight=pos_weight, reduction="none")
            loss_seg_bce = (bce * pixel_weights).sum() / pixel_weights.sum().clamp(min=1.0)
        loss_seg_dice = self.dice_seg(seg_logits, seg_targets, pixel_weights)

        loss_type = cls_logits.sum() * 0.0
        if type_logits is not None and (type_targets >= 0).any():
            loss_type = self.ce_type(type_logits.float(), type_targets)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_seg_bce) + (self.gamma * loss_seg_dice) + (self.delta * loss_type)
        return total_loss, loss_cls, loss_seg_bce, loss_seg_dice, loss_type