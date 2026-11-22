import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
        Pixel-wise binary focal loss on every image.
        FL = -alpha * y * (1-p)^gamma * log(p) - (1-alpha) * (1-y) * p^gamma * log(1-p)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.float()  # compute in float32 under AMP
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log(p_t)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class FocalTverskyLoss(nn.Module):
    """
        Focal-Tversky loss, per image, only on anomalous images.
        TI = TP / (TP + alpha*FN + beta*FP),  loss = mean_images (1 - TI)^gamma
        
        Note: equivalent to the Dice loss for alpha = beta = 0.5 and gamma = 1
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float())  # (B, H, W)
        targets = targets.float()

        # Skip normal samples (empty mask), keep only samples and targets with anomalies
        has_anomaly = targets.sum(dim=(1, 2)) > 0
        if not has_anomaly.any():  # all samples are normal
            return probs.sum() * 0.0
        probs = probs[has_anomaly]
        targets = targets[has_anomaly]

        tp = (probs * targets).sum(dim=(1, 2))
        fn = ((1 - probs) * targets).sum(dim=(1, 2))
        fp = (probs * (1 - targets)).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return torch.mean((1.0 - tversky).clamp(min=0.0) ** self.gamma)  # clamp: rounding can give TI > 1


class MultiTaskLoss(nn.Module):
    """
        Total Loss = alpha * Classification BCE + beta * Segmentation Focal + gamma * Segmentation Focal-Tversky
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, focal_alpha=0.75, focal_gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=1.33):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha      # Weight for Classification BCE loss
        self.beta = beta        # Weight for Segmentation Focal loss
        self.gamma = gamma      # Weight for Segmentation Focal-Tversky loss

        self.bce_cls = nn.BCEWithLogitsLoss()
        self.focal_seg = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_seg = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        loss_cls = self.bce_cls(cls_logits, cls_targets)
        loss_seg_focal = self.focal_seg(seg_logits, seg_targets)
        loss_seg_tversky = self.tversky_seg(seg_logits, seg_targets)

        total_loss = (self.alpha * loss_cls) + (self.beta * loss_seg_focal) + (self.gamma * loss_seg_tversky)
        return total_loss, loss_cls, loss_seg_focal, loss_seg_tversky