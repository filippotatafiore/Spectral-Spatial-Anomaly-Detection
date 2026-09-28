import torch
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_auc_score
import json
import numpy as np

from network.loss import MultiTaskLoss
from utils.globals import (
    TEST_CLASSES, DEVICE, ARTIFACTS_DIR, EPOCHS, LR, WEIGHT_DECAY,
    SOURCE_BALANCE_POWER, SEG_POS_WEIGHT, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, AMP, SEED,
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
    MODEL_VERSION, MODEL_RUN,
)
from network.trans_UNet import TransUNetSentinel2
from data.dataset import make_loader, make_source_weights, NormalImagesDataset, AnomalyImagesDataset
from utils.splits import split_normals, split_anomal


def save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, filename, val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device='cuda'):
    if isinstance(device, str):
        device = torch.device(device)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_auc": best_val_auc,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "loader_generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
        "val_cls_preds": list(val_cls_preds),
        "val_cls_targets": list(val_cls_targets),
        "val_seg_preds": np.concatenate(val_seg_preds, axis=0),
        "val_seg_targets": np.concatenate(val_seg_targets, axis=0),
    }
    torch.save(checkpoint, ARTIFACTS_DIR / filename)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-4, device='cuda'):
    if isinstance(device, str):
        device = torch.device(device)

    # ------ Model
    model = model.to(device)

    # ------ Loss function
    alpha = beta = gamma = 1.0
    criterion = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma, seg_pos_weight=SEG_POS_WEIGHT)

    # ------ Optimizer
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():  # weight decay only on convolutional layers, not on biases or batchnorm
        if not param.requires_grad:
            continue
        (decay_params if param.ndim >= 2 else no_decay_params).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    print(f"Weight decay on {len(decay_params)} tensors (none on {len(no_decay_params)})")

    # ------ Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------ Mixed precision scaler
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------ Training loop
    # Metrics History Tracker
    history = {
        "epoch": [],
        "train_loss": [],
        "train_loss_cls": [],
        "train_loss_seg_bce": [],
        "train_loss_seg_dice": [],
        "val_loss": [],
        "val_loss_cls": [],
        "val_loss_seg_bce": [],
        "val_loss_seg_dice": [],
        "val_auc": [],
        "lr": []
    }

    best_val_auc = 0.0
    best_val_seg_dice = float("inf")
    for epoch in range(1, epochs + 1):
        train_anom_samples = 0
        val_anom_samples = 0

        # ------ Train
        model.train()
        train_loss = 0.0
        train_loss_cls = 0.0
        train_loss_seg_bce = 0.0
        train_loss_seg_dice = 0.0

        for images, cls_targets, seg_targets in train_loader:
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            seg_targets = seg_targets.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                cls_logits, seg_logits = model(images)
                loss, l_cls, l_bce, l_dice = criterion(cls_logits, seg_logits, cls_targets, seg_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() - gamma * l_dice.item()  # exclude dice loss (avoid counting all normal batches with zero dice loss)
            train_loss_cls += l_cls.item()
            train_loss_seg_bce += l_bce.item()
            n_anom = (seg_targets.sum(dim=(1, 2)) > 0).sum().item()  # number of anomalous samples in the batch, actually contributing to the loss
            train_loss_seg_dice += l_dice.item() * n_anom  # weight dice loss accumulator by n_anom
            train_anom_samples += n_anom

        scheduler.step()

        # Compute averages for training epoch
        n_train_batches = len(train_loader)
        train_loss /= n_train_batches
        train_loss_cls /= n_train_batches
        train_loss_seg_bce /= n_train_batches
        train_loss_seg_dice /= max(train_anom_samples, 1)
        train_loss += gamma * train_loss_seg_dice  # add back the dice loss to the total training loss

        # ------ Validation
        model.eval()
        val_loss = 0.0
        val_loss_cls = 0.0
        val_loss_seg_bce = 0.0
        val_loss_seg_dice = 0.0
        val_cls_preds, val_cls_targets = [], []
        val_seg_preds, val_seg_targets = [], []
        
        with torch.no_grad():
            for images, cls_targets, seg_targets in val_loader:
                images = images.to(device)
                cls_targets = cls_targets.to(device)
                seg_targets = seg_targets.to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits = model(images)
                    loss, l_cls, l_bce, l_dice = criterion(cls_logits, seg_logits, cls_targets, seg_targets)

                val_loss += loss.item() - gamma * l_dice.item()  # exclude dice loss (avoid counting all normal batches with zero dice loss)
                val_loss_cls += l_cls.item()
                val_loss_seg_bce += l_bce.item()
                n_anom = (seg_targets.sum(dim=(1, 2)) > 0).sum().item()  # number of anomalous samples in the batch, actually contributing to the loss
                val_loss_seg_dice += l_dice.item() * n_anom  # weight dice loss accumulator by n_anom
                val_anom_samples += n_anom

                val_cls_preds.extend(torch.sigmoid(cls_logits).cpu().numpy())
                val_cls_targets.extend(cls_targets.cpu().numpy())
                val_seg_preds.append(torch.sigmoid(seg_logits).half().cpu().numpy())
                val_seg_targets.append(seg_targets.to(torch.uint8).cpu().numpy())

        # Compute averages for validation epoch
        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_loss_cls /= n_val_batches
        val_loss_seg_bce /= n_val_batches
        val_loss_seg_dice /= max(val_anom_samples, 1)
        val_loss += gamma * val_loss_seg_dice  # add back the dice loss to the total training loss

        try:
            val_auc = float(roc_auc_score(val_cls_targets, val_cls_preds))
        except ValueError:  # Fallback if only one class is present in validation batch
            val_auc = 0.5

        current_lr = optimizer.param_groups[0]['lr']

        # Log to history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_loss_cls"].append(train_loss_cls)
        history["train_loss_seg_bce"].append(train_loss_seg_bce)
        history["train_loss_seg_dice"].append(train_loss_seg_dice)
        history["val_loss"].append(val_loss)
        history["val_loss_cls"].append(val_loss_cls)
        history["val_loss_seg_bce"].append(val_loss_seg_bce)
        history["val_loss_seg_dice"].append(val_loss_seg_dice)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        # Save incrementally to json
        metrics_file = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_results/training_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (Cls: {train_loss_cls:.3f}, Seg BCE: {train_loss_seg_bce:.3f}, Dice: {train_loss_seg_dice:.3f}) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Image AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best_cls.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        if val_loss_seg_dice < best_val_seg_dice:
            best_val_seg_dice = val_loss_seg_dice
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best_seg.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)

        # Save checkpoint every 10 epochs
        if epoch == epochs:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "last.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        elif epoch % 10 == 0:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, f"epoch_{epoch}.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)

    print(f"\nTraining Complete. Best Validation Image AUC-ROC: {best_val_auc:.4f}")


def main():
    # Build model
    model = TransUNetSentinel2(in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, num_heads=NUM_ATT_HEADS, num_layers=NUM_TRANS_LAYERS, dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Split training images (determinitically)
    train_normal_files, val_normal_files, _ = split_normals()
    anomaly_splits = split_anomal()  # {class: {"train": [...], "val": [...], "test": [...]}}
    train_anomaly_triples = [t for cls in TEST_CLASSES for t in anomaly_splits[cls]["train"]]
    val_anomaly_triples   = [t for cls in TEST_CLASSES for t in anomaly_splits[cls]["val"]]

    print(f"Train images: {len(train_normal_files)} normal + {len(train_anomaly_triples)} anomalous")
    print(f"Val images:   {len(val_normal_files)} normal + {len(val_anomaly_triples)} anomalous")

    # Create data loaders for training and validation
    train_weights = make_source_weights(train_normal_files + train_anomaly_triples, SOURCE_BALANCE_POWER)
    if train_weights is not None:
        print(f"Balanced sampling active (power={SOURCE_BALANCE_POWER})")

    train_dataset = ConcatDataset([
        NormalImagesDataset(train_normal_files, CLIP_REFLECTANCE, augment=True),
        AnomalyImagesDataset(train_anomaly_triples, CLIP_REFLECTANCE, augment=True),
    ])
    val_dataset = ConcatDataset([
        NormalImagesDataset(val_normal_files, CLIP_REFLECTANCE),
        AnomalyImagesDataset(val_anomaly_triples, CLIP_REFLECTANCE),
    ])
    train_loader = make_loader(train_dataset, BATCH_IMAGES, NUM_WORKERS, shuffle=True, seed=SEED, weights=train_weights)
    val_loader = make_loader(val_dataset, BATCH_IMAGES, NUM_WORKERS)

    # Train the model
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)


if __name__ == "__main__":
    main()