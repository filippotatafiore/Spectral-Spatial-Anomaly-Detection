import torch
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_auc_score
import json
import numpy as np

from network.loss import MultiTaskLoss, DeviationLoss, seg_pixel_weights
from utils.globals import (
    TEST_CLASSES, DEVICE, ARTIFACTS_DIR, EPOCHS, LR, WEIGHT_DECAY,
    SOURCE_BALANCE_POWER, SEG_POS_WEIGHT, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, AMP, SEED,
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
    BACKBONE, ENCODER_LR_MULT, WARMUP_EPOCHS, TYPE_LOSS_WEIGHT,
    MODEL_VERSION, MODEL_RUN,
    DEV_MARGIN, DEV_LOSS_WEIGHT,
    SOURCE_WEIGHT_MULT, EARLY_STOP_PATIENCE, PROTO_RESEED_MIN_RATE,
    CRAT_PROB, SEG_IGNORE_BORDER, SEG_IGNORE_NODATA, HOLDOUT_CLASS,
)
from network.trans_UNet import build_model
from data.dataset import make_loader, make_source_weights, NormalImagesDataset, AnomalyImagesDataset, _load_image, build_donor_pool
from utils.splits import split_normals, split_anomal, subsample


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


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-4, device='cuda', val_hard_mask=None):
    if isinstance(device, str):
        device = torch.device(device)

    # ------ Model
    model = model.to(device)

    # ------ Loss function
    alpha = beta = gamma = 1.0
    criterion = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma, seg_pos_weight=SEG_POS_WEIGHT, delta=TYPE_LOSS_WEIGHT)
    dev_criterion = DeviationLoss(margin=DEV_MARGIN)

    # ------ Optimizer (weight decay only on conv/linear weights; lower LR for the pretrained ResNet)
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_pretrained = name.startswith("enc.backbone.")
        is_decay = param.ndim >= 2
        param_groups.setdefault((is_pretrained, is_decay), []).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": lr * (ENCODER_LR_MULT if is_pretrained else 1.0), "weight_decay": weight_decay if is_decay else 0.0}
            for (is_pretrained, is_decay), params in sorted(param_groups.items())
        ],
        lr=lr,
    )
    print({f"pretrained={p}, decay={d}": len(v) for (p, d), v in sorted(param_groups.items())})

    # ------ Learning rate scheduler (linear warmup + cosine annealing)
    if WARMUP_EPOCHS > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS),
            ],
            milestones=[WARMUP_EPOCHS],
        )
    else:
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
        "train_loss_type": [],
        "train_loss_dev": [],
        "val_loss": [],
        "val_loss_cls": [],
        "val_loss_seg_bce": [],
        "val_loss_seg_dice": [],
        "val_loss_type": [],
        "val_loss_dev": [],
        "val_auc": [],
        "val_auc_hard": [],
        "val_joint": [],
        "val_dev_auc": [],
        "proto_active": [],
        "proto_reseeds": [],
        "val_type_acc": [],
        "lr": []
    }

    best_val_auc = 0.0
    best_val_seg_dice = float("inf")
    best_val_joint, epochs_no_improve = -1.0, 0
    for epoch in range(1, epochs + 1):
        train_anom_samples = 0
        val_anom_samples = 0

        # ------ Train
        model.train()
        train_loss = 0.0
        train_loss_cls = 0.0
        train_loss_seg_bce = 0.0
        train_loss_seg_dice = 0.0
        train_loss_type = 0.0
        train_loss_dev = 0.0

        # Prototype EMA step coupled to the LR schedule (the memory moves as fast as the features it tracks)
        lr_frac = optimizer.param_groups[0]["lr"] / lr     # group 0 = non-pretrained params (base LR)
        proto_rate = 1.0 if epoch <= WARMUP_EPOCHS else lr_frac
        proto_reseed = proto_rate >= PROTO_RESEED_MIN_RATE
        proto_reseeds = 0

        for images, cls_targets, seg_targets, type_targets in train_loader:
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            seg_targets = seg_targets.to(device)
            type_targets = type_targets.to(device)

            pw = seg_pixel_weights(images, seg_targets, SEG_IGNORE_BORDER, SEG_IGNORE_NODATA) if (SEG_IGNORE_BORDER > 0 or SEG_IGNORE_NODATA) else None   # None: unweighted losses

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                cls_logits, seg_logits, type_logits, aux = model(images)
                loss, l_cls, l_bce, l_dice, l_type = criterion(cls_logits, seg_logits, cls_targets, seg_targets, type_logits, type_targets, pixel_weights=pw)
                l_dev = dev_criterion(aux["dev_map"], seg_targets, pw) if model.use_dev else loss * 0.0
                loss = loss + DEV_LOSS_WEIGHT * l_dev

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update of the normality memory with the tokens of the normal images only
            if model.use_dev:
                proto_reseeds += model.prototypes.update(aux["z"].detach()[cls_targets == 0], rate=proto_rate, reseed=proto_reseed)

            train_loss += loss.item() - gamma * l_dice.item()  # exclude dice loss (avoid counting all normal batches with zero dice loss)
            train_loss_cls += l_cls.item()
            train_loss_seg_bce += l_bce.item()
            n_anom = (seg_targets.sum(dim=(1, 2)) > 0).sum().item()  # number of anomalous samples in the batch, actually contributing to the loss
            train_loss_seg_dice += l_dice.item() * n_anom  # weight dice loss accumulator by n_anom
            train_anom_samples += n_anom
            train_loss_type += l_type.item()
            train_loss_dev += l_dev.item()

        current_lr = optimizer.param_groups[0]['lr']  # log LR

        scheduler.step()

        # Compute averages for training epoch
        n_train_batches = len(train_loader)
        train_loss /= n_train_batches
        train_loss_cls /= n_train_batches
        train_loss_seg_bce /= n_train_batches
        train_loss_seg_dice /= max(train_anom_samples, 1)
        train_loss_type /= n_train_batches
        train_loss += gamma * train_loss_seg_dice  # add back the dice loss to the total training loss
        train_loss_dev /= n_train_batches

        # ------ Validation
        model.eval()
        val_loss = 0.0
        val_loss_cls = 0.0
        val_loss_seg_bce = 0.0
        val_loss_seg_dice = 0.0
        val_loss_type, type_correct, type_total = 0.0, 0, 0
        val_cls_preds, val_cls_targets = [], []
        val_seg_preds, val_seg_targets = [], []
        val_loss_dev, val_dev_preds = 0.0, []
        
        with torch.no_grad():
            for images, cls_targets, seg_targets, type_targets in val_loader:
                images = images.to(device)
                cls_targets = cls_targets.to(device)
                seg_targets = seg_targets.to(device)
                type_targets = type_targets.to(device)

                pw = seg_pixel_weights(images, seg_targets, SEG_IGNORE_BORDER, SEG_IGNORE_NODATA) if (SEG_IGNORE_BORDER > 0 or SEG_IGNORE_NODATA) else None   # None: unweighted v11/v12 losses

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits, type_logits, aux = model(images)
                    loss, l_cls, l_bce, l_dice, l_type = criterion(cls_logits, seg_logits, cls_targets, seg_targets, type_logits, type_targets, pixel_weights=pw)
                    l_dev = dev_criterion(aux["dev_map"], seg_targets, pw) if model.use_dev else loss * 0.0
                    loss = loss + DEV_LOSS_WEIGHT * l_dev

                val_loss += loss.item() - gamma * l_dice.item()  # exclude dice loss (avoid counting all normal batches with zero dice loss)
                val_loss_cls += l_cls.item()
                val_loss_seg_bce += l_bce.item()
                n_anom = (seg_targets.sum(dim=(1, 2)) > 0).sum().item()  # number of anomalous samples in the batch, actually contributing to the loss
                val_loss_seg_dice += l_dice.item() * n_anom  # weight dice loss accumulator by n_anom
                val_anom_samples += n_anom
                val_loss_type += l_type.item()
                m = type_targets >= 0
                type_correct += (type_logits.argmax(dim=1)[m] == type_targets[m]).sum().item()
                type_total += m.sum().item()

                val_cls_preds.extend(torch.sigmoid(cls_logits).cpu().numpy())
                val_cls_targets.extend(cls_targets.cpu().numpy())
                val_seg_preds.append(torch.sigmoid(seg_logits).half().cpu().numpy())
                val_seg_targets.append(seg_targets.to(torch.uint8).cpu().numpy())

                val_loss_dev += l_dev.item()
                val_dev_preds.extend(aux["dev_score"].float().cpu().numpy())

        # Compute averages for validation epoch
        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_loss_cls /= n_val_batches
        val_loss_seg_bce /= n_val_batches
        val_loss_seg_dice /= max(val_anom_samples, 1)
        val_loss_type /= n_val_batches
        val_type_acc = type_correct / max(type_total, 1)
        val_loss += gamma * val_loss_seg_dice  # add back the dice loss to the total training loss
        val_loss_dev /= n_val_batches

        try:
            val_auc = float(roc_auc_score(val_cls_targets, val_cls_preds))
        except ValueError:  # Fallback if only one class is present in validation batch
            val_auc = 0.5

        try:
            val_dev_auc = float(roc_auc_score(val_cls_targets, val_dev_preds))   # deviation branch alone
        except ValueError:
            val_dev_auc = 0.5

        # AUC against the region-matched normals only (ben-ge normals are trivially easy) + joint model-selection score
        if val_hard_mask is not None:
            val_auc_hard = float(roc_auc_score(np.asarray(val_cls_targets)[val_hard_mask], np.asarray(val_cls_preds)[val_hard_mask]))
        else:
            val_auc_hard = val_auc
        val_joint = 0.5 * (val_auc_hard + (1.0 - val_loss_seg_dice))

        # Log to history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_loss_cls"].append(train_loss_cls)
        history["train_loss_seg_bce"].append(train_loss_seg_bce)
        history["train_loss_seg_dice"].append(train_loss_seg_dice)
        history["train_loss_type"].append(train_loss_type)
        history["train_loss_dev"].append(train_loss_dev)
        history["val_loss"].append(val_loss)
        history["val_loss_cls"].append(val_loss_cls)
        history["val_loss_seg_bce"].append(val_loss_seg_bce)
        history["val_loss_seg_dice"].append(val_loss_seg_dice)
        history["val_loss_type"].append(val_loss_type)
        history["val_loss_dev"].append(val_loss_dev)
        history["val_auc"].append(val_auc)
        history["val_dev_auc"].append(val_dev_auc)
        history["val_auc_hard"].append(val_auc_hard)
        history["val_joint"].append(val_joint)
        history["proto_active"].append(int((model.prototypes.usage > 0.1).sum().item()) if model.use_dev else 0)
        history["proto_reseeds"].append(proto_reseeds)
        history["val_type_acc"].append(val_type_acc)
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
            f"LR: {current_lr:.2e} | "
            f"Dev: {train_loss_dev:.3f} | Val Dev AUC: {val_dev_auc:.4f} | "
            f"Val AUC hard: {val_auc_hard:.4f} | Joint: {val_joint:.4f} | "
            f"Protos: {history['proto_active'][-1]} active, {proto_reseeds} reseeded, rate {proto_rate:.3f}"
        )

        # Checkpoint saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best_cls.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        if val_loss_seg_dice < best_val_seg_dice:
            best_val_seg_dice = val_loss_seg_dice
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best_seg.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        if val_joint > best_val_joint:  # best cls-seg combined score
            best_val_joint, epochs_no_improve = val_joint, 0
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best_joint.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        else:
            epochs_no_improve += 1

        # Save checkpoint every 10 epochs
        if epoch == epochs:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "last.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
        elif epoch % 10 == 0:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, f"epoch_{epoch}.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)


        # Optional early stopping on the joint score
        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE and epoch < epochs:
            print(f"Early stopping at epoch {epoch}: no joint-score improvement for {EARLY_STOP_PATIENCE} epochs")
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "last.pth", val_cls_preds, val_cls_targets, val_seg_preds, val_seg_targets, device=device)
            break

    print(f"\nTraining Complete. Best Validation Image AUC-ROC: {best_val_auc:.4f}")


def main():
    # Build model
    model = build_model(pretrained=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Split training images (determinitically)
    train_normal_files, val_normal_files, _ = split_normals()
    anomaly_splits = split_anomal()  # {class: {"train": [...], "val": [...], "test": [...]}}
    seen_classes = [c for c in TEST_CLASSES if c != HOLDOUT_CLASS]   # open-set runs: the held-out type is never seen in train/val
    train_anomaly_triples = [t for cls in seen_classes for t in anomaly_splits[cls]["train"]]
    val_anomaly_triples   = [t for cls in seen_classes for t in anomaly_splits[cls]["val"]]
    if HOLDOUT_CLASS:
        print(f"Open-set run: '{HOLDOUT_CLASS}' anomalies held out from train/val")

    print(f"Train images: {len(train_normal_files)} normal + {len(train_anomaly_triples)} anomalous")
    print(f"Val images:   {len(val_normal_files)} normal + {len(val_anomaly_triples)} anomalous")

    # Normalize with the band statistics of our (L2A) training images instead of the SSL4EO (L1C) ones (buffers -> saved in the checkpoint)
    if BACKBONE in ("cnn", "resnet50_softcon"):
        s = torch.zeros(IN_CHANNELS, dtype=torch.float64)
        s2 = torch.zeros(IN_CHANNELS, dtype=torch.float64)
        n = 0
        for f in subsample(train_normal_files + [t[0] for t in train_anomaly_triples], 2000):
            x = _load_image(f, CLIP_REFLECTANCE).double()
            s += x.sum(dim=(1, 2))
            s2 += (x ** 2).sum(dim=(1, 2))
            n += x.shape[1] * x.shape[2]
        mean = s / n
        std = (s2 / n - mean ** 2).sqrt()
        model.enc.band_mean.copy_(mean.float().view(1, -1, 1, 1))
        model.enc.band_std.copy_(std.float().view(1, -1, 1, 1))
        print(f"Band mean: {mean.numpy().round(4)}\nBand std:  {std.numpy().round(4)}")

    # Create data loaders for training and validation
    train_weights = make_source_weights(train_normal_files + train_anomaly_triples, SOURCE_BALANCE_POWER, SOURCE_WEIGHT_MULT)
    if train_weights is not None:
        print(f"Balanced sampling active (power={SOURCE_BALANCE_POWER}, multipliers={SOURCE_WEIGHT_MULT})")

    # Cross-region anomaly transplantation: donors = training anomalies with partial masks
    donors = build_donor_pool(train_anomaly_triples) if CRAT_PROB > 0 else None
    if donors:
        donor_counts = {c: len(v) for c, v in donors.items()}
        print(f"CRAT active (p={CRAT_PROB}), donors per type: {donor_counts}")

    train_dataset = ConcatDataset([
        NormalImagesDataset(train_normal_files, CLIP_REFLECTANCE, augment=True, donors=donors, crat_prob=CRAT_PROB),
        AnomalyImagesDataset(train_anomaly_triples, CLIP_REFLECTANCE, augment=True),
    ])
    val_dataset = ConcatDataset([
        NormalImagesDataset(val_normal_files, CLIP_REFLECTANCE),
        AnomalyImagesDataset(val_anomaly_triples, CLIP_REFLECTANCE),
    ])
    train_loader = make_loader(train_dataset, BATCH_IMAGES, NUM_WORKERS, shuffle=True, seed=SEED, weights=train_weights)
    val_loader = make_loader(val_dataset, BATCH_IMAGES, NUM_WORKERS)

    # Train the model
    val_hard_mask = np.r_[[p.parent.name != "ben-ge-8k" for p in val_normal_files], np.ones(len(val_anomaly_triples), dtype=bool)]
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE, val_hard_mask=val_hard_mask)


if __name__ == "__main__":
    main()