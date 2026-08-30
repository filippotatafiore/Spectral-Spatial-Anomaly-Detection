import torch
from sklearn.metrics import roc_auc_score

from network.loss import MultiTaskLoss
from utils.globals import (
    DEVICE, ARTIFACTS_DIR, EPOCHS, LR, WEIGHT_DECAY,
    SOURCE_BALANCE_POWER, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, AMP, SEED,
)
from network.trans_UNet import TransUNetSentinel2
from data.dataset import make_loader, make_source_weights, NormalImagesDataset
from utils.splits import split_normals


def save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, filename, device='cuda'):
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
    }
    torch.save(checkpoint, ARTIFACTS_DIR / filename)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-4, device='cuda'):

    # --- Build model
    model = model.to(device)

    # --- Loss function
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=1.0)

    # --- Optimizer
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

    # --- Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Mixed precision scaler
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Training loop
    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):

        # --- Train
        model.train()
        train_loss = 0.0
        
        for images, cls_targets, seg_targets in train_loader:
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            seg_targets = seg_targets.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device):
                cls_logits, seg_logits = model(images)
                loss, l_cls, l_bce, l_dice = criterion(cls_logits, seg_logits, cls_targets, seg_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # --- Validation
        model.eval()
        val_loss = 0.0
        val_cls_preds, val_cls_targets = [], []
        
        with torch.no_grad():
            for images, cls_targets, seg_targets in val_loader:
                images = images.to(device)
                cls_targets = cls_targets.to(device)
                seg_targets = seg_targets.to(device)

                with torch.autocast(device_type=device):
                    cls_logits, seg_logits = model(images)
                    loss, _, _, _ = criterion(cls_logits, seg_logits, cls_targets, seg_targets)

                val_loss += loss.item()
                val_cls_preds.extend(torch.sigmoid(cls_logits).cpu().numpy())
                val_cls_targets.extend(cls_targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_cls_targets, val_cls_preds)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Image AUC: {val_auc:.4f}")

        # Checkpoint saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "best.pth", device=device)

        # Save checkpoint every 10 epochs
        if epoch == epochs:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, "last.pth", device=device)
        elif epoch % 10 == 0:
            save_model_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, train_loader, f"epoch_{epoch}.pth", device=device)

    print(f"\nTraining Complete.\tBest Validation Image AUC-ROC: {best_val_auc:.4f}")


def main():
    model = TransUNetSentinel2(in_channels=12, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Split training images (determinitically)
    train_files, val_files, _ = split_normals()
    print(f"Train images: {len(train_files)} | val images: {len(val_files)}")

    # Create data loaders for training and validation
    train_weights = make_source_weights(train_files, SOURCE_BALANCE_POWER)
    if train_weights is not None:
        print(f"Balanced sampling active (power={SOURCE_BALANCE_POWER})")
    train_loader = make_loader(
        NormalImagesDataset(train_files, CLIP_REFLECTANCE),  # train set
        BATCH_IMAGES, NUM_WORKERS, shuffle=True, seed=SEED, weights=train_weights,
    )
    val_loader = make_loader(
        NormalImagesDataset(val_files, CLIP_REFLECTANCE),  # val set
        BATCH_IMAGES, NUM_WORKERS
    )

    # Train the model
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)


if __name__ == "__main__":
    main()