import math
import torch
import torch.nn.functional as F

import time
import json

from utils.globals import (
    PATCH_SIZE, CENTERS_PER_IMAGE, LOSS_TOKENS,
    SEED, MASK_CENTER, FEATURE_DIM, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT,
    D_MODEL, N_HEADS, N_LAYERS, FFN_HIDDEN, DROPOUT, EMBED_HIDDEN, DECODER_HIDDEN, POS_ENCODING, LR, WEIGHT_DECAY,
    EPOCHS, WARMUP_STEPS, AMP, ARTIFACTS_DIR,
    GRAD_CLIP, LOG_EVERY, EARLY_STOP_PATIENCE,
)

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_normals

from data.transforms import center_token_index, extract_patches, sample_centers, FeatureBuilder
from data.dataset import make_loader, NormalImagesDataset

from network.transformer_ae import SpectralSpatialTransformerAE


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_step(model, builder, images, generator):
    """
        Sample centers, build patches and features, reconstruct, return the loss.
    """

    # Sample CENTERS_PER_IMAGE random flat pixel indices as centers from each image in the batch
    batch = images.shape[0]
    height, width = images.shape[-2:]
    centers = sample_centers(batch, height, width, CENTERS_PER_IMAGE, images.device, generator)  # (B, CENTERS_PER_IMAGE) = (B, N)

    # Extract pixel-centered patches
    raw = extract_patches(images, centers, PATCH_SIZE)  # (B, N, T, C)
    # Build features and target from the raw patches
    features, target = builder(raw)  # (B, N, T, F), (B, N, T, C)
    tokens = PATCH_SIZE * PATCH_SIZE
    features = features.reshape(-1, tokens, features.shape[-1])  # (B*N, T, F)
    target = target.reshape(-1, tokens, target.shape[-1])  # (B*N, T, C)

    # Reconstruct the spectral bands from the features
    recon = model(features)["recon"]

    # Compute the loss
    center = center_token_index(PATCH_SIZE)  # index of center pixel in the flattened P x P patch
    if LOSS_TOKENS == "center":
        loss = F.mse_loss(recon[:, center], target[:, center])
    else:
        loss = F.mse_loss(recon, target)
    return loss, F.mse_loss(recon[:, center], target[:, center]).detach()


@torch.no_grad()
def evaluate_loss(model, builder, loader, device, val_seed=12345):
    """
        Validation loss on a fixed set of centers
    """
    model.eval()
    # Use a fixed random seed to sample the same centers for validation
    generator = torch.Generator(device=device)
    generator.manual_seed(val_seed)

    total, count = 0.0, 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        loss, _ = training_step(model, builder, images, generator)
        total += float(loss) * images.shape[0]
        count += images.shape[0]
    model.train()
    return total / max(count, 1)


def cosine_schedule(step, warmup, total):
    """
        Cosine decay with warmup.
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device} | mask_center={MASK_CENTER} | feature_dim={FEATURE_DIM}")

    # load and check stats match the current config
    stats = load_and_check_stats()

    # Split training images (determinitically)
    train_files, val_files = split_normals()
    print(f"Train images: {len(train_files)} | val images: {len(val_files)}")

    # Create data loaders for training and validation
    train_loader = make_loader(
        NormalImagesDataset(train_files, CLIP_REFLECTANCE),  # training dataset
        BATCH_IMAGES, NUM_WORKERS, shuffle=True, seed=SEED,
    )
    val_loader = make_loader(
        NormalImagesDataset(val_files, CLIP_REFLECTANCE),
        BATCH_IMAGES, NUM_WORKERS
    )

    # Feature builder
    builder = FeatureBuilder(
        N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT,
        stats["feature_mean"], stats["feature_std"],
    ).to(device)

    # Create the model
    model = SpectralSpatialTransformerAE(
        feature_dim=FEATURE_DIM, n_bands=N_BANDS, patch_size=PATCH_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ffn_hidden=FFN_HIDDEN,
        dropout=DROPOUT, embed_hidden=EMBED_HIDDEN, decoder_hidden=DECODER_HIDDEN,
        pos_encoding=POS_ENCODING, mask_center=MASK_CENTER,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader), 1) * EPOCHS

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_schedule(s, WARMUP_STEPS, total_steps)  # cosine decay with warmup
    )

    # Automatic Mixed Precision
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Random generator for sampling centers
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)


    best_val, best_epoch, bad_epochs, step = float("inf"), -1, 0, 0
    history = []
    ckpt_path = ARTIFACTS_DIR / ("best_masked.pt" if MASK_CENTER else "best.pt")

    model.train()
    for epoch in range(EPOCHS):
        running = 0.0  # running loss of current epoch
        t0 = time.time()  # start time of the epoch

        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss, center_mse = training_step(model, builder, images, generator)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Print every LOG_EVERY steps
            running += float(loss)  # running loss
            step += 1
            if step % LOG_EVERY == 0:
                print(f"  epoch {epoch:03d} step {step:06d} loss {running / LOG_EVERY:.5f} "
                      f"center_mse {float(center_mse):.5f} lr {scheduler.get_last_lr()[0]:.2e}")
                running = 0.0

        # Validation loss
        val_loss = evaluate_loss(model, builder, val_loader, device, generator)
        history.append({"epoch": epoch, "step": step, "val_loss": val_loss})
        print(f"[epoch {epoch:03d}] val_loss = {val_loss:.6f}  ({time.time() - t0:.0f}s)")

        # Save the best model checkpoint
        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "mask_center": MASK_CENTER}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best {best_epoch}).")
                break


    with (ARTIFACTS_DIR / "train_summary.json").open("w") as fh:
        json.dump({"best_val_loss": best_val, "best_epoch": best_epoch,
                   "n_params": n_params, "history": history}, fh, indent=2)
    print(f"\nBest val loss {best_val:.6f} at epoch {best_epoch}\nCheckpoint: {ckpt_path}")


if __name__ == "__main__":
    main()