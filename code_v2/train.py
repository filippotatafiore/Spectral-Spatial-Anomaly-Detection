import math
import torch
import torch.nn.functional as F

import time
import json
import platform
import argparse

from utils.globals import (
    PATCH_SIZE, CENTERS_PER_IMAGE, LOSS_TOKENS,
    SEED, VAL_SEED, MASK_CENTER, INPUT_NOISE_STD, FEATURE_DIM, SOURCE_BALANCE_POWER, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT,
    D_MODEL, N_HEADS, N_LAYERS, FFN_HIDDEN, DROPOUT, EMBED_HIDDEN, DECODER_HIDDEN, POS_ENCODING, LR, WEIGHT_DECAY,
    EPOCHS, WARMUP_STEPS, AMP, ARTIFACTS_DIR,
    GRAD_CLIP, LOG_EVERY, EARLY_STOP_PATIENCE, CKPT_EVERY,
    SPECTRAL_MASK_N, 
)
from utils.globals import config as CONFIG

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_normals, split_fingerprint

from data.transforms import center_token_index, extract_patches, sample_centers, FeatureBuilder
from data.dataset import make_loader, make_source_weights, NormalImagesDataset

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
    # spectral masking: hide SPECTRAL_MASK_N bands, make the model infer them from the visible bands (same bands are hidden in every token of the patch)
    hidden = None
    if SPECTRAL_MASK_N > 0:
        hidden = torch.randperm(N_BANDS, generator=generator, device=raw.device)[:SPECTRAL_MASK_N]
        raw_masked = raw.clone()
        raw_masked[..., hidden] = builder.band_mean[hidden]
        features, _ = builder(raw_masked)   # bands hidden
        _, target = builder(raw)            # original values
    else:
        features, target = builder(raw)  # (B, N, T, F), (B, N, T, C)
    # Corrupt the input (leave target clean) at training time (add gaussian noise to the features)
    if INPUT_NOISE_STD > 0 and model.training:
        features = features + torch.randn(features.shape, device=features.device, dtype=features.dtype, generator=generator) * INPUT_NOISE_STD
    tokens = PATCH_SIZE * PATCH_SIZE
    features = features.reshape(-1, tokens, features.shape[-1])  # (B*N, T, F)
    target = target.reshape(-1, tokens, target.shape[-1])  # (B*N, T, C)

    # Reconstruct the spectral bands from the features
    recon = model(features)["recon"]

    # Compute the loss
    center = center_token_index(PATCH_SIZE)  # index of center pixel in the flattened P x P patch
    if hidden is not None:
        loss = F.mse_loss(recon[:, center][:, hidden], target[:, center][:, hidden])
    elif LOSS_TOKENS == "center":
        loss = F.mse_loss(recon[:, center], target[:, center])
    else:
        loss = F.mse_loss(recon, target)

    if hidden is not None:
        # If doing spectral masking only the hidden bands required learning
        center_mse = F.mse_loss(recon[:, center][:, hidden], target[:, center][:, hidden]).detach()
        target_var = target[:, center][:, hidden].float().var().detach()
    else:
        center_mse = F.mse_loss(recon[:, center], target[:, center]).detach()
        target_var = target[:, center].float().var().detach()

    # Per-band error
    per_band_mse = ((recon[:, center] - target[:, center]) ** 2).mean(0).detach()


    # Naive baseline: predict center pixel as average of its 4 neighbors
    P = PATCH_SIZE
    up, down, left, right = center - P, center + P, center - 1, center + 1
    if hidden is not None:
        naive_mse = target_var.clone()
    else:
        naive = (target[:, up] + target[:, down] + target[:, left] + target[:, right]) / 4.0
        naive_mse = F.mse_loss(naive, target[:, center]).detach()

    return loss, center_mse, target_var, naive_mse, per_band_mse


@torch.no_grad()
def evaluate_loss(model, builder, loader, device, val_seed=VAL_SEED):
    """
        Validation loss on a fixed set of centers
    """
    model.eval()
    # Use a fixed random seed to sample the same centers for validation
    generator = torch.Generator(device=device)
    generator.manual_seed(val_seed)

    total, total_center, total_var, total_naive, count = 0.0, 0.0, 0.0, 0.0, 0
    last_per_band = None
    for images in loader:
        images = images.to(device, non_blocking=True)
        loss, center_mse, target_var, naive_mse, per_band_mse = training_step(model, builder, images, generator)
        last_per_band = per_band_mse
        n = images.shape[0]
        total += float(loss) * n
        total_center += float(center_mse) * n
        total_var += float(target_var) * n
        total_naive += float(naive_mse) * n
        count += n

    model.train()
    count = max(count, 1)
    val_loss = total / count
    val_center_mse = total_center / count
    val_target_var = total_var / count
    variance_explained = 1.0 - val_center_mse / max(val_target_var, 1e-12)
    naive_ve = 1.0 - (total_naive / count) / max(val_target_var, 1e-12)
    return val_loss, val_center_mse, val_target_var, variance_explained, naive_ve, (last_per_band.cpu().tolist() if last_per_band is not None else None)


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
    train_weights = make_source_weights(train_files, SOURCE_BALANCE_POWER)
    if train_weights is not None:
        print(f"Balanced sampling active (power={SOURCE_BALANCE_POWER})")
    train_loader = make_loader(
        NormalImagesDataset(train_files, CLIP_REFLECTANCE),  # training dataset
        BATCH_IMAGES, NUM_WORKERS, shuffle=True, seed=SEED, weights=train_weights,
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

    # Optimizer (weight decay on tensors with ndim >= 2, no weight decay on bias and LayerNorm weights)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay_params if param.ndim >= 2 else no_decay_params).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
    )
    print(f"Weight decay on {len(decay_params)} tensors (none on {len(no_decay_params)})")

    # Scheduler
    total_steps = max(len(train_loader), 1) * EPOCHS
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
    history, step_history = [], []
    best_model_path = ARTIFACTS_DIR / f"best.pt"
    last_model_path = ARTIFACTS_DIR / f"last.pt"
    run_start = time.time()
    early_stopped = False

    model.train()
    for epoch in range(EPOCHS):
        running, running_n = 0.0, 0
        epoch_loss_sum, epoch_n = 0.0, 0
        grad_norm = float("nan")
        t0 = time.time()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss, center_mse, target_var, _, _ = training_step(model, builder, images, generator)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_loss = float(loss.detach())
            running += batch_loss
            running_n += 1
            epoch_loss_sum += batch_loss
            epoch_n += 1
            step += 1

            # Log every LOG_EVERY steps
            if step % LOG_EVERY == 0:
                avg = running / max(running_n, 1)
                lr_now = scheduler.get_last_lr()[0]
                step_history.append({
                    "step": step, "epoch": epoch, "train_loss": avg,
                    "center_mse": float(center_mse), "lr": lr_now, "grad_norm": grad_norm,
                })
                print(f"  epoch {epoch:03d} step {step:06d} loss {avg:.5f} "
                      f"center_mse {float(center_mse):.5f} lr {lr_now:.2e}")
                running, running_n = 0.0, 0

        # Validation loss
        val_loss, val_center_mse, val_target_var, var_explained, naive_ve, per_band = evaluate_loss(model, builder, val_loader, device)
        epoch_time = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        train_loss = epoch_loss_sum / max(epoch_n, 1)

        history.append({
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_center_mse": val_center_mse,
            "val_target_var": val_target_var,
            "variance_explained": var_explained,
            "naive_variance_explained": naive_ve,
            "per_band_center_mse": per_band,
            "beats_naive": bool(var_explained > naive_ve),
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "epoch_time_s": epoch_time,
            "peak_mem_gb": peak_mem,
        })
        print(f"[epoch {epoch:03d}] train {train_loss:.6f} | val {val_loss:.6f} | "
              f"var_explained {var_explained:.4f} (naive {naive_ve:.4f}"
              f"{' BEATEN' if var_explained > naive_ve else ' NOT beaten'}) | "
              f"{epoch_time:.0f}s | {peak_mem:.2f} GB")

        # Save the best model checkpoint
        is_best = val_loss < best_val - 1e-6
        if is_best:
            best_val, best_epoch = val_loss, epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "variance_explained": var_explained,   # identity mapping diagnostic
                        "mask_center": MASK_CENTER, "config_feature_dim": FEATURE_DIM}, best_model_path)
        else:
            bad_epochs += 1

        # Save the last model checkpoint (overwrite each epoch)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch, "step": step, "best_val": best_val, "best_epoch": best_epoch,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "center_generator": generator.get_state(),
            "loader_generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
            "history": history,
            "step_history": step_history,
            "bad_epochs": bad_epochs,
            "epochs_planned": EPOCHS,
            "batch_images": BATCH_IMAGES,
        }, last_model_path)

        # Save snapshots every CKPT_EVERY epochs
        if CKPT_EVERY and (epoch + 1) % CKPT_EVERY == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch, "step": step, "best_val": best_val, "val_loss": val_loss,
                "variance_explained": var_explained,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "center_generator": generator.get_state(),
                "loader_generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
                "history": history,
                "step_history": step_history,
                "bad_epochs": bad_epochs,
                "epochs_planned": EPOCHS,
                "batch_images": BATCH_IMAGES,
            }, ARTIFACTS_DIR / f"epoch_{epoch:03d}.pt")

        # Early stopping
        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best {best_epoch}).")
            early_stopped = True
            break

    # Save training summary
    total_time = time.time() - run_start
    summary = {
        "run": {
            "tag": f"sst",
            "finished_epochs": len(history),
            "planned_epochs": EPOCHS,
            "early_stopped": early_stopped,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "n_params": n_params,
        },
        "config": {k: v for k, v in sorted(CONFIG.items())},
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "provenance": {
            "split_fingerprint": split_fingerprint(),
            "n_train_images": len(train_files),
            "n_val_images": len(val_files),
            "stats_n_pixels": stats["n_pixels"],
            "stats_train_sources": stats["train_sources"],
            "feature_mean": stats["feature_mean"],
            "feature_std": stats["feature_std"],
        },
        "timing": {
            "total_s": total_time,
            "mean_epoch_s": total_time / max(len(history), 1),
            "steps": step,
            "mean_step_s": total_time / max(step, 1),
        },
        "history": history,
        "step_history": step_history,
    }
    with (ARTIFACTS_DIR / f"train_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nBest val loss {best_val:.6f} at epoch {best_epoch}")
    print(f"Total time {total_time/60:.1f} min | checkpoint {best_model_path}")
    if history:
        print(f"Final variance_explained = {history[-1]['variance_explained']:.4f} "
              f"(-> 1.0 means the model is copying its input)")


if __name__ == "__main__":
    main()