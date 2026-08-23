import torch
import numpy as np
import time
import json
import argparse

from network.transformer_ae import SpectralSpatialTransformerAE

from data.transforms import FeatureBuilder, center_token_index, extract_patches, iter_all_centers
from data.dataset import _load_image

from evaluation.metrics import per_image_metrics, aggregate, image_level_auroc, macro_average
from evaluation.scoring import attention_statistic, fuse, RobustScaler, ATTN_VARIANTS, SIGNS
from evaluation.baselines import PCAReconstructor, rx_score, SpectralLinearPredictor

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_eval, split_fingerprint, split_normals, subsample


from utils.globals import (
    ALPHA_GRID, ARTIFACTS_DIR, CHUNK_PIXELS, CLIP_REFLECTANCE, DECODER_HIDDEN, D_MODEL,
    DROPOUT, EMBED_HIDDEN, FEATURE_DIM, FFN_HIDDEN, MASK_CENTER, MAX_IMAGES_PER_CLASS,
    MAX_NORMAL_CALIB_IMAGES, N_BANDS, N_DCT_COEFFS, N_HEADS, N_LAYERS, N_TOKENS,
    PATCH_SIZE, PCA_COMPONENTS, POS_ENCODING, SAVE_MAPS_PER_CLASS, SEED,
    SELECTION_EXCLUDE, TEST_CLASSES, USE_DCT, USE_RAW,
    SPECTRAL_MASK_N, SPECTRAL_MASK_EVAL_GROUPS,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device):
    return SpectralSpatialTransformerAE(
        feature_dim=FEATURE_DIM, n_bands=N_BANDS, patch_size=PATCH_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ffn_hidden=FFN_HIDDEN,
        dropout=DROPOUT, embed_hidden=EMBED_HIDDEN, decoder_hidden=DECODER_HIDDEN,
        pos_encoding=POS_ENCODING, mask_center=MASK_CENTER,
    ).to(device)


@torch.no_grad()
def score_image(model, builder, image, device, chunk=CHUNK_PIXELS):
    """
        Score every pixel of the image.

        image: (C, H, W)
    """
    height, width = image.shape[-2:]
    center = center_token_index(PATCH_SIZE)
    batched = image.unsqueeze(0)
    use_amp = device.type == "cuda"

    recon_map = np.empty(height * width, dtype=np.float32)
    entropy_map = np.empty(height * width, dtype=np.float32)
    received_map = np.empty(height * width, dtype=np.float32)
    
    band_groups = None
    if SPECTRAL_MASK_N > 0:
        perm = torch.randperm(N_BANDS, generator=torch.Generator().manual_seed(0))
        band_groups = [perm[i::SPECTRAL_MASK_EVAL_GROUPS].to(device) for i in range(SPECTRAL_MASK_EVAL_GROUPS)]

    for ch in iter_all_centers(height, width, chunk, device):
        raw = extract_patches(batched, ch.unsqueeze(0), PATCH_SIZE)  # (B, N, T, C) = (1, n, T, C)
        _, target = builder(raw)
        target = target.reshape(-1, N_TOKENS, target.shape[-1])
        lo, hi = int(ch[0]), int(ch[-1]) + 1

        if band_groups is None:
            features, _ = builder(raw)
            features = features.reshape(-1, N_TOKENS, features.shape[-1])
            with torch.autocast("cuda", enabled=use_amp):
                out = model(features, need_attn_stats=True)
            err = ((out["recon"][:, center].float() - target[:, center].float()) ** 2).mean(-1)
        else:
            err = torch.zeros(hi - lo, device=device)
            out = None
            for gi, hidden in enumerate(band_groups):
                raw_m = raw.clone()
                raw_m[..., hidden] = builder.band_mean[hidden]
                feats, _ = builder(raw_m)
                feats = feats.reshape(-1, N_TOKENS, feats.shape[-1])
                with torch.autocast("cuda", enabled=use_amp):
                    o = model(feats, need_attn_stats=True)
                if gi == 0:
                    out = o          # attention statistics from the first pass only
                d = o["recon"][:, center].float()[:, hidden] - target[:, center].float()[:, hidden]
                err = err + (d ** 2).sum(-1)
            err = err / N_BANDS      # every band contributed exactly once

        recon_map[lo:hi] = err.cpu().numpy()
        entropy_map[lo:hi] = out["center_entropy"].float().cpu().numpy()
        received_map[lo:hi] = out["center_received"].float().cpu().numpy()

    return {
        "recon": recon_map.reshape(height, width),  # (H, W)
        "center_entropy": entropy_map.reshape(height, width),  # (H, W)
        "center_received": received_map.reshape(height, width),  # (H, W)
    }


def collect(model, builder, items, device, label, with_masks):
    """
        Score a list of images.
    """
    out = []
    t0 = time.time()
    for i, item in enumerate(items):
        if with_masks:
            img_path, mask_path, cls = item
            maps = score_image(model, builder, _load_image(img_path, CLIP_REFLECTANCE).to(device), device)
            maps["mask"] = np.load(mask_path).astype(np.uint8)
            maps["cls"] = cls
            maps["name"] = img_path.stem
            maps["img_path"] = img_path
        else:
            maps = score_image(model, builder, _load_image(item, CLIP_REFLECTANCE).to(device), device)
            maps["name"] = item.stem

        out.append(maps)

        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  [{label}] {i + 1}/{len(items)}  ({rate:.2f}s/img, "
                  f"{rate * (len(items) - i - 1) / 60:.1f} min left)", flush=True)

    return out


def fused_scores(maps, alpha, recon_scaler, attn_scaler, variant, sign):
    attn = attention_statistic(maps, variant, N_TOKENS)
    return fuse(maps["recon"], attn, alpha, recon_scaler, attn_scaler, sign)  # Anomaly Score


def evaluate(maps_list, alpha, recon_scaler, attn_scaler, variant, sign):
    records = []
    for m in maps_list:
        s = fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign)
        r = per_image_metrics(s, m["mask"])
        if r is not None:
            records.append(r)
    return records



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--ckpt", default=None, help="checkpoint (default artifacts/best.pt)")
    ap.add_argument("-o", "--out", default=None, help="results json (default artifacts/results_<tag>.json)")
    ap.add_argument("--random-init", action="store_true", help="skip loading weights: the untrained control row")
    ap.add_argument("--fixed-alpha", type=float, default=None, help="skip selection (1.0 = recon only, 0.0 = attention only)")
    ap.add_argument("--ckpt-sweep", action="store_true", help="score every artifacts/epoch_*.pt on the val split and report the curve")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()

    stats = load_and_check_stats()

    tag = "random_init" if args.random_init else "sst"
    out_path = args.out or (ARTIFACTS_DIR / f"results_{tag}.json")

    # Build model and load weights
    builder = FeatureBuilder(N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT, stats["feature_mean"], stats["feature_std"]).to(device)
    model = build_model(device)
    if not args.random_init:
        ckpt_path = args.ckpt or (ARTIFACTS_DIR / "best.pt")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.6f})")
    else:
        print("RANDOM-INIT CONTROL (untrained weights)")
    model.eval()


    # Held-out normal images for calibration
    _, normal_val = split_normals()
    normal_val = subsample(normal_val, MAX_NORMAL_CALIB_IMAGES)
    # Evaluation splits
    splits = split_eval()
    scored = {}

    results = {
        "tag": tag,
        "split_fingerprint": split_fingerprint(),
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
        "n_normal_calib": len(normal_val),
    }


    # Calibrate on normal images
    print(f"\nCalibrating on {len(normal_val)} held-out normal images")
    normal_maps = collect(model, builder, normal_val, device, "normal", with_masks=False)
    # Scalers fit on pixel values across all normal images
    recon_scaler = RobustScaler.fit(np.concatenate([m["recon"].ravel() for m in normal_maps]))
    attn_scalers = {  # one scaler per attention variant
        v: RobustScaler.fit(np.concatenate([attention_statistic(m, v, N_TOKENS).ravel() for m in normal_maps])) for v in ATTN_VARIANTS
    }
    results["calibration"] = {"recon": recon_scaler.to_dict(), **{f"attn_{v}": s.to_dict() for v, s in attn_scalers.items()}}


    # Score evaluation splits
    for cls in TEST_CLASSES:
        scored[cls] = {}
        for part in ("val", "test"):
            items = subsample(splits[cls][part], MAX_IMAGES_PER_CLASS)
            print(f"\nScoring {cls}/{part}: {len(items)} images")
            scored[cls][part] = collect(model, builder, items, device, f"{cls}/{part}", with_masks=True)


    # Select alpha, attention variant and sign on the val set
    if args.fixed_alpha is not None:
        selection = {"alpha": args.fixed_alpha, "variant": ATTN_VARIANTS[0], "sign": 1.0, "val_macro_auroc": None, "note": "alpha fixed on the command line"}
    else:
        best = None
        for variant in ATTN_VARIANTS:
            for sign in SIGNS:
                for alpha in ALPHA_GRID:
                    per_class = {}
                    for cls in scored:
                        recs = evaluate(scored[cls]["val"], alpha, recon_scaler,
                                        attn_scalers[variant], variant, sign)
                        if recs:
                            per_class[cls] = aggregate(recs)
                    score = macro_average(per_class, "auroc_mean", exclude=SELECTION_EXCLUDE)
                    if best is None or score > best["val_macro_auroc"]:
                        best = {"alpha": alpha, "variant": variant, "sign": sign, "val_macro_auroc": score}
        selection = best
        print(f"\nSelected on VAL (excluding {SELECTION_EXCLUDE}): {selection}")
    results["selection"] = selection

    variant, sign, alpha = selection["variant"], selection["sign"], selection["alpha"]
    attn_scaler = attn_scalers[variant]


    # Report on test set
    results["per_class"] = {}
    for cls in scored:
        recs = evaluate(scored[cls]["test"], alpha, recon_scaler, attn_scaler, variant, sign)
        results["per_class"][cls] = {**aggregate(recs), "n_skipped_single_class": len(scored[cls]["test"]) - len(recs)}
    results["macro_auroc"] = macro_average(results["per_class"], "auroc_mean")
    results["macro_ap_gain"] = macro_average(results["per_class"], "ap_gain_mean")


    # Alpha sensitivity curve (report how results vary with alpha)
    results["alpha_sweep"] = {}
    for a in ALPHA_GRID:
        pc = {}
        for cls in scored:
            recs = evaluate(scored[cls]["test"], a, recon_scaler, attn_scaler, variant, sign)
            if recs:
                pc[cls] = aggregate(recs)
        results["alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")

    # Image-level detection
    def image_score(m, a, sc):
        return float(np.percentile(fused_scores(m, a, recon_scaler, sc, variant, sign), 99))

    img_best = None
    for a in ALPHA_GRID:
        norm_v = np.array([image_score(m, a, attn_scaler) for m in normal_maps])
        per_cls = {}
        for cls in scored:
            anom_v = np.array([image_score(m, a, attn_scaler) for m in scored[cls]["val"]])
            if len(anom_v):
                per_cls[cls] = {"auroc_mean": image_level_auroc(norm_v, anom_v)}
        s = macro_average(per_cls, "auroc_mean", exclude=SELECTION_EXCLUDE)
        if img_best is None or s > img_best[1]:
            img_best = (a, s)
    img_alpha = img_best[0]
    results["image_level_selection"] = {"alpha": img_alpha, "val_macro_auroc": img_best[1], "pixel_level_alpha": alpha}
    print(f"image-level alpha selected on VAL: {img_alpha} (pixel-level uses {alpha})")

    normal_img_scores = np.array([image_score(m, img_alpha, attn_scaler) for m in normal_maps])
    results["image_level_auroc"] = {
        cls: image_level_auroc(normal_img_scores, np.array([image_score(m, img_alpha, attn_scaler) for m in scored[cls]["test"]]))
        for cls in scored
    }
    # sensitivity curve, reported not used for selection
    results["image_level_alpha_sweep"] = {}
    for a in ALPHA_GRID:
        nv = np.array([image_score(m, a, attn_scaler) for m in normal_maps])
        pc = {cls: {"auroc_mean": image_level_auroc(
                        nv, np.array([image_score(m, a, attn_scaler) for m in scored[cls]["test"]]))}
              for cls in scored}
        results["image_level_alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")

    # Classical baselines on the same test images (RX, PCA, SpectralLinearPredictor)
    pca = PCAReconstructor(np.array(stats["band_mean"]), np.array(stats["band_cov"]), PCA_COMPONENTS)
    train_files, _ = split_normals()
    spec = SpectralLinearPredictor(stats["band_mean"], stats["feature_std"][:N_BANDS], train_files, lambda p: _load_image(p, CLIP_REFLECTANCE).numpy())
    print(f"SpectralLinearPredictor fitted on {spec.n_pixels_fitted:,} normal pixels")
    results["baselines"] = {"pca_explained_variance": pca.explained, "rx": {}, "pca": {}, "spectral_linear": {}}
    for cls in scored:
        rx_recs, pca_recs, spec_recs = [], [], []
        for m in scored[cls]["test"]:
            img = _load_image(m["img_path"], CLIP_REFLECTANCE).numpy()
            for fn, acc in ((rx_score, rx_recs), (pca.score, pca_recs), (spec.score, spec_recs)):
                r = per_image_metrics(fn(img), m["mask"])
                if r is not None:
                    acc.append(r)

        results["baselines"]["rx"][cls] = aggregate(rx_recs)
        results["baselines"]["pca"][cls] = aggregate(pca_recs)
        results["baselines"]["spectral_linear"][cls] = aggregate(spec_recs)

    results["baselines"]["rx_macro_auroc"] = macro_average(results["baselines"]["rx"], "auroc_mean")
    results["baselines"]["pca_macro_auroc"] = macro_average(results["baselines"]["pca"], "auroc_mean")
    results["baselines"]["spectral_linear_macro_auroc"] = macro_average(results["baselines"]["spectral_linear"], "auroc_mean")


    # Save score maps
    if SAVE_MAPS_PER_CLASS:
        maps_dir = ARTIFACTS_DIR / f"maps_{tag}"
        maps_dir.mkdir(parents=True, exist_ok=True)
        for cls in scored:
            usable = [m for m in scored[cls]["test"] if 0 < m["mask"].mean() < 1] or scored[cls]["test"]
            for m in usable[:SAVE_MAPS_PER_CLASS]:
                np.savez_compressed(
                    maps_dir / f"{cls}_{m['name']}.npz",
                    fused=fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign),
                    recon=m["recon"], entropy=m["center_entropy"],
                    received=m["center_received"], mask=m["mask"],
                )

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)


    # Print the table
    print("\n" + "=" * 92)
    print(f"{'class':<13}{'n':>5}{'skip':>6}{'base':>8}{'AUROC':>17}{'AP':>9}{'AP/base':>9}"
          f"{'imgAUROC':>10}{'RX':>8}{'PCA':>8}")
    print("-" * 92)
    for cls, m in results["per_class"].items():
        print(f"{cls:<13}{m['n_images']:>5}{m['n_skipped_single_class']:>6}"
              f"{m['base_rate_mean']:>8.3f}{m['auroc_mean']:>11.4f}+-{m['auroc_std']:<4.3f}"
              f"{m['ap_mean']:>9.4f}{m['ap_gain_mean']:>9.2f}"
              f"{results['image_level_auroc'][cls]:>10.4f}"
              f"{results['baselines']['rx'][cls]['auroc_mean']:>8.4f}"
              f"{results['baselines']['pca'][cls]['auroc_mean']:>8.4f}")
    print("-" * 92)
    print(f"macro AUROC (over classes): model {results['macro_auroc']:.4f} | "
          f"RX {results['baselines']['rx_macro_auroc']:.4f} | "
          f"PCA {results['baselines']['pca_macro_auroc']:.4f} | "
          f"SpectralLinear {results['baselines']['spectral_linear_macro_auroc']:.4f}")
    print(f"selection: alpha={alpha} variant={variant} sign={sign:+.0f}")
    print(f"wrote {out_path}")
    print("=" * 92)


    # checkpoint sweep
    if args.ckpt_sweep:
        ckpts = sorted(ARTIFACTS_DIR.glob("epoch_*.pt"))
        if not ckpts:
            print("\n(no epoch_*.pt checkpoints found; nothing to sweep)")
            return

        print("\n" + "=" * 60)
        print("CHECKPOINT SWEEP  (scored on the anomalous VAL split)")
        print(f"{'checkpoint':<18}{'macro AUROC':>14}")
        print("-" * 60)
        sweep = {}
        for cp in ckpts:
            state = torch.load(cp, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.eval()
            per_class = {}
            for cls in TEST_CLASSES:
                items = subsample(splits[cls]["val"], MAX_IMAGES_PER_CLASS)
                maps = collect(model, builder, items, device, f"sweep/{cls}", with_masks=True)
                recs = evaluate(maps, alpha, recon_scaler, attn_scaler, variant, sign)
                if recs:
                    per_class[cls] = aggregate(recs)
            score = macro_average(per_class, "auroc_mean", exclude=SELECTION_EXCLUDE)
            sweep[cp.name] = score
            print(f"{cp.name:<18}{score:>14.4f}", flush=True)
        best_cp = max(sweep, key=sweep.get)
        print("-" * 60)
        print(f"best on val: {best_cp}  ({sweep[best_cp]:.4f})")
        print("=" * 60)

        with open(ARTIFACTS_DIR / f"ckpt_sweep_{tag}.json", "w") as fh:
            json.dump({"sweep": sweep, "best": best_cp, "selection": selection}, fh, indent=2)


if __name__ == "__main__":
    main()