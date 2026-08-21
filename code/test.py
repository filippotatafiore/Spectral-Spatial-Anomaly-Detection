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
from evaluation.baselines import PCAReconstructor, rx_score

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_eval, split_fingerprint, split_normals, subsample


from utils.globals import (
    ALPHA_GRID, ARTIFACTS_DIR, CHUNK_PIXELS, CLIP_REFLECTANCE, DECODER_HIDDEN, D_MODEL,
    DROPOUT, EMBED_HIDDEN, FEATURE_DIM, FFN_HIDDEN, MASK_CENTER, MAX_IMAGES_PER_CLASS,
    MAX_NORMAL_CALIB_IMAGES, N_BANDS, N_DCT_COEFFS, N_HEADS, N_LAYERS, N_TOKENS,
    PATCH_SIZE, PCA_COMPONENTS, POS_ENCODING, SAVE_MAPS_PER_CLASS, SEED,
    SELECTION_EXCLUDE, TEST_CLASSES, USE_DCT, USE_RAW,
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

    for ch in iter_all_centers(height, width, chunk, device):
        raw = extract_patches(batched, ch.unsqueeze(0), PATCH_SIZE)  # (B, N, T, C) = (1, n, T, C)
        features, target = builder(raw)
        features = features.reshape(-1, N_TOKENS, features.shape[-1])
        target = target.reshape(-1, N_TOKENS, target.shape[-1])

        with torch.autocast("cuda", enabled=use_amp):
            out = model(features, need_attn_stats=True)

        lo, hi = int(ch[0]), int(ch[-1]) + 1
        err = ((out["recon"][:, center].float() - target[:, center].float()) ** 2).mean(-1)
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
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()

    stats = load_and_check_stats()

    tag = "random_init" if args.random_init else "sst"
    out_path = args.out or (ARTIFACTS_DIR / f"results_{tag}.json")

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
    recon_scaler = RobustScaler.fit(np.concatenate([m["recon"].ravel() for m in normal_maps]))
    attn_scalers = {
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
        selection = {"alpha": args.fixed_alpha, "variant": ATTN_VARIANTS[0], "sign": 1.0,
                     "val_macro_auroc": None, "note": "alpha fixed on the command line"}
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


    # Alpha sensitivity curve (reported, NOT used for selection)
    results["alpha_sweep"] = {}
    for a in ALPHA_GRID:
        pc = {}
        for cls in scored:
            recs = evaluate(scored[cls]["test"], a, recon_scaler, attn_scaler, variant, sign)
            if recs:
                pc[cls] = aggregate(recs)
        results["alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")

    # Image-level detection
    def image_score(m):
        return float(np.percentile(fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign), 99))

    normal_img_scores = np.array([image_score(m) for m in normal_maps])
    results["image_level_auroc"] = {
        cls: image_level_auroc(normal_img_scores, np.array([image_score(m) for m in scored[cls]["test"]])) for cls in scored
    }

    # Classical baselines on the same test images
    pca = PCAReconstructor(np.array(stats["band_mean"]), np.array(stats["band_cov"]), PCA_COMPONENTS)
    results["baselines"] = {"pca_explained_variance": pca.explained, "rx": {}, "pca": {}}
    for cls in scored:
        rx_recs, pca_recs = [], []
        for m in scored[cls]["test"]:
            img = _load_image(m["img_path"], CLIP_REFLECTANCE).numpy()
            for fn, acc in ((rx_score, rx_recs), (pca.score, pca_recs)):
                r = per_image_metrics(fn(img), m["mask"])
                if r is not None:
                    acc.append(r)

        results["baselines"]["rx"][cls] = aggregate(rx_recs)
        results["baselines"]["pca"][cls] = aggregate(pca_recs)

    results["baselines"]["rx_macro_auroc"] = macro_average(results["baselines"]["rx"], "auroc_mean")
    results["baselines"]["pca_macro_auroc"] = macro_average(results["baselines"]["pca"], "auroc_mean")


    # Save score maps for qualitative figures
    if SAVE_MAPS_PER_CLASS:
        maps_dir = ARTIFACTS_DIR / f"maps_{tag}"
        maps_dir.mkdir(parents=True, exist_ok=True)
        for cls in scored:
            for m in scored[cls]["test"][:SAVE_MAPS_PER_CLASS]:
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
          f"PCA {results['baselines']['pca_macro_auroc']:.4f}")
    print(f"selection: alpha={alpha} variant={variant} sign={sign:+.0f}")
    print(f"wrote {out_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()