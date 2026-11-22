import torch
import numpy as np
import time
import json
import argparse

from network.transformer_ae import SpectralSpatialTransformerAE

from data.transforms import FeatureBuilder, center_token_index, extract_patches, iter_all_centers
from data.dataset import _load_image
from evaluation.metrics import (per_image_metrics, aggregate, image_level_auroc, macro_average,
                                operating_point_metrics, detection_at_operating_point, threshold_at_fpr, roc_3d)
from evaluation.scoring import attention_statistic, fuse, RobustScaler, ATTN_VARIANTS, SIGNS
from evaluation.baselines import PCAReconstructor, rx_score, SpectralLinearPredictor

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_eval, split_fingerprint, split_normals, subsample


from utils.globals import (
    ALPHA_GRID, OP_FPR_GRID, OP_TARGET_FPR, ARTIFACTS_DIR, CHUNK_PIXELS, CLIP_REFLECTANCE, DECODER_HIDDEN, D_MODEL,
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
        "recon": recon_map.reshape(height, width),  # (H, W), reconstruction error per pixel
        "center_entropy": entropy_map.reshape(height, width),  # (H, W), entropy of the center token's attention distribution (how spread out the attention is over the patch)
        "center_received": received_map.reshape(height, width),  # (H, W), sum of attention weights received by the center token (how much the other tokens are attending to the center pixel)
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


def evaluate(maps_list, alpha, recon_scaler, attn_scaler, variant, sign, with_3d_roc=False):
    records = []
    for m in maps_list:
        s = fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign)
        r = per_image_metrics(s, m["mask"])
        if r is not None:
            if with_3d_roc:
                # only on the final test pass: 200 threshold steps per image is slow,
                # and it is pointless during the alpha search
                r3 = roc_3d(m["mask"], s)
                if r3:
                    r.update(r3)
            records.append(r)
    return records



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--ckpt", default=None, help="checkpoint (default artifacts/best.pt)")
    ap.add_argument("-o", "--out", default=None, help="results json (default artifacts/results_<tag>.json)")
    ap.add_argument("--random-init", action="store_true", help="skip loading weights: the untrained control row")
    ap.add_argument("--fixed-alpha", type=float, default=None, help="skip selection (1.0 = recon only, 0.0 = attention only)")
    ap.add_argument("--ckpt-sweep", action="store_true", help="score every artifacts/epoch_*.pt on the val split and report the curve")
    ap.add_argument("--use-best-sweep-ckpt", action="store_true", help="pick the checkpoint by val detection AUROC instead of by val loss")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()

    stats = load_and_check_stats()

    tag = "random_init" if args.random_init else "sst"
    out_path = args.out or (ARTIFACTS_DIR / f"results_{tag}.json")

    # -------------------- Build model and load weights
    builder = FeatureBuilder(N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT, stats["feature_mean"], stats["feature_std"]).to(device)
    model = build_model(device)
    if not args.random_init:
        ckpt_path = args.ckpt or (ARTIFACTS_DIR / "best.pt")
        if args.use_best_sweep_ckpt and args.ckpt is None:
            sweep_file = ARTIFACTS_DIR / f"ckpt_sweep_{tag}.json"
            if sweep_file.exists():
                sw = json.load(open(sweep_file))
                ckpt_path = ARTIFACTS_DIR / sw["best"]
                print(f"using the sweep-selected checkpoint: {sw['best']} "
                      f"(val macro AUROC {sw['sweep'][sw['best']]:.4f})")
            else:
                print(f"(no {sweep_file.name}; run --ckpt-sweep first, falling back to best.pt)")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.6f})")
    else:
        print("RANDOM-INIT CONTROL (untrained weights)")
    model.eval()


    # -------------------- Held-out normal images for calibration
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


    # -------------------- Calibrate on normal images (score held-out normal images to fit the scalers)
    print(f"\nCalibrating on {len(normal_val)} held-out normal images")
    normal_maps = collect(model, builder, normal_val, device, "normal", with_masks=False)  # score normal images
    # Scalers fit on pixel values across all normal images
    recon_scaler = RobustScaler.fit(np.concatenate([m["recon"].ravel() for m in normal_maps]))
    attn_scalers = {  # one scaler per attention variant
        v: RobustScaler.fit(np.concatenate([attention_statistic(m, v, N_TOKENS).ravel() for m in normal_maps])) for v in ATTN_VARIANTS
    }
    results["calibration"] = {"recon": recon_scaler.to_dict(), **{f"attn_{v}": s.to_dict() for v, s in attn_scalers.items()}}


    # -------------------- Score evaluation splits
    for cls in TEST_CLASSES:
        scored[cls] = {}
        for part in ("val", "test"):
            items = subsample(splits[cls][part], MAX_IMAGES_PER_CLASS)
            print(f"\nScoring {cls}/{part}: {len(items)} images")
            scored[cls][part] = collect(model, builder, items, device, f"{cls}/{part}", with_masks=True)  # score


    # --------------------  Select, based on the validation set, alpha and the attention variant and sign to use for the Anomaly Score (= alpha * ReconError + (1 - alpha) * AttentionOutlier)
    if args.fixed_alpha is not None:
        selection = {"alpha": args.fixed_alpha, "variant": ATTN_VARIANTS[0], "sign": 1.0, "val_macro_auroc": None, "note": "alpha fixed on the command line"}
    else:
        best = None
        for variant in ATTN_VARIANTS:   # try every attention variant used for the AttentionOutlier term (center_entropy or center_received)
            for sign in SIGNS:            # try both signs for the AttentionOutlier term (1.0 = high score = anomalous, -1.0 = low score = anomalous)
                for alpha in ALPHA_GRID:    # try every possible value in the grid for alpha in the grid (weight of the ReconError term)
                    # evaluate every class on the validation set
                    per_class = {}
                    for cls in scored:
                        recs = evaluate(scored[cls]["val"], alpha, recon_scaler, attn_scalers[variant], variant, sign)
                        if recs:
                            per_class[cls] = aggregate(recs)
                    # compute the macro-average AUROC across classes (excluding classes in SELECTION_EXCLUDE)
                    score = macro_average(per_class, "auroc_mean", exclude=SELECTION_EXCLUDE)
                    if best is None or score > best["val_macro_auroc"]:
                        best = {"alpha": alpha, "variant": variant, "sign": sign, "val_macro_auroc": score}
        selection = best
        print(f"\nSelected on VAL (excluding {SELECTION_EXCLUDE}): {selection}")
    results["selection"] = selection

    variant, sign, alpha = selection["variant"], selection["sign"], selection["alpha"]
    attn_scaler = attn_scalers[variant]


    # -------------------- Choose based on the validation set the pixel-level decision threshold for each class

    def _standardize_map(s):  # standardize a single map
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        return (s - med) / max(1.4826 * mad, 1e-8)

    # Compute pixel-level scores on the normal images and rescale to a common scale
    normal_pixel_scores = np.concatenate([
        _standardize_map(fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign)).ravel()
        for m in normal_maps])

    # For each class, select the best pixel-level decision threshold on the validation set
    pixel_thresholds = {}
    for cls in scored:
        best = None
        for f in OP_FPR_GRID:
            thr = threshold_at_fpr(normal_pixel_scores, f)
            f1s = []
            for m in scored[cls]["val"]:  # for every validation set image score
                if 0 < m["mask"].mean() < 1:
                    # Compute metrics at the selected threshold
                    s = _standardize_map(fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign))
                    f1s.append(operating_point_metrics(s, m["mask"], thr)["op_f1"])

            if f1s and (best is None or np.mean(f1s) > best[2]):
                best = (f, thr, float(np.mean(f1s)))

        if best is None:
            best = (OP_TARGET_FPR, threshold_at_fpr(normal_pixel_scores, OP_TARGET_FPR), float("nan"))

        pixel_thresholds[cls] = {"target_fpr": best[0], "value": best[1], "val_op_f1": best[2]}
        print(f"  {cls:<12} operating point selected on val: FPR={best[0]:.0%} thr={best[1]:.4f} (val op_F1 {best[2]:.4f})")

    results["pixel_thresholds"] = pixel_thresholds  # pixel thresholds for every class


    # -------------------- Evaluate on the test set, using the selected alpha, attention variant and sign, and the pixel-level thresholds selected
    results["per_class"] = {}
    for cls in scored:
        recs = evaluate(scored[cls]["test"], alpha, recon_scaler, attn_scaler, variant, sign, with_3d_roc=True)
        ops = []
        thr_cls = pixel_thresholds[cls]["value"]
        for m in scored[cls]["test"]:
            if 0 < m["mask"].mean() < 1:
                s = _standardize_map(fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign))
                ops.append(operating_point_metrics(s, m["mask"], thr_cls))
        op_mean = {k: float(np.mean([o[k] for o in ops])) for k in ops[0]} if ops else {}
        results["per_class"][cls] = {**aggregate(recs), **op_mean, "n_skipped_single_class": len(scored[cls]["test"]) - len(recs)}

    # Macro-average metrics across classes
    results["macro_auroc"] = macro_average(results["per_class"], "auroc_mean")  # macro-average AUROC across classes
    LOCAL = ["landslides"]
    SCENE = ["burn_scars", "floods", "mines"]
    results["macro_auroc_local"] = macro_average(
        {k: v for k, v in results["per_class"].items() if k in LOCAL}, "auroc_mean")  # macro-average AUROC across local-anomaly classes
    results["macro_auroc_scene"] = macro_average(
        {k: v for k, v in results["per_class"].items() if k in SCENE}, "auroc_mean")  # macro-average AUROC across whole-scene anomaly classes
    print(f"macro AUROC  local-anomaly (landslides,mines): {results['macro_auroc_local']:.4f}")
    print(f"macro AUROC  whole-scene (burn_scars,floods) : {results['macro_auroc_scene']:.4f}")

    results["macro_ap_gain"] = macro_average(results["per_class"], "ap_gain_mean")  # macro-average AP gain across classes


    # -------------------- Alpha sensitivity curve (how results vary with alpha)
    results["alpha_sweep"] = {}
    for a in ALPHA_GRID:
        pc = {}
        for cls in scored:
            recs = evaluate(scored[cls]["test"], a, recon_scaler, attn_scaler, variant, sign)
            if recs:
                pc[cls] = aggregate(recs)
        results["alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")


    # -------------------- Choose best single number score per image
    # different ways to combine pixel-level scores into a single number per image (p99, median, mean, p99/median, p99-median)
    IMG_STATS = {
        "p99":         lambda s: float(np.percentile(s, 99)),
        "median":      lambda s: float(np.median(s)),
        "mean":        lambda s: float(s.mean()),
        "p99_over_med": lambda s: float(np.percentile(s, 99) / max(np.median(s), 1e-9)),
        "p99_minus_med": lambda s: float(np.percentile(s, 99) - np.median(s)),
    }

    def image_score(m, a, sc, stat="p99"):
        # Compute a single number score per image
        return IMG_STATS[stat](fused_scores(m, a, recon_scaler, sc, variant, sign))

    img_choice = {}
    for cls in scored:
        best = None
        for stat in IMG_STATS:  # try every possible single score
            for a in ALPHA_GRID:
                nv = np.array([image_score(m, a, attn_scaler, stat) for m in normal_maps])  # single scores for normal images
                av = np.array([image_score(m, a, attn_scaler, stat) for m in scored[cls]["val"]])  # single scores for validation set anomalous images
                if not len(av):
                    continue
                # Compute image-level AUROC
                sc_ = image_level_auroc(nv, av)

                if best is None or sc_ > best[2]:
                    best = (a, stat, sc_)

        img_choice[cls] = {"alpha": best[0], "stat": best[1], "val_auroc": best[2]} if best else \
                          {"alpha": ALPHA_GRID[-1], "stat": "p99", "val_auroc": float("nan")}
        print(f"  {cls:<12} image-level: alpha={img_choice[cls]['alpha']} stat={img_choice[cls]['stat']} (val AUROC {img_choice[cls]['val_auroc']:.4f})")

    results["image_level_selection_per_class"] = img_choice
    _v = [c["val_auroc"] for c in img_choice.values() if np.isfinite(c["val_auroc"])]
    val_macro = float(np.mean(_v)) if _v else float("nan")
    results["image_level_selection"] = {
        "per_class": {c: {"alpha": v["alpha"], "stat": v["stat"]} for c, v in img_choice.items()},
        "val_macro_auroc": val_macro,
        "pixel_level_alpha": alpha,
    }
    print(f"image-level val macro AUROC (per-class alpha+stat): {val_macro:.4f} (pixel-level alpha {alpha})")


    # -------------------- Compute image-level AUROC and detection rate on test set at a fixed threshold, using the selected alpha and stat per class
    results["image_level_auroc"] = {}
    normal_img_by_cls = {}
    for cls in scored:
        a, stat = img_choice[cls]["alpha"], img_choice[cls]["stat"]
        nv = np.array([image_score(m, a, attn_scaler, stat) for m in normal_maps])
        av = np.array([image_score(m, a, attn_scaler, stat) for m in scored[cls]["test"]])
        normal_img_by_cls[cls] = (nv, av)
        results["image_level_auroc"][cls] = image_level_auroc(nv, av)

    # Compute number of anomalous images detected at a fixed threshold, and false alarms on normal images.
    results["image_level_detection"] = {}
    results["image_thresholds"] = {}
    for cls, (nv, av) in normal_img_by_cls.items():
        thr = threshold_at_fpr(nv, OP_TARGET_FPR)
        results["image_thresholds"][cls] = {"value": float(thr), "target_fpr": OP_TARGET_FPR}
        results["image_level_detection"][cls] = detection_at_operating_point(nv, av, thr)

    # Sensitivity curve
    results["image_level_alpha_sweep"] = {}
    for a in ALPHA_GRID:
        nv = np.array([image_score(m, a, attn_scaler) for m in normal_maps])
        pc = {cls: {"auroc_mean": image_level_auroc(
                        nv, np.array([image_score(m, a, attn_scaler) for m in scored[cls]["test"]]))}
              for cls in scored}
        results["image_level_alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")


    # -------------------- Classical baselines on the same test images (RX, PCA, SpectralLinearPredictor)
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



    # -------------------- Save score maps
    if SAVE_MAPS_PER_CLASS:
        maps_dir = ARTIFACTS_DIR / f"maps_{tag}"
        maps_dir.mkdir(parents=True, exist_ok=True)
        for cls in scored:
            usable = [m for m in scored[cls]["test"] if 0 < m["mask"].mean() < 1] or scored[cls]["test"]
            for m in usable[:SAVE_MAPS_PER_CLASS]:
                _s = _standardize_map(fused_scores(m, alpha, recon_scaler, attn_scaler, variant, sign))
                np.savez_compressed(
                    maps_dir / f"{cls}_{m['name']}.npz",
                    fused=_s,
                    pred_mask=(_s >= pixel_thresholds[cls]["value"]).astype(np.uint8),
                    recon=m["recon"], entropy=m["center_entropy"],
                    received=m["center_received"], mask=m["mask"],
                )


    # -------------------- Save results
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)


    # Print the table
    print("\n" + "=" * 92)
    print(f"{'class':<13}{'n':>5}{'skip':>6}{'base':>8}{'AUROC':>17}{'AP':>9}{'AP/base':>9}"
          f"{'imgAUROC':>10}{'detRate':>9}{'opF1':>8}{'trivF1':>8}{'RX':>8}{'PCA':>8}")
    print("-" * 92)
    for cls, m in results["per_class"].items():
        print(f"{cls:<13}{m['n_images']:>5}{m['n_skipped_single_class']:>6}"
              f"{m['base_rate_mean']:>8.3f}{m['auroc_mean']:>11.4f}+-{m['auroc_std']:<4.3f}"
              f"{m['ap_mean']:>9.4f}{m['ap_gain_mean']:>9.2f}"

              f"{results['image_level_auroc'][cls]:>10.4f}"
              f"{results['image_level_detection'][cls]['detection_rate']:>9.3f}"
              f"{m.get('op_f1', float('nan')):>8.3f}"
              f"{m.get('trivial_f1', float('nan')):>8.3f}"
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


    # -------------------- Sweep checkpoints on the validation set
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