import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from network.transformer_ae import SpectralSpatialTransformerAE

from data.transforms import FeatureBuilder
from data.dataset import _load_image
from evaluation.metrics import (per_image_metrics, roc_auc, aggregate, image_level_auroc, macro_average,
                                operating_point_metrics, threshold_at_fpr, roc_3d,
                                average_precision, classification_metrics, evaluate, fit_logistic, apply_logistic,
                                )
from evaluation.scoring import attention_statistic, fuse, RobustScaler, ATTN_VARIANTS, SIGNS, IMG_STATS, standardize_map, score_image_list
from evaluation.baselines import PCAReconstructor, rx_score, SpectralLinearPredictor, spectral_index_score, SPECTRAL_INDICES

from utils.artifacts import load_and_check_stats
from utils.seed import set_seed
from utils.splits import split_eval, split_fingerprint, split_normals, subsample

from utils.globals import (
    ALPHA_GRID, OP_FPR_GRID, OP_TARGET_FPR, ARTIFACTS_DIR, CHUNK_PIXELS, CLIP_REFLECTANCE,
    DECODER_HIDDEN, D_MODEL, DROPOUT, EMBED_HIDDEN, FEATURE_DIM, FFN_HIDDEN, MASK_CENTER,
    MAX_IMAGES_PER_CLASS, MAX_NORMAL_CALIB_IMAGES, MAX_NORMAL_TEST_IMAGES, N_BANDS,
    N_DCT_COEFFS, N_HEADS, N_LAYERS, N_TOKENS, PATCH_SIZE, PCA_COMPONENTS, POS_ENCODING,
    SAVE_MAPS_PER_CLASS, SEED, SELECTION_EXCLUDE, REPORT_EXCLUDE, TEST_CLASSES, USE_DCT, USE_RAW,
    SPECTRAL_MASK_N, SPECTRAL_MASK_EVAL_GROUPS, LOCAL_CLASSES, SCENE_CLASSES,
    SPECTRAL_BAND_GROUPS,
    images_per_class,
)



# ================================================================================ Aux functions
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device):
    """
        Build the model and move it to the device.
    """
    return SpectralSpatialTransformerAE(
        feature_dim=FEATURE_DIM, n_bands=N_BANDS, patch_size=PATCH_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ffn_hidden=FFN_HIDDEN,
        dropout=DROPOUT, embed_hidden=EMBED_HIDDEN, decoder_hidden=DECODER_HIDDEN,
        pos_encoding=POS_ENCODING, mask_center=MASK_CENTER,
    ).to(device)



# ================================================================================ Context object
@dataclass
class Ctx:
    """
        Shared evaluation state.
        Computes Anomaly Score and single score (at image level).
    """
    args: object
    device: object
    stats: dict
    model: object
    builder: object
    tag: str

    # filled by load_splits / score_images
    normal_val_files: list = field(default_factory=list)
    normal_test_files: list = field(default_factory=list)
    eval_splits: dict = field(default_factory=dict)
    normal_val_maps: list = field(default_factory=list)
    normal_test_maps: list = field(default_factory=list)
    scored: dict = field(default_factory=dict)
    svdd_center: object = None

    # filled by calibrate / select_fusion
    recon_scaler: object = None
    attn_scalers: dict = field(default_factory=dict)
    alpha: float = None
    variant: str = None
    sign: float = None

    # filled by the selection stages
    pixel_thresholds: dict = field(default_factory=dict)
    img_choice: dict = field(default_factory=dict)
    img_combo: dict = field(default_factory=dict)

    results: dict = field(default_factory=dict)


    def score(self, maps, alpha=None, variant=None, sign=None):
        """
            Compute anomaly score for one image.
        """
        a = self.alpha if alpha is None else alpha
        v = self.variant if variant is None else variant
        g = self.sign if sign is None else sign
        attn = attention_statistic(maps, v, N_TOKENS)
        return fuse(maps["recon"], attn, a, self.recon_scaler, self.attn_scalers[v], g)  # Anomaly Score

    def image_score(self, maps, stat, alpha=None, variant=None, sign=None):
        """
            Compute one single score per image, using one of IMG_STATS.
        """
        return IMG_STATS[stat](self.score(maps, alpha, variant, sign))



# ================================================================================ Evaluation pipeline stages
def build_context(args):
    """
        Build the evaluation context: device, stats, tag, feature builder, model.
    """
    set_seed(SEED)
    device = get_device()
    stats = load_and_check_stats()
    tag = "random_init" if args.random_init else "sst"

    # Build the model (and the feature builder)
    builder = FeatureBuilder(N_BANDS, N_DCT_COEFFS, USE_RAW, USE_DCT, stats["feature_mean"], stats["feature_std"]).to(device)
    model = build_model(device)

    # Load the model weights (unless random_init)
    if args.random_init:
        print("RANDOM-INIT CONTROL (untrained weights)")
    else:
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
        # load ckpt model weights
        model.load_state_dict(ck["model"])
        print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.6f})")

        sc = ck.get("svdd_center")
        ctx_svdd_center = None if sc is None else sc.to(device)

    model.eval()

    # Build the context object
    ctx = Ctx(args=args, device=device, stats=stats, model=model, builder=builder, tag=tag)
    ctx.results = {
        "tag": tag,
        "split_fingerprint": split_fingerprint(),
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
    }
    return ctx


def load_splits(ctx):
    """
        Load the normal and anomalous splits and store them in the context.

        Normal images:
            val: calibration, threshold reference, and every selection decision
            test: the negative class for testing
        
        Anomalous images:
            val: selection of alpha, attention variant, sign, and image-level statistic
            test: the positive class for testing
    """
    # Load normal splits
    _, normal_val, normal_test = split_normals()  # split normal images
    ctx.normal_val_files = subsample(normal_val, MAX_NORMAL_CALIB_IMAGES)
    ctx.normal_test_files = subsample(normal_test, MAX_NORMAL_TEST_IMAGES)
    # print(f"\nnormals: {len(ctx.normal_val_files)} for calibration/selection, {len(ctx.normal_test_files)} held out for reporting")

    # Load anomalous splits
    ctx.eval_splits = split_eval()  # split anomalous images

    ctx.results["n_normal_calib"] = len(ctx.normal_val_files)
    ctx.results["n_normal_test"] = len(ctx.normal_test_files)


def calibrate(ctx):
    """
        Fit the robust scalers on normal VAL images.
        Used to standardize the terms of the Anomaly Score (ReconError and AttentionOutlier).
    """
    print(f"\nCalibrating on {len(ctx.normal_val_files)} normal images")

    # Compute Anomaly Score terms (ReconError and AttentionOutlier) for every pixel of every normal VAL image
    ctx.normal_val_maps = score_image_list(ctx.model, ctx.builder, ctx.normal_val_files, ctx.device, "normal-val", with_masks=False, svdd_center=ctx.svdd_center)

    # Fit ReconError scaler
    ctx.recon_scaler = RobustScaler.fit(np.concatenate([m["recon"].ravel() for m in ctx.normal_val_maps]))
    # Fit AttentionOutlier scalers (one per variant)
    ctx.attn_scalers = {
        v: RobustScaler.fit(np.concatenate(
            [attention_statistic(m, v, N_TOKENS).ravel() for m in ctx.normal_val_maps]))
        for v in ATTN_VARIANTS
    }

    ctx.results["calibration"] = {
        "recon": ctx.recon_scaler.to_dict(),
        **{f"attn_{v}": s.to_dict() for v, s in ctx.attn_scalers.items()}
    }


def score_images(ctx):
    """
        Compute Anomaly Score terms for each pixel of each image in the anomalous val/test splits and the normal test set.
    """
    # Anomalous images: val/test
    for cls in TEST_CLASSES:
        ctx.scored[cls] = {}
        for part in ("val", "test"):
            items = subsample(ctx.eval_splits[cls][part], images_per_class(cls, part))
            print(f"\nScoring {cls}/{part}: {len(items)} images")
            ctx.scored[cls][part] = score_image_list(ctx.model, ctx.builder, items, ctx.device, f"{cls}/{part}", with_masks=True, svdd_center=ctx.svdd_center)

    # Normal images: test
    print(f"\nScoring normal/test: {len(ctx.normal_test_files)} images")
    ctx.normal_test_maps = score_image_list(ctx.model, ctx.builder, ctx.normal_test_files, ctx.device, "normal-test", with_masks=False, svdd_center=ctx.svdd_center)


def select_fusion(ctx):
    """
        Choose on VAL the parameters for the Anomaly Score (alpha, attention variant, and sign).
    """
    if ctx.args.fixed_alpha is not None:
        selection = {"alpha": ctx.args.fixed_alpha, "variant": ATTN_VARIANTS[0], "sign": 1.0,
                     "val_macro_auroc": None, "note": "alpha fixed on the command line"}
    else:
        best = None
        for variant in ATTN_VARIANTS:   # try all attention variants (entropy or received)
            for sign in SIGNS:            # try both signs for the attention outlier
                for alpha in ALPHA_GRID:    # try all alpha values (weighting of ReconError vs AttentionOutlier)

                    # Compute pixel-level metrics for each scored anomalous VAL image and aggregate them per class
                    per_class = {}
                    for cls in ctx.scored:
                        # Compute pixel-level metrics for each image in ctx.scored[cls]["val"]
                        recs = evaluate(ctx, ctx.scored[cls]["val"], alpha, variant, sign)
                        # Aggregate the metrics over images of the same class
                        if recs:
                            per_class[cls] = aggregate(recs)

                    # Compute macro-average AUROC over classes (excluding SELECTION_EXCLUDE)
                    score = macro_average(per_class, "auroc_mean", exclude=SELECTION_EXCLUDE)

                    # best combination
                    if best is None or score > best["val_macro_auroc"]:
                        best = {"alpha": alpha, "variant": variant, "sign": sign, "val_macro_auroc": score}

        selection = best
        print(f"\nSelected on VAL (excluding {SELECTION_EXCLUDE}): {selection}")

    # Store the selection
    ctx.alpha, ctx.variant, ctx.sign = selection["alpha"], selection["variant"], selection["sign"]
    ctx.results["selection"] = selection


def select_pixel_thresholds(ctx):
    """
        Choose one pixel-level decision threshold per class, on VAL, for the Anomaly Score.

        If Anomaly Score > threshold, then the pixel is flagged as anomalous.

        Two candidate decision rules:
            adaptive: threshold applied to the per-image standardised map
            global: threshold applied to the raw score
    """
    # Concatenate all standardised normal VAL pixel scores
    normal_pixel_scores = np.concatenate([standardize_map(ctx.score(m)).ravel() for m in ctx.normal_val_maps])
    # Concatenate all raw normal VAL pixel scores
    raw_normal_scores = np.concatenate([ctx.score(m).ravel() for m in ctx.normal_val_maps])

    REF = {"adaptive": normal_pixel_scores, "global": raw_normal_scores}

    def _mapped(m, mode):
        """
            Return the map to which the threshold will be applied, depending on the mode.
        """
        return standardize_map(ctx.score(m)) if mode == "adaptive" else ctx.score(m)


    for cls in ctx.scored:
        usable = [m for m in ctx.scored[cls]["val"] if 0 < m["mask"].mean() < 1]
        best = None
        for mode in ("adaptive", "global"):
            for f in OP_FPR_GRID:
                thr = threshold_at_fpr(REF[mode], f)
                f1s = [operating_point_metrics(_mapped(m, mode), m["mask"], thr)["op_f1"]
                       for m in usable]

                if f1s and (best is None or np.mean(f1s) > best[3]):
                    best = (mode, f, thr, float(np.mean(f1s)))

        if best is None:
            best = ("adaptive", OP_TARGET_FPR, threshold_at_fpr(normal_pixel_scores, OP_TARGET_FPR), float("nan"))

        ctx.pixel_thresholds[cls] = {"mode": best[0], "target_fpr": best[1], "value": best[2], "val_op_f1": best[3]}
        print(f"  {cls:<12} pixel operating point on val: rule={best[0]:<8} FPR={best[1]:.0%} thr={best[2]:.4f} (val op_F1 {best[3]:.4f})")

    ctx.results["pixel_thresholds"] = ctx.pixel_thresholds

    # global threshold for comparison
    raw_normal = np.concatenate([ctx.score(m).ravel() for m in ctx.normal_val_maps])  # raw (non-standardised) score thresholds from normal VAL pixels
    ctx.results["global_pixel_threshold"] = {f"fpr_{f}": float(threshold_at_fpr(raw_normal, f)) for f in OP_FPR_GRID}

    # Class-agnostic pixel threshold
    pooled_val = [m for cls in ctx.scored if cls not in SELECTION_EXCLUDE
                  for m in ctx.scored[cls]["val"] if 0 < m["mask"].mean() < 1]
    best = None
    for mode in ("adaptive", "global"):
        for f in OP_FPR_GRID:
            thr = threshold_at_fpr(REF[mode], f)
            f1s = [operating_point_metrics(_mapped(m, mode), m["mask"], thr)["op_f1"]
                   for m in pooled_val]

            if f1s and (best is None or np.mean(f1s) > best[3]):
                best = (mode, f, thr, float(np.mean(f1s)))
    if best:
        ctx.pixel_thresholds["__pooled__"] = {"mode": best[0], "target_fpr": best[1], "value": best[2], "val_op_f1": best[3]}
        print(f"  {'POOLED':<12} pixel operating point on val: rule={best[0]:<8} FPR={best[1]:.0%} thr={best[2]:.4f} (val op_F1 {best[3]:.4f})")



def report_pixel_level(ctx):
    """
        Pixel-level metrics on TEST for each class (at the per class selected thresholds) and macro-averaged over classes.
    """
    ctx.results["per_class"] = {}
    for cls in ctx.scored:
        # Pixel-evel metrics for each scored anomalous TEST image
        recs = evaluate(ctx, ctx.scored[cls]["test"], with_3d_roc=True)
        # Threshold selected on VAL for this class
        thr = ctx.pixel_thresholds[cls]["value"]
        # class-agnostic threshold (selected on VAL pooled over all classes)
        pooled_thr = ctx.pixel_thresholds.get("__pooled__", {}).get("value", thr)
        # Pixel-level metrics at the selected threshold, under the rule chosen on val.
        mode = ctx.pixel_thresholds[cls].get("mode", "adaptive")
        _m = lambda mm: standardize_map(ctx.score(mm)) if mode == "adaptive" else ctx.score(mm)
        ops = [operating_point_metrics(_m(m), m["mask"], thr)
               for m in ctx.scored[cls]["test"] if 0 < m["mask"].mean() < 1]
        
        op_mean = {k: float(np.mean([o[k] for o in ops])) for k in ops[0]} if ops else {}

        # Same images, class-agnostic threshold
        _pmode = ctx.pixel_thresholds.get("__pooled__", {}).get("mode", mode)
        _pm = lambda mm: standardize_map(ctx.score(mm)) if _pmode == "adaptive" else ctx.score(mm)
        p_ops = [operating_point_metrics(_pm(m), m["mask"], pooled_thr)
                 for m in ctx.scored[cls]["test"] if 0 < m["mask"].mean() < 1]
        op_mean.update({f"pooled_{k}": float(np.mean([o[k] for o in p_ops]))
                        for k in ("op_f1", "op_iou", "op_precision", "op_recall")}
                       if p_ops else {})

        # Same metrics, with a single global raw-score threshold (for comparison)
        g_thr = ctx.results["global_pixel_threshold"][f"fpr_{ctx.pixel_thresholds[cls]['target_fpr']}"]
        g_ops = [operating_point_metrics(ctx.score(m), m["mask"], g_thr)
                 for m in ctx.scored[cls]["test"] if 0 < m["mask"].mean() < 1]
        op_mean.update({f"global_{k}": float(np.mean([o[k] for o in g_ops]))
                        for k in ("op_f1", "op_iou", "op_precision", "op_recall", "op_fpr")}
                       if g_ops else {})

        # Handle skipped images (with single-class masks, AUROC undefined)
        saturated = [m for m in ctx.scored[cls]["test"] if m["mask"].mean() == 1]  # all pixels are anomalous
        empty = [m for m in ctx.scored[cls]["test"] if m["mask"].mean() == 0]  # all pixels are normal
        sat_recall = [float((_m(m) >= thr).mean()) for m in saturated]
        empty_fpr = [float((_m(m) >= thr).mean()) for m in empty]

        n_skip = len(ctx.scored[cls]["test"]) - len(recs)

        ctx.results["per_class"][cls] = {
            **aggregate(recs), **op_mean,
            "n_skipped_single_class": n_skip,
            "n_test_total": len(ctx.scored[cls]["test"]),
            "n_saturated": len(saturated),
            "saturated_recall": float(np.mean(sat_recall)) if sat_recall else float("nan"),
            "n_empty_mask": len(empty),
            "empty_mask_fpr": float(np.mean(empty_fpr)) if empty_fpr else float("nan"),
        }
        if n_skip:
            print(f"  {cls:<12} dropped {n_skip}/{len(ctx.scored[cls]['test'])} images with single-class masks (AUROC undefined)")

    pc = ctx.results["per_class"]
    ctx.results["macro_auroc_all"] = macro_average(pc, "auroc_mean")
    ctx.results["macro_ap_gain_all"] = macro_average(pc, "ap_gain_mean")
    ctx.results["macro_auroc"] = macro_average(pc, "auroc_mean", exclude=REPORT_EXCLUDE)
    ctx.results["macro_ap_gain"] = macro_average(pc, "ap_gain_mean", exclude=REPORT_EXCLUDE)
    ctx.results["report_excluded"] = REPORT_EXCLUDE

    # Distinguish between local and scene anomalies
    ctx.results["macro_auroc_local"] = macro_average(
        {k: v for k, v in pc.items() if k in LOCAL_CLASSES}, "auroc_mean", exclude=REPORT_EXCLUDE)
    ctx.results["macro_auroc_scene"] = macro_average(
        {k: v for k, v in pc.items() if k in SCENE_CLASSES}, "auroc_mean", exclude=REPORT_EXCLUDE)
    print(f"macro AUROC  headline (excluding {REPORT_EXCLUDE}): {ctx.results['macro_auroc']:.4f}")
    print(f"macro AUROC  all classes                          : {ctx.results['macro_auroc_all']:.4f}")


def select_image_scoring(ctx):
    """
        Choose how to collapse a score map into one number.
        For each anomaly type, select the alpha and statistic that maximizes image-level AUROC on VAL.
        Pooled over all anomaly types, select one alpha and statistic that maximizes image-level AUROC on VAL.
    """
    for cls in ctx.scored:
        best = None
        for stat in IMG_STATS:  # try different ways of collapsing the score map into one number (mean, max, p99, etc.)
            for a in ALPHA_GRID:  # try different alpha values (weighting of ReconError vs AttentionOutlier)
                nv = np.array([ctx.image_score(m, stat, a) for m in ctx.normal_val_maps])
                av = np.array([ctx.image_score(m, stat, a) for m in ctx.scored[cls]["val"]])
                if not len(av):
                    continue

                sc = image_level_auroc(nv, av)  # Compute image-level AUROC

                if best is None or sc > best[2]:
                    best = (a, stat, sc)

        # Store the selected alpha and statistic for this anomaly type
        ctx.img_choice[cls] = ({"alpha": best[0], "stat": best[1], "val_auroc": best[2]}
                               if best else
                               {"alpha": ALPHA_GRID[-1], "stat": "p99", "val_auroc": float("nan")})
        print(f"  {cls:<12} image-level: alpha={ctx.img_choice[cls]['alpha']} "
              f"stat={ctx.img_choice[cls]['stat']} (val AUROC {ctx.img_choice[cls]['val_auroc']:.4f})")

    # Need one statistic for all classes (normal vs anomalous image of any type)
    pooled_val = [m for cls in ctx.scored if cls not in SELECTION_EXCLUDE
                  for m in ctx.scored[cls]["val"]]
    best = None
    for stat in IMG_STATS:
        for a in ALPHA_GRID:
            nv = np.array([ctx.image_score(m, stat, a) for m in ctx.normal_val_maps])
            av = np.array([ctx.image_score(m, stat, a) for m in pooled_val])
            if not len(av):
                continue

            sc = image_level_auroc(nv, av)  # Compute image-level AUROC

            if best is None or sc > best[2]:
                best = (a, stat, sc)

    ctx.img_choice["__pooled__"] = {"alpha": best[0], "stat": best[1], "val_auroc": best[2]}
    print(f"  {'POOLED':<12} best single statistic: alpha={best[0]} stat={best[1]} (val AUROC {best[2]:.4f})")

    # Combine all statistics instead of keeping only the best one
    a_star = best[0]

    def feats(maps_list):
        return np.array([[ctx.image_score(m, s, a_star) for s in IMG_STATS] for m in maps_list])

    Xn, Xa = feats(ctx.normal_val_maps), feats(pooled_val)
    X = np.vstack([Xn, Xa])
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xa))])
    mu, sd = X.mean(0), X.std(0) + 1e-9  # standardise: the statistics have wildly different ranges
    # fit a logistic regression on the pooled VAL set, to combine the statistics into one score
    w = fit_logistic((X - mu) / sd, y)
    val_auroc = image_level_auroc(apply_logistic((Xn - mu) / sd, w), apply_logistic((Xa - mu) / sd, w))

    ctx.img_combo = {"alpha": a_star, "stats": list(IMG_STATS),
                     "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(),
                     "val_auroc": float(val_auroc)}
    print(f"  {'POOLED':<12} combined statistics:   alpha={a_star} (val AUROC {val_auroc:.4f}, single best was {best[2]:.4f})")
    ctx.results["image_level_combo"] = ctx.img_combo


    ctx.results["image_level_selection"] = ctx.img_choice


def report_image_level(ctx):
    """
        Binary classification of whole images on TEST: normal vs anomalous.
    """
    ctx.results["image_level_auroc"] = {}
    ctx.results["image_level_ap"] = {}
    ctx.results["image_level_detection"] = {}
    ctx.results["image_thresholds"] = {}


    def report(name, choice, anomalous_maps):
        # compute image-level metrics for one anomaly type

        a, stat = choice["alpha"], choice["stat"]
        nv_val = np.array([ctx.image_score(m, stat, a) for m in ctx.normal_val_maps])
        nv = np.array([ctx.image_score(m, stat, a) for m in ctx.normal_test_maps])
        av = np.array([ctx.image_score(m, stat, a) for m in anomalous_maps])
        if not len(av) or not len(nv):
            return
        thr = threshold_at_fpr(nv_val, OP_TARGET_FPR)      # chosen on VAL normals
        y = np.concatenate([np.zeros(len(nv), dtype=np.int8), np.ones(len(av), dtype=np.int8)])
        s = np.concatenate([nv, av])

        ctx.results["image_level_auroc"][name] = image_level_auroc(nv, av)
        ctx.results["image_level_ap"][name] = average_precision(y, s)
        ctx.results["image_thresholds"][name] = {"value": float(thr), "target_fpr": OP_TARGET_FPR,
                                                 "chosen_on": "normal val"}
        ctx.results["image_level_detection"][name] = classification_metrics(nv, av, thr)

    # Compute image-level metrics for each anomaly type
    for cls in ctx.scored:
        report(cls, ctx.img_choice[cls], ctx.scored[cls]["test"])
    # Compute image-level metrics for the pooled set of all anomaly types
    pooled_test = [m for cls in ctx.scored for m in ctx.scored[cls]["test"]]
    report("__pooled__", ctx.img_choice["__pooled__"], pooled_test)

    # Same pooled task, score images with the combined statistics
    if ctx.img_combo:
        c = ctx.img_combo
        mu, sd, w, a = np.array(c["mu"]), np.array(c["sd"]), np.array(c["w"]), c["alpha"]
        def combo_scores(maps_list):
            X = np.array([[ctx.image_score(m, s, a) for s in c["stats"]] for m in maps_list])
            return apply_logistic((X - mu) / sd, w)
        nv_val = combo_scores(ctx.normal_val_maps)
        nv, av = combo_scores(ctx.normal_test_maps), combo_scores(pooled_test)
        thr = threshold_at_fpr(nv_val, OP_TARGET_FPR)      # threshold from VAL normals, as everywhere
        y = np.concatenate([np.zeros(len(nv), dtype=np.int8), np.ones(len(av), dtype=np.int8)])
        ctx.results["image_level_auroc"]["__pooled_combo__"] = image_level_auroc(nv, av)
        ctx.results["image_level_ap"]["__pooled_combo__"] = average_precision(y, np.concatenate([nv, av]))
        ctx.results["image_thresholds"]["__pooled_combo__"] = {
            "value": float(thr), "target_fpr": OP_TARGET_FPR, "chosen_on": "normal val"}
        ctx.results["image_level_detection"]["__pooled_combo__"] = classification_metrics(nv, av, thr)

    d = ctx.results["image_level_detection"].get("__pooled__")
    if d:
        print("\nPOOLED image classification (normal vs any anomalous), on TEST:")
        print(f"  AUROC {ctx.results['image_level_auroc']['__pooled__']:.4f}   "
              f"recall {d['recall']:.3f}   false alarms {d['false_alarm_rate']:.3f}   "
              f"balanced acc {d['balanced_accuracy']:.3f}")
        print(f"  confusion: TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']} "
              f"(prevalence {d['prevalence']:.2f})")


def report_false_alarms(ctx):
    """
        Compute the fraction of pixels flagged as anomalous in each normal TEST image, for each anomaly type.
    """
    ctx.results["normal_scene_false_alarms"] = {}
    for cls in ctx.scored:
        thr = ctx.pixel_thresholds[cls]["value"]
        _mode = ctx.pixel_thresholds[cls].get("mode", "adaptive")
        rates = [float(((standardize_map(ctx.score(m)) if _mode == "adaptive" else ctx.score(m)) >= thr).mean())
                 for m in ctx.normal_test_maps]
        
        ctx.results["normal_scene_false_alarms"][cls] = {
            "mean_fraction_flagged": float(np.mean(rates)),
            "median_fraction_flagged": float(np.median(rates)),
            "p90_fraction_flagged": float(np.percentile(rates, 90)),
            "n_normal_images": len(rates),
        }
        print(f"  {cls:<12} clean-scene false alarms: {np.mean(rates):.1%} of pixels "
              f"(p90 {np.percentile(rates, 90):.1%})")

    # How well the flagged fraction tracks the amount of anomaly actually present
    truth, flagged = [], []
    for cls in ctx.scored:
        thr_c = ctx.pixel_thresholds[cls]["value"]
        mode_c = ctx.pixel_thresholds[cls].get("mode", "adaptive")
        
        for m in ctx.scored[cls]["test"]:
            s = standardize_map(ctx.score(m)) if mode_c == "adaptive" else ctx.score(m)
            truth.append(float(m["mask"].mean()))
            flagged.append(float((s >= thr_c).mean()))

    if len(truth) > 2 and np.std(flagged) > 1e-9:
        ctx.results["flagged_fraction_diagnostic"] = {
            "mean_flagged": float(np.mean(flagged)),
            "std_flagged": float(np.std(flagged)),
            "corr_with_true_base_rate": float(np.corrcoef(truth, flagged)[0, 1]),
            "n_images": len(truth),
        }
        print(f"  flagged fraction: mean {np.mean(flagged):.3f} std {np.std(flagged):.3f} "
              f"corr with true base rate {np.corrcoef(truth, flagged)[0, 1]:+.3f}")

    
    if any("svdd" in m for m in ctx.normal_test_maps):
        v = np.concatenate([m["svdd"].ravel() for m in ctx.normal_test_maps])
        ctx.results["svdd_collapse_check"] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "coef_variation": float(v.std() / max(v.mean(), 1e-12)),
        }
        print(f"  SVDD distance on normal scenes: mean {v.mean():.4f} std {v.std():.4f} "
              f"CV {v.std()/max(v.mean(),1e-12):.3f}  (CV < 0.05 suggests collapse)")


def report_group_attribution(ctx):
    """
        Which band group flags each anomaly type.
    """
    if not (SPECTRAL_BAND_GROUPS and any(
            m.get("group_recon") is not None for cls in ctx.scored for m in ctx.scored[cls]["test"])):
        return                                  # random-band masking: no groups to attribute to

    names = sorted(SPECTRAL_BAND_GROUPS)        # same order score_image builds band_groups in
    ctx.results["group_attribution"] = {}
    print(f"\n  spectral attribution (per-group AUROC on TEST)")
    print("    " + f"{'class':<12}" + "".join(f"{n:>14}" for n in names) + f"{'best':>14}")

    for cls in ctx.scored:
        per_group = {n: [] for n in names}
        for m in ctx.scored[cls]["test"]:
            g = m.get("group_recon")
            if g is None or not (0 < m["mask"].mean() < 1):
                continue                        # AUROC undefined on a single-class mask
            y = m["mask"].ravel()
            for gi, n in enumerate(names):
                per_group[n].append(roc_auc(y, g[gi].ravel()))
        if not per_group[names[0]]:
            continue
        means = {n: float(np.mean(v)) for n, v in per_group.items()}
        best = max(means, key=means.get)
        ctx.results["group_attribution"][cls] = {
            "per_group_auroc": means,
            "n_images": len(per_group[names[0]]),
            "best_group": best,
            "bands": {n: SPECTRAL_BAND_GROUPS[n] for n in names},
        }
        print("    " + f"{cls:<12}" + "".join(f"{means[n]:>14.4f}" for n in names) + f"{best:>14}")


def report_alpha_sweeps(ctx):
    """
        Sensitivity curves (never used for selection).
    """
    ctx.results["alpha_sweep"] = {}
    for a in ALPHA_GRID:
        pc = {}
        for cls in ctx.scored:
            recs = evaluate(ctx, ctx.scored[cls]["test"], alpha=a)
            if recs:
                pc[cls] = aggregate(recs)
        ctx.results["alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean", exclude=REPORT_EXCLUDE)

    ctx.results["image_level_alpha_sweep"] = {}
    for a in ALPHA_GRID:
        nv = np.array([ctx.image_score(m, "p99", a) for m in ctx.normal_test_maps])
        pc = {cls: {"auroc_mean": image_level_auroc(
                    nv, np.array([ctx.image_score(m, "p99", a) for m in ctx.scored[cls]["test"]]))}
              for cls in ctx.scored}
        ctx.results["image_level_alpha_sweep"][str(a)] = macro_average(pc, "auroc_mean")


def run_baselines(ctx):
    """
        Classical baseline detectors on the same TEST images: PCA, RX, and SpectralLinearPredictor.
    """
    pca = PCAReconstructor(np.array(ctx.stats["band_mean"]),
                           np.array(ctx.stats["band_cov"]), PCA_COMPONENTS)
    train_files, _, _ = split_normals()
    spec = SpectralLinearPredictor(ctx.stats["band_mean"], ctx.stats["feature_std"][:N_BANDS],
                                   train_files,
                                   lambda p: _load_image(p, CLIP_REFLECTANCE).numpy())
    print(f"SpectralLinearPredictor fitted on {spec.n_pixels_fitted:,} normal pixels")

    ctx.results["baselines"] = {"pca_explained_variance": pca.explained,
                                "rx": {}, "pca": {}, "spectral_linear": {},
                                **{k: {} for k in SPECTRAL_INDICES}}

    for cls in ctx.scored:
        acc = {k: [] for k in ("rx", "pca", "spectral_linear", *SPECTRAL_INDICES)}
        for m in ctx.scored[cls]["test"]:
            img = _load_image(m["img_path"], CLIP_REFLECTANCE).numpy()
        _index_fns = {k: (lambda img, p=p: spectral_index_score(img, p[0], p[1], p[2]))
                      for k, p in SPECTRAL_INDICES.items()}
        for key, fn in (("rx", rx_score), ("pca", pca.score), ("spectral_linear", spec.score), *_index_fns.items()):
                r = per_image_metrics(fn(img), m["mask"])
                if r is not None:
                    acc[key].append(r)
        for key in acc:
            ctx.results["baselines"][key][cls] = aggregate(acc[key])

    for key in ("rx", "pca", "spectral_linear", *SPECTRAL_INDICES):
        ctx.results["baselines"][f"{key}_macro_auroc_all"] = macro_average(
            ctx.results["baselines"][key], "auroc_mean")
        ctx.results["baselines"][f"{key}_macro_auroc"] = macro_average(
            ctx.results["baselines"][key], "auroc_mean", exclude=REPORT_EXCLUDE)


def save_score_maps(ctx):
    """
        Save a few maps per class for the qualitative figure.
    """
    if not SAVE_MAPS_PER_CLASS:
        return
    maps_dir = ARTIFACTS_DIR / f"maps_{ctx.tag}"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for cls in ctx.scored:
        usable = [m for m in ctx.scored[cls]["test"]
                  if 0 < m["mask"].mean() < 1] or ctx.scored[cls]["test"]
        for m in usable[:SAVE_MAPS_PER_CLASS]:
            s = (standardize_map(ctx.score(m))
                 if ctx.pixel_thresholds[cls].get("mode", "adaptive") == "adaptive"
                 else ctx.score(m))
            np.savez_compressed(
                maps_dir / f"{cls}_{m['name']}.npz",
                fused=s,
                pred_mask=(s >= ctx.pixel_thresholds[cls]["value"]).astype(np.uint8),
                recon=m["recon"], entropy=m["center_entropy"],
                received=m["center_received"], mask=m["mask"],
            )
    
    _pmode = ctx.pixel_thresholds.get("__pooled__", {}).get("mode", "adaptive")
    for m in ctx.normal_test_maps[:SAVE_MAPS_PER_CLASS]:
        s = standardize_map(ctx.score(m)) if _pmode == "adaptive" else ctx.score(m)
        # Use the class-agnostic threshold
        thr = ctx.pixel_thresholds.get("__pooled__", {}).get(
            "value", float(np.median([t["value"] for t in ctx.pixel_thresholds.values()])))
        np.savez_compressed(
            maps_dir / f"normal_{m['name']}.npz",
            fused=s, pred_mask=(s >= thr).astype(np.uint8),
            recon=m["recon"], entropy=m["center_entropy"],
            received=m["center_received"], mask=np.zeros_like(s, dtype=np.uint8),
        )


def print_summary(ctx, out_path):
    r = ctx.results
    print("\n" + "=" * 100)
    print(f"{'class':<13}{'n':>5}{'skip':>6}{'base':>8}{'AUROC':>17}{'AP':>9}{'AP/base':>9}"
          f"{'imgAUROC':>10}{'imgRecall':>10}{'opF1':>8}{'trivF1':>8}{'RX':>8}{'PCA':>8}")
    print("-" * 100)
    for cls, m in r["per_class"].items():
        det = r["image_level_detection"].get(cls, {})
        print(f"{cls:<13}{m['n_images']:>5}{m['n_skipped_single_class']:>6}"
              f"{m['base_rate_mean']:>8.3f}{m['auroc_mean']:>11.4f}+-{m['auroc_std']:<4.3f}"
              f"{m['ap_mean']:>9.4f}{m['ap_gain_mean']:>9.2f}"
              f"{r['image_level_auroc'].get(cls, float('nan')):>10.4f}"
              f"{det.get('recall', float('nan')):>10.3f}"
              f"{m.get('op_f1', float('nan')):>8.3f}"
              f"{m.get('trivial_f1', float('nan')):>8.3f}"
              f"{r['baselines']['rx'][cls]['auroc_mean']:>8.4f}"
              f"{r['baselines']['pca'][cls]['auroc_mean']:>8.4f}")
    print("-" * 100)
    print(f"macro AUROC (pixel): model {r['macro_auroc']:.4f} | "
          f"RX {r['baselines']['rx_macro_auroc']:.4f} | "
          f"PCA {r['baselines']['pca_macro_auroc']:.4f} | "
          f"SpectralLinear {r['baselines']['spectral_linear_macro_auroc']:.4f}")
    if "__pooled__" in r["image_level_auroc"]:
        d = r["image_level_detection"]["__pooled__"]
        print(f"POOLED image classification: AUROC {r['image_level_auroc']['__pooled__']:.4f}  "
              f"AP {r['image_level_ap']['__pooled__']:.4f}  recall {d['recall']:.3f}  "
              f"FAR {d['false_alarm_rate']:.3f}  balanced acc {d['balanced_accuracy']:.3f}")
    print(f"selection: alpha={ctx.alpha} variant={ctx.variant} sign={ctx.sign:+.0f}")
    print(f"normals: {r['n_normal_calib']} calibration / {r['n_normal_test']} reporting")
    print(f"wrote {out_path}")
    print("=" * 100)


def run_checkpoint_sweep(ctx):
    """
        Score every epoch checkpoint on VAL.
    """
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
        state = torch.load(cp, map_location=ctx.device, weights_only=False)
        ctx.model.load_state_dict(state["model"])
        ctx.model.eval()
        per_class = {}
        for cls in TEST_CLASSES:
            items = subsample(ctx.eval_splits[cls]["val"], images_per_class(cls, "val"))
            maps = score_image_list(ctx.model, ctx.builder, items, ctx.device, f"sweep/{cls}", with_masks=True, svdd_center=ctx.svdd_center)
            recs = evaluate(ctx, maps)
            if recs:
                per_class[cls] = aggregate(recs)
        sweep[cp.name] = macro_average(per_class, "auroc_mean", exclude=SELECTION_EXCLUDE)
        print(f"{cp.name:<18}{sweep[cp.name]:>14.4f}", flush=True)

    best_cp = max(sweep, key=sweep.get)
    print("-" * 60)
    print(f"best on val: {best_cp}  ({sweep[best_cp]:.4f})")
    print("=" * 60)
    with open(ARTIFACTS_DIR / f"ckpt_sweep_{ctx.tag}.json", "w") as fh:
        json.dump({"sweep": sweep, "best": best_cp, "selection": ctx.results["selection"]}, fh, indent=2)



# ================================================================================ main()
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--ckpt", default=None, help="checkpoint (default artifacts/best.pt)")
    ap.add_argument("-o", "--out", default=None, help="results json (default artifacts/results_<tag>.json)")
    ap.add_argument("--random-init", action="store_true", help="skip loading weights: the untrained control")
    ap.add_argument("--fixed-alpha", type=float, default=None, help="skip selection (1.0 = recon only, 0.0 = attention only)")
    ap.add_argument("--ckpt-sweep", action="store_true", help="score every artifacts/epoch_*.pt on val and report the curve")
    ap.add_argument("--use-best-sweep-ckpt", action="store_true", help="pick the checkpoint by val detection AUROC instead of val loss")
    return ap.parse_args()


def main():
    args = parse_args()
    ctx = build_context(args)
    out_path = args.out or (ARTIFACTS_DIR / f"results_{ctx.tag}.json")

    load_splits(ctx)    # load normal and anomalous splits
    calibrate(ctx)      # fit robust scalers on normal VAL
    score_images(ctx)   # compute Anomaly Score terms for each pixel of each image in the anomalous val/test splits and the normal test set

    select_fusion(ctx)              # choose alpha, attention variant, and sign on VAL
    select_pixel_thresholds(ctx)    # choose one pixel-level decision threshold per class on VAL
    select_image_scoring(ctx)       # choose how to collapse a score map into one number (alpha and statistic) on VAL


    report_pixel_level(ctx)     # pixel-level metrics on TEST for each class (at the per class selected thresholds) and macro-averaged over classes
    report_image_level(ctx)     # binary classification of whole images on TEST: normal vs anomalous
    report_false_alarms(ctx)    # compute the fraction of pixels flagged as anomalous in each normal TEST image, for each anomaly type

    report_group_attribution(ctx)   # which band group detects which anomaly type
    report_alpha_sweeps(ctx)        # sensitivity curves
    run_baselines(ctx)              # classical baseline detectors on the same TEST images (PCA, RX, and SpectralLinearPredictor)


    save_score_maps(ctx)    # save a few maps per class for the qualitative figure

    with open(out_path, "w") as fh:
        json.dump(ctx.results, fh, indent=2)

    print_summary(ctx, out_path)


    if args.ckpt_sweep:
        run_checkpoint_sweep(ctx)   # score every epoch checkpoint on VAL and report the curve


if __name__ == "__main__":
    main()
