import os
import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, jaccard_score, confusion_matrix, roc_curve,
)

from utils.globals import (
    DEVICE, ARTIFACTS_DIR, DATASET_PATH, TEST_CLASSES, SEG_POS_WEIGHT, BATCH_IMAGES, NUM_WORKERS, AMP,
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
    MODEL_VERSION, MODEL_RUN, TARGET_RECALL, TYPE_LOSS_WEIGHT, CLS_THRESHOLD_MODE
)
from network.trans_UNet import build_model
from network.loss import MultiTaskLoss
from data.dataset import NormalImagesDataset, AnomalyImagesDataset, make_loader
from utils.splits import split_normals, split_anomal


# list of intermediate epochs to evaluate
EVAL_EPOCHS = [10, 30, 40]
CLS_THRESHOLD = 0.5  # fallback threshold for image-level classification predictions
SEG_THRESHOLD = 0.5  # fallback threshold for pixel-level segmentation predictions

RESULTS_DIR = ARTIFACTS_DIR



def evaluate_metrics(cls_targets, cls_preds, seg_targets, seg_preds, cls_threshold=0.5, seg_threshold=0.5):
    """
        Computes Image-Level and Pixel-Level Evaluation Metrics.
    """
    cls_targets = np.array(cls_targets)
    cls_preds = np.array(cls_preds)
    cls_binary = (cls_preds >= cls_threshold).astype(int)

    # --- Image-level classification metrics
    try:
        img_auc = roc_auc_score(cls_targets, cls_preds)
    except ValueError:
        img_auc = 0.5

    img_f1 = f1_score(cls_targets, cls_binary, zero_division=0)
    img_prec = precision_score(cls_targets, cls_binary, zero_division=0)
    img_rec = recall_score(cls_targets, cls_binary, zero_division=0)
    try:
        fpr_arr, tpr_arr, _ = roc_curve(cls_targets, cls_preds)
        rec_at_fpr5 = float(np.interp(0.05, fpr_arr, tpr_arr))
    except ValueError:
        rec_at_fpr5 = float("nan")

    # confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(cls_targets, cls_binary, labels=[0, 1]).ravel()


    # --- Pixel-level segmentation/localization metrics
    seg_targets_flat = np.concatenate([t.flatten() for t in seg_targets])
    seg_preds_flat = np.concatenate([p.flatten() for p in seg_preds])
    seg_binary_flat = (seg_preds_flat >= seg_threshold).astype(int)

    try:
        pix_auc_roc = roc_auc_score(seg_targets_flat, seg_preds_flat)
    except ValueError:
        pix_auc_roc = 0.5

    precision_arr, recall_arr, _ = precision_recall_curve(seg_targets_flat, seg_preds_flat)
    pix_auc_pr = auc(recall_arr, precision_arr)

    pix_iou = jaccard_score(seg_targets_flat, seg_binary_flat, zero_division=0)
    
    # Dice Coefficient = 2 * IoU / (1 + IoU)
    intersection = np.logical_and(seg_targets_flat, seg_binary_flat).sum()
    cardinality = seg_targets_flat.sum() + seg_binary_flat.sum()
    pix_dice = (2.0 * intersection) / (cardinality + 1e-6)

    return {
        "Image_AUC_ROC": float(img_auc),
        "Image_F1": float(img_f1),
        "Image_Precision": float(img_prec),
        "Image_Recall": float(img_rec),
        "Image_Recall_at_FPR_5pct": rec_at_fpr5,
        "Pixel_AUC_ROC": float(pix_auc_roc),
        "Pixel_AUC_PR": float(pix_auc_pr),
        "Pixel_IoU": float(pix_iou),
        "Pixel_Dice": float(pix_dice),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def dice_per_image(seg_targets, seg_preds, seg_threshold):
    """
        Mean Dice over anomalous images (every image counts equally, whatever the anomaly size).
    """
    d = []
    for t, p in zip(seg_targets, seg_preds):
        t = np.asarray(t) > 0.5
        if t.any():
            b = np.asarray(p) >= seg_threshold
            d.append(2.0 * np.logical_and(t, b).sum() / (t.sum() + b.sum() + 1e-6))
    return float(np.mean(d)) if d else float("nan")


def run_test_evaluation(checkpoint_name="best.pth", device=DEVICE):
    if isinstance(device, str):
        device = torch.device(device)
    print("=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    # ------ Load model
    model = build_model(pretrained=False).to(device)

    checkpoint_path = RESULTS_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ckpt_epoch = checkpoint.get("epoch", "N/A")
    if checkpoint_name == "best.pth":
        print(f"  'best.pth' at Epoch {ckpt_epoch}")
    else:
        print(f"Loaded {checkpoint_name} (Epoch {ckpt_epoch})")

    # ------ Loss function
    alpha = beta = gamma = 1.0
    criterion = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma, seg_pos_weight=SEG_POS_WEIGHT, delta=TYPE_LOSS_WEIGHT)
    use_amp = AMP and device.type == "cuda"

    # ------ Oriented classification threshold tuning (from checkpoint's validation predictions)
    cls_threshold = CLS_THRESHOLD
    if "val_cls_preds" in checkpoint and "val_cls_targets" in checkpoint:
        v_preds = np.array(checkpoint["val_cls_preds"])
        v_targets = np.array(checkpoint["val_cls_targets"])
        precisions, recalls, thresholds = precision_recall_curve(v_targets, v_preds)
        
        if CLS_THRESHOLD_MODE == "f1":
            f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
            cls_threshold = float(thresholds[int(np.nanargmax(f1s))])
            print(f"Tuned classification threshold: {cls_threshold:.4f} (max val F1={np.nanmax(f1s):.4f})")
        else:
            recalls_aligned = recalls[:-1]
            meets_target = recalls_aligned >= TARGET_RECALL
            if meets_target.any():
                cls_threshold = float(thresholds[meets_target].max())
                print(f"Tuned classification threshold: {cls_threshold:.4f} (targeting recall>={TARGET_RECALL})")
            else:
                print(f"No threshold reaches recall>={TARGET_RECALL} on validation — falling back to threshold=0.5")
    else:
        print("No stored validation predictions in checkpoint — falling back to threshold=0.5")

    # ------ Segmentation threshold tuning (from checkpoint's validation predictions)
    seg_threshold = SEG_THRESHOLD
    if "val_seg_preds" in checkpoint and "val_seg_targets" in checkpoint:
        v_seg_preds = np.asarray(checkpoint["val_seg_preds"], dtype=np.float32).ravel()
        v_seg_targets = np.asarray(checkpoint["val_seg_targets"], dtype=np.float32).ravel()
        precision_arr, recall_arr, seg_thresholds = precision_recall_curve(v_seg_targets, v_seg_preds)
        f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
        best_idx = int(np.nanargmax(f1_arr[:-1]))
        seg_threshold = float(seg_thresholds[best_idx])
        print(f"Tuned segmentation threshold: {seg_threshold:.4f} (maximizing val pixel F1={f1_arr[best_idx]:.4f})")
    else:
        print("No stored validation pixel predictions in checkpoint — falling back to threshold=0.5")

    # ------ Normal and Anomalous Test Datasets
    _, _, normal_test_files = split_normals(root=DATASET_PATH)
    eval_test_files_dict = split_anomal(root=DATASET_PATH, classes=TEST_CLASSES)

    # ------ Evaluation
    all_cls_targets, all_cls_preds = [], []
    all_seg_targets, all_seg_preds = [], []

    class_results = {}      # Per-class evaluation
    vis_samples_dict = {}   # For test visualization

    total_cls_loss = 0.0
    total_seg_bce_loss = 0.0
    total_seg_dice_loss = 0.0
    total_overall_loss = 0.0
    total_batches = 0
    total_anom_batches = 0

    type_true, type_pred, e2e_true, e2e_pred, norm_type_pred, anom_names = [], [], [], [], [], []

    with torch.no_grad():
        # --- Test on normal images
        if len(normal_test_files) > 0:
            norm_dataset = NormalImagesDataset(normal_test_files)
            norm_loader = make_loader(norm_dataset, batch_size=BATCH_IMAGES, num_workers=NUM_WORKERS, shuffle=False)
            
            norm_cls_loss = 0.0
            norm_seg_bce = 0.0
            norm_seg_dice = 0.0
            norm_total = 0.0
            norm_batches = len(norm_loader)

            for imgs, _, _, _ in norm_loader:
                imgs = imgs.to(device)
                batch_size = imgs.size(0)

                # Targets = 0 for normals
                cls_tgt = torch.zeros(batch_size, device=device)
                seg_tgt = torch.zeros((batch_size, 120, 120), device=device)
                type_tgt = torch.full((batch_size,), -1, dtype=torch.long, device=device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits, type_logits = model(imgs)
                    loss, l_cls, l_bce, l_dice, _ = criterion(cls_logits, seg_logits, cls_tgt, seg_tgt, type_logits, type_tgt)
                
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

                t_pred = type_logits.argmax(dim=1).cpu().numpy()
                norm_type_pred.extend(t_pred)
                e2e_true.extend([0] * batch_size)
                e2e_pred.extend(np.where(cls_probs >= cls_threshold, t_pred + 1, 0))

                # Accumulate normal loss metrics
                norm_total += loss.item()
                norm_cls_loss += l_cls.item()
                norm_seg_bce += l_bce.item()
                norm_seg_dice += l_dice.item()

                all_cls_preds.extend(cls_probs)
                all_cls_targets.extend([0.0] * batch_size)
                all_seg_preds.extend(seg_probs)
                all_seg_targets.extend([np.zeros((120, 120), dtype=np.float32)] * batch_size)

            # Record metrics for normals
            class_results["normal"] = {
                "Average_Total_Loss": norm_total / norm_batches,
                "Average_Cls_Loss": norm_cls_loss / norm_batches,
                "Average_Seg_BCE_Loss": norm_seg_bce / norm_batches,
                "Average_Seg_Dice_Loss": norm_seg_dice / norm_batches,
                "Image_AUC_ROC": 1.0,           # Single-class fallback
                "Image_F1": 0.0,
                "Image_Precision": 0.0,
                "Image_Recall": 0.0,
                "Pixel_AUC_ROC": 1.0,
                "Pixel_AUC_PR": 0.0,
                "Pixel_IoU": 1.0,
                "Pixel_Dice": 1.0,
                "TN": len(normal_test_files), "FP": 0, "FN": 0, "TP": 0
            }

            # Update loss accumulators
            total_overall_loss += norm_total - gamma * norm_seg_dice  # exclude dice loss (avoid counting all normal batches with zero dice loss)
            total_cls_loss += norm_cls_loss
            total_seg_bce_loss += norm_seg_bce
            total_seg_dice_loss += norm_seg_dice
            total_batches += norm_batches

        # Snapshot normal-only predictions
        norm_cls_targets_snapshot = list(all_cls_targets)
        norm_cls_preds_snapshot = list(all_cls_preds)
        norm_seg_preds_snapshot = list(all_seg_preds)
        norm_sources_snapshot = [p.parent.name for p in normal_test_files]


        # --- Test on anomalous images
        for cls_name in TEST_CLASSES:
            cls_files = eval_test_files_dict.get(cls_name, {})
            test_triples = cls_files.get("test", [])
            if len(test_triples) == 0:
                continue

            cls_dataset = AnomalyImagesDataset(test_triples, return_meta=True)
            cls_loader = make_loader(cls_dataset, batch_size=BATCH_IMAGES, num_workers=NUM_WORKERS, shuffle=False)

            cls_targets_class, cls_preds_class = [], []
            seg_targets_class, seg_preds_class = [], []

            cls_total_loss = 0.0
            cls_cls_loss = 0.0
            cls_seg_bce = 0.0
            cls_seg_dice = 0.0
            cls_batches = len(cls_loader)
            vis_count = 0

            for imgs, cls_labels, masks, type_labels, _, names in cls_loader:
                imgs = imgs.to(device)
                masks_device = masks.to(device)
                batch_size = imgs.size(0)

                cls_tgt = cls_labels.to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits, type_logits = model(imgs)
                    loss, l_cls, l_bce, l_dice, _ = criterion(cls_logits, seg_logits, cls_tgt, masks_device, type_logits, type_labels.to(device))

                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
                masks_np = masks.numpy()

                t_pred, t_true = type_logits.argmax(dim=1).cpu().numpy(), type_labels.numpy()
                type_true.extend(t_true); type_pred.extend(t_pred)
                e2e_true.extend(np.where(t_true >= 0, t_true + 1, 0))
                e2e_pred.extend(np.where(cls_probs >= cls_threshold, t_pred + 1, 0))
                anom_names.extend(names)

                # Accumulate loss metrics for class
                cls_total_loss += loss.item()
                cls_cls_loss += l_cls.item()
                cls_seg_bce += l_bce.item()
                cls_seg_dice += l_dice.item()

                cls_targets_class.extend(cls_labels.tolist())
                cls_preds_class.extend(cls_probs)
                seg_targets_class.extend(masks_np)
                seg_preds_class.extend(seg_probs)

                # Store visualization samples
                for i in range(batch_size):
                    if vis_count < 2:       # 2 samples per class
                        img_np = imgs[i].cpu().numpy()
                        # Extract true color RGB
                        r = img_np[3]   # B4: red
                        g = img_np[2]   # B3: green
                        b = img_np[1]   # B2: blue
                        rgb = np.stack([r, g, b], axis=-1)
                        # Normalize RGB values
                        rgb_min, rgb_max = rgb.min(), rgb.max()
                        if rgb_max > rgb_min:
                            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
                        else:
                            rgb = np.zeros_like(rgb)

                        vis_samples_dict[f"{cls_name}_{vis_count}_rgb"] = rgb
                        vis_samples_dict[f"{cls_name}_{vis_count}_gt"] = masks_np[i]
                        vis_samples_dict[f"{cls_name}_{vis_count}_pred"] = seg_probs[i]
                        vis_count += 1

            # Update loss accumulators
            total_overall_loss += cls_total_loss - gamma * cls_seg_dice  # exclude dice loss (avoid counting all normal batches with zero dice loss)
            total_cls_loss += cls_cls_loss
            total_seg_bce_loss += cls_seg_bce
            total_seg_dice_loss += cls_seg_dice
            total_batches += cls_batches
            total_anom_batches += cls_batches

            # Store
            all_cls_targets.extend(cls_targets_class)
            all_cls_preds.extend(cls_preds_class)
            all_seg_targets.extend(seg_targets_class)
            all_seg_preds.extend(seg_preds_class)

            # Per-class metrics
            class_metrics = evaluate_metrics(
                cls_targets_class + norm_cls_targets_snapshot,
                cls_preds_class + norm_cls_preds_snapshot,
                seg_targets_class, seg_preds_class,
                cls_threshold=cls_threshold,
                seg_threshold=seg_threshold,
            )
            class_metrics["Pixel_Dice_per_image"] = dice_per_image(seg_targets_class, seg_preds_class, seg_threshold)
            matched = [p for p, s in zip(norm_cls_preds_snapshot, norm_sources_snapshot) if s == cls_name]
            if matched:  # AUC against the region-matched normals only
                class_metrics["Image_AUC_ROC_matched_normals"] = float(roc_auc_score(cls_targets_class + [0.0] * len(matched), cls_preds_class + matched))
            class_metrics["Average_Total_Loss"] = cls_total_loss / cls_batches
            class_metrics["Average_Cls_Loss"] = cls_cls_loss / cls_batches
            class_metrics["Average_Seg_BCE_Loss"] = cls_seg_bce / cls_batches
            class_metrics["Average_Seg_Dice_Loss"] = cls_seg_dice / cls_batches
            class_results[cls_name] = class_metrics

    # ------ Overall Metrics
    overall_metrics = evaluate_metrics(all_cls_targets, all_cls_preds, all_seg_targets, all_seg_preds, cls_threshold=cls_threshold, seg_threshold=seg_threshold)
    overall_metrics["Pixel_Dice_per_image"] = dice_per_image(all_seg_targets, all_seg_preds, seg_threshold)
    overall_metrics["Normal_Images_With_FP_Region"] = float(np.mean([(p >= seg_threshold).sum() >= 20 for p in norm_seg_preds_snapshot]))  # >= 20 px = min anomaly size
    overall_metrics["Average_Total_Loss"] = total_overall_loss / total_batches
    overall_metrics["Average_Cls_Loss"] = total_cls_loss / total_batches
    overall_metrics["Average_Seg_BCE_Loss"] = total_seg_bce_loss / total_batches
    overall_metrics["Average_Seg_Dice_Loss"] = total_seg_dice_loss / total_anom_batches
    overall_metrics["Average_Total_Loss"] += gamma * overall_metrics["Average_Seg_Dice_Loss"]  # add back the dice loss to the total loss

    overall_metrics["Classification_Threshold"] = cls_threshold
    overall_metrics["Segmentation_Threshold"] = seg_threshold

    gated = [p * float(c >= cls_threshold) for p, c in zip(all_seg_preds, all_cls_preds)]
    overall_metrics["Pixel_Dice_per_image_gated"] = dice_per_image(all_seg_targets, gated, seg_threshold)
    overall_metrics["Normal_Images_With_FP_Region_gated"] = float(np.mean([(c >= cls_threshold) and (p >= seg_threshold).sum() >= 20 for p, c in zip(norm_seg_preds_snapshot, norm_cls_preds_snapshot)]))

    src = np.array(norm_sources_snapshot)
    norm_pos = np.array(norm_cls_preds_snapshot) >= cls_threshold
    fp_by_source = {s: {"FP": int(norm_pos[src == s].sum()), "N": int((src == s).sum())} for s in sorted(set(src))}

    K, tt, tp_ = len(TEST_CLASSES), np.array(type_true), np.array(type_pred)
    v = tt >= 0
    type_metrics = {
        "Type_Accuracy": float((tt[v] == tp_[v]).mean()),
        "Type_Macro_F1": float(f1_score(tt[v], tp_[v], average="macro")),
        "Type_Confusion(rows=true)": confusion_matrix(tt[v], tp_[v], labels=list(range(K))).tolist(),
        "EndToEnd_4class_Macro_F1": float(f1_score(e2e_true, e2e_pred, average="macro")),
        "EndToEnd_4class_Confusion(0=normal)": confusion_matrix(e2e_true, e2e_pred, labels=list(range(K + 1))).tolist(),
        "Type_Pred_on_Normals_by_Source": {s: np.bincount(np.array(norm_type_pred)[src == s], minlength=K).tolist() for s in sorted(set(src))},
    } if v.any() else {}


    conf_matrix = {
        "TN": int(overall_metrics.pop("TN")),
        "FP": int(overall_metrics.pop("FP")),
        "FN": int(overall_metrics.pop("FN")),
        "TP": int(overall_metrics.pop("TP"))
    }

    ckpt_id = checkpoint_name.removesuffix(".pth")
    # Save quantitative results JSON
    test_report = {
        "overall_metrics": overall_metrics,
        "class_metrics": class_results,
        "confusion_matrix": conf_matrix,
        "fp_by_normal_source": fp_by_source,
        "type_classes": TEST_CLASSES,
        "type_metrics": type_metrics
    }
    metrics_file = RESULTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{ckpt_id}_test_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=4)
    print(f"Serialized test metrics JSON successfully saved to: {metrics_file}")

    # Save classification scores JSON
    scores = {
        "normal": [{"file": p.name, "source": p.parent.name, "cls_prob": float(c)} for p, c in zip(normal_test_files, norm_cls_preds_snapshot)],
        "anomaly": [{"file": n, "cls_prob": float(c)} for n, c in zip(anom_names, all_cls_preds[len(norm_cls_preds_snapshot):])],
    }
    with (RESULTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{ckpt_id}_scores.json").open("w") as f:
        json.dump(scores, f, indent=1)

    # Save visual samples compressed arrays
    npz_file = RESULTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{ckpt_id}_test_vis_samples.npz"
    np.savez_compressed(npz_file, **vis_samples_dict)
    print(f"Saved compressed qualitative visual samples array map to: {npz_file}")


    # ------ Print Summary
    print("\n" + "=" * 70)
    print(" OVERALL TEST PERFORMANCE SUMMARY")
    print("=" * 70)
    for k, v in overall_metrics.items():
        print(f"  * {k:20s}: {v:.4f}")

    print("\n" + "=" * 70)
    print(" PER-CLASS ANOMALY LOCALIZATION BREAKDOWN")
    print("=" * 70)
    print(f"{'Class Name':15s} | {'Img AUC':8s} | {'Pix AUC-ROC':11s} | {'Pix AUC-PR':10s} | {'IoU':6s} | {'Dice':6s}")
    print("-" * 70)
    for cls_name, m in class_results.items():
        if cls_name == "normal":
            continue
        print(
            f"{cls_name:15s} | {m['Image_AUC_ROC']:8.4f} | {m['Pixel_AUC_ROC']:11.4f} | "
            f"{m['Pixel_AUC_PR']:10.4f} | {m['Pixel_IoU']:6.4f} | {m['Pixel_Dice']:6.4f}"
        )
    print("=" * 70)

    return overall_metrics, class_results


if __name__ == "__main__":
    checkpoints_to_run = []
    for ep in EVAL_EPOCHS:
        ckpt_name = f"epoch_{ep}.pth"
        if (RESULTS_DIR / ckpt_name).exists():
            checkpoints_to_run.append(ckpt_name)
        else:
            print(f"Notice: '{ckpt_name}' was not found in {RESULTS_DIR} and will be skipped.")
    # Add last.pth
    if (RESULTS_DIR / "last.pth").exists() and "last.pth" not in checkpoints_to_run:
        checkpoints_to_run.append("last.pth")
    # Add best_cls.pth
    if (RESULTS_DIR / "best_cls.pth").exists() and "best_cls.pth" not in checkpoints_to_run:
        checkpoints_to_run.append("best_cls.pth")
    # Add best_seg.pth
    if (RESULTS_DIR / "best_seg.pth").exists() and "best_seg.pth" not in checkpoints_to_run:
        checkpoints_to_run.append("best_seg.pth")


    if not checkpoints_to_run:
        print(f"No valid target checkpoints found in {RESULTS_DIR}")
    else:
        print(f"\nSelected {len(checkpoints_to_run)} checkpoint(s) for evaluation: {checkpoints_to_run}")

        multi_summary = {}
        for ckpt_name in checkpoints_to_run:
            print("\n" + "=" * 70)
            print(f" EVALUATING CHECKPOINT: {ckpt_name}")
            print("=" * 70)

            try:
                overall_m, class_m = run_test_evaluation(checkpoint_name=ckpt_name, device=DEVICE)
                multi_summary[ckpt_name] = overall_m

            except Exception as e:
                print(f"Error evaluating {ckpt_name}: {e}")


        # Print summary
        print("\n" + "=" * 80)
        print(" EVALUATED CHECKPOINTS PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Checkpoint':20s} | {'Img AUC':8s} | {'Pix AUC-ROC':11s} | {'Pix IoU':8s} | {'Pix Dice':8s}")
        print("-" * 80)
        for ckpt_name, m in multi_summary.items():
            print(
                f"{ckpt_name:20s} | {m.get('Image_AUC_ROC', 0.0):8.4f} | "
                f"{m.get('Pixel_AUC_ROC', 0.0):11.4f} | {m.get('Pixel_IoU', 0.0):8.4f} | {m.get('Pixel_Dice', 0.0):8.4f}"
            )
            print("=" * 80)