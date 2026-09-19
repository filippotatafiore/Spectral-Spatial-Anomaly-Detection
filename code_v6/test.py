import os
import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, jaccard_score, confusion_matrix,
)

from utils.globals import (
    DEVICE, ARTIFACTS_DIR, DATASET_PATH, TEST_CLASSES, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, AMP,
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
    MODEL_VERSION, MODEL_RUN,
)
from network.trans_UNet import TransUNetSentinel2
from network.loss import MultiTaskLoss
from data.dataset import NormalImagesDataset, AnomalyImagesDataset, make_loader
from utils.splits import split_normals, split_anomal


# list of intermediate epochs to evaluate
EVAL_EPOCHS = [10, 30, 40]



def evaluate_metrics(cls_targets, cls_preds, seg_targets, seg_preds, threshold=0.5):
    """
        Computes Image-Level and Pixel-Level Evaluation Metrics.
    """
    cls_targets = np.array(cls_targets)
    cls_preds = np.array(cls_preds)
    cls_binary = (cls_preds >= threshold).astype(int)

    # --- Image-level classification metrics
    try:
        img_auc = roc_auc_score(cls_targets, cls_preds)
    except ValueError:
        img_auc = 0.5

    img_f1 = f1_score(cls_targets, cls_binary, zero_division=0)
    img_prec = precision_score(cls_targets, cls_binary, zero_division=0)
    img_rec = recall_score(cls_targets, cls_binary, zero_division=0)

    # confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(cls_targets, cls_binary, labels=[0, 1]).ravel()


    # --- Pixel-level segmentation/localization metrics
    seg_targets_flat = np.concatenate([t.flatten() for t in seg_targets])
    seg_preds_flat = np.concatenate([p.flatten() for p in seg_preds])
    seg_binary_flat = (seg_preds_flat >= threshold).astype(int)

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
        "Pixel_AUC_ROC": float(pix_auc_roc),
        "Pixel_AUC_PR": float(pix_auc_pr),
        "Pixel_IoU": float(pix_iou),
        "Pixel_Dice": float(pix_dice),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def run_test_evaluation(checkpoint_name="best.pth", device=DEVICE):
    if isinstance(device, str):
        device = torch.device(device)
    print("=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    # ------ Load model
    model = TransUNetSentinel2(in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, num_heads=NUM_ATT_HEADS, num_layers=NUM_TRANS_LAYERS, dropout=DROPOUT).to(device)

    checkpoint_path = ARTIFACTS_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ckpt_epoch = checkpoint.get("epoch", "N/A")
    if checkpoint_name == "best.pth":
        print(f"  'best.pth' at Epoch {ckpt_epoch}")
    else:
        print(f"Loaded {checkpoint_name} (Epoch {ckpt_epoch})")

    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=1.0)
    use_amp = AMP and device.type == "cuda"

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

            for imgs, _, _ in norm_loader:
                imgs = imgs.to(device)
                batch_size = imgs.size(0)

                # Targets = 0 for normals
                cls_tgt = torch.zeros(batch_size, device=device)
                seg_tgt = torch.zeros((batch_size, 120, 120), device=device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits = model(imgs)
                    loss, l_cls, l_bce, l_dice = criterion(cls_logits, seg_logits, cls_tgt, seg_tgt)
                
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

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
            total_overall_loss += norm_total
            total_cls_loss += norm_cls_loss
            total_seg_bce_loss += norm_seg_bce
            total_seg_dice_loss += norm_seg_dice
            total_batches += norm_batches

        # Snapshot normal-only predictions
        norm_cls_targets_snapshot = list(all_cls_targets)
        norm_cls_preds_snapshot = list(all_cls_preds)


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

            for imgs, cls_labels, masks, _, _ in cls_loader:
                imgs = imgs.to(device)
                masks_device = masks.to(device)
                batch_size = imgs.size(0)

                cls_tgt = cls_labels.to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cls_logits, seg_logits = model(imgs)
                    loss, l_cls, l_bce, l_dice = criterion(cls_logits, seg_logits, cls_tgt, masks_device)

                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
                masks_np = masks.numpy()

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
            total_overall_loss += cls_total_loss
            total_cls_loss += cls_cls_loss
            total_seg_bce_loss += cls_seg_bce
            total_seg_dice_loss += cls_seg_dice
            total_batches += cls_batches

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
            )
            class_metrics["Average_Total_Loss"] = cls_total_loss / cls_batches
            class_metrics["Average_Cls_Loss"] = cls_cls_loss / cls_batches
            class_metrics["Average_Seg_BCE_Loss"] = cls_seg_bce / cls_batches
            class_metrics["Average_Seg_Dice_Loss"] = cls_seg_dice / cls_batches
            class_results[cls_name] = class_metrics

    # ------ Overall Metrics
    overall_metrics = evaluate_metrics(all_cls_targets, all_cls_preds, all_seg_targets, all_seg_preds)
    overall_metrics["Average_Total_Loss"] = total_overall_loss / total_batches
    overall_metrics["Average_Cls_Loss"] = total_cls_loss / total_batches
    overall_metrics["Average_Seg_BCE_Loss"] = total_seg_bce_loss / total_batches
    overall_metrics["Average_Seg_Dice_Loss"] = total_seg_dice_loss / total_batches

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
        "confusion_matrix": conf_matrix
    }
    metrics_file = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_{ckpt_id}_test_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=4)
    print(f"Serialized test metrics JSON successfully saved to: {metrics_file}")

    # Save visual samples compressed arrays
    npz_file = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_{ckpt_id}_test_vis_samples.npz"
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
        if (ARTIFACTS_DIR / ckpt_name).exists():
            checkpoints_to_run.append(ckpt_name)
        else:
            print(f"Notice: '{ckpt_name}' was not found in {ARTIFACTS_DIR} and will be skipped.")
    # Add last.pth
    if (ARTIFACTS_DIR / "last.pth").exists() and "last.pth" not in checkpoints_to_run:
        checkpoints_to_run.append("last.pth")
    # Add best.pth
    if (ARTIFACTS_DIR / "best.pth").exists() and "best.pth" not in checkpoints_to_run:
        checkpoints_to_run.append("best.pth")


    if not checkpoints_to_run:
        print(f"No valid target checkpoints found in {ARTIFACTS_DIR}")
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