import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, jaccard_score,
)

from utils.globals import (
    DEVICE, ARTIFACTS_DIR, DATASET_PATH, TEST_CLASSES, CLIP_REFLECTANCE, BATCH_IMAGES, NUM_WORKERS, AMP,
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
)
from network.trans_UNet import TransUNetSentinel2
from data.dataset import NormalImagesDataset, AnomalyImagesDataset, make_loader
from utils.splits import split_normals, split_eval


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
        "Image_AUC_ROC": img_auc,
        "Image_F1": img_f1,
        "Image_Precision": img_prec,
        "Image_Recall": img_rec,
        "Pixel_AUC_ROC": pix_auc_roc,
        "Pixel_AUC_PR": pix_auc_pr,
        "Pixel_IoU": pix_iou,
        "Pixel_Dice": pix_dice,
    }


def run_test_evaluation(checkpoint_name="best.pth"):
    print("=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    # ------ Load model
    model = TransUNetSentinel2(in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, num_heads=NUM_ATT_HEADS, num_layers=NUM_TRANS_LAYERS, dropout=DROPOUT).to(DEVICE)

    checkpoint_path = ARTIFACTS_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ------ Normal and Anomalous Test Datasets
    _, _, normal_test_files = split_normals(root=DATASET_PATH)
    eval_test_files_dict = split_eval(root=DATASET_PATH, classes=TEST_CLASSES)

    # ------ Evaluation
    all_cls_targets, all_cls_preds = [], []
    all_seg_targets, all_seg_preds = [], []

    class_results = {}  # Per-class evaluation
    with torch.no_grad():
        # --- Test on normal images
        if len(normal_test_files) > 0:
            norm_dataset = NormalImagesDataset(normal_test_files)
            norm_loader = make_loader(norm_dataset, batch_size=BATCH_IMAGES, num_workers=NUM_WORKERS, shuffle=False)
            
            for imgs in norm_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=AMP):
                    cls_logits, seg_logits = model(imgs)
                
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

                # Store
                batch_size = imgs.size(0)
                all_cls_preds.extend(cls_probs)
                all_cls_targets.extend([0.0] * batch_size)
                all_seg_preds.extend(seg_probs)
                all_seg_targets.extend([np.zeros((120, 120), dtype=np.float32)] * batch_size)

        # --- Test on anomalous images
        for cls_name in TEST_CLASSES:
            cls_files = eval_test_files_dict.get(cls_name, {})
            test_triples = cls_files.get("test", [])
            if len(test_triples) == 0:
                continue

            cls_dataset = AnomalyImagesDataset(test_triples)
            cls_loader = make_loader(cls_dataset, batch_size=BATCH_IMAGES, num_workers=NUM_WORKERS, shuffle=False)

            cls_targets_class, cls_preds_class = [], []
            seg_targets_class, seg_preds_class = [], []
            for imgs, masks in cls_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=AMP):
                    cls_logits, seg_logits = model(imgs)

                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
                masks_np = masks.numpy()

                # Store per-class predictions and targets
                cls_targets_class.extend([1.0] * len(imgs))
                cls_preds_class.extend(cls_probs)
                seg_targets_class.extend(masks_np)
                seg_preds_class.extend(seg_probs)

            # Store
            all_cls_targets.extend(cls_targets_class)
            all_cls_preds.extend(cls_preds_class)
            all_seg_targets.extend(seg_targets_class)
            all_seg_preds.extend(seg_preds_class)

            # Per-class metrics
            class_metrics = evaluate_metrics(cls_targets_class, cls_preds_class, seg_targets_class, seg_preds_class)
            class_results[cls_name] = class_metrics

    # ------ Overall Metrics
    overall_metrics = evaluate_metrics(all_cls_targets, all_cls_preds, all_seg_targets, all_seg_preds)


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
        print(
            f"{cls_name:15s} | {m['Image_AUC_ROC']:8.4f} | {m['Pixel_AUC_ROC']:11.4f} | "
            f"{m['Pixel_AUC_PR']:10.4f} | {m['Pixel_IoU']:6.4f} | {m['Pixel_Dice']:6.4f}"
        )
    print("=" * 70)

    return overall_metrics, class_results


if __name__ == "__main__":
    run_test_evaluation()