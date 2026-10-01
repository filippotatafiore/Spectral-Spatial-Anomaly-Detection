import sys
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from utils.globals import ARTIFACTS_DIR


MODEL_VERSION = "v13"
MODEL_RUN = "0"
MODEL_ID = "best_joint_tta"       # "best_cls" | "best_seg" | "best_joint" | "last" | "epoch_XX" (add "_tta" if TTA was used)



plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"


def _plot_cm(cm, labels, title, path):
    cm = np.array(cm)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-normalized = recall per true class
    plt.figure(figsize=(1.4 * len(labels) + 2.5, 1.2 * len(labels) + 1.8), dpi=150)
    plt.imshow(cmn, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(label="Row-normalized")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _prf_from_cm(cm):
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    p = tp / np.maximum(cm.sum(axis=0), 1)
    r = tp / np.maximum(cm.sum(axis=1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return p, r, f


def plot_evaluation_results(metrics_json_path, vis_samples_path, output_dir, scores_path=None):
    metrics_json_path = Path(metrics_json_path)
    vis_samples_path = Path(vis_samples_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_json_path.exists():
        print(f"Error: Evaluation metrics JSON file not found at {metrics_json_path}")
        return

    with metrics_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data["overall_metrics"]
    class_metrics = data["class_metrics"]
    conf_matrix = data["confusion_matrix"]

    # ------ Confusion Matrix for Binary Classification
    plt.figure(figsize=(5, 4.5), dpi=150)
    cm_arr = np.array([
        [conf_matrix["TN"], conf_matrix["FP"]],
        [conf_matrix["FN"], conf_matrix["TP"]]
    ])
    
    plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Image-Level Classification Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal (0)", "Anomalous (1)"])
    plt.yticks(tick_marks, ["Normal (0)", "Anomalous (1)"])

    # Annotate counts inside the confusion matrix
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_arr[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_arr[i, j] > thresh else "black",
                     fontweight="bold")

    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


    # ------ Anomaly type: confusion matrices + precision / recall / F1
    tm = data.get("type_metrics", {})
    if tm:
        names = [c.replace("_", " ").title() for c in data["type_classes"]]
        _plot_cm(tm["Type_Confusion(rows=true)"], names, "Anomaly Type Confusion Matrix\n(anomalous test images)", output_dir / "type_confusion_matrix.png")
        _plot_cm(tm["EndToEnd_4class_Confusion(0=normal)"], ["Normal"] + names, "End-to-End Confusion Matrix\n(binary head + type head)", output_dir / "e2e_confusion_matrix.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        for ax, key, labs, title in [(axes[0], "Type_Confusion(rows=true)", names, "Anomaly type (anomalous images)"),
                                     (axes[1], "EndToEnd_4class_Confusion(0=normal)", ["Normal"] + names, "End-to-end (4 classes)")]:
            p, r, f = _prf_from_cm(tm[key])
            xl = np.arange(len(labs) + 1)
            for n, (vals, name, col) in enumerate([(p, "Precision", "#1f77b4"), (r, "Recall", "#ff7f0e"), (f, "F1", "#2ca02c")]):
                ax.bar(xl + (n - 1) * 0.25, np.r_[vals, vals.mean()], 0.25, label=name, color=col)
            ax.set_xticks(xl)
            ax.set_xticklabels(labs + ["Macro avg"], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(title)
            ax.legend(frameon=True, loc="lower left", fontsize=8)
        plt.suptitle("Anomaly Classification Metrics", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "type_metrics.png", dpi=300)
        plt.close()


    # ------ Loss Components Decomposed by Anomaly Class
    classes = [c for c in class_metrics.keys() if c != "normal"]
    cls_bce_list = [class_metrics[c]["Average_Cls_Loss"] for c in classes]
    seg_bce_list = [class_metrics[c]["Average_Seg_BCE_Loss"] for c in classes]
    seg_dice_list = [class_metrics[c]["Average_Seg_Dice_Loss"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5), dpi=150)
    plt.bar(x - width, cls_bce_list, width, label="Classification BCE", color="#2ca02c")
    plt.bar(x, seg_bce_list, width, label="Segmentation BCE", color="#9467bd")
    plt.bar(x + width, seg_dice_list, width, label="Segmentation Dice", color="#e377c2")

    plt.xlabel("Anomaly Class")
    plt.ylabel("Average Evaluation Loss")
    plt.title("Test-Set Loss Component Decomposition by Anomaly Type")
    plt.xticks(x, [c.replace("_", " ").title() for c in classes])
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_loss_comparison.png", dpi=300)
    plt.close()


    # ------ Performance Metrics Comparison (Bar Chart)
    plt.figure(figsize=(8, 5), dpi=150)
    img_auc_list = [class_metrics[c]["Image_AUC_ROC"] for c in classes]
    pix_auc_list = [class_metrics[c]["Pixel_AUC_ROC"] for c in classes]
    pix_dice_list = [class_metrics[c]["Pixel_Dice"] for c in classes]
    pix_iou_list = [class_metrics[c]["Pixel_IoU"] for c in classes]

    plt.bar(x - 1.5*width/2, img_auc_list, width/2, label="Image AUC-ROC", color="#1f77b4")
    plt.bar(x - width/4, pix_auc_list, width/2, label="Pixel AUC-ROC", color="#aec7e8")
    plt.bar(x + width/4, pix_dice_list, width/2, label="Pixel Dice", color="#ff7f0e")
    plt.bar(x + 1.5*width/2, pix_iou_list, width/2, label="Pixel IoU (Jaccard)", color="#ffbb78")

    plt.xlabel("Anomaly Class")
    plt.ylabel("Score")
    plt.title("Image Classification and Pixel Anomaly Localization Scores")
    plt.xticks(x, [c.replace("_", " ").title() for c in classes])
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "class_metrics_comparison.png", dpi=300)
    plt.close()


    # ------ Aggregated metrics: macro mean +- std over anomaly types vs pooled over all test images
    agg_keys = [("Image_AUC_ROC", "Image AUC-ROC"), ("Image_AUC_ROC_matched_normals", "Image AUC-ROC\n(matched normals)"),
                ("Pixel_AUC_ROC", "Pixel AUC-ROC"), ("Pixel_AUC_PR", "Pixel AUC-PR"),
                ("Pixel_Dice", "Pixel Dice"), ("Pixel_IoU", "Pixel IoU"), ("Pixel_Dice_per_image", "Dice / image")]
    pooled_key = {"Image_AUC_ROC_matched_normals": "Image_AUC_ROC_hard_normals"}
    macro = np.array([[class_metrics[c].get(k, np.nan) for c in classes] for k, _ in agg_keys], dtype=float)
    pooled = [overall.get(pooled_key.get(k, k), np.nan) for k, _ in agg_keys]
    xa = np.arange(len(agg_keys))
    plt.figure(figsize=(11, 5), dpi=150)
    plt.bar(xa - 0.2, np.nanmean(macro, axis=1), 0.4, yerr=np.nanstd(macro, axis=1), capsize=3, label="Macro mean ± std over anomaly types", color="#1f77b4")
    plt.bar(xa + 0.2, pooled, 0.4, label="Pooled over all test images", color="#ff7f0e")
    for i, (m, p) in enumerate(zip(np.nanmean(macro, axis=1), pooled)):
        plt.text(i - 0.2, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
        plt.text(i + 0.2, p + 0.01, f"{p:.2f}", ha="center", fontsize=7)
    plt.xticks(xa, [n for _, n in agg_keys], fontsize=8)
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title("Aggregated Detection and Localization Metrics")
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_metrics.png", dpi=300)
    plt.close()

    # ------ Per-image plots
    if scores_path is not None and Path(scores_path).exists():
        with Path(scores_path).open("r", encoding="utf-8") as f:
            sc = json.load(f)
        recs, thr = sc["records"], sc["cls_threshold"]
        normals = [r for r in recs if r["group"].startswith("normal-")]
        hard = [r for r in normals if r["group"] != "normal-ben-ge-8k"]
        anoms = [r for r in recs if not r["group"].startswith("normal-") and r["label"] == 1]

        # --- ROC curves
        def _roc(pos, neg, label, ax, **kw):
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            s = np.r_[[r["cls_prob"] for r in pos], [r["cls_prob"] for r in neg]]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, s):.3f})", **kw)

        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)
        _roc(anoms, normals, "All anomalies vs all normals", ax, linewidth=2)
        _roc(anoms, hard, "All anomalies vs region-matched normals", ax, linewidth=2, linestyle="--")
        for c in classes:
            _roc([r for r in anoms if r["group"] == c], [r for r in normals if r["group"] == f"normal-{c}"], f"{c.replace('_', ' ').title()} vs matched normals", ax, linewidth=1)
        tpr_op = np.mean([r["cls_prob"] >= thr for r in anoms])
        ax.scatter(np.mean([r["cls_prob"] >= thr for r in normals]), tpr_op, color="black", zorder=5, label=f"Operating point, all normals (thr={thr:.2f})")
        ax.scatter(np.mean([r["cls_prob"] >= thr for r in hard]), tpr_op, color="black", marker="x", zorder=5, label="Operating point, region-matched normals")
        ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("Image-Level ROC Curves")
        ax.legend(frameon=True, fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curves.png", dpi=300)
        plt.close()

        # --- Score distributions per normal source / anomaly class
        groups = [g for g in ["normal-ben-ge-8k"] + [f"normal-{c}" for c in classes] + classes if any(r["group"] == g for r in recs)]
        data_s = [[r["cls_prob"] for r in recs if r["group"] == g and (g.startswith("normal-") or r["label"] == 1)] for g in groups]
        rng = np.random.default_rng(0)
        plt.figure(figsize=(10, 5), dpi=150)
        plt.boxplot(data_s, showfliers=False, widths=0.5)
        for i, d in enumerate(data_s, start=1):
            plt.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=5, alpha=0.4, color="#d62728" if groups[i - 1].startswith("normal-") else "#1f77b4")
        plt.axhline(thr, color="black", linestyle="--", linewidth=1, label=f"Classification threshold ({thr:.2f})")
        plt.xticks(range(1, len(groups) + 1), [g.replace("normal-", "Normal\n").replace("_", " ") for g in groups], fontsize=8)
        plt.ylabel("Predicted anomaly probability")
        plt.title("Image-Level Anomaly Score Distributions (red = normal, blue = anomalous)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(output_dir / "score_distributions.png", dpi=300)
        plt.close()

        # --- Per-image Dice / IoU distributions
        seg_groups = classes + ["all"]
        dice_d = [[r["dice"] for r in anoms if "dice" in r and (g == "all" or r["group"] == g)] for g in seg_groups]
        iou_d = [[r["iou"] for r in anoms if "iou" in r and (g == "all" or r["group"] == g)] for g in seg_groups]
        xs = np.arange(len(seg_groups))
        plt.figure(figsize=(8, 5), dpi=150)
        b1 = plt.boxplot(dice_d, positions=xs - 0.18, widths=0.3, patch_artist=True, showfliers=False, showmeans=True)
        b2 = plt.boxplot(iou_d, positions=xs + 0.18, widths=0.3, patch_artist=True, showfliers=False, showmeans=True)
        for p in b1["boxes"]:
            p.set_facecolor("#ff7f0e")
        for p in b2["boxes"]:
            p.set_facecolor("#ffbb78")
        plt.xticks(xs, [g.replace("_", " ").title() for g in seg_groups])
        plt.ylim(0, 1.02)
        plt.ylabel("Per-image score")
        plt.legend([b1["boxes"][0], b2["boxes"][0]], ["Dice", "IoU"], frameon=True, loc="lower left")
        plt.title("Per-Image Localization Quality (anomalous test images)")
        plt.tight_layout()
        plt.savefig(output_dir / "per_image_seg_metrics.png", dpi=300)
        plt.close()

        # --- Robust loss decomposition: median with IQR bars
        plt.figure(figsize=(8, 5), dpi=150)
        for n, (k, name, col) in enumerate([("cls_bce", "Classification BCE", "#2ca02c"), ("seg_bce", "Segmentation BCE", "#9467bd"), ("soft_dice_loss", "Segmentation Dice", "#e377c2")]):
            vals = [np.array([r[k] for r in anoms if r["group"] == c and k in r]) for c in classes]
            med = np.array([np.median(v) for v in vals])
            err = [med - [np.percentile(v, 25) for v in vals], [np.percentile(v, 75) for v in vals] - med]
            plt.bar(x + (n - 1) * width, med, width, yerr=err, capsize=3, label=name, color=col)
        plt.xticks(x, [c.replace("_", " ").title() for c in classes])
        plt.ylabel("Median loss (IQR bars)")
        plt.title("Test-Set Loss Components by Anomaly Type (median)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(output_dir / "anomaly_loss_comparison_robust.png", dpi=300)
        plt.close()


    # ------ Qualitative Interpretability Subplots
    if vis_samples_path.exists():
        samples = np.load(vis_samples_path, allow_pickle=True)
        keys = list(samples.keys())
        grouped_samples = {}
        for k in keys:
            parts = k.split("_")
            if len(parts) >= 3:
                array_type = parts[-1]  # 'rgb', 'gt', or 'pred'
                sample_idx = parts[-2]
                class_name = "_".join(parts[:-2])
                
                group_key = f"{class_name}_{sample_idx}"
                if group_key not in grouped_samples:
                    grouped_samples[group_key] = {"class": class_name}
                grouped_samples[group_key][array_type] = str(samples[k]) if array_type == "info" else samples[k]

        if grouped_samples:
            num_groups = len(grouped_samples)
            fig, axes = plt.subplots(num_groups, 5, figsize=(15, 2.8 * num_groups), dpi=150)
            if num_groups == 1:
                axes = np.expand_dims(axes, axis=0)

            for idx, (gk, data_dict) in enumerate(sorted(grouped_samples.items())):
                class_name = data_dict["class"].replace("_", " ").title()
                rgb = data_dict["rgb"]
                gt = data_dict["gt"]
                pred = data_dict["pred"]

                # True Color RGB image
                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title(f"{class_name} [{gk.split('_')[-1]}]\n{data_dict.get('info', '')}", fontsize=8)
                axes[idx, 0].axis("off")

                # Ground Truth Anomaly Mask
                axes[idx, 1].imshow(gt, cmap="gray", interpolation="nearest", vmin=0, vmax=1)  # full masks rendered black otherwise
                axes[idx, 1].set_title("Ground Truth Mask", fontsize=9)
                axes[idx, 1].axis("off")

                # Predicted Anomaly Probability Heatmap
                im = axes[idx, 2].imshow(pred, cmap="jet", interpolation="bilinear", vmin=0, vmax=1)
                axes[idx, 2].set_title("Predicted Probability Map", fontsize=9)
                axes[idx, 2].axis("off")
                fig.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)

                # Overlay: Ground Truth vs Prediction
                axes[idx, 3].imshow(rgb)
                axes[idx, 3].contour(gt.astype(float), levels=[0.5], colors=["lime"], linewidths=1)
                axes[idx, 3].contour((pred >= overall["Segmentation_Threshold"]).astype(float), levels=[0.5], colors=["red"], linewidths=1)
                axes[idx, 3].set_title("GT (green) vs prediction (red)", fontsize=9)
                axes[idx, 3].axis("off")

                # Normality-prototype deviation map
                if "dev" in data_dict:
                    im = axes[idx, 4].imshow(data_dict["dev"], cmap="magma", interpolation="nearest", vmin=0, vmax=1)
                    fig.colorbar(im, ax=axes[idx, 4], fraction=0.046, pad=0.04)
                axes[idx, 4].set_title("Deviation map (prototypes)", fontsize=9)
                axes[idx, 4].axis("off")

            plt.suptitle("Qualitative Interpretability Maps: Model vs. Ground Truth", y=0.99, fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_dir / "sample_interpretability.png", dpi=300)
            plt.close()
            print(f"Qualitative interpretability maps saved to: {output_dir / 'sample_interpretability.png'}")
    else:
        print("Warning: Visual samples archive not found. Skipping qualitative plot generation.")


    # ------ TEXT REPORT
    print("\n" + "=" * 80)
    print("                      SSAD MODEL EVALUATION REPORT")
    print("=" * 80)
    print("OVERALL PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(f"  • Classification AUC-ROC : {overall['Image_AUC_ROC']:.4f}")
    print(f"  • Classification F1-Score: {overall['Image_F1']:.4f}")
    print(f"  • Classification Precision: {overall['Image_Precision']:.4f}")
    print(f"  • Classification Recall   : {overall['Image_Recall']:.4f}")
    print(f"  • Segmentation AUC-ROC   : {overall['Pixel_AUC_ROC']:.4f}")
    print(f"  • Segmentation AUC-PR    : {overall['Pixel_AUC_PR']:.4f}")
    print(f"  • Segmentation IoU (Jac) : {overall['Pixel_IoU']:.4f}")
    print(f"  • Segmentation Dice (F1) : {overall['Pixel_Dice']:.4f}")
    print(f"  • AUC vs matched normals : {overall.get('Image_AUC_ROC_hard_normals', float('nan')):.4f}")
    print(f"  • FPR on matched normals : {overall.get('Image_FPR_hard_normals', float('nan')):.4f}")
    print(f"  • Recall @ 5% FPR        : {overall.get('Image_Recall_at_FPR_5pct', float('nan')):.4f}")
    print(f"  • Dice per image         : {overall.get('Pixel_Dice_per_image', float('nan')):.4f}")
    if tm:
        print(f"  • Type macro-F1 / E2E macro-F1: {tm['Type_Macro_F1']:.4f} / {tm['EndToEnd_4class_Macro_F1']:.4f}")
    print("-" * 80)
    print("CLASSIFICATION LOSS EFFICIENCY SUMMARY:")
    print("  Loss objectives show the model's convergence and sensitivity under training constraints.")
    print(f"  • Avg Total Test Loss    : {overall.get('Average_Total_Loss', 0.0):.4f}")
    print(f"  • Avg Classification Loss: {overall.get('Average_Cls_Loss', 0.0):.4f}")
    print(f"  • Avg Segmentation BCE   : {overall.get('Average_Seg_BCE_Loss', 0.0):.4f}")
    print(f"  • Avg Segmentation Dice  : {overall.get('Average_Seg_Dice_Loss', 0.0):.4f}")
    print("-" * 80)
    print("CONFUSION MATRIX DETAILS:")
    print(f"  • True Negatives (TN)  : {conf_matrix['TN']}")
    print(f"  • False Positives (FP) : {conf_matrix['FP']} (Type I Error)")
    print(f"  • False Negatives (FN) : {conf_matrix['FN']} (Type II Error - Misses)")
    print(f"  • True Positives (TP)  : {conf_matrix['TP']}")
    print("-" * 80)
    print("Detailed class performance is plotted in 'class_metrics_comparison.png' and 'anomaly_loss_comparison.png'.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    metrics_json_path = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{MODEL_ID}_test_metrics.json"
    vis_samples_path = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{MODEL_ID}_test_vis_samples.npz"
    output_dir = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_ID}_eval_plots"
    scores_path = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_{MODEL_ID}_scores.json"

    plot_evaluation_results(metrics_json_path, vis_samples_path, output_dir, scores_path)