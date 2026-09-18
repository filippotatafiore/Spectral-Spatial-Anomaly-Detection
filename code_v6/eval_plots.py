import sys
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


MODEL_VERSION = "v6"
MODEL_RUN = "0"



plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

def plot_evaluation_results(metrics_json_path, vis_samples_path, output_dir):
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
                grouped_samples[group_key][array_type] = samples[k]

        if grouped_samples:
            num_groups = len(grouped_samples)
            fig, axes = plt.subplots(num_groups, 3, figsize=(9, 2.8 * num_groups), dpi=150)
            if num_groups == 1:
                axes = np.expand_dims(axes, axis=0)

            for idx, (gk, data_dict) in enumerate(sorted(grouped_samples.items())):
                class_name = data_dict["class"].replace("_", " ").title()
                rgb = data_dict["rgb"]
                gt = data_dict["gt"]
                pred = data_dict["pred"]

                # True Color RGB image
                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title(f"{class_name}\nTrue Color RGB", fontsize=9)
                axes[idx, 0].axis("off")

                # Ground Truth Anomaly Mask
                axes[idx, 1].imshow(gt, cmap="gray", interpolation="nearest")
                axes[idx, 1].set_title("Ground Truth Mask", fontsize=9)
                axes[idx, 1].axis("off")

                # Predicted Anomaly Probability Heatmap
                im = axes[idx, 2].imshow(pred, cmap="jet", interpolation="bilinear", vmin=0, vmax=1)
                axes[idx, 2].set_title("Predicted Probability Map", fontsize=9)
                axes[idx, 2].axis("off")
                fig.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)

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
    try:
        from utils.globals import ARTIFACTS_DIR
        metrics_json_path = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_test_metrics.json"
        vis_samples_path = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_test_vis_samples.npz"
        output_dir = ARTIFACTS_DIR / f"{MODEL_VERSION}_{MODEL_RUN}_evaluation"
    except ImportError:
        metrics_json_path = Path("artifacts/test_metrics.json")
        vis_samples_path = Path("artifacts/test_vis_samples.npz")
        output_dir = Path("artifacts/evaluation")

    plot_evaluation_results(metrics_json_path, vis_samples_path, output_dir)