import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils.globals import ARTIFACTS_DIR

MODEL_VERSION = "v13"
MODEL_RUN = "1"

def plot_training_metrics(metrics_path, output_dir):
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return
        
    with metrics_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
        
    epochs = history.get("epoch", [])
    if not epochs:
        print("Error: No epoch data found in metrics.")
        return
        
    # Set matplotlib style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    
    # --- Plot Total Losses (Train vs Validation)
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Multi-Task Loss")
    plt.title("Training and Validation Loss Convergence")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()
    
    # --- Plot Loss Components (BCE, Dice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    
    # Classification BCE
    axes[0].plot(epochs, history["train_loss_cls"], label="Train Cls BCE", color="#2ca02c", linewidth=2)
    axes[0].plot(epochs, history["val_loss_cls"], label="Val Cls BCE", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Classification Branch BCE")
    axes[0].legend(frameon=True)
    
    # Segmentation BCE
    axes[1].plot(epochs, history["train_loss_seg_bce"], label="Train Seg BCE", color="#9467bd", linewidth=2)
    axes[1].plot(epochs, history["val_loss_seg_bce"], label="Val Seg BCE", color="#8c564b", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Segmentation Branch BCE")
    axes[1].legend(frameon=True)
    
    # Segmentation Dice
    axes[2].plot(epochs, history["train_loss_seg_dice"], label="Train Seg Dice", color="#e377c2", linewidth=2)
    axes[2].plot(epochs, history["val_loss_seg_dice"], label="Val Seg Dice", color="#7f7f7f", linewidth=2, linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Segmentation Branch Dice")
    axes[2].legend(frameon=True)
    
    plt.suptitle("Multi-Task Loss Components Decomposition", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_components.png", dpi=300)
    plt.close()
    
    # --- Plot Validation AUC-ROC
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(epochs, history["val_auc"], label="Val AUC-ROC", color="#ff7f0e", linewidth=2, marker="o", markersize=4)
    best_idx = history["val_auc"].index(max(history["val_auc"]))
    best_epoch = epochs[best_idx]
    best_auc = history["val_auc"][best_idx]
    plt.scatter(best_epoch, best_auc, color="#d62728", s=50, zorder=5, label=f"Best AUC: {best_auc:.4f} (Ep {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.title("Validation Image-Level AUC-ROC Profile")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "val_auc_curve.png", dpi=300)
    plt.close()

    # --- Validation metrics summary: image AUC, soft Dice and type accuracy per epoch
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(epochs, history["val_auc"], label="Val image AUC-ROC", linewidth=2)
    plt.plot(epochs, [1 - v for v in history["val_loss_seg_dice"]], label="Val soft Dice", linewidth=2)
    if "val_type_acc" in history:
        plt.plot(epochs, history["val_type_acc"], label="Val type accuracy", linewidth=2)
    plt.axvline(best_epoch, color="grey", linestyle=":", linewidth=1, label=f"best_cls (ep {best_epoch})")
    plt.ylim(0, 1.02)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics During Training")
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "val_metrics.png", dpi=300)
    plt.close()

    # --- Plot type branch
    if "val_type_acc" in history:
        plt.figure(figsize=(7, 4), dpi=150)
        plt.plot(epochs, history["train_loss_type"], label="Train Type CE", linewidth=2)
        plt.plot(epochs, history["val_loss_type"], label="Val Type CE", linewidth=2, linestyle="--")
        plt.plot(epochs, history["val_type_acc"], label="Val Type Accuracy", linewidth=2, linestyle=":")
        plt.xlabel("Epoch"); plt.title("Anomaly Type Branch"); plt.legend(frameon=True); plt.tight_layout()
        plt.savefig(output_dir / "type_branch.png", dpi=300); plt.close()

    # --- Plot deviation branch
    if "val_dev_auc" in history:
        plt.figure(figsize=(7, 4), dpi=150)
        plt.plot(epochs, history["train_loss_dev"], label="Train Deviation loss", linewidth=2)
        plt.plot(epochs, history["val_loss_dev"], label="Val Deviation loss", linewidth=2, linestyle="--")
        plt.plot(epochs, history["val_dev_auc"], label="Val deviation-only AUC", linewidth=2, linestyle=":")
        plt.xlabel("Epoch"); plt.title("Normality-Prototype Deviation Branch"); plt.legend(frameon=True); plt.tight_layout()
        plt.savefig(output_dir / "deviation_branch.png", dpi=300); plt.close()
    
    # --- Plot Learning Rate
    plt.figure(figsize=(7, 4), dpi=150)
    plt.plot(epochs, history["lr"], label="Learning Rate", color="#17becf", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("LR (log scale)")
    plt.title("Learning Rate Cosine Annealing Schedule")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "lr_curve.png", dpi=300)
    plt.close()
    
    # --- Print out summary table
    print("\n" + "=" * 80)
    print("                      SSAD TRAINING PROCESS REPORT")
    print("=" * 80)
    print(f"Total Epochs Completed: {len(epochs)}")
    print(f"Best Validation AUC  : {best_auc:.4f} (achieved at Epoch {best_epoch})")
    print(f"Final Training Loss  : {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print("-" * 80)
    print(f"{'Epoch':6s} | {'Train Loss':10s} | {'Val Loss':8s} | {'Val AUC':8s} | {'Cls Loss (T/V)':16s} | {'LR':8s}")
    print("-" * 80)
    
    # Show early, middle, and final epochs to avoid cluttering stdout
    display_epochs = set([1, len(epochs)])
    if len(epochs) > 10:
        display_epochs.update([len(epochs)//4, len(epochs)//2, 3*len(epochs)//4])
    display_epochs.add(best_epoch)
    
    for idx, ep in enumerate(epochs):
        if ep in display_epochs or idx == len(epochs) - 1:
            best_marker = " [BEST]" if ep == best_epoch else ""
            cls_t_v = f"{history['train_loss_cls'][idx]:.3f}/{history['val_loss_cls'][idx]:.3f}"
            print(
                f"{ep:<6d} | {history['train_loss'][idx]:<10.4f} | "
                f"{history['val_loss'][idx]:<8.4f} | {history['val_auc'][idx]:<8.4f} | "
                f"{cls_t_v:<16s} | {history['lr'][idx]:<8.2e}{best_marker}"
            )
    print("=" * 80)
    print(f"Training visualization plots saved to: {output_dir}/")
    print("\tloss_curves.png:      Total train vs validation loss")
    print("\tloss_components.png:  Loss decomposition into Cls BCE, Seg BCE, and Seg Dice")
    print("\tval_auc_curve.png:    Image-level anomaly detection validation AUC profile")
    print("\tlr_curve.png:         Cosine Annealing learning rate schedule")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    metrics_path = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/{MODEL_VERSION}_{MODEL_RUN}_results/training_metrics.json"
    output_dir = ARTIFACTS_DIR / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results/training_plots"
    plot_training_metrics(metrics_path, output_dir)
