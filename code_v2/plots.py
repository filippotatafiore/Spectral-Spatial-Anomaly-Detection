"""
    python plots.py                 # all figures
    python plots.py --training      # training curves only
    python plots.py --evaluation    # evaluation figures only

    Reads only json/npz files.
    Figures are saved in artifacts/ssad_<version>_<run>_results/figures/ .
"""

# version of the model used to generate the artifacts
MODEL_VERSION = "v2"
MODEL_RUN = "0"


import argparse
import json
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _artifacts_dir():
    """
        Locate artifacts/ without importing utils.globals
    """
    here = Path(__file__).resolve().parent
    with (here / "config.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    d = Path(cfg.get("artifacts_dir", "artifacts"))
    return d if d.is_absolute() else (here / d).resolve()

MODEL_ARTIFACTS_DIR = _artifacts_dir() / f"ssad_{MODEL_VERSION}_{MODEL_RUN}_results"
FIG_DIR = MODEL_ARTIFACTS_DIR / "figures"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.png / .pdf")


# ======================================================= training figures

def plot_training(summary):
    hist = summary["history"]
    steps = summary.get("step_history", [])
    ep = [h["epoch"] for h in hist]

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_loss"] for h in hist], label="train", lw=2)
    ax.plot(ep, [h["val_loss"] for h in hist], label="validation", lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("Reconstruction loss")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "01_loss_curves")

    # Fine-grained training loss
    if steps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([s["step"] for s in steps], [s["train_loss"] for s in steps], lw=1)
        ax.set_xlabel("optimizer step"); ax.set_ylabel("train MSE")
        ax.set_title(f"Training loss ({len(steps)} logged points)")
        ax.grid(alpha=0.3)
        _save(fig, "02_loss_by_step")

    # Identity diagnostic: variance explained at the center token
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["variance_explained"] for h in hist], lw=2, color="#c1121f")
    ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.axhline(0.0, ls=":", c="grey", lw=1)
    ax.text(0.02, 0.96, "1.0 = the model is copying its input", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.02, 0.06, "0.0 = no better than predicting the mean", transform=ax.transAxes, va="bottom", fontsize=9, color="grey")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$1 - \mathrm{MSE}_{center}/\mathrm{Var}(target)$")
    ax.set_title("Identity diagnostic: variance explained at the center token")
    ax.grid(alpha=0.3)
    _save(fig, "03_identity_diagnostic")

    # Learning-rate schedule
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if steps:
        ax.plot([s["step"] for s in steps], [s["lr"] for s in steps], lw=1.5)
        ax.set_xlabel("optimizer step")
    else:
        ax.plot(ep, [h["lr"] for h in hist], lw=1.5); ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate"); ax.set_title("LR schedule (warmup + cosine decay)")
    ax.grid(alpha=0.3)
    _save(fig, "04_lr_schedule")

    # Gradient norm
    if steps and not all(np.isnan(s.get("grad_norm", np.nan)) for s in steps):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([s["step"] for s in steps], [s["grad_norm"] for s in steps], lw=1)
        ax.set_xlabel("optimizer step"); ax.set_ylabel("global grad norm")
        ax.set_title("Gradient norm (clip threshold dashed)")
        gc = summary.get("config", {}).get("GRAD_CLIP")
        if gc:
            ax.axhline(gc, ls="--", c="r", lw=1)
        ax.grid(alpha=0.3)
        _save(fig, "05_grad_norm")

    # Epoch time and peak GPU memory
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(ep, [h["epoch_time_s"] for h in hist], lw=2)
    a1.set_xlabel("epoch"); a1.set_ylabel("seconds"); a1.set_title("Epoch wall time"); a1.grid(alpha=0.3)
    a2.plot(ep, [h["peak_mem_gb"] for h in hist], lw=2, color="#2a9d8f")
    a2.set_xlabel("epoch"); a2.set_ylabel("GB"); a2.set_title("Peak GPU memory"); a2.grid(alpha=0.3)
    _save(fig, "06_compute")


# ======================================================= evaluation figures

def plot_evaluation(results, tag=""):
    classes = list(results["per_class"].keys())
    x = np.arange(len(classes)); w = 0.26

    # Alpha sensitivity
    sweep = results.get("alpha_sweep", {})
    if sweep:
        alphas = sorted(float(a) for a in sweep)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(alphas, [sweep[str(a)] for a in alphas], "o-", lw=2)
        chosen = results["selection"]["alpha"]
        ax.axvline(chosen, ls="--", c="r", lw=1.5, label=f"selected on val: $\\alpha$={chosen}")
        ax.set_xlabel(r"$\alpha$  (1 = reconstruction only, 0 = attention only)")
        ax.set_ylabel("macro AUROC (test)")
        ax.set_title(r"Sensitivity to $\alpha$   $S=\alpha\cdot$Recon$+(1-\alpha)\cdot$Attn")
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, f"07_alpha_sweep{tag}")

    # AUROC (per-class) against baselines
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w, [results["per_class"][c]["auroc_mean"] for c in classes], w,
           yerr=[results["per_class"][c]["auroc_std"] for c in classes], capsize=3,
           label="SST (ours)", color="#264653")
    ax.bar(x, [results["baselines"]["rx"][c]["auroc_mean"] for c in classes], w,
           label="RX", color="#e9c46a")
    ax.bar(x + w, [results["baselines"]["pca"][c]["auroc_mean"] for c in classes], w,
           label="PCA", color="#8ecae6")
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.text(len(classes) - 0.5, 0.505, "chance", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("per-image pixel AUROC"); ax.set_ylim(0, 1)
    ax.set_title("Pixel-level anomaly localization (mean $\\pm$ std over images)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, f"08_per_class_auroc{tag}")

    # AP against random baseline
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ap = [results["per_class"][c]["ap_mean"] for c in classes]
    base = [results["per_class"][c]["base_rate_mean"] for c in classes]
    ax.bar(x - w / 2, ap, w, label="average precision", color="#264653")
    ax.bar(x + w / 2, base, w, label="base rate (random AP)", color="#adb5bd")
    for i, (a, b) in enumerate(zip(ap, base)):
        ax.text(i, max(a, b) + 0.03, f"{a/b:.2f}x", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("average precision"); ax.set_ylim(0, 1.15)
    ax.set_title("AP vs its random baseline (base rates span 2%-94%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, f"09_ap_vs_baserate{tag}")


def plot_qualitative(maps_dir, n_per_class=2, tag=""):
    """
        Plot qualitative examples of the fused score and attention entropy maps.
        ground truth | fused score | attention entropy
    """
    files = sorted(maps_dir.glob("*.npz"))
    if not files:
        return
    by_cls = {}
    for f in files:
        by_cls.setdefault(f.stem.split("_")[0], []).append(f)
    chosen = [f for fs in by_cls.values() for f in fs[:n_per_class]]

    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for r, f in enumerate(chosen):
        d = np.load(f)
        axes[r, 0].imshow(d["mask"], cmap="gray"); axes[r, 0].set_ylabel(f.stem, fontsize=7)
        axes[r, 1].imshow(d["fused"], cmap="inferno")
        axes[r, 2].imshow(d["entropy"], cmap="viridis")
        if r == 0:
            for c, t in enumerate(["ground truth", "fused score S", "attention entropy"]):
                axes[r, c].set_title(t, fontsize=10)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    _save(fig, f"10_qualitative{tag}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--evaluation", action="store_true")
    args = ap.parse_args()
    do_all = not (args.training or args.evaluation)

    if do_all or args.training:
        p = MODEL_ARTIFACTS_DIR / "train_summary.json"
        if p.exists():
            print(f"Training figures from {p}")
            plot_training(json.load(open(p)))
        else:
            print(f"  (no {p}, skipping training figures)")

    if do_all or args.evaluation:
        for p in sorted(MODEL_ARTIFACTS_DIR.glob("results_*.json")):
            tag = "_" + p.stem[len("results_"):]
            print(f"Evaluation figures from {p}")
            plot_evaluation(json.load(open(p)), tag=tag)

        for d in sorted(MODEL_ARTIFACTS_DIR.glob("maps_*")):
            if d.is_dir():
                tag = "_" + d.name[len("maps_"):]
                print(f"Qualitative figures from {d}")
                plot_qualitative(d, tag=tag)


if __name__ == "__main__":
    main()