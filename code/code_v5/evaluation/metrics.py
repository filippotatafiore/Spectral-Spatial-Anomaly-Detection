import numpy as np


# ======================================== Pixel-level metrics for an image
def roc_auc(y_true, y_score):
    """
        Compute area under ROC curve.
    """

    # Convert to numpy arrays and flatten
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    # number of positive (1) and negative (0) samples
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:  # a class is missing
        return float("nan")

    # Sort based on scores
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    ranks = np.empty(s.size, dtype=np.float64)  # rank of each score
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:  # identical scores (ties)
            j += 1
        # Assign average rank to all tied scores
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1

    # sum of ranks of positive samples
    sum_ranks_pos = ranks[y_sorted == 1].sum()

    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))  # = ROC AUC


def average_precision(y_true, y_score):
    """
        Compute average precision.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    # keep only last index of each run of tied scores
    keep = np.ones(s.size, dtype=bool)
    keep[:-1] = s_sorted[1:] != s_sorted[:-1]

    tp, fp = tp[keep], fp[keep]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())  # Average precision = sum_k (R_k - R_{k-1}) * P_k


def _best_f1(y, s, n_thresholds=64):
    """
        Compute best F1 score.
    """
    thresholds = np.percentile(s, np.linspace(0.5, 99.9, n_thresholds))
    best = (0.0, 0.0, float("nan"))
    pos = y.sum()
    for t in np.unique(thresholds):
        pred = s >= t
        tp = float(np.logical_and(pred, y == 1).sum())
        if tp == 0:
            continue
        fp = float(pred.sum() - tp)
        fn = float(pos - tp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, tp / (tp + fp + fn), float(t))
    return best


def per_image_metrics(score, mask):
    """
        Pixel-level metrics for one image.
    """
    y = mask.ravel().astype(np.int8)
    s = score.ravel().astype(np.float64)
    pos = int(y.sum())
    if pos == 0 or pos == y.size:
        return None

    base = pos / y.size
    ap = average_precision(y, s)
    f1, iou, thr = _best_f1(y, s)
    return {
        "auroc": roc_auc(y, s),         # Area under ROC curve
        "ap": ap,                       # Average precision
        "base_rate": base,              # Base rate of positive pixels
        "ap_gain": ap / base,           # Average precision gain over base rate
        "best_f1": f1,                  # Best F1 score
        "iou_at_best_f1": iou,          # Intersection over union at best F1 score
        "threshold_at_best_f1": thr,    # Threshold at best F1 score
        "n_positive": pos,              # Number of positive pixels
    }


def evaluate(ctx, maps_list, alpha=None, variant=None, sign=None, beta=None, with_3d_roc=False):
    """
        Compute pixel-level metrics for each image in maps_list.
    """
    records = []
    for m in maps_list:
        s = ctx.score(m, alpha, variant, sign, beta)  # Anomaly Score for image m
        r = per_image_metrics(s, m["mask"])  # Pixel-level metrics for image m

        if r is None:  # AUROC undefined (mask is entirely one class)
            continue
        if with_3d_roc:
            # roc 3d metrics
            r3 = roc_3d(m["mask"], s)
            if r3:
                r.update(r3)
        records.append(r)

    return records



# ======================================== Aggregate metrics over images and classes
def aggregate(records):
    """
        mean / std over images.
    """
    if not records:
        return {"n_images": 0}
    out = {"n_images": len(records)}
    for k in ("auroc", "ap", "ap_gain", "best_f1", "iou_at_best_f1", "base_rate",
              "auc_df", "auc_dtau", "auc_ftau", "auc_td", "auc_bs", "auc_odp", "auc_snpr"):
        if k not in records[0]:
            continue
        v = np.array([r[k] for r in records], dtype=np.float64)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def macro_average(per_class, key, exclude=()):
    """
        Average over classes.
    """
    vals = [m[key] for c, m in per_class.items() if key in m and c not in exclude]
    return float(np.mean(vals)) if vals else float("nan")



# ======================================== Threshold selection and pixel-level metrics with the selected threshold
def threshold_at_fpr(normal_scores, target_fpr):
    """
        Choose the decision threshold using normal data only.
    """
    # Sort the normal scores and pick the threshold at the (1 - target_fpr) quantile (value such that target_fpr fraction of normal scores are above it)
    return float(np.quantile(np.asarray(normal_scores), 1.0 - target_fpr))


def operating_point_metrics(score, mask, threshold):
    """
        Compute pixel-level metrics at a fixed threshold.
    """
    y = mask.ravel().astype(np.int8)
    pred = (score.ravel() >= threshold).astype(np.int8)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int(((1 - pred) & (y == 1)).sum()); tn = int(((1 - pred) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    base = float(y.mean())
    return {
        "op_precision": prec,
        "op_recall": rec,
        "op_f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "op_iou": tp / max(tp + fp + fn, 1),
        "op_fpr": fp / max(fp + tn, 1),
        "trivial_f1": 2 * base / (1 + base),
        "trivial_iou": base,
    }



# ======================================== 
def image_level_auroc(normal_scores, anomalous_scores):
    """
        Compute image-level AUROC from the scores of normal and anomalous images.
    """
    y = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    s = np.concatenate([normal_scores, anomalous_scores])
    return roc_auc(y, s)


def detection_at_operating_point(normal_scores, anomalous_scores, threshold):
    """
        Number of anomalous images detected at a fixed threshold, and false alarms on normal images.
    """
    a = np.asarray(anomalous_scores); n = np.asarray(normal_scores)
    tp = int((a >= threshold).sum()); fp = int((n >= threshold).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(len(a), 1)
    return {
        "detection_rate": rec,                    # recall over anomalous scenes
        "false_alarm_rate": fp / max(len(n), 1),
        "precision": prec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "n_anomalous": int(len(a)), "n_normal": int(len(n)),
    }


def roc_3d(y_true, y_score, n_tau=200):
    """
        3D ROC analysis. Sweeps a normalised threshold tau over [0,1].

          AUC_(D,F)    area under (Pd, Pf)   = ordinary AUROC, detection accuracy
          AUC_(D,tau)  area under (Pd, tau)  = target detectability
          AUC_(F,tau)  area under (Pf, tau)  = background suppression, LOWER is better

          AUC_TD   = AUC_(D,F) + AUC_(D,tau)
          AUC_BS   = AUC_(D,F) - AUC_(F,tau)
          AUC_ODP  = AUC_(D,tau) + (1 - AUC_(F,tau))
          AUC_SNPR = AUC_(D,tau) / AUC_(F,tau)
    """
    y = np.asarray(y_true).ravel().astype(np.int8)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return None
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)

    taus = np.linspace(0.0, 1.0, n_tau)
    pos, neg = s[y == 1], s[y == 0]
    pd = np.array([(pos >= t).mean() for t in taus])
    pf = np.array([(neg >= t).mean() for t in taus])

    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc_df = float(trap(pd[::-1], pf[::-1]))
    auc_dtau = float(trap(pd, taus))
    auc_ftau = float(trap(pf, taus))
    return {
        "auc_df": auc_df,
        "auc_dtau": auc_dtau,
        "auc_ftau": auc_ftau,
        "auc_td": auc_df + auc_dtau,
        "auc_bs": auc_df - auc_ftau,
        "auc_odp": auc_dtau + (1.0 - auc_ftau),
        "auc_snpr": auc_dtau / max(auc_ftau, 1e-12),
    }


def classification_metrics(normal_scores, anomalous_scores, threshold):
    """
        Binary classification of whole images
    """
    n = np.asarray(normal_scores, dtype=np.float64).ravel()
    a = np.asarray(anomalous_scores, dtype=np.float64).ravel()

    tp = int((a >= threshold).sum()); fn = int(a.size - tp)
    fp = int((n >= threshold).sum()); tn = int(n.size - fp)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)            # = TPR = detection rate
    spec = tn / max(tn + fp, 1)           # = 1 - FPR
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec,
        "recall": rec,
        "detection_rate": rec,
        "specificity": spec,
        "false_alarm_rate": fp / max(fp + tn, 1),
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": 0.5 * (rec + spec),
        "n_anomalous": int(a.size),
        "n_normal": int(n.size),
        "prevalence": a.size / max(a.size + n.size, 1),
        "threshold": float(threshold),
    }



# ======================================== Logistic regression to combine the image statistics (IMG_STATS) into a single score
def fit_logistic(X, y, l2=1.0, iters=400, lr=1.0):
    """
        L2-regularised logistic regression used to combine the image statistics into a single score.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1] / len(y)
        w -= lr * g
    return w


def apply_logistic(X, w):
    """
        Decision scores (log-odds) for the fitted model.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.hstack([X, np.ones((len(X), 1))]) @ w