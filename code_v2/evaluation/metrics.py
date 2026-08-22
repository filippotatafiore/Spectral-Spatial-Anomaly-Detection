import numpy as np

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


def aggregate(records):
    """
        mean / std over images.
    """
    if not records:
        return {"n_images": 0}
    out = {"n_images": len(records)}
    for k in ("auroc", "ap", "ap_gain", "best_f1", "iou_at_best_f1", "base_rate"):
        v = np.array([r[k] for r in records], dtype=np.float64)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def image_level_auroc(normal_scores, anomalous_scores):
    """
        Compute image-level AUROC from per-image scores.
    """
    y = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    s = np.concatenate([normal_scores, anomalous_scores])
    return roc_auc(y, s)


def macro_average(per_class, key, exclude=()):
    """
        Average over classes.
    """
    vals = [m[key] for c, m in per_class.items() if key in m and c not in exclude]
    return float(np.mean(vals)) if vals else float("nan")