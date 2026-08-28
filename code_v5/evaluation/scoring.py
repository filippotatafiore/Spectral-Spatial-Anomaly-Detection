import torch
import numpy as np
import time

from data.transforms import center_token_index, extract_patches, iter_all_centers
from data.dataset import _load_image

from utils.globals import (
    CHUNK_PIXELS, PATCH_SIZE, N_TOKENS, N_BANDS, SPECTRAL_MASK_N, SPECTRAL_MASK_EVAL_GROUPS,
    CLIP_REFLECTANCE, SPECTRAL_MASK_MODE, SPECTRAL_BAND_GROUPS,
)


ATTN_VARIANTS = ("entropy", "received")  # AttentionOutlier metric (either entropy or received attention)
SIGNS = (+1.0, -1.0)
IMG_STATS = {  # Different ways of collapsing a score map into one number per image
    "p99":           lambda s: float(np.percentile(s, 99)),
    "median":        lambda s: float(np.median(s)),
    "mean":          lambda s: float(s.mean()),
    "p99_over_med":  lambda s: float(np.percentile(s, 99) / max(np.median(s), 1e-9)),
    "p99_minus_med": lambda s: float(np.percentile(s, 99) - np.median(s)),
}



# ======================================== Anomaly score terms
@torch.no_grad()
def score_image(model, builder, image, device, chunk=CHUNK_PIXELS, svdd_center=None):
    """
        Score every pixel of one image (compute terms of Anomaly Score).
        
        image: (C, H, W)

        Return:
            recon: reconstruction error (of the centre pixel's hidden bands)
            center_entropy: entropy of the center token's attention distribution (how spread out the attention is over the patch)
            center_received: sum of attention weights received by the center token (how much the other tokens are attending to the center pixel)
    """
    height, width = image.shape[-2:]
    center = center_token_index(PATCH_SIZE)
    batched = image.unsqueeze(0)
    use_amp = device.type == "cuda"

    recon_map = np.empty(height * width, dtype=np.float32)
    svdd_map = np.empty(height * width, dtype=np.float32)
    entropy_map = np.empty(height * width, dtype=np.float32)
    received_map = np.empty(height * width, dtype=np.float32)

    band_groups = None
    if SPECTRAL_MASK_MODE == "group" and SPECTRAL_BAND_GROUPS:
        # Rotate through the physical groups: every band is predicted exactly once
        band_groups = [torch.tensor(SPECTRAL_BAND_GROUPS[g], device=device, dtype=torch.long)
                       for g in sorted(SPECTRAL_BAND_GROUPS)]
    elif SPECTRAL_MASK_N > 0:
        # Disjoint groups covering all bands, so every band is predicted exactly once
        perm = torch.randperm(N_BANDS, generator=torch.Generator().manual_seed(0))
        band_groups = [perm[i::SPECTRAL_MASK_EVAL_GROUPS].to(device)
                       for i in range(SPECTRAL_MASK_EVAL_GROUPS)]


    # one row per masked group, so the per-group errors can be kept apart
    group_maps = (np.empty((len(band_groups), height * width), dtype=np.float32) if band_groups is not None else None)

    for ch in iter_all_centers(height, width, chunk, device):
        raw = extract_patches(batched, ch.unsqueeze(0), PATCH_SIZE)   # (1, n, T, C)
        _, target = builder(raw)
        target = target.reshape(-1, N_TOKENS, target.shape[-1])
        lo, hi = int(ch[0]), int(ch[-1]) + 1

        if band_groups is None:
            features, _ = builder(raw)
            features = features.reshape(-1, N_TOKENS, features.shape[-1])
            with torch.autocast("cuda", enabled=use_amp):
                out = model(features, need_attn_stats=True)
            err = ((out["recon"][:, center].float() - target[:, center].float()) ** 2).mean(-1)
        else:
            err = torch.zeros(hi - lo, device=device)
            out = None
            for gi, hidden in enumerate(band_groups):
                raw_m = raw.clone()
                raw_m[..., hidden] = builder.band_mean[hidden]
                feats, _ = builder(raw_m, hidden=hidden)
                feats = feats.reshape(-1, N_TOKENS, feats.shape[-1])
                with torch.autocast("cuda", enabled=use_amp):
                    o = model(feats, need_attn_stats=True)
                if gi == 0:
                    out = o                      # attention statistics from the first pass only
                    if svdd_center is not None and "center_embed" in o:
                        zc = o["center_embed"].float()
                        svdd_chunk = ((zc - svdd_center) ** 2).sum(-1)
                d = o["recon"][:, center].float()[:, hidden] - target[:, center].float()[:, hidden]
                err = err + (d ** 2).sum(-1)

                # Keep also per-group error
                if gi == 0:
                    group_err = torch.zeros(len(band_groups), hi - lo, device=device)
                group_err[gi, :] = (d ** 2).mean(-1)

            err = err / N_BANDS                  # every band contributed exactly once

        recon_map[lo:hi] = err.cpu().numpy()
        svdd_map[lo:hi] = (svdd_chunk.cpu().numpy() if svdd_center is not None else np.zeros(hi - lo, dtype=np.float32))
        if group_maps is not None:
            group_maps[:, lo:hi] = group_err.cpu().numpy()
        entropy_map[lo:hi] = out["center_entropy"].float().cpu().numpy()
        received_map[lo:hi] = out["center_received"].float().cpu().numpy()

    return {
        "group_recon": (group_maps.reshape(len(band_groups), height, width) if band_groups is not None else None),
        "recon": recon_map.reshape(height, width),
        "center_entropy": entropy_map.reshape(height, width),
        "center_received": received_map.reshape(height, width),
        "svdd": svdd_map.reshape(height, width),
    }


def score_image_list(model, builder, items, device, label, with_masks, svdd_center=None):
    """
        Score a list of images (call score_image for each image).
    """
    out = []
    t0 = time.time()
    for i, item in enumerate(items):
        if with_masks:
            img_path, mask_path, cls = item
            maps = score_image(model, builder,
                               _load_image(img_path, CLIP_REFLECTANCE).to(device), device, svdd_center=svdd_center)
            maps["mask"] = np.load(mask_path).astype(np.uint8)
            maps["cls"] = cls
            maps["name"] = img_path.stem
            maps["img_path"] = img_path
        else:
            maps = score_image(model, builder,
                               _load_image(item, CLIP_REFLECTANCE).to(device), device, svdd_center=svdd_center)
            maps["name"] = item.stem
        out.append(maps)

        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  [{label}] {i + 1}/{len(items)}  ({rate:.2f}s/img, "
                  f"{rate * (len(items) - i - 1) / 60:.1f} min left)", flush=True)
    return out



# ======================================== Anomaly Score computation ( = alpha * ReconError + (1-alpha) * AttentionOutlier )

def recon_error(recon, target):
    """
        Compute the reconstruction error (MSE over the spectral bands) between recon and target.
        ReconError term of Anomaly Score.
    """
    return ((recon - target) ** 2).mean(axis=-1)


def attention_statistic(maps, variant, n_tokens):
    """
        Unsigned score from attention statistics of center token in a patch.
        AttentionOutlier term of Anomaly Score.

        entropy = H(a_c)/log(T) in [0,1]
                    1.0: the center attends uniformly to its neighbourhood
                    low: attention concentrated
        received = mean_i A[i,c] * T, how much the rest of the patch relies on the center token as evidence.
                    1.0: uniform.
    """
    if variant == "entropy":
        return np.asarray(maps["center_entropy"], dtype=np.float64) / np.log(n_tokens)
    if variant == "received":
        return np.asarray(maps["center_received"], dtype=np.float64) * n_tokens
    raise ValueError(f"Unknown attention variant: {variant!r}")


class RobustScaler:
    """
        Scaler that uses median and median absolute deviation (MAD) to scale data.
        Different RobustScalers are used for ReconError center_entropy and center_received (possible AttentionOutlier terms).
        Brings the terms ReconError and AttentionOutlier to a common scale for fusion.
    """

    def __init__(self, median=0.0, scale=1.0):
        self.median = float(median)
        self.scale = float(scale)

    @classmethod
    def fit(cls, values):
        v = np.asarray(values, dtype=np.float64).ravel()
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        # 1.4826 * MAD is a consistent estimator of sigma for Gaussian data
        return cls(med, max(1.4826 * mad, 1e-8))

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.median) / self.scale

    def to_dict(self):
        return {"median": self.median, "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(d["median"], d["scale"])


def fuse(recon, attention, svdd, alpha, beta, recon_scaler, attn_scaler, svdd_scaler, attn_sign):
    """
        Combine the scores.
        Anomaly Score = alpha*ReconError + beta*AttentionOutlier + (1-alpha-beta)*Compactness
    """
    z_recon = recon_scaler.transform(recon)
    z_attn = attn_sign * attn_scaler.transform(attention)
    z_svdd = svdd_scaler.transform(svdd)
    return alpha * z_recon + beta * z_attn + max(1.0 - alpha - beta, 0.0) * z_svdd  # attention score


def standardize_map(s):
    """
        Put one score map (Anomaly Score) on a common scale (so a single threshold transfers between scenes).

        s: Anomaly score map for one image.
    """
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return (s - med) / max(1.4826 * mad, 1e-8)