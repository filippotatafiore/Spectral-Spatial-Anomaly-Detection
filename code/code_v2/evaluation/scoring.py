import numpy as np


ATTN_VARIANTS = ("entropy", "received")
SIGNS = (+1.0, -1.0)

class RobustScaler:
    """
        Scaler that uses median and median absolute deviation (MAD) to scale data.
        Put the scores from score_image() (recon, center_entropy, center_received) on a common scale.
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


def attention_statistic(maps, variant, n_tokens):
    """
        Unsigned score from attention statistics of center token in a patch.

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


def fuse(recon, attention, alpha, recon_scaler, attn_scaler, attn_sign):
    """
        Combine the scores with a weight alpha.
        Anomaly Score = alpha * ReconError + (1 - alpha) * AttentionOutlier
    """

    # Scale two terms to a common unit-free scale
    z_recon = recon_scaler.transform(recon)
    z_attn = attn_sign * attn_scaler.transform(attention)

    return alpha * z_recon + (1.0 - alpha) * z_attn  # attention score


def recon_error(recon, target):
    """
        Compute the reconstruction error (MSE over the spectral bands) between recon and target.
    """
    return ((recon - target) ** 2).mean(axis=-1)