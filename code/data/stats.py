import numpy as np

from data.transforms import SpectralDCT


def compute_stats(files, n_bands, n_dct_coeffs, clip=1.5, max_images=None, progress=True):
    """
        Statistics over the train split.
    """

    files = list(files)
    if max_images is not None:
        files = files[:max_images]

    # DCT matrix
    dct = SpectralDCT(n_bands, n_dct_coeffs).dct_matrix.numpy().astype(np.float64)  # (K, num_bands) = (K, B)

    count = 0  # total number of pixels across all images
    feature_dim = n_bands + n_dct_coeffs
    sum_feat = np.zeros(feature_dim, dtype=np.float64)  # sum of features across all pixels
    sumsq_feat = np.zeros(feature_dim, dtype=np.float64)  # sum of squared features across all pixels
    gram_raw = np.zeros((n_bands, n_bands), dtype=np.float64)  # sum of outer products of raw bands across all pixels

    for i, path in enumerate(files):
        arr = np.load(path).astype(np.float64)
        if clip is not None:
            np.clip(arr, 0.0, clip, out=arr)  # Clip values to [0, clip]

        raw = arr.reshape(n_bands, -1)  # (num_bands, num_pixels) = (B, N)
        feat = np.concatenate([raw, dct @ raw], axis=0)  # (num_bands + num_coeffs, num_pixels) = (B + K, N)

        count += raw.shape[1]  # num_pixels = N
        sum_feat += feat.sum(axis=1)
        sumsq_feat += (feat ** 2).sum(axis=1)
        gram_raw += raw @ raw.T

        if progress and (i + 1) % 500 == 0:
            print(f"  [stats] {i + 1}/{len(files)} images, {count:,} pixels")

    mean = sum_feat / count
    std = np.sqrt(np.maximum(sumsq_feat / count - mean ** 2, 1e-12))
    mean_raw = mean[:n_bands]
    cov_raw = gram_raw / count - np.outer(mean_raw, mean_raw)

    return {
        "n_images": len(files),
        "n_pixels": int(count),
        "n_bands": n_bands,
        "n_dct_coeffs": n_dct_coeffs,
        "clip": clip,
        "feature_mean": mean.tolist(),      # per-feature mean (of [raw_bands | dct_coeffs] )
        "feature_std": std.tolist(),        # per-feature std
        "band_mean": mean_raw.tolist(),     # per-band mean
        "band_cov": cov_raw.tolist(),       # per-band covariance
    }