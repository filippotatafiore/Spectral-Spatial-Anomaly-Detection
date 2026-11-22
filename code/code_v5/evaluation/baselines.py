import numpy as np


def rx_score(image, regularization=1e-6):
    """
        Compute RX anomaly score of each pixel in an image.
        R-X detector: Mahalanobis distance of each pixel to the scene's own background statistics.

        image: (C, H, W)
    """
    c, h, w = image.shape
    flat = image.reshape(c, -1).astype(np.float64)  # (C, H*W)

    # Compute mean and covariance of the image
    mean = flat.mean(axis=1, keepdims=True)
    cen = flat - mean
    cov = (cen @ cen.T) / max(flat.shape[1] - 1, 1)

    # Regularize covariance matrix to avoid singularity
    cov += regularization * np.trace(cov) / c * np.eye(c)
    inv = np.linalg.inv(cov)

    # Compute Mahalanobis distance for each pixel
    return np.einsum("bn,bc,cn->n", cen, inv, cen).reshape(h, w).astype(np.float32)


class PCAReconstructor:
    """
        Compute anomaly scores based on PCA reconstruction error.
        (PCA fitted from training mean/covariance in stats.json).
    """

    def __init__(self, mean, cov, n_components=4):
        self.mean = np.asarray(mean, dtype=np.float64)
        vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))  # eigenvalues and eigenvectors of covariance matrix
        # Sort eigenvalues
        order = np.argsort(vals)[::-1]
        # Select the top n_components eigenvectors
        self.components = vecs[:, order[:n_components]]
        self.explained = float(vals[order][:n_components].sum() / vals.sum())

    def score(self, image):
        c, h, w = image.shape
        flat = image.reshape(c, -1).astype(np.float64) - self.mean[:, None]  # (C, H*W)

        # Project onto PCA subspace
        proj = self.components.T @ flat
        # Reconstruction error
        resid = flat - self.components @ proj
        return (resid ** 2).mean(axis=0).reshape(h, w).astype(np.float32)


class SpectralLinearPredictor:
    """
        Predict each band from the other 11 of the same pixel, by least squares on normal training pixels
        Score = how badly those relations are violated.
    """

    def __init__(self, band_mean, band_std, train_files, loader, max_files=300, stride=40):
        self.mean = np.asarray(band_mean, dtype=np.float64)
        self.std = np.asarray(band_std, dtype=np.float64)
        n_bands = len(self.mean)

        rows = []
        for p in train_files[:max_files]:
            rows.append(self._standardize(loader(p))[::stride])
        X = np.concatenate(rows, 0)

        self.weights = []
        for b in range(n_bands):
            others = [j for j in range(n_bands) if j != b]
            w, *_ = np.linalg.lstsq(X[:, others], X[:, b], rcond=None)
            self.weights.append(w)
        self.n_pixels_fitted = len(X)

    def _standardize(self, img):
        c = img.shape[0]
        return (img.reshape(c, -1).T.astype(np.float64) - self.mean) / self.std

    def score(self, image):
        """image: (C, H, W) raw reflectance -> (H, W) violation of the cross-band model"""
        c, h, w = image.shape
        z = self._standardize(image)
        err = np.zeros(len(z))
        for b in range(c):
            others = [j for j in range(c) if j != b]
            err += (z[:, others] @ self.weights[b] - z[:, b]) ** 2
        return (err / c).reshape(h, w).astype(np.float32)


def spectral_index_score(image, a, b, negate=True):
    """
        Normalised difference index (x_a - x_b) / (x_a + x_b) as an anomaly score.
    """
    x = image[a].astype(np.float64)
    y = image[b].astype(np.float64)
    s = (x - y) / np.maximum(x + y, 1e-6)
    return -s if negate else s



SPECTRAL_INDICES = {
    "nbr":  (7, 11, True),    # (B8  - B12) / (B8  + B12)   burn severity
    "nbr2": (10, 11, True),   # (B11 - B12) / (B11 + B12)   burn severity, SWIR-only
    "ndvi": (7, 3, True),     # (B8  - B4)  / (B8  + B4)    vegetation
    "ndwi": (2, 7, False),    # (B3  - B8)  / (B3  + B8)    water
}