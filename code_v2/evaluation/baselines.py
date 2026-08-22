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