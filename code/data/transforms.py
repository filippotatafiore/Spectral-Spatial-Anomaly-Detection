import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_token_index(patch_size):
    """
        patch_size = P
        Compute the index of center pixel in the flattened P x P patch.
    """
    return (patch_size * patch_size) // 2


def token_coords(patch_size, device=None):
    """
        Coordinates of each token inside the patch.
    """
    rows = torch.arange(patch_size, device=device)
    cols = torch.arange(patch_size, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], dim=1)  # (P*P, 2)


def sample_centers(n_images, height, width, n_centers, device, generator=None):
    """
        For training: sample n_centers random flat pixel indices as centers from each image in the batch.
    """
    return torch.randint(
        0, height * width, (n_images, n_centers),
        device=device, generator=generator, dtype=torch.long,
    )  # (n_images, n_centers) = (batch_size, n_centers) = (B, N)


def iter_all_centers(height, width, chunk, device):
    """
        For inference: use all pixels (all flat center indices in raster order) as centers.
        Yield them in chunks.
    """
    total = height * width
    for start in range(0, total, chunk):
        yield torch.arange(start, min(start + chunk, total), device=device, dtype=torch.long)  # (chunk,)


def extract_patches(images, centers, patch_size):
    """
        Extract P x P pixel-centered patches

        images: (B, C, H, W)
        centers: (batch_size, num_centers) = (B, N)
        patch_size = P
    """
    batch, channels, height, width = images.shape  # (B, 12, 120, 120)

    # Pad the images to handle pixels near the edge
    pad = patch_size // 2
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")  # (B, C, H+2*pad, W+2*pad)
    padded_w = width + 2 * pad

    # Compute row and column indices of the centers
    n_centers = centers.shape[1]
    rows = torch.div(centers, width, rounding_mode="floor")
    cols = centers % width

    # Flat indices of every token of the patches
    offsets = token_coords(patch_size, device=images.device)  # (P*P, 2)
    flat_idx = (rows.unsqueeze(-1) + offsets[:, 0]) * padded_w + (cols.unsqueeze(-1) + offsets[:, 1])  # (B, N, P*P) = (B, N, T)
    flat_idx = flat_idx.reshape(batch, 1, -1).expand(batch, channels, -1)  # (B, C, N*T)

    # Gather the patches
    gathered = torch.gather(padded.reshape(batch, channels, -1), 2, flat_idx)  # (B, C, N*T)
    gathered = gathered.reshape(batch, channels, n_centers, patch_size * patch_size)  # (B, C, N, T)
    return gathered.permute(0, 2, 3, 1).contiguous()  # (B, N, P*P, C) = (B, N, T, C)


class SpectralDCT(nn.Module):
    """
        Apply DCT to each patch along the spectral dimension.
    """
    def __init__(self, num_bands, num_coeffs=None):
        super().__init__()
        self.num_bands = num_bands  # = C
        self.num_coeffs = num_bands if num_coeffs is None else num_coeffs  # = K

        dct_matrix = self._create_dct_matrix(num_bands)  # (num_bands, num_bands) = (C, C)
        # keep only the first K coefficients
        dct_matrix = dct_matrix[: self.num_coeffs]  # (K, num_bands) = (K, C)
        self.register_buffer("dct_matrix", dct_matrix)  # register DCT matrix as buffer (moves to GPU)

    def _create_dct_matrix(self, N):
        # Type-II DCT matrix
        
        # spectral vector:  X = (x_1, x_2, ..., x_N)
        # Formula:          DCT_output_k = norm_coeff_k * sum_n (x_n * cos(pi/N * (n+0.5) * k))
        # Matrix form:      DCT_output = M * X
        #                   M = norm_coeffs * cos(pi/N * (n+0.5) * k)   (n: (1, N), k: (N, 1), M: (N, N))

        k = torch.arange(N).unsqueeze(1).float()  # k = (0; 1; 2; ...; N-1), shape: (N, 1), k rows: output frequencies
        n = torch.arange(N).unsqueeze(0).float()  # n = (0, 1, 2, ..., N-1), shape: (1, N), n columns: input bands

        # cosine term
        M = torch.cos((math.pi / N) * (n + 0.5) * k)   # (N, N)
        # orthonormalization scaling coefficients
        norm_coeffs = torch.ones(N, 1) * math.sqrt(2 / N)   # components (frequencies k > 0) scaled by sqrt(2/N)
        norm_coeffs[0, 0] = 1 / math.sqrt(N)    # first component (frequencies k = 0) scaled by sqrt(1/N)
        M *= norm_coeffs  # element-wise multiplication

        return M    # (N, N)    with N = num_bands = C

    def forward(self, patches):
        # apply 1D DCT to the Channel dimension
        # patches = X^T : (B, N, P*P, C) = (B, N, T, C)

        if patches.shape[-1] != self.num_bands:
            raise ValueError(f"Expected last dim (num_bands) {self.num_bands}, got {patches.shape[-1]}")

        # DCT-II formula (matrix form): DCT_output^T = X^T * M^T
        return torch.matmul(patches, self.dct_matrix.t())  # (B, N, T, C) @ (C, K) = (B, N, T, K)


class FeatureBuilder(nn.Module):
    """
        Build features and targets from raw patches.

        Feature layout: [raw_bands | dct_coeffs]
    """

    def __init__(self, num_bands, num_coeffs, use_raw, use_dct, feature_mean, feature_std):
        super().__init__()
        self.num_bands = num_bands
        self.use_raw = use_raw  # include raw bands in the features
        self.use_dct = use_dct  # include DCT coefficients in the features
        self.dct = SpectralDCT(num_bands, num_coeffs) if use_dct else None

        # Standardization parameters
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.as_tensor(feature_std, dtype=torch.float32)
        if not use_raw:  # DCT-only
            mean, std = mean[num_bands:], std[num_bands:]
        elif not use_dct:  # raw-only
            mean, std = mean[:num_bands], std[:num_bands]

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        # Raw-band statistics (kept separately for the reconstruction target)
        self.register_buffer("band_mean", torch.as_tensor(feature_mean[:num_bands], dtype=torch.float32))
        self.register_buffer("band_std", torch.as_tensor(feature_std[:num_bands], dtype=torch.float32))

    def forward(self, raw_patches):
        # raw_patches: (batch_size, num_centers, P*P, C) = (B, N, T, C)
        
        parts = []
        if self.use_raw:
            parts.append(raw_patches)  # raw patches: (B, N, T, C)
        if self.use_dct:
            parts.append(self.dct(raw_patches))  # dct coeffs: (B, N, T, K)

        # Concatenate features along the last dimension
        features = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]  # (B, N, T, C+K)
        # Standardization
        features = (features - self.mean) / self.std  # (B, N, T, C+K) = (B, N, T, F)

        if self.use_raw:
            target = features[..., : self.num_bands]  # (B, N, T, C)
        else:
            target = (raw_patches - self.band_mean) / self.band_std

        return features, target  # (B, N, T, F), (B, N, T, C)

    def standardize_image(self, images):
        """
            Standardize images.

            images: (B, C, H, W)
        """
        if images.dim() != 4:
            raise ValueError(f"standardize_image expects (B, C, H, W), got {tuple(images.shape)}")
        
        return (images - self.band_mean.view(1, -1, 1, 1)) / self.band_std.view(1, -1, 1, 1)  # (B, C, H, W)