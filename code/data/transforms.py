import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PatchExtractor:
    """
        Extract patches from input image tensor.
    """
    def __init__(self, patch_size, stride=None, scale_factor=None):
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.scale_factor = scale_factor

    def __call__(self, img_tensor):
        # Rescaling
        if self.scale_factor:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False
            ).squeeze(0)
        C, H, W = img_tensor.shape

        # Padding
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Patch Extraction
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        patches = img_tensor.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)    # (1, C, H_steps, W_steps, patch_H, patch_W)
        patches = patches.contiguous().view(-1, C, self.patch_size, self.patch_size)    # (Num_Patches, C, patch_H, patch_W)
        
        return patches


class SpectralDCT(nn.Module):
    """
        Apply DCT to each patch along the spectral dimension.
    """
    def __init__(self, num_bands):
        super().__init__()
        dct_matrix = self._create_dct_matrix(num_bands)  # (num_bands, num_bands) = (C, C)
        self.register_buffer('dct_matrix', dct_matrix)  # register DCT matrix as buffer (moves to GPU)

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

        return M    # (N, N)    with N = C (num_bands)

    def forward(self, patches):
        # apply 1D DCT to the Channel dimension (dim 1)
        # patches: (Num_Patches, C, patch_H, patch_W)

        patches_permuted = patches.permute(0, 2, 3, 1)  # = X^T
        # (Num_Patches, patch_H, patch_W, C)

        # DCT-II formula (matrix form): DCT_output^T = X^T * M^T
        dct_patches = torch.matmul(patches_permuted, self.dct_matrix.t())   # (Num_Patches, patch_H, patch_W, C)

        # permute back to (Num_Patches, C, patch_H, patch_W)
        return dct_patches.permute(0, 3, 1, 2)