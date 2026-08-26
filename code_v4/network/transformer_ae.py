import math
import torch
import torch.nn as nn

from data.transforms import center_token_index, token_coords
from network.transformer_encoder import EncoderBlock

def _mlp(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def sinusoidal_2d_encoding(patch_size, d_model):
    """
        2D sinusoidal positional encoding for a P x P patch
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model={d_model}, must be divisible by 4 for 2D sinusoidal encoding.")

    # coordinates of each token inside the patch
    coords = token_coords(patch_size).float()  # (P*P, 2) = (T, 2)
    half = d_model // 2
    # Sinusoidal frequencies for each dimension
    freqs = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half))

    out = []
    for axis in range(2):
        angles = coords[:, axis: axis + 1] * freqs.unsqueeze(0)
        out.append(torch.sin(angles))
        out.append(torch.cos(angles))
    return torch.cat(out, dim=1)  # (T, d_model)


class SpectralSpatialTransformerAE(nn.Module):
    """
        Transformer autoencoder.
    """

    def __init__(self, feature_dim, n_bands, patch_size, d_model=64, n_heads=8,
                 n_layers=4, ffn_hidden=256, dropout=0.1, embed_hidden=128,
                 decoder_hidden=128, pos_encoding="sinusoidal2d", mask_center=False):
        super().__init__()

        self.patch_size = patch_size  # P
        self.n_tokens = token_coords(patch_size).shape[0]  # T
        self.center = center_token_index(patch_size)  # index of center pixel in the flattened P x P patch
        self.mask_center = mask_center

        # --------------- Embedding of the input features into the transformer dimension
        self.embed = _mlp(feature_dim, embed_hidden, d_model, dropout)

        if pos_encoding == "sinusoidal2d":
            self.register_buffer("pos", sinusoidal_2d_encoding(patch_size, d_model), persistent=False)
            self.pos_learned = None
        elif pos_encoding == "learned":
            self.pos = None
            self.pos_learned = nn.Parameter(torch.zeros(self.n_tokens, d_model))
            nn.init.trunc_normal_(self.pos_learned, std=0.02)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        # --------------- Transformer encoder blocks
        # Learned mask embedding for the center-masking variant.
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ffn_hidden, dropout) for _ in range(n_layers)
        )

        # --------------- Decoder (reconstruct the spectral bands from the transformer output)
        self.norm_out = nn.LayerNorm(d_model)
        self.decoder = _mlp(d_model, decoder_hidden, n_bands, dropout)

    def _positional(self):
        return self.pos if self.pos_learned is None else self.pos_learned

    def forward(self, features, need_attn_stats=False, return_maps=False):
        """features: (B, T, feature_dim) -> {"recon": (B, T, n_bands), ...}"""
        if features.shape[1] != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {features.shape[1]}")

        x = self.embed(features)
        if self.mask_center:
            # Replace the center token's content but keep its position, so the
            # spectrum must be predicted from the spatial neighbourhood.
            x = x.clone()
            x[:, self.center, :] = self.mask_token
        x = x + self._positional().unsqueeze(0)

        center = self.center if (need_attn_stats or return_maps) else None
        entropies, received, maps = [], [], []
        for block in self.blocks:
            x, stats = block(x, center=center, return_maps=return_maps)
            if stats:
                if "center_entropy" in stats:  # get center-token statistics
                    entropies.append(stats["center_entropy"])
                    received.append(stats["center_received"])
                if "attn" in stats:  # get attention maps
                    maps.append(stats["attn"])

        out = {"recon": self.decoder(self.norm_out(x))}
        if entropies:
            out["center_entropy"] = torch.stack(entropies, 0).mean(dim=(0, 2))
            out["center_received"] = torch.stack(received, 0).mean(dim=(0, 2))
        if maps:
            out["attn_maps"] = torch.stack(maps, 1)                 # (B, L, H, T, T)
        return out

    @property
    def max_entropy(self):
        return math.log(self.n_tokens)