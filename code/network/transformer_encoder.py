import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
        Multi-Head Self Attention. Return also reduced center-token statistics.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, center=None, return_maps=False):
        batch, tokens, dim = x.shape

        # Query, Key, Value
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        if center is None and not return_maps:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0
            )
            stats = None
        else:
            attn = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            out = self.attn_drop(attn) @ value
            stats = {}
            if center is not None:
                row = attn[:, :, center, :]  # what the center token attends TO
                col = attn[:, :, :, center]  # attention the center receives
                stats["center_entropy"] = -(row * torch.log(row.clamp_min(1e-12))).sum(-1)
                stats["center_received"] = col.mean(dim=-1)
            if return_maps:
                stats["attn"] = attn.detach()

        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out), stats


class EncoderBlock(nn.Module):
    """
        Transformer encoder block.
    """

    def __init__(self, d_model, n_heads, ffn_hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, center=None, return_maps=False):
        attended, stats = self.attn(self.norm1(x), center=center, return_maps=return_maps)
        x = x + self.drop(attended)  # residual connection
        x = x + self.drop(self.ffn(self.norm2(x)))  # layer norm, residual connection
        return x, stats