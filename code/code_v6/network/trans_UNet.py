import torch
import torch.nn as nn

from network.cnn_encoder import MultiSpectralCNNEncoder, OUT_SPATIAL_SIZE
from network.UNet_decoder import UNetDecoder


class TransUNetSentinel2(nn.Module):
    """
        Sentinel-2 Anomaly Detection.
        
        Inputs: (B, 12, 120, 120)
        Outputs:
            - Image Classification Logits: (B)
            - Pixel Segmentation Logits: (B, 120, 120)
    """
    
    def __init__(self, in_channels=12, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super(TransUNetSentinel2, self).__init__()

        self.embed_dim = embed_dim

        # --- CNN Encoder
        self.cnn_enc = MultiSpectralCNNEncoder(in_channels=in_channels, embed_dim=embed_dim)

        # --- [CLS] token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_patches = OUT_SPATIAL_SIZE * OUT_SPATIAL_SIZE  # 225 spatial tokens (15x15 grid from CNN encoder)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # --- Transformer Bottleneck
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Image Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # --- U-Net Decoder for Anomaly Localization
        self.unet_decoder = UNetDecoder(embed_dim=embed_dim)

        # --- Parameter initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)



    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- CNN Encoder
        feat, skip1, skip2, skip3 = self.cnn_enc(x)


        # Flatten
        spatial_seq = feat.flatten(2).transpose(1, 2)  # (B, 15*15, embed_dim) = (B, 225, embed_dim)
        # [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat((cls_tokens, spatial_seq), dim=1)  # (B, 226, embed_dim)
        # Positional Embedding (with dropout)
        seq = self.pos_drop(seq + self.pos_embed)


        # --- Transformer Bottleneck
        tf_out = self.transformer(seq) # (B, 226, embed_dim)
        
        # Extract CLS token
        cls_token_out = tf_out[:, 0, :]  # (B, embed_dim)
        # Extract spatial tokens
        spatial_tokens_out = tf_out[:, 1:, :] # (B, 225, embed_dim)


        # --- Classification Branch
        # Classification logits
        cls_logits = self.cls_head(cls_token_out).view(-1)  # (B)


        # --- Anomaly Localization Branch
        # Reshape spatial sequence back to 2D feature grid
        bottleneck_feat = spatial_tokens_out.transpose(1, 2).view(B, self.embed_dim, OUT_SPATIAL_SIZE, OUT_SPATIAL_SIZE)  # (B, embed_dim, 15, 15)
        
        # U-Net Decoder
        seg_logits = self.unet_decoder(bottleneck_feat, skip1, skip2, skip3)  # (B, 120, 120)
        
        return cls_logits, seg_logits