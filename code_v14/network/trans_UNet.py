import torch
import torch.nn as nn
import torch.nn.functional as F

from network.prototypes import NormalityPrototypes
from network.cnn_encoder import MultiSpectralCNNEncoder
from network.pretrained_encoder import PretrainedResNetEncoder
from utils.globals import (
    IN_CHANNELS, EMBED_DIM, NUM_ATT_HEADS, NUM_TRANS_LAYERS, DROPOUT,
    BACKBONE, INPUT_UPSCALE, TEST_CLASSES,
    NUM_PROTOTYPES, PROTO_DIM, PROTO_MOMENTUM, DEV_TOPK, USE_DEV_BRANCH,
)
from network.UNet_decoder import UNetDecoder

class TransUNetSentinel2(nn.Module):
    """
        Sentinel-2 Anomaly Detection.
        
        Inputs: (B, 12, 120, 120)
        Outputs:
            - Image Classification Logits: (B)
            - Pixel Segmentation Logits: (B, 120, 120)
    """

    def __init__(self, in_channels=12, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1, backbone="cnn", upscale=2, pretrained=True, num_types=3,
                num_prototypes=64, proto_dim=128, proto_momentum=0.99, dev_topk=8, use_dev=True):
        super(TransUNetSentinel2, self).__init__()

        self.embed_dim = embed_dim
        self.use_dev = use_dev

        # --- Encoder: from-scratch CNN (baseline) or Sentinel-2 pretrained ResNet-50 ("resnet50_softcon" | "resnet50_moco")
        if backbone == "cnn":
            self.enc = MultiSpectralCNNEncoder(in_channels=in_channels, embed_dim=embed_dim)
        else:
            self.enc = PretrainedResNetEncoder(in_channels=in_channels, embed_dim=embed_dim, weights=backbone.split("_")[1], upscale=upscale, pretrained=pretrained)
        self.out_size = self.enc.out_size  # 15 (cnn / resnet x2), 8 (resnet x1)

        # --- [CLS] token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_patches = self.out_size * self.out_size
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
            nn.Linear(embed_dim + int(use_dev), 128),      # [CLS] (+ image deviation score)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # --- Anomaly Type Head: [CLS] + bottleneck features pooled inside the predicted anomaly region
        self.type_head = nn.Sequential(
            nn.Linear(2 * embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_types)
        )

        # --- U-Net Decoder for Anomaly Localization
        self.unet_decoder = UNetDecoder(embed_dim=embed_dim + int(use_dev), skip_channels=self.enc.skip_channels)  # bottleneck (+ deviation map)

        # --- Normality-Prototype Deviation Branch (not built when use_dev=False: same layers / init order as v11)
        self.dev_topk = dev_topk
        if use_dev:
            self.dev_proj = nn.Sequential(          # token embedding for the deviation space
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, proto_dim)
            )
            self.prototypes = NormalityPrototypes(dim=proto_dim, num_prototypes=num_prototypes, momentum=proto_momentum)

        # --- Parameter initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)



    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- CNN Encoder
        feat, skip1, skip2, skip3 = self.enc(x)


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


        # Reshape spatial sequence back to 2D feature grid
        bottleneck_feat = spatial_tokens_out.transpose(1, 2).reshape(B, self.embed_dim, self.out_size, self.out_size)  # (B, embed_dim, 15, 15)

        # --- Normality-Prototype Deviation Branch
        if self.use_dev:
            z = F.normalize(self.dev_proj(spatial_tokens_out), dim=-1)              # (B, 225, proto_dim)
            dev = self.prototypes(z)                                                # (B, 225) distance to the nearest normal prototype
            dev_map = dev.reshape(B, 1, self.out_size, self.out_size)               # (B, 1, 15, 15)
            dev_score = dev.topk(self.dev_topk, dim=1).values.mean(dim=1)           # (B) image deviation score
            cls_in = torch.cat([cls_token_out, dev_score.unsqueeze(1)], dim=1)
            dec_in = torch.cat([bottleneck_feat, dev_map], dim=1)
        else:   # v11 architecture; zero deviation outputs keep the train / test code unchanged
            z = None
            dev_map = spatial_tokens_out.new_zeros(B, 1, self.out_size, self.out_size)
            dev_score = spatial_tokens_out.new_zeros(B)
            cls_in, dec_in = cls_token_out, bottleneck_feat

        # --- Classification Branch
        cls_logits = self.cls_head(cls_in).view(-1)                                 # (B)

        # --- Anomaly Localization Branch (U-Net Decoder)
        seg_logits = self.unet_decoder(dec_in, skip1, skip2, skip3)                 # (B, 120, 120)


        # --- Anomaly Type Branch
        # Average pooling of anomaly map probabilities (detached, the anomaly type loss does not modify the segmentation)
        w = F.adaptive_avg_pool2d(torch.sigmoid(seg_logits).detach().unsqueeze(1), self.out_size)  # (B, 1, 15, 15)
        # Mask-weighted average pooling of the bottleneck tokens
        region_feat = (bottleneck_feat * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-4)          # (B, embed_dim)
        # Anomaly classification head
        type_logits = self.type_head(torch.cat([cls_token_out, region_feat], dim=1))              # (B, num_types)


        aux = {"z": z, "dev_map": dev_map.squeeze(1), "dev_score": dev_score}
        return cls_logits, seg_logits, type_logits, aux



def build_model(pretrained=True):
    """
        Build the model from config.yaml
        (pretrained=False at test time)
    """
    return TransUNetSentinel2(
        in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, num_heads=NUM_ATT_HEADS, num_layers=NUM_TRANS_LAYERS, dropout=DROPOUT,
        backbone=BACKBONE, upscale=INPUT_UPSCALE, pretrained=pretrained, num_types=len(TEST_CLASSES),
        num_prototypes=NUM_PROTOTYPES, proto_dim=PROTO_DIM, proto_momentum=PROTO_MOMENTUM, dev_topk=DEV_TOPK, use_dev=USE_DEV_BRANCH,
    )