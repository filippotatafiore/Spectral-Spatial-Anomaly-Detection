import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# SSL4EO-S12 ResNet-50 checkpoints
WEIGHT_URLS = {
    "softcon": "https://huggingface.co/wangyi111/softcon/resolve/62ff465b2e7467dbfc70758ec1e9d08ab87fc46b/B13_rn50_softcon.pth",
    "moco": "https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth",
}

B10_INDEX = 10  # SSL4EO-S12 uses 13 L1C bands [B1..B8A, B9, B10, B11, B12]. L2A images have no B10

# SSL4EO-S12 per-band statistics (DN scale)
S2_MEAN = [1612.9, 1397.6, 1322.3, 1373.1, 1561.0, 2108.4, 2390.7, 2318.7, 2581.0, 837.7, 2195.2, 1537.4]
S2_STD = [791.0, 854.3, 878.7, 1144.9, 1127.5, 1164.2, 1276.0, 1249.5, 1345.9, 577.5, 1340.0, 1142.9]


class PretrainedResNetEncoder(nn.Module):
    """
        Sentinel-2 pretrained (SSL4EO-S12) ResNet-50 encoder.
        Input is upsampled by "upscale" before the ResNet (so that stride-4 stem does not erase small anomalies).
        Shallow stem on the raw (non-interpolated) input provides the full-resolution skip.

        Input: (B, 12, 120, 120)
        Output (upscale=2):
            feat (B, embed_dim, 15, 15)
            skip1 (B, 96, 120, 120)
            skip2 (B, 256, 60, 60)
            skip3 (B, 512, 30, 30)
    """

    def __init__(self, in_channels=12, embed_dim=512, weights="softcon", upscale=2, pretrained=True):
        super(PretrainedResNetEncoder, self).__init__()
        self.upscale = upscale
        self.out_size = math.ceil(120 * upscale / 16)  # spatial size after layer3 (stride 16)
        self.skip_channels = (32 + 64, 256, 512)       # (skip1, skip2, skip3)

        # --- Input normalization
        if weights == "softcon":    # pretrained on per-band standardized DNs
            mean = torch.tensor(S2_MEAN) / 10000.0
            std = torch.tensor(S2_STD) / 10000.0
        else:   # MoCo: pretrained on DN / 10000 (= our reflectance)
            mean = torch.zeros(in_channels)
            std = torch.ones(in_channels)
        self.register_buffer("band_mean", mean.view(1, -1, 1, 1))
        self.register_buffer("band_std", std.view(1, -1, 1, 1))

        # --- ResNet-50 backbone
        self.backbone = timm.create_model("resnet50", in_chans=in_channels, num_classes=0, global_pool="")
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(WEIGHT_URLS[weights], map_location="cpu")
            w = state_dict["conv1.weight"]  # (64, 13, 7, 7)
            state_dict["conv1.weight"] = torch.cat([w[:, :B10_INDEX], w[:, B10_INDEX + 1:]], dim=1)  # drop B10 -> (64, 12, 7, 7)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            assert not missing, f"Missing keys: {missing}"
            assert set(unexpected) <= {"fc.weight", "fc.bias"}, f"Unexpected keys: {unexpected}"
        self.backbone.layer4 = nn.Identity()  # don't use stride 32 (too coarse for small anomalies)

        # --- Full-resolution stem on the raw input (no interpolation)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --- Projection of layer3 features to the transformer dimension
        self.proj = nn.Sequential(
            nn.Conv2d(1024, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # Normalize input
        x = (x - self.band_mean) / self.band_std
        
        stem = self.stem(x)  # (B, 32, 120, 120)

        if self.upscale != 1:
            x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)  # (B, 12, 240, 240)

        b = self.backbone
        c1 = b.act1(b.bn1(b.conv1(x)))          # (B, 64, 120, 120)
        l1 = b.layer1(b.maxpool(c1))            # (B, 256, 60, 60)
        l2 = b.layer2(l1)                       # (B, 512, 30, 30)
        l3 = b.layer3(l2)                       # (B, 1024, 15, 15)

        if c1.shape[-2:] != stem.shape[-2:]:    # only when upscale=1
            c1 = F.interpolate(c1, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        skip1 = torch.cat([stem, c1], dim=1)    # (B, 96, 120, 120)

        return self.proj(l3), skip1, l1, l2