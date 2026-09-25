import torch
import torch.nn as nn

class UpBlock(nn.Module):
    """
        Bilinear Upsampling concatenated with CNN Skip Connection.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class UNetDecoder(nn.Module):
    """
        U-Net Decoder for Anomaly Localization.
    """

    def __init__(self, embed_dim):
        super(UNetDecoder, self).__init__()

        self.up1 = UpBlock(in_channels=embed_dim, skip_channels=256, out_channels=256)  # 15 -> 30
        self.up2 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)        # 30 -> 60
        self.up3 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)          # 60 -> 120
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, bottleneck_feat, skip1, skip2, skip3):
        d1 = self.up1(bottleneck_feat, skip3)       # (B, 256, 30, 30)
        d2 = self.up2(d1, skip2)                    # (B, 128, 60, 60)
        d3 = self.up3(d2, skip1)                    # (B, 64, 120, 120)
        seg_logits = self.seg_head(d3).squeeze(1)   # (B, 120, 120)
        return seg_logits