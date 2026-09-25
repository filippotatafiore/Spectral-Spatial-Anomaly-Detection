import torch
import torch.nn as nn


OUT_SPATIAL_SIZE = 15  # Output spatial size of the CNN encoder (15x15)


class MultiSpectralCNNEncoder(nn.Module):
    """
        Multi-spectral CNN encoder.

        Input: (B, C, H, W) = (B, 12, 120, 120)
        Output: (B, embed_dim, H', W')
    """

    def __init__(self, in_channels=12, embed_dim=512):
        super(MultiSpectralCNNEncoder, self).__init__()

        self.enc1 = nn.Sequential(  # Input: (B, 12, 120, 120)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),  # (B, 64, 120, 120)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # Output: (B, 64, 120, 120)
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),  # (B, 64, 60, 60)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),  # (B, 128, 60, 60)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # Output: (B, 128, 60, 60)
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),  # (B, 128, 30, 30)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),  # (B, 256, 30, 30)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # Output: (B, 256, 30, 30)
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),  # (B, 256, 15, 15)
            nn.Conv2d(256, embed_dim, kernel_size=3, padding=1, bias=False),  # (B, embed_dim, 15, 15)
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )  # Output: (B, embed_dim, 15, 15)

        
    def forward(self, x):
        # x: (B, 12, 120, 120)
        skip1 = self.enc1(x)        # (B, 64, 120, 120)
        skip2 = self.enc2(skip1)    # (B, 128, 60, 60)
        skip3 = self.enc3(skip2)    # (B, 256, 30, 30)
        feat = self.enc4(skip3)     # (B, embed_dim, 15, 15)
        return feat, skip1, skip2, skip3  # Return the final feature map and skip connections