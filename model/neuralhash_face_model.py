import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class NeuralHashFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1),          # Conv
            ConvBlock(16, 16, 1),               # Conv_1
            ConvBlock(16, 16, 3, 1, 1),         # Conv_2
            nn.Conv2d(16, 4, 1),                # Conv_3
            nn.Conv2d(4, 16, 1),                # Conv_4
            ConvBlock(16, 16, 1),               # Conv_5
            ConvBlock(16, 56, 1),               # Conv_6
            ConvBlock(56, 56, 3, 1, 1),         # Conv_7
            ConvBlock(56, 24, 1),               # Conv_8
            ConvBlock(24, 64, 1),               # Conv_9
            ConvBlock(64, 64, 3, 1, 1),         # Conv_10
            ConvBlock(64, 24, 1),               # Conv_11
            ConvBlock(24, 72, 1),               # Conv_12
            ConvBlock(72, 72, 5, 1, 2),         # Conv_13
            nn.Conv2d(72, 18, 1),               # Conv_14
            nn.Conv2d(18, 72, 1),               # Conv_15
            ConvBlock(72, 32, 1),               # Conv_16
            ConvBlock(32, 184, 1),              # Conv_17
            ConvBlock(184, 184, 5, 1, 2),       # Conv_18
            nn.Conv2d(184, 46, 1),              # Conv_19
            nn.Conv2d(46, 184, 1),              # Conv_20
            ConvBlock(184, 32, 1),              # ...
            ConvBlock(32, 184, 1),
            ConvBlock(184, 184, 5, 1, 2),
            nn.Conv2d(184, 46, 1),
            nn.Conv2d(46, 184, 1),
            ConvBlock(184, 32, 1),
            ConvBlock(32, 88, 1),
            ConvBlock(88, 88, 5, 1, 2),
            nn.Conv2d(88, 22, 1),
            nn.Conv2d(22, 88, 1),
            ConvBlock(88, 40, 1),
            ConvBlock(40, 112, 1),
            ConvBlock(112, 112, 5, 1, 2),
            nn.Conv2d(112, 28, 1),
            nn.Conv2d(28, 112, 1),
            ConvBlock(112, 40, 1),
            ConvBlock(40, 216, 1),
            ConvBlock(216, 216, 5, 1, 2),
            nn.Conv2d(216, 54, 1),
            nn.Conv2d(54, 216, 1),
            ConvBlock(216, 72, 1),
            ConvBlock(72, 432, 1),
            ConvBlock(432, 432, 5, 1, 2),
            nn.Conv2d(432, 108, 1),
            nn.Conv2d(108, 432, 1),
            ConvBlock(432, 72, 1),
            ConvBlock(72, 432, 1),
            ConvBlock(432, 432, 5, 1, 2),
            nn.Conv2d(432, 108, 1),
            nn.Conv2d(108, 432, 1),
            ConvBlock(432, 72, 1),
            ConvBlock(72, 432, 1),
        )
        self.final = nn.Sequential(
            nn.Conv2d(432, 1280, 1),  # Conv_53
            nn.ReLU(inplace=True),
            nn.Conv2d(1280, 500, 1),  # Conv_54
            nn.ReLU(inplace=True),
            nn.Conv2d(500, 128, 1),   # Conv_55 → 128D feature
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.final(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)             # Flatten to (B, 128)

        return x
