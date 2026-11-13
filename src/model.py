import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 128)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)

        return e1, e2, e3, b


class BaselineModel(nn.Module):
    """
    Baseline encoder-decoder model without skip connections.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = Encoder()

        self.up3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(64, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(32, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(16, 16)

        self.output = nn.Conv2d(
            16, num_classes, kernel_size=1
        )

    def forward(self, x):
        _, _, _, b = self.encoder(x)

        x = self.up3(b)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.output(x)


class SkipFusionModel(nn.Module):
    """
    Same encoder-decoder structure as the baseline,
    but adds encoder features through skip fusion.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = Encoder()

        self.up3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.fuse3 = ConvBlock(64 + 128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.fuse2 = ConvBlock(32 + 64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )
        self.fuse1 = ConvBlock(16 + 32, 16)

        self.output = nn.Conv2d(
            16, num_classes, kernel_size=1
        )

    def forward(self, x):
        e1, e2, e3, b = self.encoder(x)

        x = self.up3(b)

        x = torch.cat([x, e3], dim=1)
        x = self.fuse3(x)

        x = self.up2(x)

        x = torch.cat([x, e2], dim=1)
        x = self.fuse2(x)

        x = self.up1(x)

        x = torch.cat([x, e1], dim=1)
        x = self.fuse1(x)

        return self.output(x)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )