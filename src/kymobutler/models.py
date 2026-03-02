"""PyTorch neural network definitions translated from NeuralNetworkDefs.wl."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class LeakyReLUAlpha(nn.Module):
    """Leaky ReLU with explicit alpha matching Mathematica layer definitions."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, self.alpha * x)


class BasicBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            LeakyReLUAlpha(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    """BatchNorm -> LeakyReLU -> Conv block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            LeakyReLUAlpha(0.1),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DSConvolutionLayer(nn.Module):
    """Depthwise separable convolution layer."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                groups=in_ch,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DSConvBlock(nn.Module):
    """Depthwise separable conv block + BN + activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DSConvolutionLayer(in_ch, out_ch, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_ch),
            LeakyReLUAlpha(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNetEncoder(nn.Module):
    """UNet encoder stack used by all KymoButler UNet variants."""

    def __init__(self, base: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        block = DSConvBlock if depthwise else BasicBlock

        self.conv1 = nn.Sequential(
            BasicBlock(1, base),
            block(base, base),
            nn.Dropout2d(0.1),
        )
        self.conv2 = nn.Sequential(
            block(base, 2 * base),
            block(2 * base, 2 * base),
            nn.Dropout2d(0.1),
        )
        self.conv3 = nn.Sequential(
            block(2 * base, 4 * base),
            block(4 * base, 4 * base),
            nn.Dropout2d(0.1),
        )
        self.conv4 = nn.Sequential(
            block(4 * base, 8 * base),
            block(8 * base, 8 * base),
            nn.Dropout2d(0.1),
        )
        self.conv5 = nn.Sequential(
            block(8 * base, 16 * base),
            block(16 * base, 16 * base),
            nn.Dropout2d(0.2),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool(c1))
        c3 = self.conv3(self.pool(c2))
        c4 = self.conv4(self.pool(c3))
        c5 = self.conv5(self.pool(c4))
        return c1, c2, c3, c4, c5


class UNetDecoder(nn.Module):
    """UNet decoder stack for KymoButler models."""

    def __init__(self, base: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        block = DSConvBlock if depthwise else BasicBlock

        self.up1 = nn.ConvTranspose2d(16 * base, 8 * base, kernel_size=2, stride=2)
        self.uconv1 = nn.Sequential(block(16 * base, 8 * base), block(8 * base, 8 * base))

        self.up2 = nn.ConvTranspose2d(8 * base, 4 * base, kernel_size=2, stride=2)
        self.uconv2 = nn.Sequential(block(8 * base, 4 * base), block(4 * base, 4 * base))

        self.up3 = nn.ConvTranspose2d(4 * base, 2 * base, kernel_size=2, stride=2)
        self.uconv3 = nn.Sequential(block(4 * base, 2 * base), block(2 * base, 2 * base))

        self.up4 = nn.ConvTranspose2d(2 * base, base, kernel_size=2, stride=2)
        self.uconv4 = nn.Sequential(block(2 * base, base), block(base, base))

    def forward(self, c1: Tensor, c2: Tensor, c3: Tensor, c4: Tensor, c5: Tensor) -> Tensor:
        x = self.up1(c5)
        x = self.uconv1(torch.cat([x, c4], dim=1))

        x = self.up2(x)
        x = self.uconv2(torch.cat([x, c3], dim=1))

        x = self.up3(x)
        x = self.uconv3(torch.cat([x, c2], dim=1))

        x = self.up4(x)
        x = self.uconv4(torch.cat([x, c1], dim=1))
        return x


class UNet(nn.Module):
    """Standard KymoButler segmentation UNet with 2-class softmax output."""

    def __init__(self, base_channels: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        self.enc = UNetEncoder(base=base_channels, depthwise=depthwise)
        self.dec = UNetDecoder(base=base_channels, depthwise=depthwise)
        self.head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        c1, c2, c3, c4, c5 = self.enc(x)
        z = self.dec(c1, c2, c3, c4, c5)
        return torch.softmax(self.head(z), dim=1)


class UNetUnidirectional(nn.Module):
    """UNet with separate anterograde and retrograde 2-class heads."""

    def __init__(self, base_channels: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        self.enc = UNetEncoder(base=base_channels, depthwise=depthwise)
        self.dec = UNetDecoder(base=base_channels, depthwise=depthwise)
        self.ant_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.ret_head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        c1, c2, c3, c4, c5 = self.enc(x)
        z = self.dec(c1, c2, c3, c4, c5)
        ant = torch.softmax(self.ant_head(z), dim=1)
        ret = torch.softmax(self.ret_head(z), dim=1)
        return {"ant": ant, "ret": ret, "Antero": ant, "Retro": ret}


class ClassNet(nn.Module):
    """Simple classification network translated from `classnet` definition."""

    def __init__(self, nout: int, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        n = base_channels
        self.features = nn.Sequential(
            BasicBlock(in_channels, n),
            BasicBlock(n, n),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            BasicBlock(n, 2 * n),
            BasicBlock(2 * n, 2 * n),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            BasicBlock(2 * n, 4 * n),
            BasicBlock(4 * n, 4 * n),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(4 * n, 256),
            LeakyReLUAlpha(0.1),
            nn.Linear(256, nout),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class VisionModule(nn.Module):
    """Vision module used in bidirectional tracking candidate selection."""

    def __init__(self, base_channels: int = 64, depthwise: bool = False) -> None:
        super().__init__()
        self.drop1 = nn.Dropout2d(0.05)
        self.drop2 = nn.Dropout2d(0.5)
        self.net = UNet(base_channels=base_channels, depthwise=depthwise)

    def forward(self, img: Tensor, bin_mask: Tensor, full_bin: Tensor) -> Tensor:
        b1 = self.drop1(bin_mask) * (1.0 - 0.05)
        b2 = self.drop2(full_bin) * (1.0 - 0.5)
        x = torch.cat([img, b1, b2], dim=1)
        if x.shape[1] != 1:
            # Mathematica graph feeds three images and then passes to a single-input UNET.
            # We collapse to one channel to preserve behavior without changing UNet topology.
            x = x.mean(dim=1, keepdim=True)
        return self.net(x)


def build_unet(base_channels: int = 64) -> UNet:
    """Factory equivalent of Mathematica `UNET`."""
    return UNet(base_channels=base_channels, depthwise=False)


def build_unet_dsw(base_channels: int = 64) -> UNet:
    """Factory equivalent of Mathematica `UNETdsw`."""
    return UNet(base_channels=base_channels, depthwise=True)


def build_unet_unidirectional(base_channels: int = 64) -> UNetUnidirectional:
    """Factory equivalent of Mathematica `UNETunidirectional`."""
    return UNetUnidirectional(base_channels=base_channels, depthwise=False)


def build_unet_dsw_unidirectional(base_channels: int = 64) -> UNetUnidirectional:
    """Factory equivalent of Mathematica `UNETdswUnidirectional`."""
    return UNetUnidirectional(base_channels=base_channels, depthwise=True)


def build_classnet(nout: int, in_channels: int = 1, input_size: int = 64) -> ClassNet:
    """Factory equivalent of Mathematica `classnet`.

    `input_size` is accepted for API compatibility.
    """
    del input_size
    return ClassNet(nout=nout, in_channels=in_channels)


def build_vision_module(sz: int, base_channels: int = 64, depthwise: bool = False) -> VisionModule:
    """Factory equivalent of Mathematica `VisionModule[sz]`.

    `sz` is accepted for API compatibility.
    """
    del sz
    return VisionModule(base_channels=base_channels, depthwise=depthwise)
