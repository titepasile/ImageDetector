"""Small, explicit encoder adapters. Every encoder returns one vector per image."""

import torch
from torch import nn

from src.models.resnet.res_net import ResNet


class ConvEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, 3, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = in_channels

    def forward(self, x):
        return self.pool(self.layers(x)).flatten(1)


class ResidualEncoder(nn.Module):
    def __init__(self, base_channels, blocks, preserve_detail=False):
        super().__init__()
        self.network = ResNet(blocks, 2, base_channels, preserve_detail)
        self.network.fc = nn.Identity()
        self.feature_dim = self.network.feature_dim

    def forward(self, x):
        return self.network.forward_features(x)


class ImageNetEncoder(nn.Module):
    def __init__(self, pretrained=True, preserve_detail=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        self.feature_dim = self.network.fc.in_features
        self.network.fc = nn.Identity()
        if preserve_detail:
            self.network.conv1.stride = (1, 1)
            self.network.maxpool = nn.Identity()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x):
        return self.network((x - self.mean) / self.std)


def build_encoder(config, download_weights=True):
    if config.name == "cnn":
        return ConvEncoder(config.channels)
    if config.name == "resnet":
        return ResidualEncoder(
            config.base_channels, config.blocks, config.preserve_detail
        )
    if config.name == "imagenet_resnet18":
        return ImageNetEncoder(
            config.pretrained and download_weights, config.preserve_detail
        )
    raise ValueError(f"Unknown encoder: {config.name}")
