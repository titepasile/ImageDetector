"""Local encoder

Only the start of the network differs from res_net.py:

    global ResNet:  7x7 conv, stride 2  ->  max pool, stride 2
                    (128x128 crop is 32x32 before the first block)
    local ResNet:   3x3 conv, stride 1, no max pool
                    (first block still sees all 128x128 pixels)

The pixel-level traces of generators (upsampling patterns, missing sensor
noise) are only a few pixels wide. Downsampling 4x before any residual block
would average them away, which is fine for the global view but defeats the
purpose of the local one.

The residual blocks (res_block.py) are used unchanged.

    model = LocalResNet()
    logits = model(x)            # (B, 2)   used in stage A training
    z2 = model.features(x)       # (B, model.feature_dim)   used in stage B
"""

import torch
import torch.nn.functional as F
from torch import nn

from .res_block import ResBlock


class LocalResNet(nn.Module):
    def __init__(self, n_blocks_list=(2, 2, 2, 2), n_classes=2, base_channels=32,
                 first_stage_stride=1):
        super().__init__()
        self.config = {"n_blocks_list": list(n_blocks_list), "n_classes": n_classes,
                       "base_channels": base_channels,
                       "first_stage_stride": first_stage_stride}
        # No BN/ReLU here: every pre-activation block starts with BN -> ReLU.
        self.init_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1,
                                   stride=1, bias=False)

        layers = []
        in_channels = base_channels
        for i, n_blocks in enumerate(n_blocks_list):
            stride = first_stage_stride if i == 0 else 2
            out_channels = in_channels if i == 0 else in_channels * 2
            layers.append(self._make_stage(in_channels, out_channels, n_blocks, stride))
            in_channels = out_channels
        self.res_blocks = nn.Sequential(*layers)

        self.bn = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = in_channels
        self.fc = nn.Linear(in_channels, n_classes)  # temporary head (stage A)

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride):
        blocks = [ResBlock(in_channels, out_channels, stride)]
        blocks += [ResBlock(out_channels, out_channels, stride=1) for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def features(self, x):
        """Crop batch (B, 3, H, W) -> feature vectors z2 (B, feature_dim)."""
        z = self.init_conv(x)
        z = self.res_blocks(z)
        z = F.relu(self.bn(z))
        z = self.avgpool(z)
        return torch.flatten(z, 1)

    def forward(self, x):
        return self.fc(self.features(x))


def load_encoder(checkpoint_path, map_location="cpu"):
    """Rebuild a trained LocalResNet from a checkpoint written by train_local.py."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model = LocalResNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
