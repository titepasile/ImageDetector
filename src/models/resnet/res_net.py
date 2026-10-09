from torch import nn
from res_block import ResBlock


class ResNet(nn.Module):
    def __init__(self, n_blocks_list, n_classes, base_channels):
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, padding=3, bias=False, stride=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        layers = []
        in_channels = base_channels
        for i, layer in enumerate(n_blocks_list):
            stride = 1 if i == 0 else 2
            out_channels = in_channels * 2
            layers.append(self._make_block(in_channels, out_channels, layer, stride))
            in_channels = out_channels
        self.res_blocks = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, n_classes)

    def _make_block(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for i in n_blocks - 1:
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.init_conv(x)
        z = self.res_blocks(z)
        z = self.avgpool(z)
        z = self.fc(z)
