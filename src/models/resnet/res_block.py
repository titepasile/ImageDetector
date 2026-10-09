from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        self.relu = nn.ReLU
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )

        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        z = self.conv1(self.relu(self.bn1(x)))
        z = self.conv2(self.relu(self.bn2(x)))

        z += self.skip(x)
        return z
