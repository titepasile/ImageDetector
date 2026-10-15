from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
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

        self.is_proj = stride != 1 or in_channels != out_channels

        if self.is_proj:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        pre = self.relu(self.bn1(x))
        z = self.conv1(pre)
        z = self.conv2(self.relu(self.bn2(z)))

        s = pre if self.is_proj else x
        z += self.skip(s)
        return z
