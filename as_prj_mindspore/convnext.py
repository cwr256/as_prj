import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P


class ConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode="pad", padding=0)
        self.pad = P.Pad(paddings=((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvNext(nn.Cell):
    def __init__(self, num_classes=38):
        super(ConvNext, self).__init__()
        self.conv1 = ConvBlock(3, 64, 7, 2, 3)
        self.conv2 = ConvBlock(64, 64, 3, 1, 1)
        self.conv3 = ConvBlock(64, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 56*56, num_classes, weight_init=TruncatedNormal(sigma=0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# net = ConvNext(num_classes=38)
# print(net)