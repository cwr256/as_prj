import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.nn import Conv2d, BatchNorm2d, PReLU, Flatten, Dense, SequentialCell, Cell

class ConvBNPReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0):
        super(ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode=pad_mode, padding=padding,
                              has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class InvertedResidual(nn.Cell):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expansion)
        self.conv1 = ConvBNPReLU(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = ConvBNPReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, pad_mode='pad', padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def construct(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.bn(x)
        if self.use_shortcut:
            x = x + identity
        x = self.prelu(x)
        return x


class MobileNetV2(nn.Cell):
    def __init__(self, num_classes=38, width_mult=1.0):
        super(MobileNetV2, self).__init__()

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],

            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        # building first layer
        layers = [ConvBNPReLU(3, input_channel, kernel_size=3, stride=2, pad_mode='pad', padding=1)]

        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    layers.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
                input_channel = output_channel

        # building last several layers
        layers.append(ConvBNPReLU(input_channel, 1280, kernel_size=1))

        self.features = nn.SequentialCell(layers)

        # Replace global average pooling with depthwise separable convolution
        self.conv_final = nn.Conv2d(1280, num_classes, kernel_size=1, has_bias=True)
        self.bn_final = nn.BatchNorm2d(num_classes)
        self.softmax = nn.Softmax(axis=1)
        self.prelu_final = PReLU()
    def construct(self, x):
        x = self.features(x)
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.prelu_final(x)
        x = P.ReduceMean(keep_dims=True)(x, (2, 3))
        x = P.Squeeze(axis=(2, 3))(x)  # 对输出进行squeeze操作，去掉后两个维度
        return x
# net =MobileNetV2(num_classes=38)
# print(net)