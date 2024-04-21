import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNPReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNPReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNPReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GDConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1):
        super(GDConv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = F.avg_pool2d(out, out.size()[2:])
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [2, 24, 2, 2],
                [2, 32, 2, 2],
                [2, 64, 3, 2],
                [1, 96, 2, 1],
                [1, 160, 2, 2],
                [1, 320, 1, 1],
            ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNPReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNPReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # 添加 Dropout 层
            GDConv(self.last_channel, num_classes),  # 使用 GDConv 替换最后一层的分离卷积
            nn.BatchNorm2d(num_classes),
            nn.PReLU(num_classes),
        )

        # L1 正则化
        self.l1_regularization = nn.L1Loss(size_average=False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, target=None):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)

        # 计算 L1 正则化项
        if target is not None:
            l1_loss = self.l1_regularization(x, target)
            return x, l1_loss
        else:
            return x
