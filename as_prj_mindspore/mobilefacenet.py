"""
Mobilefacenet.
"""
from mindspore import nn
from mindspore.nn import Dense, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, SequentialCell, Cell
from mindspore.common.initializer import initializer, HeNormal
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from PIL import Image
import numpy as np

__all__ = ["get_mbf", "get_mbf_large"]


class Flatten(Cell):
    """
    Flatten.
    """

    def construct(self, x):
        """
        construct.
        """
        return x.view(x.shape[0], -1)


class ConvBlock(Cell):
    """
    ConvBlock.
    """

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super().__init__()
        self.layers = nn.SequentialCell(
            Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad',
                   padding=padding, has_bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(channel=out_c)
        )

    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)


class LinearBlock(Cell):
    """
    LinearBlock.
    """

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super().__init__()
        self.layers = nn.SequentialCell(
            Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad',
                   padding=padding, has_bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)


class DepthWise(Cell):
    """
    DepthWise.
    """

    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2),
                 padding=(1, 1, 1, 1), group=1):
        super().__init__()
        self.residual = residual
        self.layers = nn.SequentialCell(
            ConvBlock(in_c, out_c=group, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1)),
            ConvBlock(group, group, group=group, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(group, out_c, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1))
        )

    def construct(self, x):
        """
        construct.
        """
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Cell):
    """
    Residual.
    """

    def __init__(self, c, num_block, group, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)):
        super().__init__()
        cells = []
        for _ in range(num_block):
            cells.append(DepthWise(c, c, True, kernel, stride, padding, group))
        self.layers = SequentialCell(*cells)

    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)

class GDC(Cell):
    """
    GDC.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.SequentialCell(
            LinearBlock(512, 512, kernel=(7, 7), stride=(1, 1), padding=(0, 0, 0, 0), group=512),
            Flatten(),
            Dense(247808, out_channels, has_bias=False),
            BatchNorm1d(out_channels)
        )

    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)


class MobileFaceNet(Cell):
    """
    Build the mobileface model.

    Args:
        num_classes (int): The number of output classes.
        num_features (int): The num of features. Default: 512.
        blocks (Tuple): The architecture of backbone. Default: (1, 4, 6, 2).
        scale (int): The scale of network blocks. Default: 2.

    Examples:
        >>> net = MobileFaceNet(num_classes=38, num_features=512, blocks=(1, 4, 6, 2), scale=2)
    """
    def __init__(self, num_classes, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super().__init__()
        self.scale = scale
        self.layers = nn.CellList()
        # Add input convolution layer to handle 224x224 color images
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1),
                        padding=(1, 1, 1, 1), group=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], group=128, kernel=(3, 3),
                        stride=(1, 1), padding=(1, 1, 1, 1)),
            )

        self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1, 1, 1), group=128),
            Residual(64 * self.scale, num_block=blocks[1], group=128, kernel=(3, 3), stride=(1, 1),
                        padding=(1, 1, 1, 1)),
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1, 1, 1), group=256),
            Residual(128 * self.scale, num_block=blocks[2], group=256, kernel=(3, 3),
                        stride=(1, 1), padding=(1, 1, 1, 1)),
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2),
                        padding=(1, 1, 1, 1), group=512),
            Residual(128 * self.scale, num_block=blocks[3], group=256, kernel=(3, 3),
                        stride=(1, 1), padding=(1, 1, 1, 1)),
        ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1),
                        padding=(0, 0, 0, 0))
        self.features = GDC(247808, num_classes)  # Modify GDC initialization

    def _initialize_weights(self):
        """
        initialize_weights
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                        cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape,
                        cell.bias.data.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.data.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.data.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                        cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape,
                        cell.bias.data.dtype))

    def construct(self, x):
        """
        construct.
        """
        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x


    # Remain unchanged

def get_mbf(num_features=512, blocks=(1, 4, 6, 2), scale=2):
    """
    Get the mobilefacenet-0.45G.

    Examples:
        >>> net = get_mbf(512)
    """
    return MobileFaceNet(num_features, blocks, scale=scale)


def get_mbf_large(num_features=512, blocks=(2, 8, 12, 4), scale=4):
    """
    Get the large mobilefacenet.

    Examples:
        >>> net = get_mbf_large(512)
    """
    return MobileFaceNet(num_features, blocks, scale=scale)


net = MobileFaceNet(num_classes=38)
param_dict = load_checkpoint("arcface_mobilefacenet-24_71.ckpt")
load_param_into_net(net, param_dict)

img_path = "dataarcface/B21041636/B21041636.jpg" # 此处可以更换为自己的图像路径
img = Image.open(img_path)
img = img.resize((112, 112), Image.BICUBIC)
img = np.array(img).transpose(2,0,1)
img = ms.Tensor(img, ms.float32)

img = ((img / 255) - 0.5) / 0.5
img = ms.Tensor(img, ms.float32)
if len(img.shape) == 4:
    pass
elif len(img.shape) == 3:
    img = img.expand_dims(axis=0)

net_out = net(img)
embeddings = net_out.asnumpy()
print(embeddings)
