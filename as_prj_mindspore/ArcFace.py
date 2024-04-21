"""
Arcface loss.
"""
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype

from ce_loss import SoftMaxCE

class ArcFace(nn.Cell):
    """
    Implement of large margin arc distance.

    Args:
        world_size (Int): Size of each input sample.
        s (Float): Norm of input feature. Default: 64.0.
        m (Float): Margin. Default: 0.5.

    Examples:
        # >>> margin_softmax = ArcFace(world_size=world_size)
    """

    def __init__(self, world_size, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.shape = ops.Shape()
        self.mul = ops.Mul()
        self.cos = ops.Cos()
        self.acos = ops.ACos()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        # self.tile = ops.Tile().shard(((8, 1),))
        self.on_value = Tensor(m, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.ce_loss = SoftMaxCE(world_size=world_size)

    def construct(self, cosine, label):
        """
        Construct.
        """
        # 将标签转换为MindSpore张量
        label_tensor = Tensor(label, dtype=mstype.int32)
        # print(label_tensor)

        m_hot = self.onehot(label_tensor, self.shape(cosine)[1],
                            self.on_value, self.off_value)
        # print(m_hot)
        cosine = self.acos(cosine)
        cosine += m_hot[0]
        # print(cosine)
        cosine = self.cos(cosine)
        cosine = self.mul(cosine, self.s)
        # cosinet = Tensor(cosine,mstype.float32)
        # print(cosine)
        # print(type(cosine))
        # print(label_tensor.shape[0])
        loss = self.ce_loss(cosine, label_tensor)

        return loss

    # def construct(self, cosine, label):
    #     m_hot = self.onehot(label, self.shape(cosine)[1],
    #                         self.on_value, self.off_value)
    #
    #     cosine = self.acos(cosine)
    #     cosine += m_hot
    #     cosine = self.cos(cosine)
    #     cosine = self.mul(cosine, self.s)
    #
    #     loss = self.ce_loss(cosine, label)
    #
    #     return loss
loss = ArcFace(world_size=1)
# print(loss)
