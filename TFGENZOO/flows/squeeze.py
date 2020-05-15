import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowBase


class Squeezing(FlowBase):
    """Squeezing Layer

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L338-L352
        https://arxiv.org/pdf/1605.08803.pdf Figure 3

    Note:

        * forward formula
            | z = reshape(x, [B, H // 2, W // 2, C * 4])

        * inverse formula
            | x = reshape(z, [B, H, W, C])

        * checkerboard spacing

                e.g.

                | [[[[1], [2], [5], [6]],
                | [[3], [4], [7], [8]],
                | [[9], [10], [13], [14]],
                | [[11], [12], [15], [16]]]]

                to

                | [[[ 1,  5],
                | [ 9, 13]]]

                | [[[ 2,  6],
                | [10, 14]]]

                | [[[ 3,  7],
                | [11, 15]]]

                | [[[ 4,  8],
                | [12, 16]]]

    """

    def __init__(self, with_zaux=False):
        super(Squeezing, self).__init__()
        self.with_zaux = with_zaux

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        x = tf.nn.space_to_depth(x, 2)
        if self.with_zaux and zaux is not None:
            zaux = tf.nn.space_to_depth(zaux, 2)
            return x, zaux
        return x

    def inverse(self, z: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        z = tf.nn.depth_to_space(z, 2)
        if self.with_zaux and zaux is not None:
            zaux = tf.nn.depth_to_space(zaux, 2)
            return z, zaux
        return z


# TODO: move to squeeze_test.py
def main():
    layer = Squeezing()
    x = tf.keras.Input([32, 32, 1])
    y = layer(x)
    print(y)
    tf.keras.Model(x, y).summary()

    layer = Squeezing(with_zaux=True)
    x = tf.keras.Input([32, 32, 1])
    zaux = tf.keras.Input([32, 32, 1])
    y, _zaux = layer(x, zaux=zaux)
    tf.keras.Model([x, zaux], [y, _zaux]).summary()

    layer = Squeezing()
    x = tf.random.normal([16, 32, 32, 1])
    y = layer(x)
    print(y.shape)
    _x = layer(y, inverse=True)
    print(tf.reduce_sum(x - _x))
