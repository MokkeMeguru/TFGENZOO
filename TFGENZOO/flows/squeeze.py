import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowBase


class Squeezing(FlowBase):
    """
    """

    def __init__(self, with_zaux=False):
        super(Squeezing, self).__init__()
        self.with_zaux = with_zaux

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        x = tf.nn.space_to_depth(x, 2)
        if self.with_zaux:
            zaux = tf.nn.space_to_depth(zaux, 2)
            return x, zaux
        return x

    def inverse(self, z: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        z = tf.nn.depth_to_space(z, 2)
        if self.with_zaux and zaux is not None:
            zaux = tf.nn.depth_to_space(zaux, 2)
            return z, zaux
        return z


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
