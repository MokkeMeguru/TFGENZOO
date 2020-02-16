import tensorflow as tf

from TFGENZOO.flows.flowbase import FactorOutBase


class FactorOut(FactorOutBase):
    def build(self, input_shape: tf.TensorShape):
        self.split_size = input_shape[-1] // 2
        super(FactorOut, self).build(input_shape)

    def __init__(self, with_zaux=False):
        super(FactorOut, self).__init__()
        self.with_zaux = with_zaux

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        new_z = x[..., :self.split_size]
        x = x[..., self.split_size:]
        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux

    def inverse(self, z: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        new_z = zaux[..., -self.split_size:]
        zaux = zaux[..., :-self.split_size]
        z = tf.concat([new_z, z], axis=-1)
        if self.with_zaux:
            return z, zaux
        else:
            return z


def main():
    layer = FactorOut()
    x = tf.random.normal([16, 4, 4, 128])
    y, zaux = layer(x, zaux=None, inverse=False)
    _x = layer(y, zaux=zaux, inverse=True)
    print(x.shape)
    print(y.shape)
    print(zaux.shape)
    print(_x.shape)
    print(tf.reduce_sum(x - _x))
    layer = FactorOut(with_zaux=True)
    x = tf.random.normal([16, 8, 8, 8])
    zaux = tf.random.normal([16, 8, 8, 8])
    z, zaux = layer(x, zaux=zaux, inverse=False)
    _x, _zaux = layer(z, zaux=zaux, inverse=True)
    print(x.shape)
    print(z.shape)
    print(zaux.shape)
    print(_x.shape)
    print(tf.reduce_mean(x - _x))
    layer = FactorOut()
    x = tf.keras.Input([32, 32, 1])
    z, zaux = layer(x, zaux=None)
    model = tf.keras.Model(x, z)
    model.summary()
    return model
