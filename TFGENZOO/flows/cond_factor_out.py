import tensorflow as tf

from TFGENZOO.flows.flowbase import FactorOutBase
from TFGENZOO.flows.utils import gaussianize
from TFGENZOO.flows.utils.conv_zeros import Conv2DZeros
from TFGENZOO.flows.utils.util import split_feature


class ConditionalFactorOut(FactorOutBase):
    def build(self, input_shape: tf.TensorShape):
        self.split_size = input_shape[-1] // 2
        self.reduce_axis = list(range(1, len(input_shape)))
        super(ConditionalFactorOut, self).build(input_shape)

    def __init__(self, with_zaux=False):
        super(ConditionalFactorOut, self).__init__()
        self.with_zaux = with_zaux
        self.conv = Conv2DZeros(width_scale=2)

    def split2d_prior(self, z: tf.Tensor):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        new_z = x[..., :self.split_size]
        x = x[..., self.split_size:]
        mean, logsd = self.split2d_prior(x)
        ll = gaussianize.gaussian_likelihood(mean, logsd, new_z)
        ll = tf.reduce_sum(ll, axis=self.reduce_axis)
        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux, ll

    def inverse(self, z: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        if zaux is not None:
            new_z = zaux[..., -self.split_size:]
            zaux = zaux[..., :-self.split_size]
            z = tf.concat([new_z, z], axis=-1)
            if self.with_zaux:
                return z, zaux
            else:
                return z
        else:
            # Sampling
            tf.print("Sampling with one side")
            mean, logsd = self.split2d_prior(z)
            new_z = gaussianize.gaussian_sample(mean, logsd)
            z = tf.concat([new_z, z], axis=-1)
            return z


def main():
    layer = ConditionalFactorOut()
    x = tf.random.normal([16, 4, 4, 128])
    y, zaux, ll = layer(x, zaux=None, inverse=False)
    # _x = layer(y, zaux=zaux, inverse=True)
    _x = layer(y, zaux=None, inverse=True)
    print(x.shape)
    print(y.shape)
    print(zaux.shape)
    print(_x.shape)
    print(tf.reduce_mean(x - _x))
    layer = ConditionalFactorOut()
    x = tf.keras.Input([4, 4, 128])
    z, zaux, ll = layer(x, zaux=None)
    model = tf.keras.Model(x, [z, zaux, ll])
    model.trainable = True
    model.summary()
    return model
