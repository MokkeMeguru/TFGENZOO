from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from TFGENZOO.flows.utils.actnorm_activation import ActnormActivation

Layer = layers.Layer


class Conv2D(Layer):
    """Convolution layer for NHWC image
    Note:
        this layer applies
        - data-dependent normalization (actnorm, openai's Glow)
        - weight normalization for stable training
    """

    def __init__(self,
                 width: int = None,
                 width_scale: int = 1,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: str = "SAME",
                 do_actnorm: bool = True,
                 do_weightnorm: bool = False,
                 weight_std: float = 0.05):
        super(Conv2D, self).__init__()
        self.width = width
        self.width_scale = width_scale
        self.kernel_size = list(kernel_size)
        self.stride = [1] + list(stride) + [1]
        self.padding = padding
        self.do_actnorm = do_actnorm
        self.do_weightnorm = do_weightnorm
        self.weight_std = weight_std
        if self.do_actnorm:
            self.activation = ActnormActivation()

    def build(self, input_shape: tf.TensorShape):
        n_in = input_shape[-1]
        n_out = (self.width
                 if self.width is not None
                 else n_in * self.width_scale)
        filters = self.kernel_size + [n_in, n_out]
        self.kernel = self.add_weight(name="kernel",
                                      shape=filters,
                                      dtype=tf.float32)
        self.reduce_axis = list(range(len(input_shape) - 1))
        if not self.do_actnorm:
            self.bias = self.add_weight(
                name="bias",
                shape=[1 for _ in range(len(input_shape) - 1)] + [n_out])
        self.built = True

    def call(self, x: tf.Tensor):
        if self.do_weightnorm:
            kernel = tf.nn.l2_normalize(self.kernel, self.reduce_axis)
        else:
            kernel = self.kernel
        x = tf.nn.conv2d(
            x, filters=kernel, strides=self.stride, padding=self.padding)
        if self.do_actnorm:
            x = self.activation(x)
        else:
            x += self.bias
        return x


def main():
    x = tf.keras.Input([16, 16, 2])
    conv = Conv2D(width_scale=2)
    y = conv(x)
    model = tf.keras.Model(x, y)
    model.summary()
    x = tf.random.normal([32, 16, 16, 2])
    y = conv(x)
    print(y.shape)


if __name__ == '__main__':
    main()
