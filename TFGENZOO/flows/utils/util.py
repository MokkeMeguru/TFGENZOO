from typing import Callable, Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from TFGENZOO.flows.utils.conv import Conv2D
from TFGENZOO.flows.utils.conv_zeros import Conv2DZeros

Layer = layers.Layer


def bits_x(log_likelihood: tf.Tensor,
           log_det_jacobian: tf.Tensor, pixels: int, n_bits: int = 8):
    """bits/dims
    Args:
        log_likelihood: shape is [batch_size,]
        log_det_jacobian: shape is [batch_size,]
        pixels: e.g. HWC image => H * W * C
        n_bits: e.g [0 255] image => 8 = log(256)

    Returns:
        bits_x: shape is [batch_size,]

    formula:
        (log_likelihood + log_det_jacobian)
          / (log 2 * h * w * c) + log(2^n_bits) / log(2.)
    """
    nobj = - 1.0 * (log_likelihood + log_det_jacobian)
    _bits_x = nobj / (np.log(2.) * pixels) + n_bits
    return _bits_x


def split_feature(x: tf.Tensor, type: str = "split"):
    """type = [split, cross]
    """
    channel = x.shape[-1]
    if type == "split":
        return x[..., :channel//2], x[..., channel // 2:]
    elif type == "cross":
        return x[..., 0::2], x[..., 1::2]


class ShallowResNet(Model):
    """ResNet of OpenAI's Glow
    """

    def __init__(self, width: int = 512, out_scale: int = 2):
        super(ShallowResNet, self).__init__()
        self.width = width
        self.conv1 = Conv2D(width=self.width)
        self.conv2 = Conv2D(width=self.width, kernel_size=[1, 1])
        self.conv_out = Conv2DZeros(width_scale=out_scale)

    def call(self, x: tf.Tensor):
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = self.conv_out(x)
        return x


class ResidualBlock(Model):
    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) != 4:
            raise NotImplementedError(
                'this resblock can be applyed NHWC tensor')

        num_channels = input_shape[-1]

        if not self.out_channels:
            self.out_channels = num_channels

        self.conv1 = layers.Conv2D(
            filters=self.out_channels, kernel_size=1, strides=(1, 1),
            padding='same',
            # kernel_regularizer=regularizers.l2(self.l2_reg_scale),
            # activity_regularizer=regularizers.l2(self.l2_reg_scale)
        )
        self.conv2 = layers.Conv2D(
            filters=self.out_channels, kernel_size=3, strides=(1, 1),
            padding='same',
            # kernel_regularizer=regularizers.l2(self.l2_reg_scale),
            # activity_regularizer=regularizers.l2(self.l2_reg_scale)
        )
        self.conv3 = layers.Conv2D(
            filters=self.out_channels, kernel_size=3, strides=(1, 1),
            padding='same',
            # kernel_regularizer=regularizers.l2(self.l2_reg_scale),
            # activity_regularizer=regularizers.l2(self.l2_reg_scale)
        )

        if self.skip_connection:
            if not self.out_channels == num_channels:
                self.shortcut = layers.Conv2D(
                    filters=self.out_channels, kernel_size=1, strides=(1, 1),
                    padding='same',
                    # kernel_regularizer=regularizers.l2(self.l2_reg_scale),
                    # activity_regularizer=regularizers.l2(self.l2_reg_scale)
                )
            else:
                self.shortcut = None
        super(ResidualBlock, self).build(input_shape)

    def __init__(self,
                 activation_fn: Callable = tf.nn.relu,
                 l2_reg_scale: float = 0.01,
                 skip_connection: bool = True,
                 out_channels: Union[int, None] = None):
        super(ResidualBlock, self).__init__()
        self.skip_connection = skip_connection
        self.activation_fn = activation_fn
        self.l2_reg_scale = l2_reg_scale
        self.out_channels = out_channels

    def get_config(self):
        return {
            'activation_fn': self.activation_fn,
            'l2_reg_scale': self.l2_reg_scale,
            'skip_connection': self.skip_connection,
            'out_channels': self.out_channels}

    def call(self, x, training=None, mask=None, initialize=False):
        x_init = x
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        if self.skip_connection:
            if self.shortcut:
                x_init = self.shortcut(x_init)
            h = h + x_init
        return self.activation_fn(h)


class ResidualNet(Model):
    def build(self, input_shape: tf.TensorShape):
        num_channels = input_shape[-1]
        num_units = num_channels * self.units_factor
        self.resblk_kwargs['out_channels'] = num_units
        layers = []
        for _ in range(self.num_block):
            layers.append(ResidualBlock(**self.resblk_kwargs))
        self.res_layers = layers
        self.output_fn = tf.keras.layers.Conv2D(
            filters=num_channels * 2,
            kernel_size=3, strides=(1, 1),
            padding='same',
            activation=tf.nn.sigmoid)  # None
        super(ResidualNet, self).build(input_shape)

    def __init__(self,
                 num_block: int = 1,
                 units_factor: int = 2,
                 resblk_kwargs: Dict = None):
        super(ResidualNet, self).__init__()
        self.num_block = num_block
        self.units_factor = units_factor
        self.resblk_kwargs = resblk_kwargs if resblk_kwargs else {}

    def get_config(self):
        return {
            'num_block': self.num_block,
            'units_factor': self.units_factor}

    def call(self, x, training=None, mask=None, initialize=False):
        for layer in self.res_layers:
            x = layer(x, training=training, initialize=initialize)
        x = self.output_fn(x, training=training)
        return x


def main():
    x = tf.keras.Input([32, 32, 3])
    model = ResidualNet(num_block=3, units_factor=6)
    y = model(x, training=True, mask=None, initialize=False)
    model.summary()
    x = tf.random.normal([16, 32, 32, 3])
    y = model(x)
    print(tf.reduce_max(y))
    print(tf.reduce_min(y))
    return model


if __name__ == '__main__':
    main()
