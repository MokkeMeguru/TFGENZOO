from enum import Enum

import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, regularizers

from TFGENZOO.flows.actnorm import Actnorm
from TFGENZOO.flows.flowbase import FlowComponent
from TFGENZOO.flows.utils.actnorm_activation import ActnormActivation

Layer = layers.Layer
Conv2D = layers.Conv2D


class LogScale(Layer):
    def build(self, input_shape: tf.TensorShape):
        shape = [1, input_shape[-1]]
        self.logs = self.add_weight(
            name="log_scale", shape=tuple(shape), initializer="zeros", trainable=True
        )

    def __init__(self, log_scale_factor: float = 3.0, **kwargs):
        super(LogScale, self).__init__(**kwargs)
        self.log_scale_factor = log_scale_factor

    def call(self, x: tf.Tensor):
        return x * tf.exp(self.logs * self.log_scale_factor)

class AffineCouplingMask(Enum):
    ChannelWise = 1


class AffineCoupling(FlowComponent):
    """Affine Coupling Layer

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN(x1)
            | scale = sigmoid(log_scale + 2.0)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN(z1)
            | scale = sigmoid(log_scale + 2.0)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
    """

    def __init__(
        self,
        mask_type: AffineCouplingMask = AffineCouplingMask.ChannelWise,
        scale_shift_net: Layer = None,
        **kwargs
    ):
        super(AffineCoupling, self).__init__(**kwargs)
        if not scale_shift_net:
            raise ValueError
        self.scale_shift_net = scale_shift_net
        self.mask_type = mask_type

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        super(AffineCoupling, self).build(input_shape)

    def forward(self, x: tf.Tensor, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        h = self.scale_shift_net(x1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = tf.nn.sigmoid(log_scale + 2.0)
            # scale = tf.exp(tf.clip_by_value(log_scale, -15.0, 15.0))
            z2 = (x2 + shift) * scale

            # scale's shape is [B, H, W, C]
            # log_det_jacobian's hape is [B]
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
            return tf.concat([z1, z2], axis=-1), log_det_jacobian
        else:
            raise NotImplementedError()

    def inverse(self, z: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net(z1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
            scale = tf.nn.sigmoid(log_scale + 2.0)
            # scale = tf.exp(- tf.clip_by_value(log_scale, -15.0, 15.0))
            x2 = (z2 / scale) - shift

            # scale's shape is [B, H, W, C // 2]
            # inverse_log_det_jacobian's hape is [B]
            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
            return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
        else:
            raise NotImplementedError()
