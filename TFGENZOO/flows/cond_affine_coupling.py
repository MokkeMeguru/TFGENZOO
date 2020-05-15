from typing import Union, Tuple
from enum import Enum
import tensorflow as tf

from tensorflow.keras import Model, Sequential, layers, regularizers

from TFGENZOO.flows.actnorm import Actnorm
from TFGENZOO.flows.utils.actnorm_activation import ActnormActivation
from TFGENZOO.flows.affine_coupling import LogScale
from TFGENZOO.flows.affine_coupling import AffineCouplingMask

Layer = layers.Layer
Conv2D = layers.Conv2D


class ConditionalAffineCoupling(Layer):
    """Conditional Affine Coupling Layer

    Sources:

        https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py#L191

    Note:
       * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN([x1, c])
            | scale = sigmoid(log_scale + 2.0)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat(z1, z2)
            | LogDetJacobian = sum(log(scale))
            | , where
            |  x is input [B, H, W, C]
            |  c is conditional input [B, H, W, C']

       * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN([z1, c])
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
        super().__init__(**kwargs)
        if not scale_shift_net:
            raise ValueError
        self.scale_shift_net = scale_shift_net
        self.mask_type = mask_type

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        super().build(input_shape)

    def forward(self, x: tf.Tensor, condition: tf.Tensor, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        # TODO: DUAL resnet
        # https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py#L191
        h = self.scale_shift_net(x1, condition=condition)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
            scale = tf.nn.sigmoid(log_scale + 2.0)
            z2 = (x2 + shift) * scale
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
        else:
            raise NotImplementedError()

        return tf.concat([z1, z2], axis=-1), log_det_jacobian

    def inverse(self, z: tf.Tensor, condition: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net(z1, condition=condition)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
            scale = tf.nn.sigmoid(log_scale + 2.0)
            x2 = (z2 / scale) - shift
            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
        else:
            raise NotImplementedError()

        return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian


class ConditionalDualAffineCoupling(Layer):
    """Conditional Affine Coupling Layer (Dual)

    Sources:

        https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py#L191

    Note:
       * forward formula
            | [x1, x2] = split(x)
            |
            | log_scale_a, shift_a = NN([x1, c])
            | scale_a = sigmoid(log_scale_a + 2.0)
            | z2 = (x2 + shift_a) * scale_a
            |
            | log_scale_b, shift_b = NN([z2, c])
            | z1 = (x1 + shift_b) * scale_b
            |
            | z = concat([z1, z2])
            |
            | LogDetJacobian = sum(log(scale_a)) + sum(log(scale_b))
            | , where
            |  x is input [B, H, W, C]
            |  c is conditional input [B, H, W, C']

       * inverse formula
            | [z1, z2] = split(x)
            |
            | log_scale_b, shift_b = NN([z2, c])
            | scale_b = sigmoid(log_scale_b + 2.0)
            | x1 = x1 /  scale_b - shift_b
            |
            | log_scale_a, shift_a = NN([z1, c])
            | scale_a = sigmoid(log_scale_a + 2.0)
            | x2 = x2 / scale_a - shift_a
            |
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale_a)) - sum(log(scale_b))

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
        raise NotImplementedError
        super().__init__(**kwargs)
        if not scale_shift_net:
            raise ValueError
        self.scale_shift_net = scale_shift_net
        self.mask_type = mask_type

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        super().build(input_shape)

    def forward(self, x: tf.Tensor, condition: tf.Tensor, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        # TODO: DUAL resnet
        # https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py#L191

        h = self.scale_shift_net(x1, condition=condition)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
            scale = tf.nn.sigmoid(log_scale + 2.0)
            z2 = (x2 + shift) * scale
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
        else:
            raise NotImplementedError()

        return tf.concat([z1, z2], axis=-1), log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net(z1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
            scale = tf.nn.sigmoid(log_scale + 2.0)
            x2 = (z2 / scale) - shift
            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
        else:
            raise NotImplementedError()

        return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
