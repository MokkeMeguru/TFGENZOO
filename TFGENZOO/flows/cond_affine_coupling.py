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

        h = self.scale_shift_net(tf.concat([x1, condition], axis=-1))
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
