from enum import Enum

import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, regularizers

from TFGENZOO.flows.actnorm import Actnorm
from TFGENZOO.flows.flowbase import FlowComponent

Layer = layers.Layer
Conv2D = layers.Conv2D


class LogScale(Layer):
    def build(self, input_shape: tf.TensorShape):
        shape = [1, input_shape[-1]]
        self.logs = self.add_weight(
            shape=tuple(shape),
            initializer='zeros',
            # regularizer=tf.keras.regularizers.l2(0.01),
            trainable=True)

    def __init__(self, log_scale_factor: float = 3.0, **kwargs):
        super(LogScale, self).__init__(**kwargs)
        self.log_scale_factor = log_scale_factor

    def call(self, x: tf.Tensor):
        return x * tf.exp(self.logs * self.log_scale_factor)


class SequentialWithKwargs(Sequential):
    def __init__(self, layers=None, name=None):
        super(SequentialWithKwargs, self).__init__(
            layers=layers, name=name)

    def call(self,
             inputs, training=None, mask=None):
        if self._is_graph_network:
            if not self.built:
                self._init_graph_network(
                    self.inputs, self.outputs, name=self.name)
            return super(SequentialWithKwargs, self).call(
                inputs, training=training, mask=mask)

        outputs = inputs  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # During each iteration,
            # `inputs` are the inputs to `layer`, and `outputs`
            # are the outputs of `layer`
            # applied to `inputs`. At the end of each
            # iteration `inputs` is set to `outputs`
            # to prepare for the next layer.
            kwargs = {}
            argspec = self._layer_call_argspecs[layer].args
            if 'mask' in argspec:
                kwargs['mask'] = mask
            if 'training' in argspec:
                kwargs['training'] = training

            outputs = layer(inputs, **kwargs)

            # `outputs` will be the inputs to the next layer.
            inputs = outputs
            mask = outputs._keras_mask

        return outputs


class GlowNN(Layer):
    """
    attributes:
    - depth: int
    convolution depth
    """

    def build(self, input_shape: tf.TensorShape):
        res_block = SequentialWithKwargs()
        filters = int(input_shape[-1])
        for i in range(self.depth):
            res_block.add(
                layers.Conv2D(filters=filters, kernel_size=3,
                              strides=(1, 1),
                              padding='same',
                              use_bias=False,
                              activation=None))
            res_block.add(
                Actnorm(calc_ldj=False))
            res_block.add(
                layers.ReLU())
        res_block.add(
            layers.Conv2D(filters=filters * 2, kernel_size=3,
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer='zeros',
                          bias_initializer='zeros',
                          use_bias=True,
                          activation=None)
        )
        res_block.add(LogScale())
        self.res_block = res_block
        super(GlowNN, self).build(input_shape)

    def __init__(self, depth=2, **kwargs):
        super(GlowNN, self).__init__(kwargs)
        self.depth = depth

    def call(self, x: tf.Tensor, **kwargs):
        return self.res_block(x, **kwargs)


class AffineCouplingMask(Enum):
    ChannelWise = 1


class AffineCoupling(FlowComponent):
    """Affine Coupling Layer

    refs: pixyz
    https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    notes:
    - forward formula
    [x1, x2] = split(x)
    log_scale, shift = NN(x1)
    scale = sigmoid(log_scale + 2.0)
    z1 = x1
    z2 = (x2 + shift) * scale
    => z = concat([z1, z2])
    => log_det_jacobian = sum(log(scale))

    - inverse formula
    [z1, z2] = split(x)
    log_scale, shift = NN(z1)
    scale = sigmoid(log_scale + 2.0)
    x1 = z1
    x2 = z2 / scale - shift
    => z = concat([x1, x2])
    => inverse_log_det_jacobian = - sum(log(scale))

    notes:
    in Glow's Paper, scale is calculated by exp(log_scale),
    but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
    """

    def __init__(self,
                 mask_type: AffineCouplingMask =
                 AffineCouplingMask.ChannelWise,
                 scale_shift_net: Layer = None,
                 **kwargs):
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
        # scale = tf.nn.sigmoid(log_scale + 2.0)
        scale = tf.exp(tf.clip_by_value(log_scale, -15.0, 15.0))
        z2 = (x2 + shift) * scale
        log_det_jacobian = tf.reduce_sum(
            tf.math.log(scale), axis=self.reduce_axis)
        return tf.concat([z1, z2], axis=-1), log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net(z1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]
        # scale = tf.nn.sigmoid(log_scale + 2.0)
        scale = tf.exp(- tf.clip_by_value(log_scale, -15.0, 15.0))
        x2 = (z2 * scale) - shift
        inverse_log_det_jacobian = - 1 * tf.reduce_sum(
            tf.math.log(scale), axis=self.reduce_axis)
        return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
