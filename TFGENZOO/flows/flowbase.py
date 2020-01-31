from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class FlowBase(Layer, metaclass=ABCMeta):
    """
    """
    @abstractmethod
    def __init__(self, args):
        ""
        super(FlowBase, self).__init__()

    @abstractmethod
    def build(self, **kwargs):
        self.built = True

    def call(self, inputs, inverse=False, **kwargs):
        if inverse:
            return self.inverse(inputs, **kwargs)
        else:
            self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, inputs, **kwargs):
        return inputs

    @abstractmethod
    def inverse(self, inputs, **kwargs):
        return inputs


class FlowComponent(FlowBase):
    @abstractmethod
    def __init__(self, args):
        "docstring"
        super(FlowComponent, self).__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        log_det_jacobian = tf.zeros(x.shape[0:1])
        z = x
        return z, log_det_jacobian

    @abstractmethod
    def inverse(self, z, **kwargs):
        inverse_log_det_jacobian = tf.zeros(z.shape[0:1])
        x = z
        return x, inverse_log_det_jacobian

    def assert_tensor(self, x: tf.Tensor, z: tf.Tensor):
        if self.with_debug:
            tf.debugging.assert_shapes([(x, z.shape)])

    def assert_log_det_jacobian(self, log_det_jacobian: tf.Tensor):
        """assert log_det_jacobian's shape
        TODO:
        tf-2.0's bug
        tf.debugging.assert_shapes([(tf.constant(1.0), (None, ))])
        # => None (true)
        tf.debugging.assert_shapes([(tf.constant([1.0, 1.0]), (None, ))])
        # => None (true)
        tf.debugging.assert_shapes([(tf.constant([[1.0], [1.0]]), (None, ))])
        # => Error
        """
        if self.with_debug:
            tf.debugging.assert_shapes(
                [(log_det_jacobian, (None, ))])


class FlowModule(FlowBase):
    def build(self, **kwargs):
        super(FlowModule, self).build()

    def __init__(self, components: List[FlowComponent]):
        self.components = components

    def forward(self, x, **kwargs):
        z = x
        log_det_jacobian = []
        for component in self.components:
            z, ldj = component(x, inverse=False, **kwargs)
            log_det_jacobian.append(ldj)
        log_det_jacobian = sum(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z, **kwargs):
        x = z
        inverse_log_det_jacobian = []
        for component in reversed(self.components):
            z, ildj = component(x, inverse=True, **kwargs)
            inverse_log_det_jacobian.append(ildj)
        inverse_log_det_jacobian = sum(inverse_log_det_jacobian)
        return z, inverse_log_det_jacobian


class FactorOutBase(FlowBase):
    @abstractmethod
    def __init__(self, args):
        super(FactorOutBase, self).__init__()

    @abstractmethod
    def build(self, kwargs):
        super(FactorOutBase, self).build()

    @abstractmethod
    def forward(self, x: tf.Tensor, zs: List[tf.Tensor], **kwargs):
        z1, z2 = tf.split(x, 2, axis=-1)
        zs = [z2] + zs
        log_det_jacobian = tf.zeros(tf.shape(z1)[0:1])
        return z1, log_det_jacobian, zs

    @abstractmethod
    def inverse(self, x: tf.Tensor, zs: List[tf.Tensor], **kwargs):
        z1 = x
        z2 = zs[0]
        zs = zs[1:]
        z = tf.concat([z1, z2], axis=-1)
        inverse_log_det_jacobian = tf.zeros(tf.shape(z)[0:1])
        return z, inverse_log_det_jacobian, zs


class ConditionalFactorOutBase(FlowBase):
    @abstractmethod
    def __init__(self, args):
        super(ConditionalFactorOutBase, self).__init__()

    @abstractmethod
    def build(self, kwargs):
        super(ConditionalFactorOutBase, self).build()

    @abstractmethod
    def forward(self, x: tf.Tensor, c: tf.Tensor, **kwargs):
        z = x
        log_det_jacobian = tf.zeros(tf.shape(z)[0:1])
        return x, log_det_jacobian

    @abstractmethod
    def inverse(self, z: tf.Tensor, c: tf.Tensor, **kwargs):
        x = z
        inverse_log_det_jacobian = tf.zeros(tf.shape(z)[0:1])
        return x, inverse_log_det_jacobian
