from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class FlowBase(Layer, metaclass=ABCMeta):
    """
    """
    @abstractmethod
    def __init__(self, **kwargs):
        super(FlowBase, self).__init__(kwargs)

    def initialize_parameter(self, x: tf.Tensor):
        pass

    def build(self, input_shape: tf.TensorShape):
        self.initialized = self.add_weight(
            name='initialized',
            dtype=tf.bool,
            trainable=False)
        self.initialized.assign(False)
        self.built = True

    def call(self, x: tf.Tensor,
             inverse=False,
             **kwargs):
        if not self.initialized:
            if not inverse:
                self.initialize_parameter(x)
                self.initialized.assign(True)
            else:
                raise Exception('Invalid initialize')
        if inverse:
            return self.inverse(x, **kwargs)
        else:
            return self.forward(x, **kwargs)

    @abstractmethod
    def forward(self, inputs, **kwargs):
        return inputs

    @abstractmethod
    def inverse(self, inputs, **kwargs):
        return inputs


class FlowComponent(FlowBase):
    @abstractmethod
    def __init__(self, **kwargs):
        "docstring"
        super(FlowComponent, self).__init__(**kwargs)

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
    def build(self, input_shape: tf.TensorShape):
        super(FlowModule, self).build(
            input_shape=input_shape)

    def __init__(self, components: List[FlowComponent]):
        super(FlowModule, self).__init__()
        self.components = components

    def forward(self, x, **kwargs):
        z = x
        log_det_jacobian = []
        for component in self.components:
            z, ldj = component(z, inverse=False, **kwargs)
            log_det_jacobian.append(ldj)
        log_det_jacobian = sum(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z, **kwargs):
        x = z
        inverse_log_det_jacobian = []
        for component in reversed(self.components):
            x, ildj = component(x, inverse=True, **kwargs)
            inverse_log_det_jacobian.append(ildj)
        inverse_log_det_jacobian = sum(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


class FactorOutBase(FlowBase):
    @abstractmethod
    def __init__(self, **kwargs):
        super(FactorOutBase, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        super(FactorOutBase, self).build(input_shape)

    def call(self, x: tf.Tensor,
             zaux: tf.Tensor = None,
             inverse=False,
             temparature=1.0,
             **kwargs):
        if not self.initialized:
            if not inverse:
                self.initialize_parameter(x)
                self.initialized.assign(True)
            else:
                raise Exception('Invalid initialize')
        if inverse:
            return self.inverse(x, zaux, temparature=temparature, **kwargs)
        else:
            return self.forward(x, zaux, **kwargs)

    @abstractmethod
    def forward(self, x: tf.Tensor, zaux: tf.Tensor, **kwargs):
        pass

    @abstractmethod
    def inverse(self, x: tf.Tensor, zaux: tf.Tensor,
                temparature: float, **kwargs):
        pass
