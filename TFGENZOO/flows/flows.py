import tensorflow as tf
from tensorflow.keras import layers
from abc import ABC, abstractmethod
from typing import List
Layer = layers.Layer
Model = tf.keras.Model


class FlowAbst(ABC):
    """the base of Flow Layer (abstruction)
    this layer is for FlowList, Blockwise, etc...
    """

    def __init__(self, with_debug: bool = True, **kwargs):
        """initialization the base of Flow Layer
        Args:
        - with_debug: bool, take some assertion.
        """
        self.with_debug = with_debug
        super(FlowAbst, self).__init__()

    @abstractmethod
    def build(self, input_shape):
        self.shape = input_shape
        super(FlowAbst, self).build(input_shape)

    def __call__(self, x: tf.Tensor, **kwargs):
        return self.call(x, **kwargs)

    @abstractmethod
    def call(self, x: tf.Tensor, **kwargs):
        log_det_jacobian = tf.broadcast_to(0.0, [x.shape[0]])
        self.assert_tensor(x, x)
        self.assert_log_det_jacobian(log_det_jacobian)
        return x, log_det_jacobian

    @abstractmethod
    def inverse(self, z: tf.Tensor, **kwargs):
        inverse_log_det_jacobian = tf.broadcast_to(0.0, [z.shape[0]])
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        self.assert_tensor(z, z)
        return z, inverse_log_det_jacobian

    @abstractmethod
    def setStat(self, x: tf.Tensor, **kwargs):
        pass

    def assert_tensor(self, x: tf.Tensor, z: tf.Tensor):
        if self.with_debug:
            tf.debugging.assert_shapes([(x, z.shape)])

    def assert_log_det_jacobian(self, log_det_jacobian: tf.Tensor):
        """assert log_det_jacobian's shape
        TODO:
        tf-2.0's bug
        tf.debugging.assert_shapes([(tf.constant(1.0), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([1.0, 1.0]), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([[1.0], [1.0]]), (None, ))]) # => Error
        """
        if self.with_debug:
            tf.debugging.assert_shapes(
                [(log_det_jacobian, (None, ))])


class Flow(ABC, Layer):
    """the base of Flow Layer
    formula:
    z  = f(x)
    where (data) x -> z (latent space)

    >> flow = Flow()
     >> z, log_det_jacobian = flow(x)
    where
    z is f(x)
    log_det_jacobian is log|det J_f|
    >> x, inverse_log_det_jacobian = flow.inverse(z)
    where
    x is f^{-1}(z)
    inverse_log_det_jacobian is log|det J_f^{-1}|
    """

    def __init__(self, with_debug: bool = True, **kwargs):
        """initialization the base of Flow Layer
        Args:
        - with_debug: bool, take some assertion.
        """
        self.with_debug = with_debug
        super(Flow, self).__init__()

    @abstractmethod
    def build(self, input_shape):
        self.shape = input_shape
        super(Flow, self).build(input_shape)

    @abstractmethod
    def call(self, x: tf.Tensor, **kwargs):
        log_det_jacobian = tf.broadcast_to(0.0, [x.shape[0]])
        self.assert_tensor(x, x)
        self.assert_log_det_jacobian(log_det_jacobian)
        return x, log_det_jacobian

    @abstractmethod
    def inverse(self, z: tf.Tensor, **kwargs):
        inverse_log_det_jacobian = tf.broadcast_to(0.0, [z.shape[0]])
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        self.assert_tensor(z, z)
        return z, inverse_log_det_jacobian

    def assert_tensor(self, x: tf.Tensor, z: tf.Tensor):
        if self.with_debug:
            tf.debugging.assert_shapes([(x, z.shape)])

    def assert_log_det_jacobian(self, log_det_jacobian: tf.Tensor):
        """assert log_det_jacobian's shape
        TODO:
        tf-2.0's bug
        tf.debugging.assert_shapes([(tf.constant(1.0), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([1.0, 1.0]), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([[1.0], [1.0]]), (None, ))]) # => Error
        """
        if self.with_debug:
            tf.debugging.assert_shapes(
                [(log_det_jacobian, (None, ))])


class FlowList(Flow):
    """Flow Layer's list
    TODO: Flow or FlowAbst
    formula:
    z  = f_n o ...  o f_2 o f_1(x)
    where (data) x -> z (latent space)

    >> flowlist = FlowList([flow_1, flow_2, flow_3, ...])
     >> z, log_det_jacobian = flowlist(x)
    where
    z is x->flow_1->flow_2->flow_3->....
    log_det_jacobian is log|det J_{f_1}| + log|det J_{f_2}| + log|det J_{f_3}| + ...
    >> x, inverse_log_det_jacobian = flow.inverse(z)
    where
    x is z->...->flow_3^{-1}->flow_2^{-1}->flow_1^{-1}
    inverse_log_det_jacobian is log|det J_{f_1}^{-1}| + log|det J_{f_2}^{-1}| + ...
    """

    def __init__(self, flow_list: List[Flow], with_debug: bool = True, **kwargs):
        super(FlowList, self).__init__(with_debug=with_debug)
        self.flow_list = flow_list

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, x: tf.Tensor, **kwargs):
        log_det_jacobians = [] #tf.broadcast_to(0.0,  tf.shape(x)[0:1])
        for flow in self.flow_list:
            x,  _log_det_jacobian = flow(x, **kwargs)
            log_det_jacobians.append(_log_det_jacobian)
        log_det_jacobian = sum(log_det_jacobians)
        self.assert_log_det_jacobian(log_det_jacobian)
        return x, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        inverse_log_det_jacobians = [] # tf.broadcast_to(0.0, tf.shape(z)[0:1])
        for flow in reversed(self.flow_list):
            z, _inverse_log_det_jacobian = flow.inverse(z)
            inverse_log_det_jacobians.append(_inverse_log_det_jacobian)
        inverse_log_det_jacobian = sum(inverse_log_det_jacobians)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return z, inverse_log_det_jacobian

    def setStat(self, x: tf.Tensor, **kwargs):
        for flow in self.flow_list:
            if callable(getattr(flow, 'setStat', None)):
                tf.print('[List] Flow List setStat')
                flow.setStat(x)
            else:
                tf.print('[List] Flow List not called setStat ', flow.name)
            x, _log_det_jacobian = flow(x, **kwargs)

    def assert_tensor(self, x: tf.Tensor, z: tf.Tensor):
        if self.with_debug:
            tf.debugging.assert_shapes([(x, z.shape)])

    def assert_log_det_jacobian(self, log_det_jacobian: tf.Tensor):
        """assert log_det_jacobian's shape
        TODO:
        tf-2.0's bug
        tf.debugging.assert_shapes([(tf.constant(1.0), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([1.0, 1.0]), (None, ))]) # => None (true)
        tf.debugging.assert_shapes([(tf.constant([[1.0], [1.0]]), (None, ))]) # => Error
        """
        if self.with_debug:
            tf.debugging.assert_shapes(
                [(log_det_jacobian, (None, ))])
