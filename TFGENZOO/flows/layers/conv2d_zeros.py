from typing import List

import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class Conv2DZeros(Layer):
    """
    Sources:
        https://github.com/openai/glow/blob/master/tfops.py#L293-L313
    Diffs:
        - remove add_edge_padding
          too complex implementation
    """

    def __init__(self, width: int,
                 filter_size: List, stride: List,
                 pad: str = "SAME", logscale_factor: int = 3,
                 skip=1, edge_bias: bool = True):
        """
        """
        super().__init__()
        assert len(stride) == 2, "not supported stride size {}".format(stride)
        assert len(filter_size) == 2,\
            "not supported filter size {}".format(filter_size)
        self.width = width
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.logscale_factor = logscale_factor
        self.skip = skip
        self.edge_bias = edge_bias


    def build(self, input_shape: tf.TensorShape):
        n_in  = input_shape[3]
        self.stride_shape = [1] + self.stride + [1]
        self.filter_size = self.filter_size + [n_in, self.width]
        self.W = self.add_weight(
            name="W",
            shape=self.filter_size,
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32)
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.width),
            initializer=tf.keras.initializers.Zeros())
        self.logs = self.add_weight(
            name="logs",
            shape=(1, self.width),
            initializer=tf.keras.initializer.Zeros())
        super().build(input_shape)

    def call(self, x: tf.Tensor, **kwargs):
        x = tf.nn.conv2d(x, self.w, self.stride_shape, self.pad,
                         data_format="NHWC")
        x += self.bias
        x *= tf.exp(self.logs * self.logscale_factor)
        return x
