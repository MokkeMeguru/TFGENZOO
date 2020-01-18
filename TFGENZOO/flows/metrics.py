import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List
from TFGENZOO.flows.flows import Flow
tfd = tfp.distributions


class Process(Flow):
    """
    """

    def __init__(self, n_bins: float = 256.0, with_debug: bool = True):
        """
        """
        self.n_bins = n_bins
        super(Process, self).__init__(with_debug=with_debug)

    def build(self, input_shape):
        log_det_jacobian = - np.log(self.n_bins) * np.prod(input_shape[1:])
        self.log_det_jacobian = tf.cast(log_det_jacobian, tf.float32)
        super(Process, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs):

        def preprocess(x):
            x = tf.cast(x, tf.float32)
            x = x / self.n_bins - 0.5
            x += tf.random.uniform(tf.shape(x), 0, 1.0 / self.n_bins)
            return x

        z = preprocess(x)
        log_det_jacobian = tf.broadcast_to(self.log_det_jacobian, tf.shape(x)[0:1]) 
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        """
        TODO: implement inverse_log_det_jacobian
        """
        def postprocess(z: tf.Tensor):
            """postprocess for generate image as rgb [0, 256]
            args:
            - z: tf.Tensor [N, H, W, C]
            - n_bins: int
            rgb[256] => 256
            note:
            ref. https://github.com/openai/glow/blob/master/model.py
            """
            return tf.clip_by_value(
                # tf.floor((z + 0.5) * self.n_bins) 
                (tf.clip_by_value(z, -0.5, 0.5) + 0.5) * self.n_bins * (256. / self.n_bins), 0, 255)

        x = postprocess(z)
        return x, tf.broadcast_to(0.0, tf.shape(z)[0:1])


