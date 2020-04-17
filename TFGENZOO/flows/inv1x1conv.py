from typing import Tuple

import numpy as np
import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowComponent


def regular_matrix_init(
        shape: Tuple[int, int], dtype=None):
    """initialize with orthogonal matrix
    Args:
        shape: generated matrix's shape [C, C]
        dtype:
    Returns:
        w_init: orthogonal matrix [C, C]
    Source:
        https://github.com/openai/glow/blob/master/model.py#L445-L451
    """
    assert len(shape) == 2, 'this initialization for 2D matrix'
    assert shape[0] == shape[1], (
        'this initialization for 2D matrix, C \times C'
    )
    c = shape[0]
    w_init = np.linalg.qr(np.random.randn(c, c))[0].astype("float32")
    return w_init


class Inv1x1Conv(FlowComponent):
    """Invertible 1x1 Convolution Layer
    Sources:
        https://arxiv.org/pdf/1807.03039.pdf
        https://github.com/openai/glow/blob/master/model.py#L457-L472

    Notes:
    - forward formula

        forall i, j: z_{i, j} = W x_{i, j}
        LogDetJacobian = H W log|det(W)|

        where
            x_{i, j}, y_{i, j} in [C, C]
            W in [C, C]
    - inverse formula

        forall i, j: x_{i, j} = W^{-1} z_{i, j}
        InverseLogDetJacobian = - H W log|det(W)|

         where
             x_{i, j}, y_{i, j} in [C, C]
             W in [C, C]
    """

    def build(self, input_shape: tf.TensorShape):
        _, h, w, c = input_shape
        self.h = h
        self.w = w
        self.c = c
        self.W = self.add_weight(
            name='W',
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init
        )
        super().build(input_shape)

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: tf.Tensor, **kwargs):
        _W = tf.reshape(self.W, [1, 1, self.c, self.c])
        z = tf.nn.conv2d(x, _W, [1, 1, 1, 1], "SAME")
        # scalar
        log_det_jacobian = tf.cast(
            tf.linalg.slogdet(tf.cast(self.W, tf.float64))[1]
            * self.h * self.w, tf.float32)
        # expand as batch
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        _W = tf.reshape(tf.linalg.inv(self.W), [1, 1, self.c, self.c])
        x = tf.nn.conv2d(z, _W, [1, 1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * tf.linalg.slogdet(tf.cast(self.W, tf.float64))[1]
            * self.h * self.w, tf.float32)
        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(z)[0:1])
        return x, inverse_log_det_jacobian
