from typing import Tuple

import numpy as np
import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowComponent


def regular_matrix_init(
        shape: Tuple[int, int], dtype=tf.float32):
    """initialize with orthogonal matrix
    args:
    - shape: Tuple
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
    ref. Glow https://arxiv.org/pdf/1807.03039.pdf

    notes:
    - forward formula
    => forall i, j: z_{i, j} = W x_{i, j}
    => log_det_jacobian = H * W * log|det(W)|
    where
    x_{i, j}, y_{i, j} in [C, C]
    W in [C, C]
    - inverse formula
    => forall i, j: x_{i, j} = W^{-1} z_{i, j}
    => inverse_log_det_jacobian = - 1 * H * W * log|det(W)|
    where
    x_{i, j}, y_{i, j} in [C, C]
    W in [C, C]
    """

    def build(self, input_shape: tf.TensorShape):
        h, w, c = input_shape[1:]
        self.h = h
        self.w = w
        self.c = c
        self.W = self.add_weight(
            name='W',
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init
        )
        super(Inv1x1Conv, self).build(input_shape)

    def __init__(self, **kwargs):
        super(Inv1x1Conv, self).__init__()

    def forward(self, x: tf.Tensor, **kwargs):
        _W = tf.reshape(self.W, [1, 1, self.c, self.c])
        z = tf.nn.conv2d(x, _W, [1, 1, 1, 1], "SAME")
        log_det_jacobian = tf.linalg.slogdet(self.W)[1] * self.h * self.w
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        _W = tf.reshape(tf.linalg.inv(self.W), [1, 1, self.c, self.c])
        x = tf.nn.conv2d(z, _W, [1, 1, 1, 1], "SAME")
        inverse_log_det_jacobian = -1 * \
            tf.linalg.slogdet(self.W)[1] * self.h * self.w
        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(z)[0:1])
        return x, inverse_log_det_jacobian


def test_():

    origin_x = tf.random.uniform([16, 12, 12, 12])

    x = origin_x
    ws = []
    inv_ws = []
    for i in range(200):
        w = np.linalg.qr(np.random.randn(12, 12))[0].astype("float32")
        inv_w = tf.linalg.inv(w)
        ws.append(w)
        inv_ws.append(inv_w)
    z = x
    for w in ws:
        z = tf.nn.conv2d(z, tf.reshape(
            w, [1, 1, 12, 12]), [1, 1, 1, 1], 'SAME')
    x = z
    for inv_w in reversed(inv_ws):
        x = tf.nn.conv2d(x, tf.reshape(
            inv_w, [1, 1, 12, 12]), [1, 1, 1, 1], 'SAME')
    print(tf.reduce_sum((x - origin_x)**2))
