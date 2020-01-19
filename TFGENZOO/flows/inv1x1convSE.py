import tensorflow as tf
from TFGENZOO.flows import flows
from tensorflow.keras import layers
from tensorflow.keras import initializers
from typing import List
import numpy as np

Flow = flows.Flow

def regular_matrix_init(shape, dtype=tf.float32):
    c = shape[0]
    w_init = np.linalg.qr(np.random.randn(c, c))[0]
    return w_init

class Inv1x1ConvSE(Flow):
    """Invertible 1x1 Convolution layer
    ref. Glow https://arxiv.org/pdf/1807.03039.pdf
    formula:
    forall i, j: y_{i, j} = W x_{i, j}
    log_determinant_jacobian = h * w * log|det (W)|
    where
    x_{i, j}, y_{i, j} in [C, C]
    W in [C, C]
    """
    def build(self, input_shape, **kargs):
        s, e = input_shape[1:]
        self.s = s
        self.e = e
        self.W = self.add_weight(
            name='W',
            shape=(e, e),
            initializer=regular_matrix_init
        )

    def __init__(self, with_debug: bool = True, **kargs):
        """
        """
        super(Inv1x1ConvSE, self).__init__(with_debug=with_debug)

    def call(self, x: tf.Tensor, **kargs):
        """Invertible 1x1 Convolution
        Args:
        - x: tf.Tensor
        input data
        Returns:
        - z: tf.Tensor
        ouptut latent
        - log_det_jacobian
        formula:
        z = W x
        log_det_jacobian = log|det W|
        """
        _W = tf.reshape(self.W, [1, self.s, self.e])
        z = tf.nn.conv1d(x, _W, [1, 1, 1], 'SAME')
        log_det_jacobian = self.h * self.w * tf.math.log(abs(tf.linalg.det(self.W)))
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kargs):
        """De Invertible 1x1 Convolution
        Args:
        - z: tf.Tensor
        input latent
        Returns:
        - x: tf.Tensor
        output data
        - inverse_log_det_jacobian
        formula:
        x = W^{-1} x
        inverse_log_det_jacobian = - log |det (W)|
        """
        _inv_W = tf.linalg.inv(self.W)
        _inv_W = tf.reshape(_inv_W, [1, self.c, self.c])
        x = tf.nn.conv1d(z, _inv_W, [1, 1, 1], 'SAME')
        ildj = - self.h * self.w * \
            tf.math.log(abs(tf.linalg.det(self.W)))
        ildj = tf.broadcast_to(ildj, tf.shape(x)[0:1])
        self.assert_tensor(z, x)
        self.assert_log_det_jacobian(ildj)
        return x, ildj


def test_Inv1x1Conv():
    conv1x1 = Inv1x1Conv()
    x = tf.keras.Input([40, 128])
    y, log_det_jacobian = conv1x1(x)
    model = tf.keras.Model(x, [y, log_det_jacobian])
    model.summary()
    x = tf.random.normal([12, 40, 128])
    z, log_det_jacobian = conv1x1(x)
    _x, ildj = conv1x1.inverse(z)
    assert log_det_jacobian.shape == z.shape[0:1], "log_det_jacobian's shape is invalid"
    assert ildj.shape == _x.shape[0:1], "log_det_jacobian's shape is invalid"
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_mean(log_det_jacobian + ildj)))

