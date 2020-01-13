import tensorflow as tf
from flows import flows
Flow = flows.Flow


class RevPermute(Flow):
    """Reverse Permute Layer
    formula:
    f: z = x
    log_det_jacobian = 0
    """

    def __init__(self, with_debug: bool = True, **kwargs):
        """
        """
        super(RevPermute, self).__init__(with_debug=with_debug)

    def build(self, input_shape):
        super(RevPermute, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs):
        log_det_jacobian = tf.broadcast_to(tf.constant(0.0), tf.shape(x)[0:1])
        z = tf.reverse(x, axis=[-1])
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        inverse_log_det_jacobian = tf.broadcast_to(
            tf.constant(0.0), tf.shape(z)[0:1])
        x = tf.reverse(z, axis=[-1])
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian
