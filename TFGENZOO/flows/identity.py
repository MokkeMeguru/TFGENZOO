import tensorflow as tf
from TFGENZOO.flows import flows
Flow = flows.Flow


class Identity(Flow):
    """Identity Flow Layer
    formula:
    f: z = x
    log_det_jacobian = 0
    """

    def __init__(self, with_debug: bool = True, **kwargs):
        super(Identity, self).__init__(with_debug=with_debug)

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, x: tf.Tensor, **kwargs):
        log_det_jacobian = tf.broadcast_to(tf.constant(0.0), tf.shape(x)[0:1])
        return x, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        inverse_log_det_jacobian = tf.broadcast_to(
            tf.constant(0.0), tf.shape(z)[0:1])
        return z, inverse_log_det_jacobian


def test_Identity():
    iflow = Identity()
    x = tf.keras.Input(shape=(32, 32, 1))
    y, log_det_jacobian = iflow(x)
    model = tf.keras.Model(x, (y, log_det_jacobian))
    model.summary()

    x = tf.random.normal([128, 32, 32, 1])
    y_, ldj_ = iflow(x)
    assert y_.shape == x.shape, "in-out shapes are not same"
    assert ldj_.shape == x.shape[0:1],\
        "log detarminant Jacobian's shape is invalid"
    x_, ildj_ = iflow.inverse(y_)
    assert ldj_.shape == x.shape[0:1], \
        "log_det_jacobian's shape is invalid"
    assert ildj_.shape == x.shape[0:1], \
        "inverse_log_det_jacobian's shape is invalid"
    assert x.shape == x_.shape, "inversed shape are invalid"
    assert tf.reduce_sum(
        (x_ - x)**2) < 1e-5, "this flow layer is inverative function"
