import tensorflow as tf
from TFGENZOO.flows import flows
Flow = flows.Flow


class SqueezeHWC(Flow):
    """Squeeze Flow Layer for image [H, W, C]
    formula:
    f: z = x
    ref_1: https://arxiv.org/pdf/1605.08803.pdf
    ref_2: https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py
    """

    def __init__(self, with_debug: bool = True, **kargs):
        super(SqueezeHWC, self).__init__(with_debug=with_debug)

    def build(self, input_shape):
        _, self.h, self.w, self.c = input_shape
        assert self.h % 2 == 0, "height must be even"
        assert self.w % 2 == 0, "width must be even"
        super(SqueezeHWC, self).build(input_shape)

    def call(self, x: tf.Tensor, **kargs):
        z = tf.reshape(
            x,
            [-1, self.h // 2, 2, self.w // 2, 2, self.c]
        )
        z = tf.transpose(z, [0, 1, 3, 5, 2, 4])
        z = tf.reshape(
            z,
            [-1, self.h // 2, self.w // 2, self.c * 2 * 2]
        )
        log_det_jacobian = tf.broadcast_to(0.0, tf.shape(x)[0:1])
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kargs):
        x = tf.reshape(z, [-1, self.h // 2, self.w // 2, self.c, 2, 2])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [-1, self.h, self.w, self.c])
        ildj = tf.broadcast_to(0.0, tf.shape(x)[0:1])
        self.assert_log_det_jacobian(ildj)
        return x, ildj

class UnSqueezeHWC(Flow):
    """Squeeze Flow Layer for image [H, W, C]
    formula:
    f: z = x
    ref_1: https://arxiv.org/pdf/1605.08803.pdf
    ref_2: https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py
    """

    def __init__(self, with_debug: bool = True, **kargs):
        super(UnSqueezeHWC, self).__init__(with_debug=with_debug)

    def build(self, input_shape):
        _, _h, _w, _c = input_shape
        self.h = _h * 2
        self.w = _w * 2
        self.c = _c // 4
        assert _c >= 4, "channel size should be >= 4"
        assert _c % 4 == 0, "channel size is devisible by 4"

    def inverse(self, z: tf.Tensor, **kargs):
        x = tf.reshape(
            z,
            [-1, self.h // 2, 2, self.w // 2, 2, self.c]
        )
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(
            x,
            [-1, self.h // 2, self.w // 2, self.c * 2 * 2]
        )
        log_det_jacobian = tf.broadcast_to(0.0, tf.shape(z)[0:1])
        self.assert_log_det_jacobian(log_det_jacobian)
        return x, log_det_jacobian

    def call(self, x: tf.Tensor, **kargs):
        z = tf.reshape(x, [-1, self.h // 2, self.w // 2, self.c, 2, 2])
        z = tf.transpose(z, [0, 1, 4, 2, 5, 3])
        z = tf.reshape(z, [-1, self.h, self.w, self.c])
        ildj = tf.broadcast_to(0.0, tf.shape(x)[0:1])
        self.assert_log_det_jacobian(ildj)
        return z, ildj


def test_SqueeseHWC():
    sflow = SqueezeHWC()
    x = tf.keras.Input(shape=(4, 4, 1))
    z, log_det_jacobian = sflow(x)
    model = tf.keras.Model(x, (z, log_det_jacobian))
    model.summary()
    # ref. https://arxiv.org/pdf/1605.08803.pdf
    # You can check its correctness
    x = tf.Variable([[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14],
                      [11, 12, 15, 16]]])
    x = tf.expand_dims(x, axis=-1)
    z, ldj = sflow(x)
    print(x[0])
    print(z[0])
    x = tf.random.uniform([32, 4, 4, 1])
    z, ldj = sflow(x)
    print(z.shape)
    _x, ildj = sflow.inverse(z)
    assert ldj.shape == x.shape[0:1],\
        "log detarminant Jacobian's shape is invalid"
    assert ldj.shape == x.shape[0:1], \
        "log_det_jacobian's shape is invalid"
    assert ildj.shape == x.shape[0:1], \
        "inverse_log_det_jacobian's shape is invalid"
    assert x.shape == _x.shape, "inversed shape are invalid"
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_mean(ldj + ildj)))

def test_UnSqueeseHWC():
    sflow = UnSqueezeHWC()
    x = tf.keras.Input(shape=(2, 2, 4))
    z, log_det_jacobian = sflow(x)
    model = tf.keras.Model(x, (z, log_det_jacobian))
    model.summary()
    x = tf.random.uniform([32, 2, 2, 4])
    z, ldj = sflow(x)
    print(z.shape)
    _x, ildj = sflow.inverse(z)
    assert ldj.shape == x.shape[0:1],\
        "log detarminant Jacobian's shape is invalid"
    assert ldj.shape == x.shape[0:1], \
        "log_det_jacobian's shape is invalid"
    assert ildj.shape == x.shape[0:1], \
        "inverse_log_det_jacobian's shape is invalid"
    assert x.shape == _x.shape, "inversed shape are invalid"
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_mean(ldj + ildj)))
    ssflow = SqueezeHWC()
    __x, lldj = ssflow(x)
    print('diff: {}'.format(tf.reduce_mean(x - __x)))
