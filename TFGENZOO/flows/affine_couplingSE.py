import tensorflow as tf
from TFGENZOO.flows import flows
from typing import List
Flow = flows.Flow
Conv1D = layers.Conv1D

def split_feature(x: tf.Tensor, type="split"):
    """
    """
    E = tf.shape(x)[-1]
    if type == "split":
        return x[:, :, :E // 2], x[:, :, E // 2]
    elif type == "cross":
        return x[:, :, 0::2], x[:, :,  1::2]

class NNSE(Layer):
    """NN Layer for AffineCouplingSE
    text has the shape [S, E]
    this layer is not for [H, W, C]
    """
    def nn_add_weight(self, name, shape, initializer):
        return self.add_weight(
            name=name,
            shape=shape,
            initializer=initializer,
            regularizer=None,
            constraint=None,
            trainable=True,
            dtype=self.dtype
        )

    def build(self, input_shape):
        """
        """
        assert len(input_shape) == 3, \
            'this model is for text which shape is [S, E]'
        self.shape = input_shape
        self.filters = input_shape[-1]

        self.kernel = self.nn_add_weight(
            name='kernel',
            shape=self.kernel_size + [self.n_hidden[-1]] + [input_shape[-1] * 2],
            initializer=initializers.get('zeros'))

        self.bias = self.nn_add_weight(
            name='bias',
            shape=(self.filters * 2,),
            initializer=initializers.get('zeros'))

        self.logs = self.add_weight(
            'logs',
            shape=[1 for i in range(len(input_shape) - 2)] + [input_shape[-1] * 2],
            initializer = initializers.get('zeros'))

    def __init__(self,
                 input_shape,
                 n_hidden=[512, 512],
                 kernel_size=[3, 3],
                 strides=[1, 1],
                 activation='relu',
                 name=None
    ):
        self.n_hidden=n_hidden
        if name:
            super(NNSE, self).__init__(name=name)
        else:
            super(NNSE, self).__init__('NNSE')
        layer_list = []
        for i, (hidden, kernel, stride) in enumerate(zip(n_hidden, kernel_size, strides)):
            layer_list.append(
                Conv1D(
                    hidden,
                    kernel_size=kernel,
                    activation=activation,
                    padding='SAME',
                    name="dense_{}".format(i)
                ))
        self.logscale_factor=3.0
        self.layer_list = layer_list
        self.setup_output_layer()

    def setup_output_layer(self):
        """setup output layer
        """
        self.kernel_size = [3]
        self.strides = [1]
        self.padding = "SAME"

    def call(self, x, **kwargs):
        y = x
        for layer in self.layer_list:
           y = layer(y)
        y = tf.nn.conv1d(y, self.kernel, self.strides, self.padding, data_format='NHWC')
        y  = y + self.bias
        y = y * tf.exp(self.logs * self.logscale_factor)
        shift, scale = split_feature(y, 'cross')
        scale = tf.nn.sigmoid(scale + 2.0)
        return  scale, shift


def test_NNSE():
    nn = NNSE([512, 512])
    x = tf.keras.Input([40, 128])
    s, t = nn(x)
    model = tf.keras.Model(x, [s, t])
    model.summary()


class AffineCouplingSE(Flow):
    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        if self.nn_layer == 'encoder':
            raise Exception()
        else:
            self.nn = NNSE(input_shape[-1] // 2)

    def __init__(
            self,
            n_hidden = [64, 64],
            nn_layer='encoder',
            dff=2048,
            **kwargs):
        super(AffineCouplingSE, self).__init__()
        self.n_hidden = n_hidden
        self.nn_layer = nn_layer

    def call(self, x: tf.Tensor, **kwargs):
        """AffineCouplingHWC
        Args:
        - x: tf.Tensor
        input data
        Returns:
        - z: tf.Tensor
        output latent
        - log_det_jacobian: tf.Tensor
        log determinant jacobian

        formula:
        x_a, x_b = split(x, axis=-1)
        (s, t) = NN(x_b)
        y_a = s o x_a + t
        y_b = x_b
        y = concat(y_a, y_b)
        where
        (data) x <-> y (latent)
        x_a, x_b in [S, E // 2]
        """
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        s, t = self.nn(x_b)
        y_a = x_a * s + t
        y = tf.concat([y_a, y_b], axis=-1)
        log_det_jacobian = tf.reduce_sum(
            tf.math.log(s), axis=list(range(len(['B' ,'S', 'E']))) [1:])
        self.assert_tensor (x, y)
        self.assert_log_det_jacobian (log_det_jacobian)
        return y, log_det_jacobian

    def inverse (self, y: tf.Tensor, **kwargs):
        y_a, y_b = tf.split (y, 2, axis=-1)
        x_b =y_b
        s, t = self.nn (y_b)
        x = tf.concat ([x_a, x_b], axis=-1)
        inverse_log_det_jacobian = - tf.reduce_sum (
            tf.math.log (s), axis = list (range (len (['B', 'S', 'E'])))[1:])
        self.assert_tensor (y, x)
        self.assert_log_det_jacobian (inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian

def test_AffineCouplingSE ():
    aff = AffineCouplingSE(
                           n_hidden = [64, 64],
            nn_layer='nn')
    x = tf.keras.Input([32, 128])
    y, log_det_jacobian = aff(x)
    model = tf.keras.Model(x, [y, log_det_jacobian])
    model.summary()
    from pprint import pprint
    pprint([v.name for v in model.trainable_variables])
    x = tf.random.normal([128, 32, 128])
    y, log_det_jacobian = aff(x)
    x_, ildj = aff.inverse(y)
    assert log_det_jacobian.shape == y.shape[0:1], \
        "log_det_jacobian's shape is invalid"
    assert ildj.shape == y.shape[0:1], "log_det_jacobian's shape is invalid"
    print('diff: {}'.format(tf.reduce_mean(x - x_)))
    print('sum: {}'.format(tf.reduce_mean(log_det_jacobian + ildj)))

    with tf.GradientTape() as tape:
        y, ldj = aff(x)
    grads = tape.gradient(ldj, aff.trainable_variables)
    print(len(grads), len(aff.trainable_variables))
