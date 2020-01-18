import tensorflow as tf
from TFGENZOO.flows import flows
from tensorflow.keras import layers
from tensorflow.keras import initializers
from typing import List

Flow = flows.Flow
Conv2D = layers.Conv2D
Layer = layers.Layer


def split_feature(x: tf.Tensor, type="split"):
    """
    """
    C = tf.shape(x)[-1]
    if type == "split":
        return x[:, :, :, :C // 2], x[:, :, :, C // 2]
    elif type == "cross":
        return x[:, :, :,  0::2], x[:, :, :,  1::2]

class NNHWC(Layer):
    """NN Layer for AffineCouplingHWC
    IMG has the shape [H, W, C]
    this Layer is not for [H, W], [S, E]
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
        assert len(input_shape) == 4, \
            'this model is for Image which shape is [H, W, C]'
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
                 n_hidden=[512, 512],
                 kernel_size=[[3, 3], [1, 1]],
                 strides=[[1, 1], [1, 1]],
                 activation='relu',
                 logscale_factor=3.0,
                 name=None,
                 **kargs):
        """
        """
        if name:
            super(NNHWC, self).__init__(name=name)
        else:
            super(NNHWC, self).__init__()
        layer_list = []
        self.kernel_initializer = initializers.get('zeros')
        self.bias_initializer = initializers.get('zeros')
        self.n_hidden = n_hidden
        self.logscale_factor=3.0
        self.rn_initialzier = tf.random_normal_initializer(mean=0, stddev =0.05)
        for i, (hidden, kernel, stride) in enumerate(
                zip(n_hidden, kernel_size, strides)):
            layer_list.append(
                Conv2D(
                    hidden,
                    kernel_size=kernel,
                    strides=stride,
                    activation=activation,
                    padding='SAME',
                    kernel_initializer=self.kernel_initializer,
                    name=self.name + "dense_{}_1".format(i)))
        self.layer_list = layer_list
        self.setup_output_layer()

    def setup_output_layer(self):
        """setup output layer
        """
    #     self.s_activation = tf.nn.sigmoid
    #     self.t_activation = tf.nn.relu6
        self.kernel_size = [3, 3]
    #     self.rank = 2
        self.strides = (1, 1, 1, 1)
        self.padding = "SAME"

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        y = tf.nn.conv2d(y, self.kernel, self.strides, self.padding, data_format='NHWC')
        y = y + self.bias
        y = y * tf.exp(self.logs * self.logscale_factor)
        shift, scale = split_feature(y, 'cross')
        scale = tf.nn.sigmoid(scale + 2.0)
        return scale, shift


def test_NNHWC():
    nn = NNHWC([64, 64])
    x = tf.keras.Input([32, 32, 1])
    s, t = nn(x)
    model = tf.keras.Model(x, [s, t])
    model.summary()
    for i in model.trainable_variables:
        print(i.name)


class AffineCouplingHWC(Flow):
    """AffineCouplingHWC Layer
    AffineCoupling for Image [H, W, C]
    ref. Glow https://arxiv.org/abs/1807.03039
    formula:
    x_a, x_b = split(x, axis=-1)
    (log s, t) = NN(x_b)
    s = exp(log s)
    y_a = s o x_a + t
    y_b = x_b
    y = concat(y_a, y_b)
    where
    (data) x <-> z (latent)
    x_a, x_b in [H, W, C // 2]

    log_det_jacobian
    sum(log(|s|))
    """

    def build(self, input_shape):
        if self.with_debug:
            assert input_shape[-1] % 2 == 0, 'last dimention should be even'

    def __init__(self, n_hidden: List[int], with_debug: bool = True, **kargs):
        """
        """
        super(AffineCouplingHWC, self).__init__(with_debug=with_debug)
        self.NN = NNHWC(n_hidden)

    def call(self, x: tf.Tensor, **kargs):
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
        x_a, x_b in [H, W, C // 2]
        """
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        s, t = self.NN(x_b)
        y_a = x_a * s + t
        y = tf.concat([y_a, y_b], axis=-1)
        log_det_jacobian = tf.reduce_sum(
            tf.math.log(s), axis=list(range(len(['B', 'H', 'W', 'C'])))[1:])
        self.assert_tensor(x, y)
        self.assert_log_det_jacobian(log_det_jacobian)
        return y, log_det_jacobian

    def inverse(self, y: tf.Tensor, **kargs):
        """De-AffineCouplingHWC
        Args:
        - y: tf.Tensor
        input latent
        Returns:
        - x: tf.Tensor
        output data
        - inverse_log_det_jacobian: tf.Tensor
        log determinant jacobian

        formula:
        y_a, y_b = split(y, axis=-1)
        (log s, t) = NN(y_b)
        s = exp(log s)
        x_a = (y_a - t) / s
        x_b = y_b
        x = concat(x_a, x_b)
        where
        (data) x <-> z (latent)
        y_a, y_b in [H, W, C // 2]
        """
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        s, t = self.NN(y_b)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        inverse_log_det_jacobian = - tf.reduce_sum(
            tf.math.log(s), axis=list(range(len(['B', 'H', 'W', 'C'])))[1:])
        self.assert_tensor(y, x)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


def test_AffineCouplingHWC():
    aff = AffineCouplingHWC([64, 64])
    x = tf.keras.Input([16, 16, 4])
    y, log_det_jacobian = aff(x)
    model = tf.keras.Model(x, [y, log_det_jacobian])
    model.summary()
    from pprint import pprint
    pprint([v.name for v in model.trainable_variables])
    x = tf.random.normal([32, 16, 16, 4])
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
