import tensorflow as tf
from flows import flows
from tensorflow.keras import layers
from tensorflow.keras import initializers
from typing import List

Flow = flows.Flow
Conv2D = layers.Conv2D
Layer = layers.Layer


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
        self.log_s_kernel = self.nn_add_weight(
            name='log_s/kernel',
            shape=self.kernel_size + [self.n_hidden[-1]] + [input_shape[-1]],
            initializer=self.kernel_initializer)
        self.log_s_bias = self.nn_add_weight(
            name='log_s/bias',
            shape=(self.filters,),
            initializer=self.bias_initializer)
        self.t_kernel = self.nn_add_weight(
            name='t/kernel',
            shape=self.kernel_size + [self.n_hidden[-1]] + [input_shape[-1]],
            initializer=self.kernel_initializer)
        self.t_bias = self.nn_add_weight(
            name='t/bias',
            shape=(self.filters,),
            initializer=self.bias_initializer)

    def __init__(self,
                 n_hidden=[512, 512],
                 kernel_size=[[3, 3], [1, 1]],
                 strides=[[1, 1], [1, 1]],
                 activation='relu',
                 name=None,
                 **kargs):
        """
        """
        if name:
            super(NNHWC, self).__init__(name=name)
        else:
            super(NNHWC, self).__init__()
        layer_list = []
        self.n_hidden = n_hidden
        for i, (hidden, kernel, stride) in enumerate(
                zip(n_hidden, kernel_size, strides)):
            layer_list.append(
                Conv2D(
                    hidden,
                    kernel_size=kernel,
                    strides=stride,
                    activation=activation,
                    padding='SAME',
                    kernel_initializer='zeros',
                    name=self.name + "dense_{}_1".format(i)))
        self.layer_list = layer_list
        self.setup_output_layer()

    def setup_output_layer(self):
        """setup output layer
        """
        self.log_s_activation = tf.nn.tanh
        self.t_activation = tf.nn.relu6
        self.kernel_size = [3, 3]
        self.rank = 2
        self.strides = (1, 1, 1, 1)
        self.padding = "SAME"
        self.kernel_initializer = initializers.get('zeros')
        self.bias_initializer = initializers.get('zeros')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = tf.nn.conv2d(y, self.log_s_kernel,
                             self.strides, self.padding, data_format='NHWC')
        log_s = log_s + self.log_s_bias
        log_s = self.log_s_activation(log_s)
        t = tf.nn.conv2d(y, self.t_kernel, self.strides,
                         self.padding, data_format='NHWC')
        t = t + self.t_bias
        t = self.t_activation(t)
        return log_s, t


def test_NNHWC():
    nn = NNHWC([64, 64])
    x = tf.keras.Input([32, 32, 1])
    log_s, t = nn(x)
    model = tf.keras.Model(x, [log_s, t])
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
        - y: tf.Tensor
        output latent
        - log_det_jacobian: tf.Tensor
        log determinant jacobian

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
        """
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        log_s, t = self.NN(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        log_det_jacobian = tf.reduce_sum(
            log_s, axis=list(range(len([32, 32, 32, 1])))[1:])
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
        log_s, t = self.NN(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        inverse_log_det_jacobian = tf.reduce_sum(
            log_s, axis=list(range(len([32, 32, 32, 1])))[1:])
        self.assert_tensor(y, x)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


def test_AffineCouplingHWC():
    aff = AffineCouplingHWC([64, 64])
    x = tf.keras.Input([16, 16, 2])
    y, log_det_jacobian = aff(x)
    model = tf.keras.Model(x, [y, log_det_jacobian])
    model.summary()
    from pprint import pprint
    pprint([v.name for v in model.trainable_variables])
    x = tf.random.normal([32, 16, 16, 2])
    y, log_det_jacobian = aff(x)
    x_, ildj = aff.inverse(y)
    print('diff: {}'.format(tf.reduce_mean(x - x_)))
