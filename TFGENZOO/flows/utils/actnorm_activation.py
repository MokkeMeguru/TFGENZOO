import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class ActnormActivation(Layer):
    """Actnorm
    Attributes:
        scale (float)          : scaling
        logscale_factor (float): logscale_factor
    Note:
        y = (x + bias) * exp(logs)
        bias ans logs is initialized by first batch
    """

    def __init__(self,
                 scale: float = 1.0,
                 batch_variance: bool = False,
                 logscale_factor=3.0):
        super(ActnormActivation, self).__init__()
        self.scale = scale
        self.logscale_factor = logscale_factor
        self._init_critical_section = tf.CriticalSection(name='init_mutex')

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) == 4:
            reduce_axis = [0, 1, 2]
        else:
            raise NotImplementedError()

        self.reduce_axis = reduce_axis
        bias_shape = [1 for i in range(len(input_shape))]
        bias_shape[-1] = input_shape[-1]
        self.bias = self.add_weight(name="bias",
                                    shape=tuple(bias_shape))
        self.logs = self.add_weight(name="logs",
                                    shape=tuple(bias_shape))
        self._initialized = self.add_weight(name="initialized",
                                            initializer="zeros",
                                            dtype=tf.bool,
                                            shape=None,
                                            trainable=False)
        self.built = True

    def initialize_parameter(self, x: tf.Tensor):
        with tf.control_dependencies([
                tf.debugging.assert_equal(
                    self._initialized,
                    False,
                    message="The layer has been initialized")
        ]):
            tf.print("layer initialization {}".format(self.name))
            # ref.
            # https://github.com/tensorflow/tensorflow/issues/18222#issuecomment-579303485
            ctx = tf.distribute.get_replica_context()
            n = ctx.num_replicas_in_sync

            mean = tf.reduce_mean(x, axis=self.reduce_axis, keepdims=True)
            mean_square = tf.reduce_mean(
                tf.square(x), axis=self.reduce_axis, keepdims=True)
            mean, mean_square = ctx.all_reduce(
                tf.distribute.ReduceOp.SUM, [mean, mean_square])
            mean = mean / n
            mean_square = mean_square / n

            x_var = mean_square - tf.square(mean)
            x_mean = mean

            # WARN: this formula leads to raise NaN error
            logs = (tf.math.log(self.scale /
                                (tf.math.sqrt(x_var) + 1e-6) /
                                self.logscale_factor)
                    * self.logscale_factor)
            # logs = tf.math.log(self.scale * tf.math.rsqrt(x_var + 1e-6))

            # WARN: Tensorflow says this operation hasn't been required,
            # self.add_update(self.bias.assign(- x_mean))
            # self.add_update(self.logs.assign(logs))
            # self.add_update(self._initialized.assign(True))
            assign_tensors = [self.bias.assign(- x_mean),
                              self.logs.assign(logs),
                              self._initialized.assign(True)]
            return assign_tensors

    def call(self, inputs: tf.Tensor):
        def _do_nothing():
            return self.logs, self.bias

        def _do_update():
            with tf.control_dependencies(
                    self.initialize_parameter(inputs)):
                return self.logs, self.bias
        logs, bias = self._init_critical_section.execute(
            lambda: tf.cond(self._initialized,
                            _do_nothing,
                            _do_update))
        inputs = inputs + bias
        inputs = inputs * tf.exp(logs)
        return inputs


def main():
    aa = ActnormActivation()
    x = tf.keras.Input([16, 16, 2])
    y = aa(x)
    model = tf.keras.Model(x, y)
    model.summary()
    print(model.variables)
    y_ = model(tf.random.normal([32, 16, 16, 2]))
    print(y_.shape)
    print(model.variables)
