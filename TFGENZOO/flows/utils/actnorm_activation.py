import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class ActnormActivation(Layer):
    """Actnorm Layer without inverse function

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L71-L163

    Attributes:
        scale (float)          : scaling
        logscale_factor (float): logscale_factor

    Note:
        * initialize
            | mean = mean(first_batch)
            | var = variance(first-batch)
            | logs = log(scale / sqrt(var)) / log-scale-factor
            | bias = -mean

        * forward formula (forward only)
            | logs = logs * log_scale_factor
            | scale = exp(logs)
            | z = (x + bias) * scale
    """

    def __init__(self, scale: float = 1.0, logscale_factor=3.0, **kwargs):
        super(ActnormActivation, self).__init__()
        self.scale = scale
        self.logscale_factor = logscale_factor

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) == 4:
            reduce_axis = [0, 1, 2]
            b, h, w, c = list(input_shape)
        else:
            raise NotImplementedError()

        self.reduce_axis = reduce_axis

        logs_shape = [1 for _ in range(len(input_shape))]
        logs_shape[-1] = input_shape[-1]

        self.logs = self.add_weight(
            name="logscale",
            shape=tuple(logs_shape),
            initializer="zeros",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self.bias = self.add_weight(
            name="bias", shape=tuple(logs_shape), initializer="zeros", trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self.initialized = self.add_weight(
            name="initialized", dtype=tf.bool, trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.initialized.assign(False)
        self.build = True

    def initialize_parameter(self, x: tf.Tensor):
        tf.print("[Info] initialize parameter at {}".format(self.name))
        ctx = tf.distribute.get_replica_context()
        n = ctx.num_replicas_in_sync
        x_mean, x_mean_sq = ctx.all_reduce(
            tf.distribute.ReduceOp.SUM,
            [
                tf.reduce_mean(x, axis=self.reduce_axis, keepdims=True) / n,
                tf.reduce_mean(tf.square(x), axis=self.reduce_axis, keepdims=True) / n,
            ],
        )

        # var(x) = x^2 - mean(x)^2
        x_var = x_mean_sq - tf.square(x_mean)
        logs = (
            tf.math.log(self.scale * tf.math.rsqrt(x_var + 1e-6)) / self.logscale_factor
        )
        self.logs.assign(logs)
        self.bias.assign(-x_mean)

    def call(self, x: tf.Tensor):
        if not self.initialized:
            self.initialize_parameter(x)
            self.initialized.assign(True)

        logs = self.logs * self.logscale_factor
        x = x + self.bias
        x = x * tf.exp(logs)
        return x


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
