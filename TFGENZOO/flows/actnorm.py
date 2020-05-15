import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowComponent


class Actnorm(FlowComponent):
    """Actnorm Layer
    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L71-L163

    Note:

        * initialize
            | mean = mean(first_batch)
            | var = variance(first_batch)
            | logs = log(scale / sqrt(var)) / logscale_factor
            | bias = - mean

        * forward formula
            | logs = logs * logscale_factor
            | scale = exp(logs)
            z = (x + bias) * scale
            log_det_jacobain = sum(logs) * H * W

        * inverse formula
            | logs = logs * logsscale_factor
            | inv_scale = exp(-logs)
            | z = x * inv_scale - bias
            | inverse_log_det_jacobian = sum(- logs) * H * W

    Attributes:
        calc_ldj: bool
            flag of calculate log det jacobian
        scale: float
            initialize batch's variance scaling
        logscale_factor: float
            barrier log value to - Inf
    """

    def __init__(self, scale: float = 1.0, logscale_factor: float = 3.0, **kwargs):
        super(Actnorm, self).__init__(**kwargs)
        self.scale = scale
        self.logscale_factor = logscale_factor

    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) == 4:
            reduce_axis = [0, 1, 2]
            b, h, w, c = list(input_shape)
            self.logdet_factor = tf.constant(h * w, dtype=tf.float32)
        else:
            raise NotImplementedError()

        self.reduce_axis = reduce_axis

        # logs_shape = [1, 1, 1, C] if input_shape == [B, H, W, C]
        logs_shape = [1 for _ in range(len(input_shape))]
        logs_shape[-1] = input_shape[-1]

        self.logs = self.add_weight(
            name="logscale",
            shape=tuple(logs_shape),
            initializer="zeros",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=tuple(logs_shape), initializer="zeros", trainable=True
        )

        super().build(input_shape)

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

        self.bias.assign(-x_mean)
        self.logs.assign(logs)

    def forward(self, x: tf.Tensor, **kwargs):
        logs = self.logs * self.logscale_factor

        # centering
        z = x + self.bias
        # scaling
        z = z * tf.exp(logs)

        log_det_jacobian = tf.reduce_sum(logs) * self.logdet_factor
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        logs = self.logs * self.logscale_factor

        # inverse scaling
        x = z * tf.exp(-1 * logs)
        # inverse centering
        x = x - self.bias

        inverse_log_det_jacobian = -1 * tf.reduce_sum(logs) * self.logdet_factor
        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(x)[0:1]
        )
        return x, inverse_log_det_jacobian
