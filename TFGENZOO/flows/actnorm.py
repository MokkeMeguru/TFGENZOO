from functools import reduce

import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowComponent


class Actnorm(FlowComponent):
    """Actnorm Layer
    attributes:
    - calc_ldj: bool
    flag of calculate log det jacobian
    - scale: float
    initialize batch's variance scaling
    - logscale_factor: float
    barrier log value to - Inf

    notes:
    - initialize
    mean = mean(first_batch)
    var = variance(first_batch)
    logs = log(scale / sqrt(var) / log_scale_factor)
    bias = - mean

    - forward formula
    logs = logs * log_scale_factor
    scale = exp(logs)
    z = (x + bias) * scale
    log_det_jacobain = sum(logs) * H * W

    - inverse
    logs = logs * log_scale_factor
    inv_scale = exp(-logs)
    z = x * inv_scale - bias
    inverse_log_det_jacobian = sum(- logs) * H * W
    """

    def __init__(self,
                 calc_ldj: bool = True,
                 scale: float = 1.0,
                 logscale_factor: float = 3.0,
                 **kwargs):
        self.calc_ldj = calc_ldj
        self.scale = scale
        self.logscale_factor = logscale_factor
        super(Actnorm, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        reduce_axis = list(range(len(input_shape)))
        reduce_axis.pop(-1)
        self.reduce_axis = reduce_axis
        self.logdet_factor = tf.constant(
            reduce(lambda x, y: x*y,
                   list(input_shape[1:-1])), tf.float32)
        logs_shape = [1 for i in range(len(input_shape))]
        logs_shape[-1] = input_shape[-1]
        self.logs = self.add_weight(shape=tuple(logs_shape),
                                    initializer='zeros',
                                    # regularizer=tf.keras.regularizers.l2(0.01),
                                    trainable=True)
        self.bias = self.add_weight(shape=tuple(logs_shape),
                                    initializer='zeros',
                                    # regularizer=tf.keras.regularizers.l2(0.02),
                                    trainable=True)
        super(Actnorm, self).build(input_shape)

    def initialize_parameter(self, x: tf.Tensor):
        tf.print('[Info] initialize parameter at {}'.format(self.name))
        ctx = tf.distribute.get_replica_context()
        n = ctx.num_replicas_in_sync
        x_mean, x_mean_sq = ctx.all_reduce(
            tf.distribute.ReduceOp.SUM,
            [tf.reduce_mean(
                x, axis=self.reduce_axis, keepdims=True) / n,
             tf.reduce_mean(
                 tf.square(x), axis=self.reduce_axis, keepdims=True) / n]
        )
        x_var = x_mean_sq - tf.square(x_mean)
        logs = tf.math.log(self.scale * tf.math.rsqrt(x_var + 1e-6))
        # logs = tf.math.log(
        #  self.scale * tf.math.rsqrt(x_var + 1e-6) / self.log_scale_factor)
        #  * self.log_scale_factor
        self.add_update(self.bias.assign(- x_mean))
        self.add_update(self.logs.assign(logs))

    def forward(self, x: tf.Tensor, **kwargs):
        z = x + self.bias
        z = z * tf.exp(self.logs)
        if self.calc_ldj:
            log_det_jacobian = tf.reduce_sum(self.logs) * self.logdet_factor
            log_det_jacobian = tf.broadcast_to(
                log_det_jacobian, tf.shape(x)[0:1])
            return z, log_det_jacobian
        else:
            return z

    def inverse(self, z: tf.Tensor, **kwargs):
        x = z * tf.exp(- 1 * self.logs)
        x = x - self.bias
        if self.calc_ldj:
            inverse_log_det_jacobian = (
                -1 * tf.reduce_sum(self.logs) * self.logdet_factor)
            inverse_log_det_jacobian = tf.broadcast_to(
                inverse_log_det_jacobian, tf.shape(x)[0:1])
            return x, inverse_log_det_jacobian
        else:
            return x
