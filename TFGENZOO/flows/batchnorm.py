"""BatchNorm flow Layer with Custom BatchNormalization
ref.
https://github.com/tensorflow/tensorflow/issues/18222
https://github.com/jkyl/biggan-deep/blob/master/src/custom_layers/batch_normalization.py
https://github.com/tensorflow/community/blob/master/rfcs/20181016-replicator.md#global-batch-normalization
This Layer is for BatchNormalization Bijector with Big Batch (multi GPU/TPU)
"""
import tensorflow as tf
from TFGENZOO.flows import flows
from tensorflow.keras import layers
import numpy as np
Flow = flows.Flow
Layer = layers.Layer



class BatchNormalization(Flow):
    """the rough BatchNormalization flow layer
    formula:
    z = f(x) =  X - mean(X)/ std(X)
    x = f^{-1}(z) = Y * std(X) + mean(X)
    log_determinat_jacobian = log (Sigma(std(X)_i^2))^{-1/2}
    = - 0.5 * log (Sigma(std(X)_i^{2}))
    """

    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: float = 1e-6,
                 name: str = 'BatchNorm',
                 normaxis: int = -1,
                 momentum: float = 0.99,
                 with_debug: bool = True,
                 **kwargs):
        """
        args:
        - center: bool
        use mean statistics
        - scale: bool
        use stddev statistics
        - epsilon: float
        epsilon for zero division
        - name: str
        layer's name
        - normaxis: int
        layer's feature axis
        if data is NHWC => C (-1)
        - momentum: float
        momentum of batchnormalization
        - with_debug: bool
        debug for Flow Layer
        """
        super(BatchNormalization, self).__init__(
            name=name,
            **kwargs
        )
        self.axis = normaxis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.momentum = momentum

    def _get_broadcast_fn(self, input_shape):
        """broadcasting for de-normalization
        Note:
        If data's shape is [None, 32, 32, 3],
        BatchNormalization is normalize axis -1
        So, to de-normalizing the data, we should use the Tensor [1, 1, 1, 3]
        But BatchNormalization's mean or variance 's shape is [3,]
        Then, we need broadcast [3,] -> [1,1, 1, 3]
        """
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims)
                          if i not in [self.axis]]
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape[self.axis]

        def _broadcast(v: tf.Tensor):
            if (reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v
        return _broadcast
    def build(self, input_shape):
        """
        args:
        - input_shape: list
        example. [None, H, W, C] = [None, 32, 32, 3] (cifer 10)
        """
        super(BatchNormalization, self).build(input_shape)
        self.built = True
        axes = list(range(len(input_shape)))
        axes.pop(self.axis)
        pixels = input_shape.as_list()
        pixels.pop(self.axis)
        self.pixels = np.prod(pixels[1:])
        self.axes = axes # ERROR
        self.feature_dim = input_shape[self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                shape=(self.feature_dim,),
                name='gamma',
                initializer='ones',
                constraint='non_neg'
            )
        if self.center:
            self.beta = self.add_weight(
                shape=(self.feature_dim,),
                name='beta',
                initializer='zeros',
            )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.feature_dim,),
            initializer=tf.initializers.zeros,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=(self.feature_dim,),
            initializer=tf.initializers.ones,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)

        self._broadcast_fn = self._get_broadcast_fn(input_shape)

    def _assign_moving_average(self, variable: tf.Tensor, value: tf.Tensor):
        return variable.assign(variable * self.momentum
                               + value * (1.0 - self.momentum))

    def call(self, x: tf.Tensor, training=True, **kwargs):
        """
        args:
        - x: tf.Tensor
        input data
        - training: bool
        switch of training
        Returns:
        - z: tf.Tensor
        output latent
        - log_det_jacobian: tf.Tensor
        log determinant jacobian
        note:
        z = (gamma (x - mu) / sigma) + beta
        """
        if training:
            # axes = list(range(len(self.input_shape)))
            # axes.pop(self.axis)
        
            ctx = tf.distribute.get_replica_context()
            n = ctx.num_replicas_in_sync
            mean, mean_sq = ctx.all_reduce(
                tf.distribute.ReduceOp.SUM,
                [tf.reduce_mean(x, axis=self.axes) / n,
                 tf.reduce_mean(tf.square(x),
                                axis=self.axes) / n]
            )
            variance = mean_sq - mean ** 2
            mean_update = self._assign_moving_average(self.moving_mean, mean)
            variance_update = self._assign_moving_average(
                self.moving_variance, variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        z = tf.nn.batch_normalization(x, mean=mean,
                                      variance=variance,
                                      offset=self.beta,
                                      scale=self.gamma + 1e-6,
                                      variance_epsilon=self.epsilon)
        log_variance = tf.reduce_sum(tf.math.log(variance + self.epsilon))
        log_gamma = tf.reduce_sum(tf.math.log(self.gamma + 1e-6))
        log_det_jacobian = log_gamma - 0.5 * log_variance
        # ref. https://github.com/tensorflow/probability/blob/r0.8/tensorflow_probability/python/bijectors/batch_normalization.py
        # I think this formula is something wrong, but the google do it...
        log_det_jacobian = log_det_jacobian * self.pixels
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, training: bool = False, **kargs):
        """De-BatchNormalization
        args:
        - z: tf.Tensor
        input latent
        - training: bool
        switch of training
        note:
        x = sigma (z - beta) / gamma  + mu
        """
        mean = self._broadcast_fn(self.moving_mean)
        variance = self._broadcast_fn(self.moving_variance)
        beta = self._broadcast_fn(self.beta)
        gamma = self._broadcast_fn(self.gamma)
        sigma = tf.sqrt(variance + self.epsilon)
        x = (z - beta) * sigma / (gamma + 1e-6) + mean
        log_variance = tf.reduce_sum(tf.math.log(
            self.moving_variance + self.epsilon))
        log_gamma = tf.reduce_sum(tf.math.log(self.gamma + 1e-6))
        log_det_jacobian = log_gamma - 0.5 * log_variance
        inverse_log_det_jacobian = - log_det_jacobian
        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(z)[0:1])
        self.assert_tensor(z, x)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


def test_Batchnorm():
    batchnorms = BatchNormalization()
    x = tf.keras.Input([32, 32, 3])
    z, log_det_jacobian = batchnorms(x, training=True)
    model = tf.keras.Model(x, [z, log_det_jacobian])
    model.summary()
    from pprint import pprint
    print('model_params:')
    pprint(model.trainable_variables)
    x = tf.random.normal([12, 32, 32, 3])
    z, log_det_jacobian = batchnorms(x, training=True)
    _x, ildj = batchnorms.inverse(z)
    print('=> with_training_diff: {}'.format(tf.reduce_mean(x - _x)))
    # not zero
    x = tf.random.normal([12, 32, 32, 3])
    z, log_det_jacobian = batchnorms(x, training=False)
    _x, ildj = batchnorms.inverse(z, training=False)
    print('=> without_training_diff: {}'.format(tf.reduce_mean(x - _x)))
    # nearly zero
    print('whole shape: x: {} / z: {} ldj: {} / ildj: {}'
          .format(_x.shape, z.shape, log_det_jacobian.shape, ildj.shape))
    assert log_det_jacobian.shape == z.shape[0:1], \
        "log_det_jacobian's shape is invalid"
    assert ildj.shape == _x.shape[0:1], "log_det_jacobian's shape is invalid"
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_mean(log_det_jacobian + ildj)))
