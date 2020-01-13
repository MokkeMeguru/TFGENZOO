"""BatchNorm flow Layer with tf.keras.layers.BatchNormalization
WARNING: PLEASE DON'T USE THIS LAYER
Tensorflow's official tf.keras.layers.BatchNormalization is THE INVALID LAYER
WE SHOULD RE-IMPLEMENT BatchNormalization Layer
"""
import tensorflow as tf
from flows import flows
from tensorflow.keras import layers
Flow = flows.Flow
BatchNorm = layers.BatchNormalization


class BatchNormalization(Flow):
    """the rough BatchNormalization flow layer
    TODO: WE SHOULD NOT USE official BatchNormalization Layer...
    this layer has many and many bugs.
    formula:
    z = f(x) =  X - mean(X)/ std(X)
    x = f^{-1}(z) = Y * std(X) + mean(X)
    log_determinat_jacobian = log (Sigma(std(X)_i^2))^{-1/2}
    = - 0.5 * log (Sigma(std(X)_i^{2}))
    """

    def __init__(self,
                 batchnorm_layer: tf.keras.layers.Layer = None,
                 batchnorm_axis: int = -1, with_debug: bool = True,  **kargs):
        """
        """
        super(BatchNormalization, self).__init__(with_debug=with_debug)
        def g_constraint(x): return tf.nn.relu(x) + 1e-6
        if batchnorm_layer is not None:
            self.batchnorm = batchnorm_layer
        else:
            self.batchnorm = BatchNorm(
                axis=batchnorm_axis,
                gamma_constraint=g_constraint)

    def build(self, input_shape):
        """build this Layer
        Note:
        input_shape is [H, W, C] = [32, 32, 3] (ex. Image such as MNIST)
        """
        self.shape = input_shape
        self._broadcast_fn = self._get_broadcast_fn()

    def _get_broadcast_fn(self):
        """broadcasting for de-normalization
        Note:
        If data's shape is [None, 32, 32, 3],
        BatchNormalization is normalize axis -1
        So, to de-normalizing the data, we should use the Tensor [1, 1, 1, 3]
        But BatchNormalization's mean or variance 's shape is [3,]
        Then, we need broadcast [3,] -> [1,1, 1, 3]
        """
        if not self.batchnorm.built:
            self.batchnorm.build(self.shape)
        ndims = len(self.shape)
        reduction_axes = [i for i in range(ndims)
                          if i not in self.batchnorm.axis]
        broadcast_shape = [1] * ndims
        broadcast_shape[self.batchnorm.axis[0]] = \
            self.shape[self.batchnorm.axis[0]]

        def _broadcast(v: tf.Tensor):
            if (reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v
        return _broadcast

    def de_normalize_inverse_log_det_jacobian(self, z: tf.Tensor):
        log_var = tf.math.log(
                     self.batchnorm.moving_variance + self.batchnorm.epsilon)
        log_gamma = tf.math.log(self.batchnorm.gamma)
        log_det_jacobian = tf.reduce_sum(log_gamma - 0.5 * log_var)
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(z)[0:1])
        return - log_det_jacobian

    def de_normalize(self, z: tf.Tensor, training):
        """Batch de-Normalize
        formula:
        x = (z / scale) + mean
        where
        scale and mean are stored value calculated at forward process
        """
        # mean, variance, beta, gamma are the shape [z.shape[-1],]
        # epsilon is the shape ()
        mean = self._broadcast_fn(self.batchnorm.moving_mean)
        variance = self._broadcast_fn(self.batchnorm.moving_variance)
        beta = self._broadcast_fn(self.batchnorm.beta) \
            if self.batchnorm.center else None
        gamma = self._broadcast_fn(self.batchnorm.gamma)\
            if self.batchnorm.scale else None
        epsilon = self.batchnorm.epsilon

        rescale = tf.sqrt(variance + epsilon)
        if self.batchnorm.scale is not None:
            rescale = rescale / gamma
        x = z * rescale + \
            (mean - beta * rescale if beta is not None else mean)
        inverse_log_det_jacobian = \
            self.de_normalize_inverse_log_det_jacobian(z)
        return x, inverse_log_det_jacobian

    def normalize_log_det_jacobian(self, x: tf.Tensor, training):
        """Batch Normalization's log determinant jacobian
        formula:
        log (PI(sigma^2_i + epsilon))^{-1/2}
        = -0.5 * Sigma {log(sigma^2_i + epsilon)}
        reference: https://arxiv.org/pdf/1605.08803.pdf
        """
        # log_var is the shape [x.shape[-1],]
        # log_gamma is the shape [x.shape[-1],]
        # log_det_jacobian is the shape [] -> [batch_size, ] (broadcasting)
        event_ndims = self.batchnorm.axis
        reduction_axes = [i for i in range(len(self.shape))
                          if i not in event_ndims]
        log_var = tf.math.log(
            tf.where(tf.logical_not(training),
                     self.batchnorm.moving_variance,
                     tf.nn.moments(x=x, axes=reduction_axes)[1])
            + self.batchnorm.epsilon)
        log_gamma = tf.math.log(self.batchnorm.gamma)
        log_det_jacobian = tf.reduce_sum(log_gamma - 0.5 * log_var)
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return log_det_jacobian

    def normalize(self, x: tf.Tensor, training):
        """Batch Normalization
        """
        # print('batch-normalize:', training)
        z = self.batchnorm(x, training=training)
        log_det_jacobian = self.normalize_log_det_jacobian(x, training)
        return z, log_det_jacobian

    def call(self, x: tf.Tensor, training: bool = True, **kargs):
        """BatchNormalization
        Args:
        - x: tf.Tensor
        input data
        - training: bool
        switch of training
        Returns:
        - z: tf.Tensor
        output latent
        - log_det_jacobian: tf.Tensor
        log determinant jacobian

        TODO: in tensorflow probability,
        forward step is De-Batchnormalization
        I don't know how to calculate batch's mean and variance
        """
        z, log_det_jacobian = self.normalize(x, training)
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, training: bool = False, **kargs):
        """De-BatchNormalization
        Args:
        - z: tf.Tensor
        input latent
        - training: bool
        switch of training
        """
        x, inverse_log_det_jacobian = self.de_normalize(z, training)
        self.assert_tensor(x, z)
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
