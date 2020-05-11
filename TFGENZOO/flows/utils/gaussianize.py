import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def gaussian_likelihood(mean: tf.Tensor, logsd: tf.Tensor, x: tf.Tensor):
    """calculate negative log likelihood of Gaussian Distribution.
    Args:
        mean  (tf.Tensor[B, ...]): mean
        logsd (tf.Tensor[B, ...]): log standard deviation
        x     (tf.Tensor[B, ...]): tensor
    Returns:
        ll    (tf.Tensor[B, ...]): log likelihood
    Note:
        ll = - 1/2 * {
           k * ln(2 * PI) + ln |var| + (x - mu)^T (Var ^ -1) (x - mu)}
        , where
            k = 1 (Independent)
            var is a variance = exp(2. * logsd)
    """
    c = np.log(2 * np.pi)
    ll = -0.5 * (c + 2.0 * logsd + ((x - mean) ** 2) / tf.math.exp(2.0 * logsd))
    return ll


def gaussian_sample(mean: tf.Tensor, logsd: tf.Tensor, temparature: float = 1.0):
    """sampling from mean / logsd * temparature
    Args:
        mean (tf.Tensor[B, ...]): mean
        logsd(tf.Tensor[B, ...]): log standard deviation
        temparature      (float): temparature
    Returns:
        new_z(tf.Tensor[B, ...]): sampled latent variable
    Noto:
        I cann't gurantee it's correctness.
        Please open the tensorflow probability's Issue.
    """
    return tfp.distributions.Normal(mean, tf.math.exp(logsd) * temparature).sample()
