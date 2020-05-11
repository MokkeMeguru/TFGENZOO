import numpy as np
import tensorflow as tf


def bits_x(
    log_likelihood: tf.Tensor, log_det_jacobian: tf.Tensor, pixels: int, n_bits: int = 8
):
    """bits/dims
    Args:
        log_likelihood: shape is [batch_size,]
        log_det_jacobian: shape is [batch_size,]
        pixels: e.g. HWC image => H * W * C
        n_bits: e.g [0 255] image => 8 = log(256)

    Returns:
        bits_x: shape is [batch_size,]

    formula:
        (log_likelihood + log_det_jacobian)
          / (log 2 * h * w * c) + log(2^n_bits) / log(2.)
    """
    nobj = -1.0 * (log_likelihood + log_det_jacobian)
    _bits_x = nobj / (np.log(2.0) * pixels) + n_bits
    return _bits_x


def split_feature(x: tf.Tensor, type: str = "split"):
    """type = [split, cross]
    """
    channel = x.shape[-1]
    if type == "split":
        return x[..., : channel // 2], x[..., channel // 2 :]
    elif type == "cross":
        return x[..., 0::2], x[..., 1::2]
