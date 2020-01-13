import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from typing import List
# for Flow-based Model
from flows.flows import Flow, FlowList
# from Flow-Step
from flows.squeezeHWC import SqueezeHWC, UnSqueezeHWC
from flows.batchnorm import BatchNormalization
from flows.affine_couplingHWC import AffineCouplingHWC

from flows.metrics import Process
from flows.utils import RevPermute


def single_scale_realnvp(x: tf.keras.Input, K: int = 32,
                         n_hidden: List[int] = [64, 64],
                         with_debug: bool = False,
                         preprocess: bool = True):
    """single scale realnvp
    - x: tf.keras.Input
    has shape [H, W, C]
    - n_hidden: List[int]
    affine_coupling's hidden wize
    - with_debug: bool
    assertion flag
    - preprocess: bool
    image preprocess [0, 255] -> [-0.5, 0.5]
    DEFAULT TRUE
    """
    def _realnvp_step(n_hidden: List[int]):
        step = []
        step.append(BatchNormalization(with_debug=with_debug))
        step.append(RevPermute(with_debug=with_debug))
        step.append(AffineCouplingHWC(n_hidden, with_debug=with_debug))
        return step
    process = Process(n_bins=256.0, with_debug=with_debug)
    squeeze = SqueezeHWC(with_debug=with_debug)
    steps = []
    for i in range(K):
        steps += _realnvp_step(n_hidden)
    unsqueeze = UnSqueezeHWC(with_debug=with_debug)
    if preprocess:
        flows = [process, squeeze] + steps + [unsqueeze]
    else:
        flows = [squeeze] + steps + [unsqueeze]
    flowmodel = FlowList(flows, with_debug=with_debug)
    y, log_det_jacobian = flowmodel(x)
    return tf.keras.Model(x, [y, log_det_jacobian]), flowmodel


def _single_scale_realnvp(x: tf.keras.Input, K: int, n_hidden: List[int] = [64, 64], with_debug: bool = False):
    """this is for checking graph network
    """
    def _realnvp_step(x: tf.Tensor, log_det_jacobian: tf.Tensor, n_hidden: List[int]):
        x, _log_det_jacobian = BatchNormalization(with_debug=with_debug)(x)
        log_det_jacobian += _log_det_jacobian
        x = tf.reverse(x, axis=[-1])
        x, _log_det_jacobian = AffineCouplingHWC(
            n_hidden, with_debug=with_debug)(x)
        log_det_jacobian += _log_det_jacobian
        return x, log_det_jacobian
    y, log_det_jacobian = Process(n_bins=256.0, with_debug=with_debug)(x)
    y, _log_det_jacobian = SqueezeHWC(with_debug=with_debug)(y)
    log_det_jacobian += _log_det_jacobian
    for i in range(K):
        y, log_det_jacobian = _realnvp_step(y, log_det_jacobian, n_hidden)
    y, _log_det_jacobian = UnSqueezeHWC(with_debug=with_debug)(y)
    log_det_jacobian += _log_det_jacobian
    return tf.keras.Model(x, [y, log_det_jacobian])


def test__single_scale_realnvp():
    model = _single_scale_realnvp(tf.keras.Input([32, 32, 1]), K=3)
    tf.keras.utils.plot_model(
        model, to_file='single_realnvp.png',  show_shapes=True)
    x = tf.random.uniform([64, 32, 32, 1])
    y, log_det = model(x)
    tfd = tfp.distributions
    target_dist = tfd.MultivariateNormalDiag(
        tf.zeros(np.prod(x.shape[1:])), tf.ones(np.prod(x.shape[1:])))
    ll = target_dist.log_prob(tf.reshape(y, [-1, np.prod(x.shape[1:])]))
    nll = -ll
    nobj = nll - log_det
    bits_x = nobj / (np.log(2.) * np.prod(x.shape[1:]))
    print('bits_x: {}'.format(tf.reduce_mean(bits_x)))
   # => <tf.Tensor: shape=(), dtype=float32, numpy=10.463169><tf.Tensor: shape=(), dtype=float32, numpy=10.463169>


def test_single_scale_realnvp():
    model, _ = single_scale_realnvp(tf.keras.Input([32, 32, 1]), K=3)
    tf.keras.utils.plot_model(
        model, to_file='single_realnvp_with_list.png', show_shapes=True)
    x = tf.random.uniform([64, 32, 32, 1])
    y, log_det = model(x)
    tfd = tfp.distributions
    target_dist = tfd.MultivariateNormalDiag(
        tf.zeros(np.prod(x.shape[1:])), tf.ones(np.prod(x.shape[1:])))
    ll = target_dist.log_prob(tf.reshape(y, [-1, np.prod(x.shape[1:])]))
    nll = -ll
    nobj = nll - log_det
    bits_x = nobj / (np.log(2.) * np.prod(x.shape[1:]))
    print('bits_x: {}'.format(tf.reduce_mean(bits_x)))
    _, flow = single_scale_realnvp(
        tf.keras.Input([32, 32, 1]), K=12, preprocess=False)
    x = tf.random.uniform([64, 32, 32, 1])
    z, ldj = flow(x, training=False)
    _x, ildj = flow.inverse(z)
    print('inference diff without preprocess\n\tdiff                                 :{}\n\tlog_det_jacobian:{}'.format(
        tf.reduce_mean((x - _x) ** 2), tf.reduce_mean(ldj + ildj)))
    # => <tf.Tensor: shape=(), dtype=float32, numpy=10.463169><tf.Tensor: shape=(), dtype=float32, numpy=10.463169>
    return model
