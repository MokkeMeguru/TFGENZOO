import numpy as np
import tensorflow as tf
from typing import List
# for Flow-based Model
from TFGENZOO.flows.flows import Flow, FlowList
# for Multi-Scale Architecture
from TFGENZOO.flows.flowblock import FlowBlockHalf
from TFGENZOO.flows.identity import Identity
# for Flow-Step
from TFGENZOO.flows.squeezeHWC import SqueezeHWC, UnSqueezeHWC
from TFGENZOO.flows.batchnorm import BatchNormalization
from TFGENZOO.flows.affine_couplingHWC import AffineCouplingHWC
from TFGENZOO.flows.inv1x1conv import Inv1x1Conv
from TFGENZOO.flows.metrics import Process
from TFGENZOO.flows.actnorm import Actnorm


def gen_flowStep(n_hidden: List[int] = [64, 64], with_debug: bool = False):
    flow_step = FlowList(flow_list=[
        Actnorm(with_debug=with_debug), #  BatchNormalization(with_debug=with_debug),
        Inv1x1Conv(with_debug=with_debug),
        AffineCouplingHWC(n_hidden, with_debug=with_debug),
    ], with_debug=with_debug)
    return flow_step


def gen_flowSteps(num_step: int = 2,
                  n_hidden: List[int] = [64, 64],
                  with_debug: bool = False):
    flow_steps = [gen_flowStep(n_hidden=n_hidden, with_debug=with_debug)
                  for _ in range(num_step)]
    return FlowList(flow_list=flow_steps, with_debug=with_debug)


def gen_MultiScaleFlow(
        L=2,
        K=16,
        n_hidden=[64, 64],
        with_debug: bool = False,
        preprocess: bool = False
):
    """Multi-Scale-Architecture in RealNVP
    ref:
    https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/examples/img/multi-scale-arch.png
    """
    def _gen_MSF(level, num_step, n_hidden):
        if level <= 0:
            sHWC = SqueezeHWC(with_debug=with_debug)
            usHWC = UnSqueezeHWC(with_debug=with_debug)
            flow_steps = gen_flowSteps(num_step=num_step,
                                       n_hidden=n_hidden,
                                       with_debug=with_debug)
            return FlowList([sHWC, flow_steps, usHWC])
        else:
            ident = Identity(with_debug=with_debug)
            flow_steps = gen_flowSteps(num_step=num_step,
                                       n_hidden=n_hidden,
                                       with_debug=with_debug)
            msf = _gen_MSF(level-1, num_step, n_hidden)
            block = FlowBlockHalf([ident, msf])
            sHWC = SqueezeHWC(with_debug=with_debug)
            usHWC = UnSqueezeHWC(with_debug=with_debug)
            return FlowList([sHWC, flow_steps, block, usHWC])
    if preprocess:
        preprocess = Process(n_bins=256.0, with_debug=with_debug)
        return FlowList([preprocess, _gen_MSF(L - 1, K, n_hidden)])
    else:
        return _gen_MSF(L - 1, K, n_hidden)


class Glow(tf.keras.Model):
    def __init__(self, args, input_shape=[32, 32, 1]):
        super(Glow, self).__init__(name='Glow')
        self.glow = gen_MultiScaleFlow(
            L=args['L'],
            K=args['K'],
            n_hidden=args['n_hidden'],
            with_debug=False,
            preprocess=True
        )
    def setStat(self, x, **kwargs):
       tf.print("called set Stat")
       self.glow.setStat(x)

    def call(self, x, training=True,**kwargs):
        z, log_det_jacobian = self.glow(x, training=training)
        return z, log_det_jacobian

    def inverse(self, z, training=False, **kwargs):
        x, inverse_log_det_jacobian = self.glow.inverse(z, training=training)
        return x, inverse_log_det_jacobian


def test_MultiScaleFlow():
    flow_step = gen_MultiScaleFlow(with_debug=True)
    x = tf.keras.Input([32, 32, 1])
    z, log_det_jacobian = flow_step(x)
    model = tf.keras.Model(x, (z, log_det_jacobian))
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file='rawglow.png',
        show_shapes=True,
        expand_nested=True,
    )
    # params = 7716(7708/8)

    x = tf.random.normal([16, 32, 32, 1])
    z, ldj = flow_step(x, training=False)
    _x, ildj = flow_step.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))

    print('[debug] in training, batchnorm layer will return different value')
    x = tf.random.normal([16, 32, 32, 1])
    z, ldj = flow_step(x, training=True)
    # print(tf.reduce_max(ldj))
    # print(tf.reduce_min(ldj))
    # print(tf.reduce_mean(ldj))
    _x, ildj = flow_step.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))


def test_flowStep():
    sHWC = SqueezeHWC(with_debug=True)
    flow_step = gen_flowStep(with_debug=True)
    usHWC = UnSqueezeHWC(with_debug=True)
    x = tf.keras.Input([32, 32, 1])
    z, _ = sHWC(x)
    z, log_det_jacobian = flow_step(z)
    z, _ = usHWC(z)
    model = tf.keras.Model(x, (z, log_det_jacobian))
    model.summary()
    # params = 7716(7708/8)

    x = tf.random.normal([16, 32, 32, 1])
    flowStep = FlowList([sHWC, flow_step, usHWC], with_debug=True)
    z, ldj = flowStep(x, training=False)
    _x, ildj = flowStep.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))

    print('[debug] in training, batchnorm layer will return different value')
    x = tf.random.normal([16, 32, 32, 1])
    z, ldj = flowStep(x, training=True)
    _x, ildj = flowStep.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))


def test_flowSteps():
    flow_steps = gen_flowSteps(with_debug=True)
    sHWC = SqueezeHWC(with_debug=True)
    usHWC = UnSqueezeHWC(with_debug=True)
    flow_steps = [sHWC] + [flow_steps] + [usHWC]
    flow_steps = FlowList(flow_steps)
    x = tf.keras.Input([32, 32, 1])
    z, log_det_jacobian = flow_steps(x)
    model = tf.keras.Model(x, (z, log_det_jacobian))
    model.summary()

    x = tf.random.normal([16, 32, 32, 1])
    z, ldj = flow_steps(x, training=False)
    _x, ildj = flow_steps.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))

    print('[debug] in training, batchnorm layer will return different value')
    x = tf.random.normal([16, 32, 32, 1])
    z, ldj = flow_steps(x, training=True)
    _x, ildj = flow_steps.inverse(z)
    print('diff: {}'.format(tf.reduce_mean(x - _x)))
    print('sum: {}'.format(tf.reduce_sum(ldj + ildj)))


def _test_flowStep():
    with_debug = True
    n_hidden = [64, 64]
    x = tf.keras.Input([32, 32, 1])
    sHWC = SqueezeHWC()
    bn = BatchNormalization(with_debug=with_debug)
    inv = Inv1x1Conv(with_debug=with_debug)
    aff = AffineCouplingHWC(n_hidden, with_debug=with_debug)
    uns = UnSqueezeHWC(with_debug=with_debug)
    z = x
    ldj = 0.0
    z, log_det_jacobian = sHWC(z)
    ldj += log_det_jacobian
    z, log_det_jacobian = bn(z, training=True)
    ldj += log_det_jacobian
    z, log_det_jacobian = inv(z)
    ldj += log_det_jacobian
    z, log_det_jacobian = aff(z)
    ldj += log_det_jacobian
    z, log_det_jacobian = uns(z)
    ldj += log_det_jacobian

    model = tf.keras.Model(x, (z, ldj))
    model.summary()
    # params = 7716(7708/8)
