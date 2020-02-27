from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from TFGENZOO.flows.actnorm import Actnorm
from TFGENZOO.flows.affine_coupling import AffineCoupling
from TFGENZOO.flows.factor_out import FactorOut
from TFGENZOO.flows.flowbase import FactorOutBase, FlowModule
from TFGENZOO.flows.inv1x1conv import Inv1x1Conv
from TFGENZOO.flows.quantize import LogitifyImage
from TFGENZOO.flows.squeeze import Squeezing
from TFGENZOO.flows.utils import ResidualNet


class SingleFlow(Model):
    def __init__(self, K: int = 5, L: int = 1, resblk_kwargs: Dict = None):
        super().__init__()
        if resblk_kwargs is None:
            resblk_kwargs = {
                'num_block': 3,
                'units_factor': 6
            }
        self.resblk_kwargs = resblk_kwargs
        self.K = K
        self.L = L
        layers = []
        layers.append(LogitifyImage())
        for _ in range(3):
            layers.append(Squeezing())
            for _ in range(5):
                layers.append(Actnorm())
                layers.append(Inv1x1Conv())
                layers.append(AffineCoupling(scale_shift_net=ResidualNet(
                    **self.resblk_kwargs)))
        self.flows = layers

    def call(self, x, zaux=None, inverse=False, training=True):
        if inverse:
            return self.inverse(x, zaux, training)
        else:
            return self.forward(x, training)

    def inverse(self, x, zaux, training):
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        for flow in reversed(self.flows):
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True)
                else:
                    x = flow(x, inverse=True)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True)
                else:
                    x = flow(x, zaux=zaux, inverse=True)
            else:
                x, ldj = flow(x, inverse=True, training=training)
                inverse_log_det_jacobian += ldj
        return x, inverse_log_det_jacobian

    def forward(self, x, training):
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
            elif isinstance(flow, FactorOutBase):
                x, zaux = flow(x, zaux)
            else:
                x, ldj = flow(x, training=training)
                log_det_jacobian += ldj
        return x, log_det_jacobian  # , zaux


class BasicGlow(Model):
    def __init__(self, K: int = 5, L: int = 3, resblk_kwargs: Dict = None):
        super(BasicGlow, self).__init__()
        if resblk_kwargs is None:
            resblk_kwargs = {
                'num_block': 3,
                'units_factor': 6
            }
        self.resblk_kwargs = resblk_kwargs
        self.K = K
        self.L = L
        layers = []
        layers.append(LogitifyImage())
        for l in range(self.L):
            if l == 0:
                layers.append(Squeezing(with_zaux=False))
            else:
                layers.append(Squeezing(with_zaux=True))
            fml = []
            for k in range(self.K):
                fml.append(Actnorm())
                fml.append(Inv1x1Conv())
                fml.append(AffineCoupling(scale_shift_net=ResidualNet(
                    **self.resblk_kwargs)))
            layers.append(FlowModule(fml))
            if l == 0:
                layers.append(FactorOut())
            elif l != self.L - 1:
                layers.append(FactorOut(with_zaux=True))
        self.flows = layers

    def call(self, x, zaux=None, inverse=False, training=True):
        if inverse:
            return self.inverse(x, zaux, training)
        else:
            return self.forward(x, training)

    def inverse(self, x, zaux, training):
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        for flow in reversed(self.flows):
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True)
                else:
                    x = flow(x, inverse=True)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True)
                else:
                    x = flow(x, zaux=zaux, inverse=True)
            else:
                x, ldj = flow(x, inverse=True, training=training)
                inverse_log_det_jacobian += ldj
        return x, inverse_log_det_jacobian

    def forward(self, x, training):
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
            elif isinstance(flow, FactorOutBase):
                x, zaux = flow(x, zaux=zaux)
            else:
                x, ldj = flow(x, training=training)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux


def basic_glow_Test():
    tf.debugging.enable_check_numerics()
    x = tf.keras.Input([24, 24, 1])
    model = BasicGlow()
    model.build(x.shape)
    z, ldj, zaux = model(x)
    print(z.shape)
    print(ldj.shape)
    model.summary()

    train, test = tf.keras.datasets.mnist.load_data()
    train_image = train[0] / 255.0
    train_image = train_image[..., tf.newaxis]
    train_image = tf.compat.v1.image.resize_bilinear(
        train_image, size=(24, 24))
    # forward -> inverse
    train_image = train_image[0:12]
    forward, ldj, zaux = model(train_image, inverse=False)
    print('max-min')
    tf.print(tf.reduce_max(forward))
    tf.print(tf.reduce_min(forward))
    tf.print(tf.reduce_sum(tf.cast(tf.math.is_inf(forward), tf.float32)))
    tf.print(tf.reduce_sum(tf.cast(tf.math.is_nan(forward), tf.float32)))
    inverse, ildj = model(forward, zaux=zaux, inverse=True)
    print(forward.shape)
    print(zaux.shape)
    print(ldj.shape)
    print(ildj.shape)
    print('diffs-ldj-rec')
    tf.print(tf.reduce_mean(ldj + ildj))
    tf.print(tf.reduce_mean(train_image - inverse))
    train_image = inverse
    print(train_image.shape)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 18))
    for i in range(9):
        img = tf.squeeze(train_image[i])
        fig.add_subplot(3, 3, i + 1)
        plt.title(train[1][i])
        plt.tick_params(bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False)
        plt.imshow(img, cmap='gray_r')
    plt.show(block=True)

    tf.debugging.disable_check_numerics()
    # return model

def main():
    basic_glow_Test()


if __name__ == '__main__':
    main()
