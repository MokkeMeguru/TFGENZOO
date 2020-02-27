import numpy as np
import tensorflow as tf

from TFGENZOO.flows.affine_coupling import AffineCoupling, GlowNN
from TFGENZOO.flows.flowbase import FlowComponent


class GlowNNTest(tf.test.TestCase):
    def setUp(self):
        super(GlowNNTest, self).setUp()
        self.nn = GlowNN(depth=2)
        self.nn.build((None, 16, 16, 4))

    def testGlowNN(self):
        x = tf.random.normal([1024, 16, 16, 4])
        self.nn(x)
        y = self.nn(x)
        self.assertShapeEqual(
            np.zeros([1024, 16, 16, 8]), y)


class AffineCouplingTest(tf.test.TestCase):
    def setUp(self):
        super(AffineCouplingTest, self).setUp()
        self.affine_coupling = AffineCoupling(scale_shift_net=GlowNN(depth=2))
        self.affine_coupling.build((None, 16, 16, 4))

    def testAffineCouplingInitialize(self):
        assert self.affine_coupling.initialized == False
        x = tf.random.uniform((1024, 16, 16, 4))
        self.affine_coupling(x)
        assert self.affine_coupling.initialized == True
        for i in self.affine_coupling.scale_shift_net.res_block.layers:
            if isinstance(i, FlowComponent):
                self.assertTrue(i.initialized)

    def testAffineCouplingInv(self):
        x = tf.random.normal((1024, 16, 16, 4))
        self.affine_coupling(x)
        x = tf.random.normal((1024, 16, 16, 4))
        z,  ldj = self.affine_coupling(x)
        rev_x, ildj = self.affine_coupling(x, inverse=True)
        # print(tf.reduce_max((x - rev_x) ** 2))
        self.assertShapeEqual(
            np.zeros([1024, 16, 16, 4]), z)
        self.assertShapeEqual(
            np.zeros([1024]), ldj)
        self.assertAllClose(x, rev_x,
                            rtol=1e-8, atol=1)
        self.assertAllClose(ldj + ildj,
                            tf.zeros([1024]), rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
    tf.test.main(argv=None)
