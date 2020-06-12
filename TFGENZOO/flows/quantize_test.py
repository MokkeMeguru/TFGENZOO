#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from TFGENZOO.flows.quantize import LogitifyImage


class LogitifyImageTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.li = LogitifyImage()
        self.li.build([None, 32, 32, 1])

    def testLogitifyImageOutputShape(self):
        x = tf.nn.tanh(tf.random.normal([1024, 32, 32, 1]))
        z, ldj = self.li(x, inverse=False)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testLogitifyImageOutput(self):
        # x's range is [0, 1]
        x = tf.nn.sigmoid(tf.random.normal([1024, 32, 32, 1]))
        z, ldj = self.li(x, inverse=False)
        rev_x, ildj = self.li(z, inverse=True)
        self.assertAllClose(x, rev_x, rtol=8e-1, atol=1e-2)
        self.assertAllClose(ldj + ildj, tf.zeros([1024]))
