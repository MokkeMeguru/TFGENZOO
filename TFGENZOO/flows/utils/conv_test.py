import numpy as np
import tensorflow as tf

from .conv_zeros import Conv2DZeros


class Conv2DZerosTest(tf.test.TestCase):
    def setup(self):
        super(Conv2DZerosTest, self).setup()
        self.conv = Conv2DZeros(width_scale=2)
        self.conv.build((None, 32, 32, 1))
        
    def testConv2DZerosOutputShape(self):
        x = tf.random.normal([32, 32, 32, 1])
        y = self.conv(x)
        self.assertShapeEqual(
            np.zeros(32, 32, 32, 2), y)



def main():
    x = tf.keras.Input([16, 16, 2])
    conv = Conv2DZeros(width_scale=2)
    y = conv(x)
    model = tf.keras.Model(x, y)
    model.summary()
    x = tf.random.normal([32, 16, 16, 2])
    y = conv(x)
    print(y.shape)


if __name__ == '__main__': 
    tf.test.main(argv=None)
