import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowBase


class LogitifyImage(FlowBase):
    """Apply Tapani Raiko's dequantization and express
    image in terms of logits"""

    def build(self, input_shape: tf.TensorShape):
        super(LogitifyImage, self).build(input_shape)
        if len(input_shape) == 4:
            self.reduce_axis = [1, 2, 3]
        elif len(input_shape) == 2:
            self.reduce_axis = [1]
        else:
            raise NotImplementedError()

        super(LogitifyImage, self).build(input_shape)

    def __init__(self, corruption_level=1.0, alpha=1e-5):
        super(LogitifyImage, self).__init__()
        self.corruption_level = corruption_level
        self.alpha = alpha

    def forward(self, x: tf.Tensor, **kwargs):
        z = x * 255.0
        z = z + self.corruption_level * tf.random.uniform(tf.shape(x))
        z = z / (255.0 + self.corruption_level)
        z = z * (1 - self.alpha) + self.alpha * 0.5
        new_z = tf.math.log(z) - tf.math.log(1 - z)
        logdet_jacobian = tf.reduce_sum(- tf.math.log(z) - tf.math.log(1 - z),
                                        self.reduce_axis)
        return new_z, logdet_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        denominator = 1 + tf.exp(-z)
        x = 1 / denominator
        inverse_log_det_jacobian = tf.reduce_sum(
            -2 * tf.math.log(denominator) - z,
            self.reduce_axis)
        return x, inverse_log_det_jacobian


def _main():
    layer = LogitifyImage()  # BasicGlow()
    x = tf.keras.Input((32, 32, 1))
    y = layer(x, training=True)
    model = tf.keras.Model(x, y)

    train, test = tf.keras.datasets.mnist.load_data()
    train_image = train[0] / 255.0
    train_image = train_image[..., tf.newaxis]
    # forward -> inverse
    train_image = train_image[0:12]
    forward, ldj = layer.forward(train_image)
    inverse, ildj = layer.inverse(forward)
    print(ldj)
    print(ildj)
    print(ldj + ildj)
    print(tf.reduce_mean(ldj + ildj))
    print(tf.reduce_mean(train_image - inverse))
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

    model.summary()
    return model
