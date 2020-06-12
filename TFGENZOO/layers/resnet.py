import tensorflow as tf
from tensorflow.keras import Model, layers

from TFGENZOO.flows.utils.conv import Conv2D
from TFGENZOO.flows.utils.conv_zeros import Conv2DZeros

Layer = layers.Layer


def ShallowResNet(inputs, width: int = 512, out_scale: int = 2):
    """ResNet of OpenAI's Glow
    Sources:
        https://github.com/openai/glow/blob/master/model.py#L420-L426
    Notes:
        This layer is not Residual Network
        because this layer does not have Skip connection
    """

    conv1 = Conv2D(width=width)
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width_scale=out_scale)

    outputs = inputs
    outputs = tf.nn.relu(conv1(outputs))
    outputs = tf.nn.relu(conv2(outputs))
    outputs = conv_out(outputs)
    return tf.keras.Model(inputs, outputs)

    # class ShallowResNet(Model):
    #     """ResNet of OpenAI's Glow
    #     Sources:
    #         https://github.com/openai/glow/blob/master/model.py#L420-L426
    #     Notes:
    #         This layer is not Residual Network
    #         because this layer does not have Skip connection
    #     """

    #     def build(self, input_shape: tf.TensorShape):
    #         self.conv_out = Conv2DZeros(width=input_shape[-1] * self.out_scale)
    #         self.built = True

    #     def __init__(self, width: int = 512, out_scale: int = 2):
    #         super().__init__()
    #         self.out_scale = out_scale
    #         self.width = width
    #         self.conv1 = Conv2D(width=self.width)
    #         self.conv2 = Conv2D(width=self.width, kernel_size=[1, 1])

    #     def get_config(self):
    #         config = super().get_config().copy()
    #         config_update = {
    #             "width": self.width,
    #             "conv1": self.conv1.get_config(),
    #             "conv2": self.conv2.get_config(),
    #             "conv_out": self.conv_out.get_config(),
    #         }
    #         config.update(config_update)
    #         return config

    # def call(self, x: tf.Tensor, cond: tf.Tensor = None):
    #     if cond is not None:
    #         x = tf.concat([x, cond], axis=-1)
    #     x = tf.nn.relu(self.conv1(x))
    #     x = tf.nn.relu(self.conv2(x))
    #     x = self.conv_out(x)
    #     return x


class ShallowConnectedResNet(Model):
    """ResNet of OpenAI's Glow + Skip Connection
    Sources:
        https://github.com/openai/glow/blob/master/model.py#L420-L426
    Notes:
        This layer is not Residual Network
        because this layer does not have shortcut connection
    """

    def build(self, input_shape: tf.TensorShape):
        self.conv_out = Conv2DZeros(width=input_shape[-1] * self.out_scale)
        self.built = True

    def __init__(self, width: int = 512, out_scale: int = 2):
        super().__init__()
        self.out_scale = out_scale
        self.width = width
        self.conv1 = Conv2D(width=self.width)
        self.conv2 = Conv2D(width=self.width, kernel_size=[1, 1])

    def get_config(self):
        config = super().get_config().copy()
        config_update = {
            "width": self.width,
            "conv1": self.conv1.get_config(),
            "conv2": self.conv2.get_config(),
            "conv_out": self.conv_out.get_config(),
        }
        config.update(config_update)
        return config

    def call(self, x: tf.Tensor, cond: tf.Tensor = None):
        if cond is not None:
            x = tf.concat([x, cond], axis=-1)
        shortcut = x
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(x + shortcut)
        x = self.conv_out(x)
        return x
