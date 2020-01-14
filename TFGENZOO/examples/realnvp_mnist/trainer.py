"""whole model task
Architecture:
ref. https://github.com/tensorflow/models/blob/master/official/transformer/v2/transformer_main.py
"""
from TFGENZOO.TFGENZOO.examples.realnvp_mnist import model
from TFGENZOO.TFGENZOO.examples.realnvp_mnist import args
from TFGENZOO.TFGENZOO.examples.realnvp_mnist import dataset
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import datetime

Mean = tf.metrics.Mean
Adam = tf.optimizers.Adam
tfd = tfp.distributions


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """for warmup learning rate
    ref:
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def bits_x(log_likelihood: tf.Tensor, log_det_jacobian: tf.Tensor, pixels: int):
    """bits/dims
    args:
    - log_likelihood: tf.Tensor
    shape is [batch_size,]
    - log_det_jacobian: tf.Tensors
    shape is [batch_size,]
    - pixels: int
    pixels
    ex. HWC image => H * W * C
    returns:
    - bits_x: tf.Tensor
    shape is [batch_size,]
    formula:
    - (log_likelihood + log_det_jacobian) / (log 2 * h * w * c)
    """
    nobj = - 1.0 * (log_likelihood + log_det_jacobian)
    _bits_x = nobj / (np.log(2.) * pixels)
    return _bits_x


class RealNVPTask(object):
    def __init__(self, args=args.args, training: bool = True):
        """
        """
        self.args = args
        train_dataset, test_dataset = dataset.read_mnist_dataset(
            show_example=False)
        self.train_dataset = self.preprocess_dataset(train_dataset)
        self.test_dataset = self.preprocess_dataset(test_dataset)

        self.pixels = np.prod(self.args['input_shape'])
        self.model, self.flow = model.single_scale_realnvp(
            tf.keras.Input(self.args['input_shape']),
            K=self.args['K'],
            n_hidden=[64, 64],
            with_debug=False,
            preprocess=True
        )
        self.target_dist = tfd.MultivariateNormalDiag(
            tf.zeros([self.pixels]),
            tf.ones([self.pixels]))

        self.log_prob_loss = Mean(name='log_prob', dtype=tf.float32)
        self.ldj_loss = Mean(name='log_det_jacobian', dtype=tf.float32)
        self.train_loss = Mean(name='train_loss', dtype=tf.float32)

        self.learning_rate_schedule = CustomSchedule(
            d_model=np.prod(self.args['input_shape']))
        self.optimizer = Adam(self.learning_rate_schedule)

        self.setup_checkpoint()

        if training:
            self.writer = tf.summary.create_file_writer(
                logdir='../logs/realnvp_log/' +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.model.summary()

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        """
        """
        dataset = dataset.shuffle(buffer_size=100000)
        dataset = dataset.batch(
            self.args['batch_size']).prefetch(buffer_size=100000)
        return dataset

    def setup_checkpoint(self):
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, '../checkpoints/realnvp_mnist',
            max_to_keep=3
        )
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('[Model] Latest checkpoint restored !!')
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    def train(self):
        self.writer.set_as_default()

        @tf.function
        def train_step(x: tf.Tensor):
            with tf.GradientTape() as tape:
                z, log_det_jacobian = self.model(x)
                log_likelihood = self.target_dist.log_prob(
                    tf.reshape(z, [-1, self.pixels]))
                bx = bits_x(log_likelihood, log_det_jacobian, self.pixels)
            grads = tape.gradient(bx, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
            self.log_prob_loss(log_likelihood)
            self.ldj_loss(log_det_jacobian)
            self.train_loss(bx)
        initialize = False
        for epoch in range(self.args['epochs']):
            for x in tqdm(self.train_dataset):
                if not initialize:
                    self.flow.setStat(x['img'])
                    self.initialize = True
                train_step(x['img'])
            self.ckpt_manager.save()
            print('EPOCH {}: train_loss {}'.format(
                epoch, self.train_loss.result()))
            tf.summary.scalar(
                'loss[train]', self.train_loss.result(), step=epoch)
            tf.summary.scalar(
                'loss/ll', self.log_prob_loss.result(), step=epoch)
            tf.summary.scalar('loss/ldj', self.ldj_loss.result(), step=epoch)
            self.inference(epoch)
            self.train_loss.reset_states()
            self.log_prob_loss.reset_states()
            self.ldj_loss.reset_states()

    def inference(self, epoch: int = 1):
        generated = self.flow.inverse(tf.reshape(
            self.target_dist.sample(9), [9, 32, 32, 1]))[0]
        fig = plt.figure(figsize=(8, 8))
        for i in range(9):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(tf.squeeze(
                generated[i], axis=-1), aspect="auto", cmap="gray_r")
        fig.savefig('../logs/realnvp_log/generated-{:0=5}.png'.format(epoch))
        plt.close()


def main():
    task = RealNVPTask()
    task.train()


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__()
        pass

    def build(self, input_shape):
        super(CustomLayer, self).build(input_shape)
        self.s = input_shape

    def call(self, x, **kwargs):
        return x, tf.zeros([32])


def customLayerToSAVE():
    from flows.batchnorm import BatchNormalization
    print(tf.__version__)
    x = tf.keras.Input([32, 32, 1])
    layers = []
    for i in range(10):
        # layers.append(tf.keras.layers.Conv2D(3, 1))
        layers.append(
            CustomLayer()

        )
        # layers.append(tf.keras.layers.BatchNormalization())
        layers.append(BatchNormalization())
    y = x
    for layer in layers:
        y, t = layer(y)
    model = tf.keras.Model(x, y)
    optimizer = tf.optimizers.Adam()
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt.save('test')
