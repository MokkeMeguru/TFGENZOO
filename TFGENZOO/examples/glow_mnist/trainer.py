"""
Note: does tfp serve correct distribution?
I think tfp can calculate log probability as same as PyTorch
ref:
"""
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import metrics, optimizers
from tqdm import tqdm

from TFGENZOO.examples.glow_mnist import args, model
from TFGENZOO.examples.utils import load_mnist

Mean = metrics.Mean
Adam = optimizers.Adam
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


def bits_x(log_likelihood: tf.Tensor,
           log_det_jacobian: tf.Tensor, pixels: int):
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

# def bits_per_dims(log_prob, pixels):
#     return ((log_prob / pixels) - np.log(128.)) / np.log(2.)


class Glow_trainer:
    def __init__(self, args=args.args, training=True):
        self.args = args

        print(self.args)

        # build model
        self.glow = model.BasicGlow(
            K=self.args['K'], L=self.args['L'],
            resblk_kwargs=self.args['resblk_kwargs'])

        # determin input data size
        x = tf.keras.Input(self.args['input_shape'])
        self.pixels = np.prod(self.args['input_shape'])
        self.glow.build(x.shape)
        z, ldj, zaux = self.glow(x)
        self.z_shape = list(z.shape[1:])
        self.zaux_shape = list(zaux.shape[1:])
        print ("z",  self.z_shape)
        print ("z",  self.zaux_shape)
        self.z_dims = np.prod(z.shape[1:])
        self.zaux_dims = np.prod(zaux.shape[1:])

        print("z_f's shape: ", z.shape)
        print("log det jacobian's shape: ", ldj.shape)
        print("z_aux's shape: ", zaux.shape)

        # show the model's summary
        self.glow.summary()

        # load dataset
        self.train_dataset, self.test_dataset = load_mnist.load_dataset(
            BATCH_SIZE=self.args['batch_size'])

        # load target distribution (not same as original Glow's implementation)
        self.setup_target_dist()

        # setup loss metrics
        self.train_loss = Mean(name='nll', dtype=tf.float32)
        self.valid_loss = Mean(name='nll', dtype=tf.float32)
        self.train_log_det_jacobian = Mean(
            name='log_det_jacobian', dtype=tf.float32)
        self.valid_log_det_jacobian = Mean(
            name='log_det_jacobian', dtype=tf.float32)

        # optimizer
        self.learning_rate_schedule = CustomSchedule(
                d_model=np.prod(self.args['input_shape']))
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_schedule)

        # setup checkpoint
        checkpoint_path = args.get(
            'checkpoint_path', 'checkpoints/glow')
        self.setup_checkpoint(checkpoint_path)

        if training:
            for sample in self.train_dataset.take(1):
                print('init loss (log_prob bits/dims) {}'.format(
                    tf.reduce_mean(self.val_step(sample['img']))))
            self.writer = tf.summary.create_file_writer(
                logdir="glow_log/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.writer.set_as_default()

    def setup_target_dist(self):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.ones([self.z_dims]))
        zaux_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.zaux_dims]), tf.ones([self.zaux_dims]))
        self.target_distribution = (z_distribution, zaux_distribution)

    def setup_checkpoint(self, checkpoint_path: Path):
        print("checkpoint will be saved at", checkpoint_path)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self.glow, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=3
        )
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('[Glow] Latest checkpoint restored!!!')
            for sample in self.train_dataset.take(1):
                print('restored loss (log_prob bits/dims) {}'.format(
                    tf.reduce_mean(self.val_step(sample['img']))))
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    @tf.function
    def val_step(self, x):
        tf.debugging.enable_check_numerics()
        z, log_det_jacobian, zaux = self.glow(x, training=False)
        z = tf.reshape(z, [-1, self.z_dims])
        zaux = tf.reshape(zaux, [-1, self.zaux_dims])
        lp = self.target_distribution[0].log_prob(z)
        lpaux = self.target_distribution[1].log_prob(zaux)
        loss = bits_x(lp + lpaux, log_det_jacobian, self.pixels)
        self.valid_loss(loss)
        self.valid_log_det_jacobian(log_det_jacobian)
        tf.debugging.disable_check_numerics()
        return loss

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, log_det_jacobian, zaux = self.glow(x, training=True)
            z = tf.reshape(z, [-1, self.z_dims])
            zaux = tf.reshape(zaux, [-1, self.zaux_dims])
            lp = self.target_distribution[0].log_prob(z)
            lpaux = self.target_distribution[1].log_prob(zaux)
            loss = bits_x(lp + lpaux, log_det_jacobian, self.pixels)

        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(
            zip(grads, variables)
        )
        self.train_loss(loss)
        self.train_log_det_jacobian(log_det_jacobian)

    def generate_image(self, beta_z: float = 0.2, beta_zaux: float = 0.2):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.broadcast_to(beta_z, [self.z_dims]))
        zaux_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.zaux_dims]),
            tf.broadcast_to(beta_zaux, [self.zaux_dims]))
        z = z_distribution.sample(4)
        z = tf.reshape(z, [-1] + self.z_shape)
        zaux = zaux_distribution.sample(4)
        zaux = tf.reshape(zaux, [-1] + self.zaux_shape)
        x, ildj = self.glow.inverse(z, zaux, training=False)
        with self.writer.as_default():
            print("mean", tf.reduce_mean(x),
                  "std", tf.math.reduce_std(x),
                  "max", tf.reduce_max(x),
                  "min", tf.reduce_min(x))
            tf.summary.image("generated image", x,
                             max_outputs=4,
                             step=self.optimizer.iterations)
            for x in self.test_dataset.take(1):
                tf.summary.image("reference image", x['img'][:4],
                             max_outputs=4,
                             step=self.optimizer.iterations)
                z, log_det_jacobian, zaux = self.glow(x['img'][:4],
                                                      training=False)
                x, ildj = self.glow.inverse(z, zaux, training=False)
                tf.summary.image("reversed image", x,
                             max_outputs=4,
                             step=self.optimizer.iterations)

    def train(self, beta_z=0.2, beta_zaux=0.2):
        for epoch in range(self.args['epochs']):
            for x in tqdm(self.train_dataset):
                self.train_step(x['img'])
            for x in tqdm(self.test_dataset):
                self.val_step(x['img'])
            self.generate_image(beta_z, beta_zaux)
            ckpt_save_path = self.ckpt_manager.save()
            tf.summary.scalar('train/nll',
                              self.train_loss.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('valid/nll',
                              self.valid_loss.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('train/ldj',
                              self.train_log_det_jacobian.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('valid/ldj',
                              self.valid_log_det_jacobian.result(),
                              step=self.optimizer.iterations)
            print('epoch {}: train_loss={}, val_loss={}, saved_at={}'.format(
                epoch, self.train_loss.result().numpy(),
                self.valid_loss.result().numpy(),
                ckpt_save_path))
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_log_det_jacobian.reset_states()
            self.valid_log_det_jacobian.reset_states()


def main():
    args.args['resblk_kwargs'] = None
    trainer = Glow_trainer(args=args.args)
    trainer.train()
