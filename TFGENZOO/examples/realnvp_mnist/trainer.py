"""whole model task
Architecture:
ref. https://github.com/tensorflow/models/blob/master/official/transformer/v2/transformer_main.py
"""
from TFGENZOO.examples.realnvp_mnist import model
from TFGENZOO.examples.realnvp_mnist import args
import tensorflow_probability as tfp
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import datetime

Mean = tf.metrics.Mean
Adam = tf.optimizers.Adam
tfd = tfp.distributions

class RealNVPTask(object):
    pass
