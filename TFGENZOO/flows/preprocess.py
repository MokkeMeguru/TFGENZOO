import numpy
import tensorflow as tf
import tensorflow_probability as tfp

from TFGENZOO.flows.flowbase import FlowComponent

 from typing import List

class PreProcess(FlowComponent):
    def __init__(self, args):
        super(PreProcess, self).__init__()
