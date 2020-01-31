import tensorflow as tf
from TFGENZOO.flowbase import FlowComponent
from enum import Enum
from tensorflow.keras import layers
Layer = layers.Layer

class AffineCouplingMask(Enum):
  ChannelWise=1


class AffineCoupling(FlowComponent):
  """
  Affine Coupling Layer
  
  refs: pixyz
  https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py
  
  notes:
  - forward formula
  [x1, x2] = split(x)
  z1 = x1
  log_scale, shift = NN(x1)
  scale = sigmoid(log_scale + 2.0)
  z2 = (x2 + shift) * scale
  => z = concat([z1, z2])
  => log_det_jacobian = sum(log(scale))
  
  - inverse formula
  [z1, z2] = split(x)
  log_scale, shift = NN(x2)
  x1 = z1
  scale = sigmoid(log_scale + 2.0)
  x2 = z2 / scale - shift
  => z = concat([x1, x2])
  => inverse_log_det_jacobian = - sum(log(scale))
  
  notes:
  in Glow's Paper, scale is calculated by exp(log_scale), 
  but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
  """
  def __init__(self, 
    mask_type: AffineCouplingMask=AffineCouplingMask.ChannelWise,
    scale_shift_net: Layer=None):
    if not scale_shift_net:
      raise ValueError
    self.scale_shift_net = scale_shift_net
    self.mask_type = mask_type
    
