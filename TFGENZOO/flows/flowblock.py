import tensorflow as tf
from TFGENZOO.flows import flows
from TFGENZOO.flows import identity
from typing import List
FlowAbst = flows.FlowAbst
Flow = flows.Flow
Identity = identity.Identity


class FlowBlockHalf(Flow):
    """the Half blockwised Flow Layer
    TODO: Flow or FlowAbst
    formula:
    [x_1, x_2] = split(x)
    z = [f(x_1), f(x_2)]
    where
    (data) x -> z (latent space)
    x is tf.Tensor[batch_size, ... , C]
    x_1 and x_2 are tf.Tensor[batch_size, ..., C // 2]
    C is even number

    >> flow=FlowBlockHalf([flow_1, flow_2])
    >> z, log_det = flow(x)
    where
    z is [flow_1(x_1), flow_2(x_2)]
    log_det is log|det J_{f_1}| + log|det J_{f_2}|
    >> x, inverse_log_det = flow.inverse(z)
    where
    x is [flow_1(z_1), flow_2(z_2)]
    inverse_log_det is log|det J_{f_1}^{-1}| + log|det J_{f_2}^{-1}|
    """

    def __init__(self, flows: List[Flow], with_debug: bool = True, **kargs):
        """initialization the Half blockwised Flow Layer
        Args:
        - flows: List[Flow]
        - with_debug: bool, take some assertion
        """
        super(FlowBlockHalf, self).__init__(with_debug=with_debug)
        assert len(flows) == 2, "this model can has only two Flow Layer"
        self.flows = flows

    def build(self, input_shape):
        if input_shape[-1] % 2 != 0:
            raise Exception("last dimention's size must be even")

    def call(self, x: tf.Tensor, **kwargs):
        """Half blockwised Flow Layer
        Args:
        - x: tf.Tensor
        input data
        Returns:
        - z: tf.Tensor
        output latent
        - log_det_jacobian: tf.Tensor
        formula:
        x_1, x_2 = split(x)
        z_1, log_det_jacobian_1 = flow_1(x_1)
        z_2, log_det_jacobian_2 = flow_2(x_2)
        z = concat(z_1, z_2)
        log_det_jacobian = log_det_jacobian_1 + log_det_jacobian_2
        where
        x in [H, W, C]
        x_1, x_2 in [H, W, C // 2]
        """
        x_1, x_2 = tf.split(x, 2, axis=-1)
        z_1, z_1_log_det_jacobian = self.flows[0](x_1, **kwargs)
        z_2, z_2_log_det_jacobian = self.flows[1](x_2, **kwargs)
        z = tf.concat([z_1, z_2], axis=-1)
        log_det_jacobian = z_1_log_det_jacobian + z_2_log_det_jacobian
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        """De Half blockwised Flow Layer
        Args:
        - z: tf.Tensor
        input latent
        Returns:
        - x: tf.Tensor
        output data
        - ildj: tf.Tensor
        inverse log_det_jacobian
        formula:
        z_1, x_2 = split(x)
        x_1, ildj_1 = flow_1(z_1)
        x_2, ildj_2 = flow_2(z_2)
        x = concat(x_1, x_2)
        ildj = ildj_1 + ildj_2
        where
        z in [H, W, C]
        z_1, z_2 in [H, W, C // 2]
        """
        z_1, z_2 = tf.split(z, 2, axis=-1)
        x_1, x_1_inverse_log_det_jacobian = self.flows[0].inverse(z_1, **kwargs)
        x_2, x_2_inverse_log_det_jacobian = self.flows[1].inverse(z_2, **kwargs)
        x = tf.concat([x_1, x_2], axis=-1)
        inverse_log_det_jacobian = (
            x_1_inverse_log_det_jacobian + x_2_inverse_log_det_jacobian)
        self.assert_tensor(z, x)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian

    def setStat(self, x: tf.Tensor, **kwargs):
        x_1, x_2 = tf.split(x, 2, axis=-1)
        if callable(getattr(self.flows[0], 'setStat', None)):
            self.flows[0].setStat(x_1)
        if callable(getattr(self.flows[1], 'setStat', None)):
            self.flows[1].setStat(x_2)
