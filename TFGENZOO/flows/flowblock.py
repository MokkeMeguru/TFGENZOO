import tensorflow as tf
from flows import flows
from typing import List
Flow = flows.Flow


class FlowBlockHalf(Flow):
    """the Half blockwised Flow Layer
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

    def build(self, input_shape):
        if input_shape[-1] % 2 == 1:
            raise Exception("last dimention's size must be even")

    def call(self, x: tf.Tensor, **kargs):
        x_1, x_2 = tf.split(x, 2, axis=-1)
        z_1, z_1_log_det_jacobian = flows[0](x_1, **kargs)
        z_2, z_2_log_det_jacobian = flows[1](x_2, **kargs)
        z = tf.concat([z_1, z_2], axis=-1)
        log_det_jacobian = z_1_log_det_jacobian + z_2_log_det_jacobian
        self.assert_tensor(x, z)
        self.assert_log_det_jacobian(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kargs):
        z_1, z_2 = tf.split(z, 2, axis=-1)
        x_1, x_1_inverse_log_det_jacobian = flows[0].inverse(z_1, **kargs)
        x_2, x_2_inverse_log_det_jacobian = flows[1].inverse(z_2, **kargs)
        x = tf.concat([x_1, x_2], axis=-1)
        inverse_log_det_jacobian = x_1_inverse_log_det_jacobian + x_2_inverse_log_det_jacobian
        self.assert_tensor(z, x)
        self.assert_log_det_jacobian(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian
