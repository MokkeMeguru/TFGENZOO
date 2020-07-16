import tensorflow as tf

from TFGENZOO.flows.flowbase import FlowBase


class SqueezeBase(FlowBase):
    def __init__(self, n_squeeze: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_queeze = n_squeeze

    def get_config(self):
        config = super().get_config()
        config_update = {"n_squeeze": self.n_squeeze}
        config.update(config_update)
        return config

    def build(self, input_shape):
        if input_shape[-2] % self.n_squeeze != 0:
            tf.print(
                "Invalid shape size: Timestep-size {} % {} == 0".format(
                    input_shape[-2], self.n_squeeze
                )
            )
        super().build(input_shape)


class Squeeze(SqueezeBase):
    """Squeeze Layer

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L338-L352
        https://arxiv.org/pdf/1605.08803.pdf Figure 3

    Note:

        * forward formula
            | z = reshape(x, [B, H // 2, W // 2, C * 4])

        * inverse formula
            | x = reshape(z, [B, H, W, C])

        * checkerboard spacing

                e.g.

                | [[[[1], [2], [5], [6]],
                | [[3], [4], [7], [8]],
                | [[9], [10], [13], [14]],
                | [[11], [12], [15], [16]]]]

                to

                | [[[ 1,  5],
                | [ 9, 13]]]

                | [[[ 2,  6],
                | [10, 14]]]

                | [[[ 3,  7],
                | [11, 15]]]

                | [[[ 4,  8],
                | [12, 16]]]

    """

    def __init__(self, **kwargs):
        super().__init__(n_squeeze=2, **kwargs)

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        """
        Args:
            x     (tf.Tensor): input tensor [B, H, W, C]
            zaux  (tf.Tensor): pre-latent tensor [B, H, W, C'']
        Returns:
            tf.Tensor: reshaped input tensor [B, H // 2, W // 2, C * 4]
            tf.Tensor: reshaped pre-latent tensor [B, H // 2, H // 2, C'' * 4]
        """
        x = tf.nn.space_to_depth(x, 2)
        if zaux is not None:
            zaux = tf.nn.space_to_depth(zaux, 2)
            return x, zaux
        return x

    def inverse(self, z: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        """
        Args:
            z    (tf.Tensor): input tensor [B, H // 2, W // 2, C * 4]
            zaux (tf.Tensor): pre-latent tensor [B, H // 2, W // 2, C'' * 4]
        Returns:
            tf.Tensor: reshaped input tensor [B, H // 2, W // 2, C * 4]
            tf.Tensor: reshaped pre-latent tensor [B, H // 2, W // 2, C'' * 4]
        """
        z = tf.nn.depth_to_space(z, 2)
        if zaux is not None:
            zaux = tf.nn.depth_to_space(zaux, 2)
            return z, zaux
        return z


class Squeeze2DWithMask(SqueezeBase):
    def __init__(self, n_squeeze: int = 2, **kwargs):
        super().__init__(n_squeeze=n_squeeze, **kwargs)

    def forward(
        self, x: tf.Tensor, zaux: tf.Tensor = None, mask: tf.Tensor = None, **kwargs
    ):
        """
        Args:
            x     (tf.Tensor): input tensor [B, T, C]
            zaux  (tf.Tensor): pre-latent tensor [B, T, C'']
            mask  (tf.Tensor): mask tensor [B, T, M] where M may be 1
        Returns:
            tf.Tensor: reshaped input tensor [B, T // n_squeeze, C * n_squeeze]
            tf.Tensor: reshaped pre-latent tensor [B, T // n_squeeze, C'' * n_squeeze]
            tf.Tensor: reshaped mask tensor [B, T // n_squeeze, M]
        """
        # We cannot use x.shape because TF's problem.
        # b, t, c = tf.shape(x)
        shapes = tf.shape(x)
        b = shapes[0]
        t = shapes[1]
        c = shapes[2]

        t = (t // self.n_squeeze) * self.n_squeeze
        x = x[:, :t, :]  # [B, T_round, C]

        z = tf.reshape(
            tf.reshape(x, [-1, t // self.n_squeeze, self.n_squeeze, c]),
            [-1, t // self.n_squeeze, c * self.n_squeeze],
        )

        if mask is not None:
            mask = mask[:, self.n_squeeze - 1 :: self.n_squeeze, :]
            mask = tf.cast(mask, x.dtype)
        else:
            mask = tf.ones([b, t // self.n_squeeze, 1], dtype=x.dtype)

        if zaux is not None:
            _, t, c = tf.shape(zaux)
            zaux = tf.reshape(
                tf.reshape(zaux, [-1, t // self.n_squeeze, self.n_squeeze, c]),
                [-1, t // self.n_squeeze, c * self.n_squeeze],
            )
            return z * mask, mask, zaux
        else:
            return z * mask, mask

    def inverse(
        self, z: tf.Tensor, zaux: tf.Tensor = None, mask: tf.Tensor = None, **kwargs
    ):
        """
        Args:
            z    (tf.Tensor): input tensor [B, T // n_squeeze, C * n_squeeze]
            zaux (tf.Tensor): pre-latent tensor [B, T // n_squeeze, C'' * n_squeeze]
            mask (tf.Tensor): pre-latent tensor [B, T // n_squeeze, 1]
        Returns:
            tf.Tensor: reshaped input tensor [B, T, C]
            tf.Tensor: reshaped pre-latent tensor [B, T, C'']
            tf.Tensor: mask tensor [B, T, 1]
        """
        # b, t, c = tf.shape(x)
        shapes = tf.shape(z)
        b = shapes[0]
        t = shapes[1]
        c = shapes[2]

        x = tf.reshape(
            tf.reshape(z, [-1, t, self.n_squeeze, c // self.n_squeeze]),
            [-1, t * self.n_squeeze, c // self.n_squeeze],
        )

        if mask is not None:
            mask = tf.expand_dims(mask, -1)  # [B, T, 1, 1]
            mask = tf.tile(mask, [1, 1, 1, self.n_squeeze])
            mask = tf.reshape(mask, [b, t * self.n_squeeze, 1])
            mask = tf.cast(mask, dtype=x.dtype)
        else:
            mask = tf.ones([b, t * self.n_squeeze, 1], dtype=x.dtype)

        if zaux is not None:
            _, t, c = tf.shape(zaux)
            zaux = tf.reshape(
                tf.reshape(zaux, [-1, t, self.n_squeeze, c // self.n_squeeze]),
                [-1, t * self.n_squeeze, c // self.n_squeeze],
            )
            return x * mask, mask, zaux
        else:
            return x * mask, mask
