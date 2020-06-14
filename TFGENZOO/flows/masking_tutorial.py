#!/usr/bin/env python3

# this is the masking tutorial.
#
# masking is too complex problem in Flow-based Model

import tensorflow as tf

# 1. create tensor [B, T, C] where B is batch-size, T is time-steps-size, C is channel-depth
x = tf.random.normal([1, 12, 4])

# 2-1. define lengths
sequence_lengths = [10]

# 2-2. convert lengths to the mask in TF's context
mask = tf.expand_dims(tf.cast(tf.sequence_mask([10], 12), tf.float32), [-1])

# mask
# =>
# <tf.Tensor: shape=(1, 12, 1), dtype=float32, numpy=
# array([[[1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [0.]]], dtype=float32)>

# 3. apply masking
y = x * mask

# y
# =>
# <tf.Tensor: shape=(1, 12, 4), dtype=float32, numpy=
# array([[[ 0.43332544, -1.1774725 ,  0.8200461 , -1.6072778 ],
#         [ 1.0042578 ,  0.41701305,  0.5996525 , -0.31023723],
#         [-0.73475283, -0.3390424 ,  1.574429  ,  1.3448967 ],
#         [ 0.214637  ,  0.25557062,  0.98831767, -0.69492155],
#         [-0.02621194, -0.26923105, -0.37685397,  0.6796588 ],
#         [ 0.7080794 , -0.21232349,  0.8423167 ,  0.01970931],
#         [-0.72115535, -1.0133871 , -1.5454832 ,  1.9395523 ],
#         [ 0.24893883, -0.6955964 ,  0.68318325, -1.2815914 ],
#         [-0.54904926, -0.35209   , -1.0603873 , -0.45499185],
#         [-2.161542  ,  0.43661335, -1.3146498 ,  0.06375913],
#         [ 0.        , -0.        ,  0.        , -0.        ],
#         [ 0.        , -0.        ,  0.        , -0.        ]]],
#       dtype=float32)>
