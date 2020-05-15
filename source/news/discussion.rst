==========
Discussion
==========

Is 1x1Conv compatible with tensordot?
=====================================
1x1conv is the convolution with filter [1, 1].

So this procedure is same as tensordot.

Experiments
------------

* `TFGENZOO_KNOWLEDGES (CAST 1x1conv to TensorDot)(colab notebook) <https://colab.research.google.com/gist/MokkeMeguru/ee7af2a1c4947e4b0305efb892daf6b2/tfgenzoo_knowledges.ipynb>`__
SomeTimes, matrixdeterminant will be crash
==========================================

Sometimes, matrixdeterminant will be 0. which is cannot apply backpropergation.

We can use decomposed matrix of W, but it consume training time more and more...

References
----------

* `Tensorflow Probability - ScaledMatvecLU <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/scale_matvec_lu.py>`__
