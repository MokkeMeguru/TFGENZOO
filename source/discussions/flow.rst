================
Flow-based Model
================

Is 1x1Conv compatible with tensordot?
=====================================
1x1conv is the convolution with filter [1, 1].

So this procedure is as same as tensordot.

Experiments
------------

* `TFGENZOO_KNOWLEDGES (CAST 1x1conv to TensorDot)(colab notebook) <https://colab.research.google.com/gist/MokkeMeguru/ee7af2a1c4947e4b0305efb892daf6b2/tfgenzoo_knowledges.ipynb>`__
SomeTimes, matrix\_determinant will be crash
==========================================

Sometimes, matrix\_determinant will be 0. which is cannot apply backpropergation in InvConv1x1.

We can use decomposed matrix of W, but it consume training time more and more...

References
----------

* `Tensorflow Probability - ScaledMatvecLU <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/scale_matvec_lu.py>`__


In Qunantize, the log-det-jacobian has better formulation than current one in the view of time complexity.
==========================================================================================================

Now, the forward formula in LogitifyImage is below.

.. math::
    z_1 &= x + \epsilon \\\\
    z_2 &= (1- \alpha) z_1  + 0.5 \alpha \\\\
    z &= z_3 = \text{logit}z = \log{z_2} - \log(1 - z_2)

And then, the log det jacobian is below.

.. math::
    \text{log_det_jacobian} = \text{softplus}(z_3) + \text{softplus}(- z_3) - \text{softplus}(\log{\alpha} - \log(1 - \alpha))

But we can propose below formula to decrease the time complexity.


.. math::
    \text{log_det_jacobian} &= - \log(z_2) - \log(1 - z_2) - \text{softplus}(\log{\alpha} - \log(1 - \alpha))\\\\
        &= \text{softplus}(z_3) + \text{softplus}(- z_3) - \text{softplus}(\log{\alpha} - \log(1 - \alpha)) \\\\
        &= \log(1 + \exp(z_3)) + \log(1 + \exp(- z_3)) - t \\\\
        &= \log(1 + \exp(\log{z_2} + \log(1 - z_2))) - \log(1 + \exp(- \log{z_2} + \log(1 - z_2))) - t \\\\
        &= \log(1 + \cfrac{z_2}{1 - z_2}) + \log(1 + \cfrac{1 - z_2}{z_2}) - t \\\\
        &= \log(\cfrac{1}{1 - z_2}) + \log(\cfrac{1}{z_2})  - t\\\\
        &= - \log(z_2) - \log(1 - z_2) - t \\\\
        & ,where \ t =   \text{softplus}(\log{\alpha} - \log(1 - \alpha)) \\\\

However, if :math:`1 - z_2 < \epsilon` , :math:`\log(1 - z_2)` will cancel significant digits because :math:`\log(1 - z_2) \approx -\text{Inf}`

Besides, if :math:`z_2 < \epsilon`, :math:`\log(z_2)` will cancel significant digits because :math:`\log(z_2) \approx -\text{Inf}`

So that, we should use **softplus** instead of **log** in this case.
