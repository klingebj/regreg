.. _fused-lasso:

===========
Fused LASSO
===========

The signal approximator problem minimizes the following
as a function of :math:`\beta`

.. math::

   \frac{1}{2}\|y - \beta\|^{2}_{2}  + \lambda_1 \|D\beta\|_1

It does this by solving the dual problem, which minimizes
the following as a function of *u*

.. math::

   \frac{1}{2}\|y - D'u\|^{2}_{2}  \ \ \text{s.t.} \ \ \|u\|_{\infty}
   \leq  \lambda_1

The one-dimensional fused LASSO is a method that
fits a piecewise constant function to data. Applications
include change-point models, CGH analysis, etc.

.. plot:: users/plots/fused_lasso1.py

We can add a sparsity penalty.

.. plot:: users/plots/fused_lasso2.py

The matrix can be changed to favor piecewise linear functions.

.. plot:: users/plots/fused_lasso3.py

Again, we can add a sparsity penalty.

.. plot:: users/plots/fused_lasso4.py

