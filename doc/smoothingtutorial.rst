.. _smoothingtutorial:

Smoothing the seminorm tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates the :ref:`fusedlassoapprox` problem
and the use of smoothing the seminorm as in :ref:`NESTA`.

The sparse fused lasso minimizes the objective

    .. math::
       \frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta\|_1

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

To solve this problem using RegReg we begin by loading the necessary numerical libraries. Much of this follows the :ref:`fusedlassoapprox` tutorial, so
we will skip some comments.


.. ipython::

   import numpy as np
   import pylab	
   from scipy import sparse

   import regreg.api as R

   # generate the data

   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = R.l2normsq.shift(-Y,coef=1)
   sparsity = R.l1norm(len(Y), lagrange=1.8)

   # fused
   D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
   D
   D = sparse.csr_matrix(D)
   fused = R.l1norm.linear(D, lagrange=25.5)


The penalty can be smoothed to create a 
smooth_function object which can be solved with FISTA.

.. ipython::

   smoothed_sparsity = R.smoothed_atom(sparsity, epsilon=0.01)
   smoothed_fused = R.smoothed_atom(fused, epsilon=0.01)

The smoothing is defined as (Yosida regularization?)

.. math::

   \begin{aligned}
   h^{\epsilon}_{K}(D\beta+\alpha) &= \sup_{u \in K} u^T(D\beta+\alpha) - \frac{\epsilon}{2}\|u\|^2_2 \\
   &= \epsilon \left(\|(D\beta+\alpha)/\epsilon\|^2_2 - \|(D\beta+\alpha)/\epsilon-P_K((D\beta+\alpha)/\epsilon)\|^2_2\right)
   \end{aligned}

with gradient

.. math::

   \nabla_{\beta} h^{\epsilon}_{K}(D\beta+\alpha) = D^TP_K((D\beta+\alpha)/\epsilon)

Finally, we can create the final problem object,

.. ipython::

   problem = R.smooth_function(loss, smoothed_sparsity, smoothed_fused)
   solver = R.FISTA(problem)
   _ip.magic('time solver.fit()')

which has both the loss function and the seminorm represented in it. 
We will estimate :math:`\beta` for various values of :math:`epsilon`

.. ipython::

   for eps in [.5**i for i in range(15)]:
       smoothed_fused.epsilon = smoothed_sparsity = eps
       solver.fit()

We can then plot solution to see the result of the regression,

.. plot:: ./examples/smoothingtutorial.py

