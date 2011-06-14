.. _elasticnettutorial:

Elastic Net tutorial
~~~~~~~~~~~~~~~~~~~~

The Elastic Net problem minimizes the objective

    .. math::
       \frac{1}{2}||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1} + \lambda_2 \|\beta\|_2^2

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np

and the RegReg classes necessary for this problem,

.. ipython::

   import regreg.api as rr

[TODO: Add some real or more interesting data.]

Next, let's generate some example data,

.. ipython::
 
   X = np.random.normal(0,1,500000).reshape((500,1000))
   Y = np.random.normal(0,1,500)

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = rr.l2normsq.affine(X,-Y, coef=0.5)
   grouping = rr.l2normsq(1000, coef=1.)
   sparsity = rr.l1norm(1000, lagrange=5.)

The penalty contains the regularization parameter that can be easily accessed and changed,

.. ipython::

   grouping.coef
   grouping.coef += 1 
   grouping.coef
   sparsity.lagrange
 

Now we can create the final problem object by comining the smooth functions and the :math:`\ell_1` seminorm,

.. ipython::

   problem = rr.container(loss, grouping, sparsity)

The penalty parameters can still be changed by accessing grouping and sparsity directly.

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = rr.FISTA(problem)
   obj_vals = solver.fit(max_its=100, tol=1e-5)
   solution = solver.composite.coefs

Here max_its represents primal iterations, and tol is the primal tolerance. 

.. ipython::

   obj_vals




