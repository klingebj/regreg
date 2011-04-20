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

   from regreg.algorithms import FISTA
   from regreg.atoms import l1norm
   from regreg.seminorm import seminorm
   from regreg.smooth import squaredloss, smooth_function, l2normsq

The l2normsq class is used to represent the :math:`\ell_2` squared norm, the l1norm class is used to represent the :math:`\ell_1` norm and thesquaredloss class represents the loss function. The classes seminorm and smooth_function are containers for combining functions. FISTA is a first-order algorithm and seminorm is a class for combining different seminorm penalties. 

[TODO: Add some real or more interesting data.]

Next, let's generate some example data,

.. ipython::
 
   X = np.random.normal(0,1,500000).reshape((500,1000))
   Y = np.random.normal(0,1,500)

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = squaredloss(X,Y)
   grouping = l2normsq(1000, l=1.)
   sparsity = l1norm(1000, l=5.)

The penalty contains the regularization parameter that can be easily accessed and changed,

.. ipython::

   grouping.l 
   grouping.l += 1 
   grouping.l 
   sparsity.l
 

Now we can create the final problem object by comining the smooth functions and the :math:`\ell_1` seminorm,

.. ipython::

   problem = smooth_function(loss, grouping).add_seminorm(seminorm(sparsity))

The penalty parameters can still be changed by accessing grouping and sparsity directly.

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = FISTA(problem)
   obj_vals = solver.fit(max_its=100, tol=1e-5)
   solution = solver.problem.coefs

Here max_its represents primal iterations, and tol is the primal tolerance. 

.. ipython::

   obj_vals




