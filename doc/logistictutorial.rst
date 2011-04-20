.. _logisticl2tutorial:

:math:`\ell_2` regularized logistic regression tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`\ell_2` regularized logistic regression problem minimizes the objective

    .. math::
       -2\left(Y^TX\beta - \sum_i \log \left[ 1 + \exp(x_i^T\beta) \right] \right) + \lambda \|\beta\|_2^2

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np
   import pylab	
   from scipy import sparse

and the RegReg classes necessary for this problem,

.. ipython::

   from regreg.algorithms import FISTA
   from regreg.smooth import logistic_loglikelihood, smooth_function, l2normsq

The l2normsq class is used to represent the :math:`\ell_2` squared norm, the logistic_loglikelihood class represents the loss function and smooth_function is a container class for combining smooth functions. FISTA is a first-order algorithm and seminorm is a class for combining different seminorm penalties. 

[TODO: Add some real or more interesting data.]

Next, let's generate some example data,

.. ipython::
 
   X = np.random.normal(0,1,500000).reshape((500,1000))
   Y = np.random.randint(0,2,500)

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = logistic_loglikelihood(X,Y)
   penalty = l2normsq(1000, l=1.)

The penalty contains the regularization parameter that can be easily accessed and changed,

.. ipython::

   penalty.l 

.. ipython::

   problem = smooth_function(loss, penalty)

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = FISTA(problem)
   obj_vals = solver.fit(max_its=100, tol=1e-5)
   solution = solver.problem.coefs

Here max_its represents primal iterations, and tol is the primal tolerance. 

.. ipython::

   obj_vals




