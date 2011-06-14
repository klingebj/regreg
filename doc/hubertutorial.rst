.. _hubertutorial:

Huberized lasso tutorial
~~~~~~~~~~~~~~~~~~~~~~~~

The Huberized lasso minimizes the following objective

    .. math::
	H_\delta(Y - X\beta) + \lambda \|\beta\|_1

where :math:`H(\cdot)` is a function applied element-wise,

    .. math::
        H_\delta(r) = \left\{\begin{array}{ll} r^2/2 & \mbox{ if } |r| \leq \delta \\ \delta r - \delta^2/2 & \mbox{ else}\end{array} \right.

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
   Y = np.random.randint(0,2,500)

Now we can create the problem object, beginning with the loss function

.. ipython::

   penalty = rr.l1norm(1000,lagrange=5.)
   loss = rr.smoothed_atom(l1norm.affine(X, -Y), epsilon=1.)

The penalty contains the regularization parameter that can be easily accessed and changed,

.. ipython::

   penalty.lagrange

Now we can create the final problem object

.. ipython::

   problem = rr.container(loss, penalty)

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = rr.FISTA(problem)
   obj_vals = solver.fit(max_its=200, tol=1e-6)
   solution = solver.composite.coefs

Here max_its represents primal iterations, and tol is the primal tolerance. 

.. ipython::

   obj_vals




