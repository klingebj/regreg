.. _basispursuit_tutorial:

Basis pursuit tutorial
~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we demonstrate how to solve the basis pursuit problem
via a smoothing approach as in TFOCS.
The basis pursuit problem is

.. math::

   \text{minimize}_{\beta: \|y-X\beta\| \leq \lambda} \|\beta\|_1

Let's generate some data first, setting the first 100 coefficients
to be large.

.. ipython::


   import numpy as np
   import scipy.linalg

   X = np.random.standard_normal((500,1000))

   beta = np.zeros(1000)
   beta[:100] = 3 * np.sqrt(2 * np.log(1000))

   Y = np.random.standard_normal((500,)) + np.dot(X, beta)

   # Later, we will need this for a Lipschitz constant
   Xnorm = scipy.linalg.eigvalsh(np.dot(X.T,X), eigvals=(998,999)).max()

The approach in TFOCS is to smooth the :math:`\ell_1` objective
yielding a dual problem

.. math::

   \text{minimize}_{u} \left(\|\beta\|_1 + \frac{\epsilon}{2} \|\beta\|^2_2 \right)^* \biggl|_{\beta=-X'u} + y'u + \lambda \|u\|_2

Above, :math:`f^*` denotes the convex conjugate. In this case,
it is a smoothed version of the unit :math:`\ell_{\infty}` ball constraint,
as its conjugate is the :math:`\ell_1` norm. Suppose
we want to minimize the :math:`\ell_1` norm achieving
an explanation of 90\% of the norm of *Y*. That is,

.. math::

   \|Y - X\beta\|^2_2 \leq 0.1 \cdot \|Y\|^2_2

The code to construct the loss function looks like this

.. ipython::

   import regreg.api as R
   from regreg.smooth import linear
   smooth_l1 = R.smoothed_constraint(R.l1norm(1000, lagrange=1).dual_atom, epsilon=0.01,
   	                                  store_argmin=True)
   loss = R.affine_smooth(smooth_l1, -X.T, None)
   smooth_f = R.smooth_function(loss, linear(Y))

We store the argmin above in *smoothed_constraint* because
it will actually be our primal solution. The penalty is specified as

.. ipython::

   norm_Y = np.linalg.norm(Y)
   l2_constraint_value = np.sqrt(0.1) * norm_Y
   l2_lagrange = R.l2norm(500, lagrange=l2_constraint_value)

The container puts these together, then solves the problem by
decreasing the smoothing.

.. ipython::

   basis_pursuit = R.container(smooth_f, l2_lagrange)
   solver = R.FISTA(basis_pursuit.problem(initial=np.random.standard_normal(500)))
   tol = 1.0e-08

   for epsilon in [0.6**i for i in range(20)]:
       smooth_l1.epsilon = epsilon
       solver.problem.L = 1.1/epsilon * Xnorm
       h = solver.fit(max_its=2000, tol=tol, min_its=10, backtrack=False)

   basis_pursuit_soln = smooth_l1.argmin

The solution should explain about 90% of the norm of *Y*

.. ipython::

   print 1 - (np.linalg.norm(Y-np.dot(X, basis_pursuit_soln)) / norm_Y)**2



Now, let's solve the corresponding bound form of the LASSO and verify
we obtain the same solution.

.. ipython::

   sparsity = R.l1norm(1000, lagrange=np.fabs(basis_pursuit_soln).sum(), constraint=True)
   loss = R.l2normsq.affine(X, -Y)
   lasso = R.container(loss, sparsity)
   lasso_solver = R.FISTA(lasso.problem())
   h = lasso_solver.fit(max_its=2000, tol=1.0e-10)
   lasso_soln = lasso_solver.problem.coefs

   print np.fabs(lasso_soln).sum(), np.fabs(basis_pursuit_soln).sum()
   print np.linalg.norm(Y-np.dot(X, lasso_soln)), np.linalg.norm(Y-np.dot(X, basis_pursuit_soln))


.. plot::

   import numpy as np
   import scipy.linalg
   import pylab

   X = np.random.standard_normal((500,1000))

   beta = np.zeros(1000)
   beta[:100] = 3 * np.sqrt(2 * np.log(1000))

   Y = np.random.standard_normal((500,)) + np.dot(X, beta)
   Xnorm = scipy.linalg.eigvalsh(np.dot(X.T,X), eigvals=(998,999)).max()

   import regreg.api as R
   from regreg.smooth import linear
   smooth_l1 = R.smoothed_constraint(R.l1norm(1000, lagrange=1).dual_atom, epsilon=0.01,
   	                                  store_argmin=True)
   loss = R.affine_smooth(smooth_l1, -X.T, None)
   smooth_f = R.smooth_function(loss, linear(Y))


   norm_Y = np.linalg.norm(Y)
   l2_constraint_value = np.sqrt(0.1) * norm_Y
   l2_lagrange = R.l2norm(500, lagrange=l2_constraint_value)

   basis_pursuit = R.container(smooth_f, l2_lagrange)
   solver = R.FISTA(basis_pursuit.problem(initial=np.random.standard_normal(500)))
   tol = 1.0e-08

   solver = R.FISTA(basis_pursuit.problem(initial=np.random.standard_normal(500)))
   for epsilon in [0.6**i for i in range(20)]:
       smooth_l1.epsilon = epsilon
       solver.problem.L = 1.1/epsilon * Xnorm
       solver.fit(max_its=2000, tol=tol, min_its=10, backtrack=False)

   basis_pursuit_soln = smooth_l1.argmin

   sparsity = R.l1norm(1000, lagrange=np.fabs(basis_pursuit_soln).sum(), constraint=True)
   loss = R.l2normsq.affine(X, -Y)
   lasso = R.container(loss, sparsity)
   lasso_solver = R.FISTA(lasso.problem())
   lasso_solver.fit(max_its=2000, tol=1.0e-10)
   lasso_soln = lasso_solver.problem.coefs

   pylab.plot(basis_pursuit_soln, label='Basis pursuit')
   pylab.plot(lasso_soln, label='LASSO')
   pylab.legend()
