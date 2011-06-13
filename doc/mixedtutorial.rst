.. _mixedtutorial:

Mixing seminorms tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to use RegReg to solve problems that have seminorms in both the objective and the contraint. We illustrate with an example:

.. math::

       \frac{1}{2}||y - \beta||^{2}_{2} + \lambda \|\beta\|_1 \text{ subject to} \  ||D\beta||_{1} \leq \delta   

with

.. math::

       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np
   import pylab	
   from scipy import sparse
   import regreg.api as R

The l1norm class is used to represent the :math:`\ell_1` norm, the signal_approximator class represents the loss function and smooth_function is a container class for combining smooth functions. FISTA is a first-order algorithm and container is a class for combining different seminorm penalties. 

Next, let's generate an example signal, and solve the Lagrange form of the problem

.. ipython::
 
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = R.l2normsq.shift(-Y, coef=0.5)

   sparsity = R.l1norm(len(Y), lagrange=1.4)
   # TODO should make a module to compute typical Ds
   D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
   fused = R.l1norm.linear(D, lagrange=25.5)
   problem = R.container(loss, sparsity, fused)
   
   solver = R.FISTA(problem.composite())
   solver.fit(max_its=100, tol=1e-10)
   solution = solver.composite.coefs

We will now solve this problem in constraint form, using the 
achieved  value :math:`\delta = \|D\widehat{\beta}\|_1.
By default, the container class will try to solve this problem with the two-loop strategy.

.. ipython::

   delta = np.fabs(D * solution).sum()
   sparsity = R.l1norm(len(Y), lagrange=1.4)
   fused_constraint = R.l1norm.linear(D, bound=delta)
   constrained_problem = R.container(loss, fused_constraint, sparsity)
   constrained_solver = R.FISTA(constrained_problem.composite())
   constrained_solver.composite.lipshitz = 1.01
   vals = constrained_solver.fit(max_its=10, tol=1e-06, backtrack=False, monotonicity_restart=False)
   constrained_solution = constrained_solver.composite.coefs


We can now check that the obtained value matches the constraint,

.. ipython::

   constrained_delta = np.fabs(D * constrained_solution).sum()
   print delta, constrained_delta

We can also solve this using the conjugate function :math:`\mathcal{L}_\epsilon^*`

.. ipython::

   loss = R.l2normsq.shift(-Y, coef=0.5)
   true_conjugate = R.l2normsq.shift(Y, coef=0.5, constant=-np.linalg.norm(Y)**2/2.)
   problem = R.container(loss, fused_constraint, sparsity)
   solver = R.FISTA(problem.conjugate_composite(true_conjugate))
   solver.fit(max_its=200, tol=1e-08)
   conjugate_coefs = problem.conjugate_primal_from_dual(solver.composite.coefs)

Let's also solve this with the generic constraint class, which is called by default when conjugate_problem is called without an argument

.. ipython::

   loss = R.l2normsq.shift(-Y, coef=0.5)
   problem = R.container(loss, fused_constraint, sparsity)
   solver = R.FISTA(problem.conjugate_composite())
   solver.fit(max_its=200, tol=1e-08)
   conjugate_coefs_gen = problem.conjugate_primal_from_dual(solver.composite.coefs)


   print np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)
   print np.linalg.norm(solution - conjugate_coefs_gen) / np.linalg.norm(solution)
   print np.linalg.norm(conjugate_coefs - conjugate_coefs_gen) / np.linalg.norm(conjugate_coefs)


.. plot:: ./examples/mixedtutorial.py

