.. _constrainedtutorial:

Constrained tutorial
~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to use RegReg to solve problems using the container class. We illustrate with an example:

.. math::

       \frac{1}{2}||y - \beta||^{2}_{2} \ \text{subject to} \  ||D\beta||_{1} \leq \delta_1,   \|\beta\|_1 \leq \delta_2

This problem is solved by solving a dual problem, following the 
general derivation in the TFOCS paper

.. math::

       \frac{1}{2}||y - D^Tu_1 - u_2||^{2}_{2} + \delta_1 \|u_1\|_{\infty} + \delta_2 \|u_2\|_{\infty}

For a general loss function, the general objective has the form

.. math::

    {\cal L}_{\epsilon}(\beta) \ \text{subject to} \  ||D\beta||_{1} \leq \delta_1,   \|\beta\|_1 \leq \delta_2

which is solved by minimizing the dual

.. math::

    {\cal L}^*_{\epsilon}(-D^Tu_1-u_2) + \delta_1 \|u_1\|_{\infty} + \delta_2 \|u_2\|_{\infty}


Recall that for the sparse fused LASSO

.. math::

       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np
   from scipy import sparse
   import regreg.api as rr

Next, let's generate an example signal, and solve the Lagrange
form of the problem

.. ipython::
 
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = rr.quadratic.shift(-Y, coef=0.5)

   sparsity = rr.l1norm(len(Y), 1.4)
   # TODO should make a module to compute typical Ds
   D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
   fused = rr.l1norm.linear(D, 25.5)
   problem = rr.container(loss, sparsity, fused)
   
   solver = rr.FISTA(problem)
   solver.fit(max_its=100)
   solution = problem.coefs

We will now solve this problem in constraint form, using the 
achieved  values :math:`\delta_1 = \|D\widehat{\beta}\|_1, \delta_2=\|\widehat{\beta}\|_1`.
By default, the container class will try to solve this problem with the two-loop strategy.

.. ipython::

   delta1 = np.fabs(D * solution).sum()
   delta2 = np.fabs(solution).sum()
   fused_constraint = rr.l1norm.linear(D, bound=delta1)
   sparsity_constraint = rr.l1norm(500, bound=delta2)
   constrained_problem = rr.container(loss, fused_constraint, sparsity_constraint)
   constrained_solver = rr.FISTA(constrained_problem)
   constrained_solver.composite.lipschitz = 1.01
   vals = constrained_solver.fit(max_its=10, tol=1e-06, backtrack=False, monotonicity_restart=False)
   constrained_solution = constrained_solver.composite.coefs


We can also solve this problem approximately by smoothing one or more of the constraints with the smoothed_atom method. The smoothed constraint is then treated as a differentiable function which can be faster in some problems.

.. ipython::

   smoothed_fused_constraint = rr.smoothed_atom(fused_constraint, epsilon=1e-2)
   smoothed_constrained_problem = rr.container(loss, smoothed_fused_constraint, sparsity_constraint)
   smoothed_constrained_solver = rr.FISTA(smoothed_constrained_problem)
   vals = smoothed_constrained_solver.fit(tol=1e-06)
   smoothed_constrained_solution = smoothed_constrained_solver.composite.coefs
   print np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)
   print np.linalg.norm(solution - smoothed_constrained_solution) / np.linalg.norm(solution)

.. plot:: ./examples/constrainedtutorial.py

