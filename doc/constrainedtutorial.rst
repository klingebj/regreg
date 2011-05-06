.. _constrainedtutorial:

Constrained tutorial
~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to use RegReg to solve
constrained problems. We illustrate with an example:

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
   import pylab	
   from scipy import sparse

   from regreg.algorithms import FISTA
   from regreg.atoms import l1norm
   from regreg.seminorm import seminorm 
   from regreg.constraint import constraint
   from regreg.smooth import l2normsq

The l1norm class is used to represent the :math:`\ell_1` norm, the signal_approximator class represents the loss function and smooth_function is a container class for combining smooth functions. FISTA is a first-order algorithm and seminorm is a class for combining different seminorm penalties. 

Next, let's generate an example signal, and solve the Lagrange
form of the problem

.. ipython::
 
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = l2normsq.shift(-Y, l=0.5)

   sparsity = l1norm(len(Y), 1.4)
   # TODO should make a module to compute typical Ds
   D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
   fused = l1norm.linear(D, 25.5)
   penalty = seminorm(sparsity, fused)
   problem = loss.add_seminorm(penalty)
   solver = FISTA(problem)
   solver.fit(max_its=100, tol=1e-10)
   solution = solver.problem.coefs

We will now solve this problem in constraint form, using the 
achieved  values :math:`\delta_1 = \|D\widehat{\beta}\|_1, \delta_2\|\widehat{\beta}\|_1`.
The loss for the constraint form is specified in terms of the 
conjugate of the original loss. For the signal approximator loss, we see

.. math::

   (\frac{1}{2}\|Y-\cdot\|_2^2)^*(\alpha) = \sup_{\beta} \beta'\alpha-\frac{1}{2} \|Y-\beta\|^2_2 = \frac{1}{2} \|Y+\alpha\|^2_2

In general, one can find the conjugate by solving a minimization problem. If
:math:`{\cal L}` is not strictly convex we can add a small ridge term to 
achieve this by setting

.. math::

   {\cal L}_{\epsilon}(\beta) = {\cal L}(\beta) + \frac{\epsilon}{2} \|\beta\|^2_2

For general strongly convex :math:`{\cal L}_{\epsilon}`, a standard result of convexity yields the following formula for the gradient of :math:`{\cal L}_{\epsilon}^*` which is all that
is needed to solve the necessary dual problem

.. math::

   \newcommand{\argmin}{\mathop{\mathrm{argmin}}}
   \newcommand{\argmax}{\mathop{\mathrm{argmax}}}
   \nabla_{\eta} {\cal L}_{\epsilon}^*(\eta) = \argmax_{\beta} \left( \eta^T\beta-{\cal L}_{\epsilon}(\beta)  \right)= \argmin_{\beta} \left( {\cal L}_{\epsilon}(\beta) - \eta^T\beta \right)

This can be implemented fairly compactly:

.. literalinclude:: ../code/regreg/conjugate.py
   :pyobject: conjugate


Now, back to solving our problem.

.. ipython::

   conjugate = l2normsq.shift(Y, l=0.5)

   delta1 = np.fabs(D * solution).sum()
   delta2 = np.fabs(solution).sum()

   fused_constraint = l1norm.linear(D, delta1)
   sparsity_constraint = l1norm(500, delta2)
   
   sparse_fused = constraint(conjugate, fused_constraint, sparsity_constraint)
   constrained_solver = FISTA(sparse_fused.dual_problem())
   vals = constrained_solver.fit(max_its=1000, tol=1e-06)
   constrained_solution = sparse_fused.primal_from_dual(constrained_solver.problem.coefs)

Let's try fitting it with the generic conjugate class

.. ipython::

   from regreg.conjugate import conjugate

   loss = l2normsq.shift(-Y, l=0.5)
   generic = conjugate(loss)
   sparse_fused_gen = constraint(generic, fused_constraint, sparsity_constraint)
   constrained_solver_gen = FISTA(sparse_fused_gen.dual_problem())
   gen_vals = constrained_solver_gen.fit(max_its=1000, tol=1e-06)
   constrained_solution_gen = sparse_fused.primal_from_dual(constrained_solver_gen.problem.coefs)
   print np.linalg.norm(constrained_solution_gen - constrained_solution) / np.linalg.norm(constrained_solution)

.. plot::



   import numpy as np
   import pylab	
   from scipy import sparse

   from regreg.algorithms import FISTA
   from regreg.atoms import l1norm
   from regreg.seminorm import seminorm 
   from regreg.constraint import constraint
   from regreg.conjugate import conjugate
   from regreg.smooth import l2normsq

   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = l2normsq.shift(-Y, l=0.5)

   sparsity = l1norm(len(Y), 1.4)
   # TODO should make a module to compute typical Ds
   D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
   fused = l1norm.linear(D, 25.5)
   penalty = seminorm(sparsity, fused)
   problem = loss.add_seminorm(penalty)
   solver = FISTA(problem)
   solver.fit(max_its=1000, tol=1e-06)
   solution = solver.problem.coefs

   conjugate_loss = l2normsq.shift(Y, l=0.5)

   delta1 = np.fabs(D * solution).sum()
   delta2 = np.fabs(solution).sum()

   fused_constraint = l1norm.linear(D, delta1)
   sparsity_constraint = l1norm(500, delta2)

   sparse_fused = constraint(conjugate_loss, fused_constraint, sparsity_constraint)
   constrained_solver = FISTA(sparse_fused.dual_problem())
   constrained_solver.fit(max_its=1000, tol=1e-06)
   constrained_solution = sparse_fused.primal_from_dual(constrained_solver.problem.coefs)

   pylab.scatter(np.arange(Y.shape[0]), Y)
   pylab.plot(solution, c='r', linewidth=5)	
   pylab.plot(constrained_solution, c='black', linewidth=3)	

   # blank line, blah
   from regreg.conjugate import conjugate
   loss = l2normsq.shift(-Y, l=0.5)
   # loss.L = 1.1
   generic = conjugate(loss, epsilon=0.)
   sparse_fused_gen = constraint(generic, fused_constraint, sparsity_constraint)
   p = sparse_fused_gen.dual_problem()
   # p.L = 10. / generic.epsilon
   constrained_solver_gen = FISTA(p)
   gen_vals = constrained_solver_gen.fit(max_its=1000, tol=1e-06)

   constrained_solution_gen = sparse_fused.primal_from_dual(constrained_solver_gen.problem.coefs)
   pylab.plot(constrained_solution_gen, c='gray', linewidth=1)		
