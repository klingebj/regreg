.. _fusedapproxtutorial:

Sparse fused lasso tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates the :ref:`fusedlassoapprox` problem.

The sparse fused lasso minimizes the objective

    .. math::

       \frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta\|_1

    with

    .. math::

       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np
   import pylab	
   from scipy import sparse

and the RegReg classes necessary for this problem,

.. ipython::

   from regreg.algorithms import FISTA
   from regreg.atoms import l1norm
   from regreg.container import container
   from regreg.smooth import signal_approximator, smooth_function

The l1norm class is used to represent the :math:`\ell_1` norm, the signal_approximator class represents the loss function and smooth_function is a container class for combining smooth functions. FISTA is a first-order algorithm and container is a class for combining different seminorm penalties. 

Next, let's generate an example signal,

.. ipython::
 
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

which looks like

.. plot::

   import numpy as np
   import pylab
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   pylab.scatter(np.arange(Y.shape[0]), Y)

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = smooth_function(signal_approximator(Y))

there are other loss functions (squared error, logistic, etc) and any differentiable function can be specified. Next, we specifiy the seminorm for this problem by instantiating two l1norm objects,

.. ipython::

   sparsity = l1norm(len(Y), 0.8)

which creates an l1norm object with :math:`\lambda_2=0.8`. The first argument specifies the length of the coefficient vector. The object sparsity now has a coefficient associated with it that we can access and change,

.. ipython::

   sparsity.l
   sparsity.l += 1
   sparsity.l

Next, we create the fused lasso matrix and the associated l1norm object,

.. ipython::

   D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
   D
   D = sparse.csr_matrix(D)
   fused = l1norm.linear(D, 25.5)

Here we first created D, converted it a sparse matrix, and then created an l1norm object with the sparse version of D and :math:`\lambda_1 = 25.5`. We can now combine the two l1norm objects and the loss function using the  container class

.. ipython::

   problem = container(loss, sparsity, fused)

We could still easily access the penalty parameter

.. ipython::
   
   problem.atoms
   problem.atoms[0].l

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = FISTA(problem.problem())
   solver.fit(max_its=100, tol=1e-10)
   solution = solver.problem.coefs

Here max_its represents primal (outer) iterations, and tol is the primal tolerance. 

We can then plot solution to see the result of the regression,

.. plot::

   import numpy as np
   import pylab	
   from scipy import sparse
   from regreg.algorithms import FISTA
   from regreg.atoms import l1norm
   from regreg.container import container
   from regreg.smooth import signal_approximator, smooth_function

   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = smooth_function(signal_approximator(Y))
   sparsity = l1norm(len(Y), l=0.8)
   sparsity.l
   sparsity.l += 1
   sparsity.l
   D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
   D = sparse.csr_matrix(D)
   fused = l1norm.linear(D, l=25.5)
   problem = container(loss, sparsity, fused)
   solver = FISTA(problem.problem())
   solver.fit(max_its=100, tol=1e-10)
   solution = solver.problem.coefs
   pylab.plot(solution, c='g', linewidth=3)	
   pylab.scatter(np.arange(Y.shape[0]), Y)


