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

   import regreg.api as rr

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

   loss = rr.quadratic.shift(-Y,coef=0.5)

there are other loss functions (squared error, logistic, etc) and any differentiable function can be specified. Next, we specifiy the seminorm for this problem by instantiating two l1norm objects,

.. ipython::

   sparsity = rr.l1norm(len(Y), lagrange=0.8)

which creates an l1norm object with :math:`\lambda_2=0.8`. The first argument specifies the length of the coefficient vector. The object sparsity now has a coefficient associated with it that we can access and change,

.. ipython::

   sparsity
   sparsity.lagrange += 1
   sparsity.lagrange

Next, we create the fused lasso matrix and the associated l1norm object,

.. ipython::

   D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
   D
   D = sparse.csr_matrix(D)
   fused = rr.l1norm.linear(D, lagrange=25.5)

Here we first created D, converted it a sparse matrix, and then created an l1norm object with the sparse version of D and :math:`\lambda_1 = 25.5`. We can now combine the two l1norm objects and the loss function using the  container class

.. ipython::

   problem = rr.container(loss, sparsity, fused)

We could still easily access the penalty parameter

.. ipython::
   
   problem.nonsmooth_atoms
   problem.nonsmooth_atoms[0].lagrange

Next, we can select our algorithm of choice and use it solve the problem,

.. ipython::

   solver = rr.FISTA(problem)
   solver.fit(max_its=100, tol=1e-10)
   solution = problem.coefs

Here max_its represents primal (outer) iterations, and tol is the primal tolerance. 

We can then plot solution to see the result of the regression,

.. plot:: ./examples/fusedtutorial.py

