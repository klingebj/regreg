.. _normalizetutorial:

Using an affine transform to normalize a matrix :math:`X`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to use an affine transform to normalize a data matrix :math:`X` without actually storing the normalized matrix.

Suppose that we would like to solve the LASSO

    .. math::
       \frac{1}{2}||y - X\beta||^{2}_{2} + \lambda||\beta||_{1}

after :math:`X` has been normalized to have column mean 0 and standard deviation 1. To begin, let's create a sample matrix :math:`X`.

.. ipython::

   import numpy as np
   import regreg.api as rr

   n = 100
   p = 10
   X = np.random.standard_normal((n,p)) + np.random.standard_normal(p) + 7


We can always manually center and scale the columns of :math:`X`

.. ipython::

   Xnorm = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
   print np.mean(Xnorm, axis=0)
   print np.std(Xnorm, axis=0)

However if :math:`X` is very large we may not want to store the normalized copy. This is especially true if :math:`X` is sparse because centering the columns will likely make the matrix dense. Instead we can use the normalize affine transformation

.. ipython::

    Xnorm_rr = rr.normalize(X, center=True, scale=True) # the default


We can verify that multiplications with Xnorm_rr are done correctly

.. ipython::

    test_vec1 = np.random.standard_normal(p)
    test_vec2 = np.random.standard_normal(n)
    print np.linalg.norm( np.dot(Xnorm, test_vec1) - Xnorm_rr.linear_map(test_vec1))
    print np.linalg.norm( np.dot(Xnorm.T, test_vec2) - Xnorm_rr.adjoint_map(test_vec2))

Finally, we can solve the LASSO with both matrices and see that the solutions are the same,

.. ipython::

    Y = np.random.standard_normal(n)
    loss = rr.quadratic.affine(Xnorm, -Y, coef=0.5)
    sparsity = rr.l1norm(p, lagrange = 3.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit()
    coefs1 = solver.composite.coefs

    loss = rr.quadratic.affine(Xnorm_rr, -Y, coef=0.5)
    sparsity = rr.l1norm(p, lagrange = 3.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit()
    coefs2 = solver.composite.coefs

    print np.linalg.norm(coefs1-coefs2)