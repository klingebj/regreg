.. _svmtutorial:

Support vector machine tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This tutorial illustrates one version of the support vector machine, a linear
example. 
The minimization problem for the support vector machine,
following *ESL* is 

.. math::

       \text{minimize}_{\beta,\gamma} \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{\lambda}{2} \|\beta\|^2_2

We use the :math:`C` parameterization in (12.25) of *ESL*

.. math::

       \text{minimize}_{\beta,\gamma} C \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{1}{2} \|\beta\|^2_2

This is an example of the positive part atom combined with a smooth
quadratic penalty. Above, the :math:`x_i` are rows of a matrix of features
and the :math:`y_i` are labels coded as :math:`\pm 1`.

Let's generate some data appropriate for this problem.

.. ipython::

   import numpy as np

   np.random.seed(400) # for reproducibility
   N = 500
   P = 2

   Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
   X = np.random.standard_normal((N,P))
   X[Y==1] += np.array([3,-2])[np.newaxis,:]

We now specify the hinge loss part of the problem

.. ipython::

   import regreg.api as rr
   X_1 = np.hstack([X, np.ones((N,1))])
   transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
   C = 0.2 # = 1/\lambda
   hinge = rr.positive_part(N, lagrange=C)
   hinge_loss = rr.linear_atom(hinge, transform)

and the quadratic penalty

.. ipython::

   quadratic = rr.l2normsq.linear(rr.selector(slice(0,P), (P+1,)), coef=0.5)

Now, let's solve it

.. ipython::

   problem = rr.container(quadratic, hinge_loss)
   solver = rr.FISTA(problem)
   vals = solver.fit()
   solver.composite.coefs

This determines a line in the plane, specified as :math:`\beta_1 \cdot x + \beta_2 \cdot y + \gamma = 0` and the classifications are determined by which
side of the line a point is on.

.. ipython::

   fits = np.dot(X_1, problem.coefs)
   labels = 2 * (fits > 0) - 1
   accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
   accuracy

.. plot:: ./examples/svm.py

Sparse SVM
~~~~~~~~~~

We can also fit a sparse SVM by adding a sparsity penalty to the original problem, solving the problem

.. math::

       \text{minimize}_{\beta,\gamma} C \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{1}{2} \|\beta\|^2_2 + \lambda \|\beta\|_1

Let's generate a bigger dataset

.. ipython::

   N = 1000
   P = 200

   Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
   X = np.random.standard_normal((N,P))
   X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]

The hinge loss is defined similarly, and we only need to add a sparsity penalty

.. ipython::

   X_1 = np.hstack([X, np.ones((N,1))])
   transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
   C = 0.2
   hinge = rr.positive_part(N, lagrange=C)
   hinge_loss = rr.linear_atom(hinge, transform)

   s = rr.selector(slice(0,P), (P+1,))
   sparsity = rr.l1norm.linear(s, lagrange=0.2)
   quadratic = rr.l2normsq.linear(s, coef=0.5)

.. ipython::

   problem = rr.container(quadratic, hinge_loss, sparsity)
   solver = rr.FISTA(problem)
   solver.fit()
   solver.composite.coefs

In high dimensions, it becomes easier to separate
points.

.. ipython::

   fits = np.dot(X_1, problem.coefs)
   labels = 2 * (fits > 0) - 1
   accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
   accuracy


Sparse Huberized SVM
~~~~~~~~~~~~~~~~~~~~


We can also smooth the hinge loss to yield a Huberized version of SVM.
In fact, it is easier to write the python code to specify the problem then
to write it out formally.

The hinge loss is defined similarly, and we only need to add a sparsity penalty

.. ipython::

   X_1 = np.hstack([X, np.ones((N,1))])
   transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
   C = 0.2
   hinge = rr.positive_part(N, lagrange=C)
   hinge_loss = rr.linear_atom(hinge, transform)
   smoothed_hinge_loss = rr.smoothed_atom(hinge_loss)


   s = rr.selector(slice(0,P), (P+1,))
   sparsity = rr.l1norm.linear(s, lagrange=0.2)
   quadratic = rr.l2normsq.linear(s, coef=0.5)

Now, let's fit it. For this problem, we can use a known bound for the Lipschitz
constant. We'll first get a bound on the largest squared singular value of X

.. ipython::

   from regreg.affine import power_L
   singular_value_sq = power_L(X)
   # the other smooth piece is a quadratic with identity
   # for quadratic form, so its lipschitz constant is 1

   lipschitz = 1.05 * singular_value_sq + 1

Now, we can solve the problem without having to backtrack.

.. ipython::

   problem = rr.container(quadratic, 
                          smoothed_hinge_loss, sparsity)
   solver = rr.FISTA(problem)
   solver.composite.lipschitz = lipschitz
   vals = solver.fit(backtrack=False)
   solver.composite.coefs

In high dimensions, it becomes easier to separate
points.

.. ipython::

   fits = np.dot(X_1, problem.coefs)
   labels = 2 * (fits > 0) - 1
   accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
   accuracy
