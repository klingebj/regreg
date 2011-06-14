.. _svmtutorial:

Support vector machine tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This tutorial illustrates one version of the support vector machine, a linear
example. 
The minimization problem for the support vector machine,
following *ESL* is 

    .. math::

       \text{minimize}_{\beta,\gamma} \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{\lambda}{2} \|\beta\|^2_2

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
   solver.fit()
   solver.problem.coefs

This determines a line in the plane, specified as :math:`\beta_1 \cdot x + \beta_2 \cdot y + \gamma = 0` and the classifications are determined by which
side of the line a point is on.

.. ipython::

   fits = np.dot(X_1, problem.coefs)
   labels = 2 * (fits > 0) - 1
   accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
   accuracy

.. plot:: ./doc/examples/svm.py