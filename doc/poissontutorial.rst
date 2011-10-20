.. _poissontutorial:

Poisson regression tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Poisson regression problem minimizes the objective

    .. math::
       -2 \left(Y^TX\beta - \sum_{i=1}^n \mbox{exp}(x_i^T\beta) \right), \qquad Y_i \in {0,1,2,\ldots}

which corresponds to the usual Poisson regression model

   .. math::
       P(Y_i=j) = \frac{\mbox{exp}(jx_i^T\beta-\mbox{exp}(x_i^T\beta))}{j!}

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np

and the RegReg classes necessary for this problem,

.. ipython::

   import regreg.api as rr

The only code needed to add Poisson regression is a class
with one method which computes the objective and its gradient.

.. literalinclude:: ../code/regreg/smooth.py
   :pyobject: poisson_loglikelihood
   

[TODO: Add some real or more interesting data.]

Next, let's generate some example data,

.. ipython::
 
   n = 1000
   p = 50
   X = np.random.standard_normal((n,p))
   Y = np.random.randint(0,100,n)

Now we can create the problem object, beginning with the loss function

.. ipython::

   loss = rr.poisson_loglikelihood.linear(X, counts=Y)

Next, we can fit this model in the usual way

.. ipython::

   problem = rr.container(loss)
   solver = rr.FISTA(problem)
   obj_vals = solver.fit()
   solution = solver.composite.coefs

