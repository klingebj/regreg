.. _example:


Testing
~~~~~~~

.. plot::

   import pylab, numpy as np
   X = np.random.standard_normal(50)
   Y = np.random.standard_normal(50)
   pylab.scatter(X,Y)

.. ipython::

   X = 34
   Y = X + 3
   Y


.. rcode::

   X = rnorm(40)
   Y = rnorm(40)
   summary(lm(Y~X))

.. rplot::

   par(mfrow=c(2,2))
   plot(lm(Y~X))
