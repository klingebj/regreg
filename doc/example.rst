.. _example:


Testing
~~~~~~~

.. plot::

   import pylab, numpy as np
   X = np.random.standard_normal(50)
   Y = np.random.standard_normal(50)
   pylab.scatter(X,Y)

.. plot:: ./pyplots/plotex.py
   :include-source:

.. ipython::

   X = 34
   Y = X + 3
   Y

.. literalinclude:: ../code/regreg/examples.py
   :pyobject: lasso_example
