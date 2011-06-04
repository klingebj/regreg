.. _nearly-isotonic:


Nearly isotonic signal approximator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solves

    .. math::
       \frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}\|D\beta\|_+

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)


.. plot:: ./examples/nearlyisotonic.py

.. literalinclude:: ../examples/nearlyisotonic.py


