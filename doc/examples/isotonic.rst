.. _isotonic:


Isotonic signal approximator
~~~~~~~

Solves

    .. math::
       ||y - \beta||^{2}_{2} \quad \mbox{subject to } D\beta \geq 0

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)


.. plot:: ./examples/isotonic.py

.. literalinclude:: ../examples/isotonic.py


