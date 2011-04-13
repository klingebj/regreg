.. _gallery:


RegReg examples gallery
~~~~~~~



* LASSO signal approximator

    .. math::
       ||y - \beta||^{2}_{2} + \lambda_{1}||\beta||_{1}

* LASSO

    .. math::
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}


* :ref:`fusedlassoapprox`

    .. math::
       ||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1}

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

* Linear trend filtering

* Non-negative signal approximator

* :ref:`isotonic`

    .. math::
       ||y - \beta||^{2}_{2} \quad \mbox{subject to } D\beta \geq 0

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)


* :ref:`nearly-isotonic`

    .. math::
       ||y - \beta||^{2}_{2} + \lambda_{1}\|D\beta\|^+

    with

    .. math::
       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)


* Concave signal approximator

* Nearly concave signal approximator

* Group LASSO 