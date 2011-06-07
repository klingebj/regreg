.. _algorithms:

RegReg algorithms
~~~~~~~~~~~~~~~~~

This page gives an overview of the types of problems RegReg is designed to solve and discusses some of the algorithmic tools available to users.

Regularized regression
----------------------

RegReg is a framework for solving regularized regression problems such as the `LASSO <http://www-stat.stanford.edu/~tibs/lasso.html>`_

.. math::

   \frac{1}{2}||y - X\beta||^{2}_{2} + \lambda||\beta||_1

and many others. The general problem is to minimize a combination of a differentiable function and certain non-differentiable penalties or constraints. This includes many problems popular in applied statistics, such as 

* the LASSO in constraint form

    .. math::

       \frac{1}{2}||y - X\beta||^{2}_{2} \ \text{subject to} \  ||\beta||_{1} \leq \delta

* the `group LASSO <http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00532.x/full>`_

    .. math::

       \frac{1}{2}||y - X\beta||^{2}_{2}  + \lambda \sum_j \|\beta_j\|_{K_J}, \qquad \|\eta\|_{K_j} = \sqrt{\eta^T K \eta}


* :math:`\ell_2` regularized logistic regression 

    .. math::
       -2\left(Y^TX\beta - \sum_i \log \left[ 1 + \exp(x_i^T\beta) \right] \right) + \lambda \|\beta\|_2^2, \qquad Y_i \in \{0,1\}

The generic problem
^^^^^^^^^^^^^^^^^^^

RegReg is designed to solve the generic problem


.. math::
   
   \mbox{minimize}_\beta \quad \mathcal{L}(\beta) + \mathcal{P}(\beta)

where 

.. math::

   \mathcal{P}(\beta) = \sum_{i \in \mathcal{I}} \lambda_i h_{K_i}(D_i \beta+ \alpha_i)   

and the :math:`K_i` are convex sets and the :math:`h_K` are support functions

.. math::

   h_K(x) = \sup_{v \in K} v^T x.

Many popular seminorms fall into this framework, for example

* the :math:`\ell_1` norm

* the :math:`\ell_2` norm

* the element-wise sum of the positive part of a vector

Constraints related to these seminorms can also be represented as affine compositions to support functions, for example,

* :math:`\|D \beta + \alpha\|_1 \leq c`

* :math:`\|D\beta + \alpha\|_2 \leq c`

* :math:`\beta \geq 0` element-wise.

While the formulation may seem overly abstract, it allows us to use facts about convex geometry to better understand the class of problems RegReg is designed to solve [semipaper]_.


Algorithms
----------

The core RegReg algorithm is generalized gradient descent. In particular, the RegReg implementation is based on [FISTA]_. This can be used to solve many problems directly, and to solve subproblems in more complicated strategies such as solving the dual problem associated with the general problem described above

.. math::

   \mbox{minimize}_u \quad \frac{1}{2} \| z - \sum_{i \in \mathcal{I}} D_i^T u_i\|_2^2 \quad \mbox{s.t.} \quad u_i \in \lambda_i K_i

or an [ADMM]_ approach

.. math::

   \frac{1}{2}||y - \beta||^{2}_{2}  + \sum_i \lambda_i \|z_i\|_1 + \sum_i u_i^T(z_i - D_i \beta) + \frac{\rho}{2} \sum_i \|z_i - D_i\beta\|_2^2 


Optimization strategies
^^^^^^^^^^^^^^^^^^^^^^^

RegReg offers support for a variety of optimization strategies including


* Primal generalized gradient descent 

  * The current implementation is the based on the [FISTA]_ framework 
  * The primal problem is solved by solving a series of proximal problems
  * Often the proximal problems are very easy to solve. Sometimes they are more complicated - then the proximal problems can be solved via their dual, which can be solved directly by generalized gradient descent

* Dual generalized gradient ascent

  * If the conjugate of the differentiable part of the objective is known, then generalized gradient descent can be used to solve the dual problem
  * If the conjugate is not known, it can be evaluated by solving an optimization problem with gradient descent 

* ADMM 

  * The general problem can be solved directly with ADMM
  * This approach has the advantage of making it easy to parallelize many problems across features or observations
  * `IPython <http://ipython.scipy.org/moin/>`_ provides an amazingly simple and rich   `parallel computing environment <http://ipython.scipy.org/doc/rel-0.10.1/html/parallel/index.html>`_ that can be used with RegReg for distributed optimization. TODO: Add tutorial.

* Seminorm smoothing

  * The function :math:`P(\beta)` can be approximated by a smooth function
  * This approximate problem can be solved directly using simple gradient methods
  * See :ref:`smoothingtutorial`


.. [ADMM] Boyd, S., Parikh, N., Chu, E., Peleato, B., Eckstein, J. "*Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*" (http://www.stanford.edu/~boyd//papers/pdf/admm_distr_stats.pdf)
.. [semipaper] Mazumder, R., Taylor, T., Tibshirani, R.J. "*Simple, flexible and scalable algorithms for penalized regression.*" **In preparation**
.. [FISTA] Beck, A., Teboulle, M. "*A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*" (http://iew3.technion.ac.il/~becka/papers/71654.pdf) TODO: Add formal citation
