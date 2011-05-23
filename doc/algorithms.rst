.. _algorithms:

RegReg algorithms
~~~~~~~~~~~~~~~~~

RegReg is designed to solve the generic problem 

.. math::
   
   \mbox{minimize}_\beta \quad \mathcal{L}(\beta) + \mathcal{P}(\beta)

where 

.. math::

   \mathcal{P}(\beta) = \sum_{i \in \mathcal{I}} \lambda_i h_{K_i}(D_i \beta)   

and the :math:`K_i` are closed convex sets with

.. math::

   h_K(x) = \sup_{v \in K} v^T x.

Many popular seminorms fall into this framework, for example

* the :math:`\ell_1` norm

* the :math:`\ell_2` norm

* the positive part of a vector.

The RegReg strategy is to solve this problem in a generalized gradient framework by majorizing the original objective and minimizing

.. math::

   \quad L \|z-\beta\|_2^2 + \mathcal{P}(\beta)

via the dual problem

.. math::

   \mbox{minimize}_u \quad \frac{1}{2} \| z - \sum_{i \in \mathcal{I}} D_i^T u_i\|_2^2 \quad \mbox{s.t.} \quad u_i \in \lambda_i K_i.

This strategy is described in [semipaper]_. RegReg provides several general strategies for solving this dual problem. 

* Generalized gradient descent 

  * The dual problem can be solved directly with generalized gradient methods
  * The current implementation is the based on the [FISTA]_ framework 
  * For an example, see the :ref:`tutorial`

* Block-wise descent

  * The dual can be solved by cyclically solving problems in each of the :math:`u_i`
  * Each :math:`u_i` can be obtained by generalized gradient descent
  * Alternatively, if special purpose solvers are available for the subproblem in :math:`u_i` then these can be used to solve the subproblem
  * For an example, see the [TODO: add blockwise tutorial]

* Seminorm smoothing

  * The function :math:`P(\beta)` can be approximated by a smooth function
  * This approximate problem can be solved directly using simple gradient methods
  * TODO: add smoothing tutorial


* ADMM

  * Two different versions of the problem. Here are the Lagrangians of the 
  two versions. The first is based on the substitution :math:`\beta_j=\beta`,
  i.e. each atom having its own copy of :math:`\beta`.

.. math::

   L_{\rho}(\beta, \beta_j, u_j) = {\cal L}(\beta) + \sum_j h_{K_j}(D_j\beta_j + \alpha_j) + u_j^T(\beta_j-\beta) +\frac{\rho}{2} \|\beta_j-\beta\|^2_2 

   The second involves two substitutions: :math:`\beta_j=\beta`, 
   :math:`D_j\beta_j=\nu_j`. Hence, there are more Lagrange
   multipliers

.. math::

   L_{\rho}(\beta, \beta_j, u_j, w_j) = {\cal L}(\beta) + \sum_j h_{K_j}(\nu_j + \alpha_j) + u_j^T(\beta_j-\beta) + w_j^T(D_j\beta_j-\nu_j) + \frac{\rho}{2} \|\beta_j-\beta\|^2_2 




.. [semipaper] Mazumder, R., Taylor, T., Tibshirani, R.J. "*Simple, flexible and scalable algorithms for penalized regression.*" **In preparation**
.. [FISTA] Beck, A., Teboulle, M. "*A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*" (http://iew3.technion.ac.il/~becka/papers/71654.pdf) TODO: Add formal citation
