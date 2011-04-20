.. _algorithms:

RegReg algorithms
~~~~~~~~~~~~~~~~~

RegReg is designed to solve the generic problem 

.. math::
   
   \mbox{minimize}_\beta \quad \mathcal{L}(\beta) + \mathcal{P}(\beta)

where 

.. math::

   \mathcal{P}(\beta) = \sum_{i \in \mathcal{I}} h_{K_i}(D^T_i \beta)   

via the dual problem

.. math::

   \mbox{minimize}_u \quad \frac{1}{2} \| z - \sum_{i \in \mathcal{I}} D_i u_i\|_2^2 \quad \mbox{s.t.} \quad u_i \in \lambda K_i.

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


.. [semipaper] Mazumder, R., Taylor, T., Tibshirani, R.J. "*Simple, flexible and scalable algorithms for penalized regression.*" **In preparation**
.. [FISTA] Beck, A., Teboulle, M. "*A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*" (http://iew3.technion.ac.il/~becka/papers/71654.pdf) TODO: Add formal citation
