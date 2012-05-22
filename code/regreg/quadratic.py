import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded

from .affine import affine_transform
from .smooth import smooth_atom
from .composite import smooth_conjugate
from .cones import zero
from .identity_quadratic import identity_quadratic

class quadratic(smooth_atom):
    """
    The square of the l2 norm

    Q: array
       positive definite matrix 


    """

    objective_template = r"""\ell^{Q}\left(%(var)s\right)"""

    def __init__(self, primal_shape, coef=1., Q=None, Qdiag=False,
                 offset=None,
                 quadratic=None,
                 initial=None):
        smooth_atom.__init__(self,
                             primal_shape,
                             coef=coef,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

        self.Q = Q
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, None, Qdiag)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        x = self.apply_offset(x)
        if self.Q is None:
            if mode == 'both':
                f, g  = self.scale(np.linalg.norm(x)**2) / 2., self.scale(x)
                return f, g
            elif mode == 'grad':
                f, g = None, self.scale(x)
                return g
            elif mode == 'func':
                f, g = self.scale(np.linalg.norm(x)**2) / 2., None
                return f
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) / 2., self.scale(self.Q_transform.linear_map(x))
                return f, g
            elif mode == 'grad':
                f, g = None, self.scale(self.Q_transform.linear_map(x))
                return g
            elif mode == 'func':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) / 2., None
                return f
            else:
                raise ValueError("mode incorrectly specified")


    def get_conjugate(self, factor=False):

        if self.Q is None:
            q = identity_quadratic(self.coef, -self.offset, 0, 0).collapsed()
            totalq = q + self.quadratic
            print 'totalq: ', totalq
            totalq_conj = totalq.conjugate.collapsed()
            return quadratic(self.primal_shape, 
                             offset=totalq_conj.linear_term/totalq_conj.coef,
                             coef=totalq_conj.coef,
                             quadratic=identity_quadratic(0,0,0,totalq_conj.constant_term-q.constant_term))
#                             quadratic=identity_quadratic(0,0,0,totalq_conj.constant_term))
        else:
            sq = self.quadratic.collapsed()
            if self.offset is not None:
                sq.linear_term += self.scale(self.Q_transform.linear_map(self.offset))
            if self.Q_transform.diagD:
                return quadratic(self.primal_shape,
                                 Q=1./(self.coef*self.Q_transform.linear_operator + sq.coef),
                                 offset=offset,
                                 quadratic=outq, 
                                 coef=1.,
                                 Qdiag=True)
            elif factor:
                return quadratic(self.primal_shape,
                             Q=cholesky(self.coef * self.Q + sq.coef*np.identity(self.primal_shape)),
                             Qdiag=False,
                             offset=offset,
                             quadratic=outq,
                             coef=1.)
            else:
                raise ValueError('factor is False, so no factorization was done')

class cholesky(object):

    '''

    Given :math:`Q > 0`, returns a linear transform
    that is multiplication by :math:`Q^{-1}` by
    first computing the Cholesky decomposition of :math:`Q`.

    Parameters
    ----------

    Q: array
       positive definite matrix 

    '''

    def __init__(self, Q, cholesky=None, banded=False):
        self.primal_shape = Q.shape[0]
        self.dual_shape = Q.shape[0]
        self.affine_offset = None
        self._Q = Q
        self.banded = banded
        if cholesky is None:
            if not self.banded:
                self._cholesky = cho_factor(Q)
            else:
                self._cholesky = cholesky_banded(Q)
        else:
            self._cholesky = cholesky

    def linear_map(self, x):
        if not self.banded:
            return cho_solve(self._cholesky, x)
        else:
            return cho_solve_banded(self._cholesky, x)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        return self.linear_map(x)
