import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded
from affine import affine_transform
from smooth import smooth_atom

class quadratic(smooth_atom):
    """
    The square of the l2 norm
    """

    def __init__(self, primal_shape, coef=1., Q=None, Qdiag=False,
                 constant_term=0,
                 offset=None,
                 linear_term=None):
        self.offset = offset
        self.linear_term = linear_term
        self.constant_term = constant_term
        self.Q = Q
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, None, Qdiag)
        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.coef = coef

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
                f, g  = self.scale(np.linalg.norm(x)**2) + self.constant_term, self.scale(2 * x)
            elif mode == 'grad':
                f, g = None, self.scale(2 * x)
            elif mode == 'func':
                f, g = self.scale(np.linalg.norm(x)**2) + self.constant_term, None
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term, self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'grad':
                f, g = None, self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'func':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term, None
            else:
                raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, x, mode)

    def get_conjugate(self, epsilon=0, factor=False):
        if self.offset is not None:
            linear_term = -self.offset
        else:
            linear_term = None
        if self.linear_term is not None:
            offset = -self.linear_term
        else:
            offset = None

        if self.Q is None:
            return quadratic(self.primal_shape, offset=offset,
                             linear_term=linear_term, coef=0.25/(self.coef+epsilon))
        elif self.Q_transform.diagD:
            return quadratic(self.primal_shape,
                             Q=1./(self.Q_transform.linear_operator + epsilon),
                             offset=offset,
                             linear_term=linear_term,
                             coef=0.25/self.coef,
                             Qdiag=True)
        elif factor:
            return quadratic(self.primal_shape,
                             Q=cholesky(self.Q_transform.linear_operator +
                                        epsilon*np.identity(self.primal_shape[0])),
                             Qdiag=False,
                             offset=offset,
                             linear_term=linear_term,
                             coef=.25/self.coef)
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
