import numpy as np
import warnings
try:
    from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded
except ImportError:
    warnings.warn('cannot import some cholesky solvers from scipy')

from ..affine import affine_transform
from ..smooth import smooth_atom
from ..problems.composite import smooth_conjugate
from ..atoms.cones import zero
from ..identity_quadratic import identity_quadratic


class quadratic(smooth_atom):
    """
    The square of the l2 norm

    Q: array
       positive definite matrix 

    """

    objective_vars = smooth_atom.objective_vars.copy()
    objective_vars['Q'] = 'Q'
    objective_template = r"""\frac{%(coef)s}{2} \cdot %(var)s^T %(Q)s %(var)s"""

    def __init__(self, shape, coef=1., Q=None, Qdiag=False,
                 offset=None,
                 quadratic=None,
                 initial=None):
        smooth_atom.__init__(self,
                             shape,
                             coef=coef,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)
        
        self.Q = Q
        self.Qdiag = Qdiag
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, None, Qdiag)

    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, coef=%s, Q=%s, Qdiag=%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.coef),
                 repr(self.Q),
                 repr(self.Qdiag),
                 str(self.offset))
        else:
            return "%s(%s, coef=%s, Q=%s, Qdiag=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.coef),
                 repr(self.Q),
                 repr(self.Qdiag),
                 str(self.offset),
                 self.quadratic)

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


    def get_conjugate(self, factor=False, as_quadratic=False):

        if self.Q is None:
            q = identity_quadratic(self.coef, self.offset, 0, 0).collapsed()
            totalq = q + self.quadratic
            totalq_conj = totalq.conjugate.collapsed()
            if as_quadratic:
                return quadratic(self.shape, 
                                 offset=totalq_conj.linear_term/totalq_conj.coef,
                                 coef=totalq_conj.coef,
                                 quadratic=identity_quadratic(0,0,0,-totalq.constant_term))
            else:
                return smooth_conjugate(zero(self.shape,
                                             quadratic=totalq))
        else:
            sq = self.quadratic.collapsed()
            if self.offset is not None:
                sq.linear_term -= self.scale(self.Q_transform.linear_map(self.offset))
            if self.Q_transform.diagD:
                return quadratic(self.shape,
                                 Q=1./(self.coef*self.Q_transform.linear_operator + sq.coef),
                                 offset=offset,
                                 quadratic=outq, 
                                 coef=1.,
                                 Qdiag=True)
            elif factor:
                return quadratic(self.shape,
                             Q=cholesky(self.coef * self.Q + sq.coef*np.identity(self.shape)),
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
        self.input_shape = Q.shape[0]
        self.output_shape = Q.shape[0]
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

def squared_error(X, Y, coef=1):
    """
    Least squares with design $X$

    .. math::

       \frac{C}{2} \|X\beta-Y\|^2_2

    """
    atom = quadratic.affine(X, -Y, coef=coef)
    atom.atom.objective_vars['offset'] = 'Y'
    atom.atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom

def signal_approximator(signal, coef=1):
    """
    Least squares with design $I$

    .. math::

       \frac{C}{2} \|\beta-Y\|^2_2

    """
    atom = quadratic.shift(-signal, coef=coef)    
    atom.atom.objective_vars['offset'] = 'Y'
    atom.atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom


