import numpy as np
from scipy import sparse
from copy import copy
import warnings

from .seminorms import seminorm as unweighted_seminorm

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import (linear_transform, identity as identity_transform, 
                     affine_transform, selector)
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)


class seminorm(unweighted_seminorm):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def __init__(self, shape, weights, lagrange=None, bound=None, 
                 offset=None, 
                 quadratic=None,
                 initial=None):

        unweighted_seminorm.__init__(self, shape,
                                 lagrange=lagrange,
                                 bound=bound,
                                 quadratic=quadratic,
                                 initial=initial,
                                 offset=offset)

        self.weights = np.asarray(weights)
        if self.weights.shape != self.shape:
            raise ValueError('weights should have same shape as shape')

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            weights_equal = np.all(np.equal(self.weights, other.weights))
            if self.bound is not None:
                return self.bound == other.bound and weights_equal
            return self.lagrange == other.lagrange and weights_equal
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              self.weights.copy(),
                              quadratic=self.quadratic,
                              initial=self.coefs,
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset))
    
    def __repr__(self):
        if self.lagrange is not None:
            if not self.quadratic.iszero:
                return "%s(%s, %s, lagrange=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     str(self.weights),
                     self.lagrange,
                     str(self.offset))
            else:
                return "%s(%s, %s, lagrange=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     str(self.weights),
                     self.lagrange,
                     str(self.offset),
                     self.quadratic)
        else:
            if not self.quadratic.iszero:
                return "%s(%s, %s, bound=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape),
                     str(self.weights),
                     self.bound,
                     str(self.offset))
            else:
                return "%s(%s, %s, bound=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape),
                     str(self.weights),
                     self.bound,
                     str(self.offset),
                     self.quadratic)


    def get_conjugate(self):
        if self.quadratic.coef == 0:
            inv_weights = 1./self.weights

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            if self.bound is None:
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.shape, 
                           inv_weights, 
                           bound=self.lagrange, 
                           lagrange=None,
                           offset=offset,
                           quadratic=outq)
            else:
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.shape,
                           inv_weights,
                           lagrange=self.bound, 
                           bound=None,
                           offset=offset,
                           quadratic=outq)
        else:
            atom = smooth_conjugate(self)

        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)
    
    def form_transform(self, subsample=False):
        '''
        By subsampling we can get rid of some variables that have 0 weights.
        '''
        if not hasattr(self, '_linear_transform'):
            if self.weights is not None:
                test = self.weights == 0
                if test.sum() and subsample:
                    self._linear_transform = selector(~test, self.shape)
                else:
                    self._linear_transform = identity_transform(self.shape)
            else:
                self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    linear_transform = property(form_transform)


@objective_doc_templater()
class l1norm(seminorm):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|W%(var)s\|_1"""

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (x != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(x[finite] * self.weights[finite]).sum()

    @doc_template_user
    def constraint(self, x, bound=None):
        bound = seminorm.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        return np.sign(x) * np.maximum(np.fabs(x)-lagrange * self.weights /lipschitz, 0)

    @doc_template_user
    def bound_prox(self, x, bound=None):
        raise NotImplementedError


@objective_doc_templater()
class supnorm(seminorm):

    r"""
    The :math:`\ell_{\infty}` norm
    """

    objective_template = r"""\|W%(var)s\|_{\infty}"""

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (x != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(x[finite] * self.weights[finite]).max()

    @doc_template_user
    def constraint(self, x, bound=None):
        bound = seminorm.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1+self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        raise NotImplementedError

    @doc_template_user
    def bound_prox(self, x, bound=None):
        bound = seminorm.bound_prox(self, x, bound)
        return np.clip(x, -bound/self.weights, bound/self.weights)


conjugate_weighted_pairs = {}
for n1, n2 in [(l1norm,supnorm)]:
    conjugate_weighted_pairs[n1] = n2
    conjugate_weighted_pairs[n2] = n1
