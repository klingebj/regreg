import numpy as np
from scipy import sparse
from composite import composite, nonsmooth
from affine import (linear_transform, identity as identity_transform, 
                    affine_transform, selector)
from copy import copy
import warnings
from atoms import atom as unweighted_atom

try:
    from projl1_cython import projl1
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1

class atom(unweighted_atom):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def __init__(self, primal_shape, weights, lagrange=None, bound=None, 
                 linear_term=None,
                 constant_term=0., offset=None):

        unweighted_atom.__init__(self, primal_shape,
                                 lagrange=lagrange,
                                 bound=bound,
                                 linear_term=linear_term,
                                 constant_term=constant_term,
                                 offset=offset)

        self.weights = np.asarray(weights)
        if self.weights.shape != self.primal_shape:
            raise ValueError('weights should have same shape as primal_shape')

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            weights_equal = np.all(np.equal(self.weights, other.weights))
            if self.bound is not None:
                return self.bound == other.bound and weights_equal
            return self.lagrange == other.lagrange and weights_equal
        return False

    def __copy__(self):
        return self.__class__(copy(self.primal_shape),
                              self.weights.copy(),
                              linear_term=copy(self.linear_term),
                              constant_term=copy(self.constant_term),
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset))
    
    def __repr__(self):
        if self.lagrange is not None:
            return "%s(%s, %s, lagrange=%f, linear_term=%s, offset=%s, constant_term=%f)" % \
                (self.__class__.__name__,
                 `self.primal_shape`, 
                 str(self.weights),
                 self.lagrange,
                 str(self.linear_term),
                 str(self.offset),
                 self.constant_term)

        else:
            return "%s(%s, %s, bound=%f, linear_term=%s, offset=%s, constant_term=%f)" % \
                (self.__class__.__name__,
                 `self.primal_shape`,
                 str(self.weights),
                 self.bound,
                 str(self.linear_term),
                 str(self.offset),
                 self.constant_term)
    
    def get_conjugate(self):
        if not hasattr(self, "_conjugate"):
            inv_weights = 1./self.weights
            if self.offset is not None:
                linear_term = -self.offset
            else:
                linear_term = None
            if self.linear_term is not None:
                offset = -self.linear_term
            else:
                offset = None
            if self.bound is None:
                cls = conjugate_seminorm_pairs[self.__class__]
                atom = cls(self.primal_shape, 
                           inv_weights, 
                           bound=self.lagrange, 
                           lagrange=None,
                           linear_term=linear_term,
                           offset=offset)
            else:
                cls = conjugate_seminorm_pairs[self.__class__]
                atom = cls(self.primal_shape,
                           inv_weights,
                           lagrange=self.bound, 
                           bound=None,
                           linear_term=linear_term,
                           offset=offset)

            if offset is not None and linear_term is not None:
                _constant_term = (linear_term * offset).sum()
            else:
                _constant_term = 0.
            atom.constant_term = self.constant_term - _constant_term
            self._conjugate = atom
            self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)
    
    @property
    def linear_transform(self):
        if not hasattr(self, '_linear_transform'):
            if self.weights is not None:
                test = self.weights == 0
                if test.sum():
                    self._linear_transform = selector(~test, self.primal_shape)
                else:
                    self._linear_transform = identity_transform(self.primal_shape)
            else:
                self._linear_transform = identity_transform(self.primal_shape)
        return self._linear_transform

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    
class l1norm(atom):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_1"""
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (x != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(x[finite] * self.weights[finite]).sum()                
    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        return np.sign(x) * np.maximum(np.fabs(x)-lagrange * self.weights /lipschitz, 0)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x, lipschitz=1, bound=None):
        raise NotImplementedError
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class supnorm(atom):

    """
    The :math:`\ell_{\infty}` norm
    """

    objective_template = r"""\|%(var)s\|_{\infty}"""
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'\beta + \alpha'}


    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (x != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(x[finite] * self.weights[finite]).max()
            
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1+self.tol)
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        raise NotImplementedError
    
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        return np.clip(x, -bound/self.weights, bound/self.weights)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict


conjugate_seminorm_pairs = {}
for n1, n2 in [(l1norm,supnorm)# ,
               # (l2norm,l2norm),
               # (positive_part, constrained_max),
               # (constrained_positive_part, max_positive_part)
               ]:
    conjugate_seminorm_pairs[n1] = n2
    conjugate_seminorm_pairs[n2] = n1
