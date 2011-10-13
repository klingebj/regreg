"""
This module contains the implementation of block norms, i.e.
l1/l*, linf/l* norms. These are used in multiresponse LASSOs.

"""

from md5 import md5
import numpy as np
import atoms
from copy import copy

class block_sum(atoms.atom):

    _doc_dict = {'linear':r' + \text{Tr}(\eta^T X)',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':r'p \times q',
                 'var':r'X'}

    objective_template = r"""\|%(var)s\|_{1,h}"""
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def __init__(self, atom_cls, primal_shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 linear_term=None,
                 constant_term=0.):
        self.linear_term = linear_term
        self.offset = offset
        self.constant_term = constant_term
        self.primal_shape = primal_shape
        self.atom = atom_cls(primal_shape[1:], lagrange=lagrange,
                             bound=bound,
                             offset=None,
                             linear_term=None,
                             constant_term=0.)

    def seminorms(self, x, lagrange=None, check_feasibility=False):
        value = np.empty(self.primal_shape[0])
        for i in range(self.primal_shape[0]):
            value[i] = self.atom.seminorm(x[i], lagrange=lagrange,
                                          check_feasibility=False)
        return value

    def seminorm(self, x, check_feasibility=False,
                 lagrange=None):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.sum( \
            self.seminorms(x, check_feasibility=check_feasibility,
                           lagrange=1.))

    def constraint(self, x):
        # XXX should we check feasibility here?
        v = np.sum(self.seminorms(x, check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        lagrange = atoms.atom.lagrange_prox(self, x, lipschitz, lagrange)
        v = np.empty(x.shape)
        for i in xrange(self.primal_shape[0]):
            v[i] = self.atom.lagrange_prox(x[i], lipschitz=lipschitz,
                                           lagrange=lagrange)
        return v

    def bound_prox(self, x, lipschitz=1, bound=None):
        raise NotImplementedError('requires a little thought -- should be like l1prox')

    @property
    def lagrange(self):
        return self.atom.lagrange

    @property
    def bound(self):
        return self.atom.bound

    @property
    def conjugate(self):
        if not hasattr(self, "_conjugate"):
            if self.offset is not None:
                linear_term = -self.offset
            else:
                linear_term = None
            if self.linear_term is not None:
                offset = -self.linear_term
            else:
                offset = None

            conj_atom = self.atom.conjugate
            atom_cls = conj_atom.__class__
            cls = conjugate_block_pairs[self.__class__]
            atom = cls(atom_cls, self.primal_shape, 
                       linear_term=linear_term,
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound)

            if offset is not None and linear_term is not None:
                _constant_term = (linear_term * offset).sum()
            else:
                _constant_term = 0.
            atom.constant_term = self.constant_term - _constant_term
            self._conjugate = atom
            self._conjugate._conjugate = self
        return self._conjugate


class block_max(block_sum):

    objective_template = r"""\|%(var)s\|_{\infty,h}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.max(self.seminorms(x,  
                                                lagrange=1.,
                                                check_feasibility=check_feasibility))

    def constraint(self, x, bound=None):
        bound = atoms.atom.constraint(self, x, bound=bound)
        # XXX should we check feasibility here?
        v = np.max(self.seminorms(x, lagrange=1., check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        raise NotImplementedError('requires a little thought -- should be like l1prox')

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atoms.atom.bound_prox(self, x, lipschitz=lipschitz, 
                                      bound=bound)
        v = np.empty(x.shape)
        for i in xrange(self.primal_shape[0]):
            v[i] = self.atom.bound_prox(x[i], lipschitz=lipschitz,
                                        bound=bound)
        return v

class linf_l2(block_max):
    
    objective_template = r"""\|%(var)s\|_{\infty,2}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def __init__(self, primal_shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 linear_term=None,
                 constant_term=0.):
        block_max.__init__(self, atoms.l2norm,
                           primal_shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           linear_term=linear_term,
                           constant_term=constant_term)

    def constraint(self, x):
        norm_max = np.sqrt((x**2).sum(1)).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.sqrt((x**2).sum(1)).max()
        return lagrange * norm_max

    def bound_prox(self, x, lipschitz=1, bound=None):
        norm = np.sqrt((x**2).sum(1))
        bound = atoms.atom.bound_prox(self, x, lipschitz=lipschitz, 
                                      bound=bound)
        v = x.copy()
        v[norm >= bound] *= bound / norm[norm >= bound][:,np.newaxis]
        return v

    @property
    def conjugate(self):
        if not hasattr(self, "_conjugate"):
            if self.offset is not None:
                linear_term = -self.offset
            else:
                linear_term = None
            if self.linear_term is not None:
                offset = -self.linear_term
            else:
                offset = None

            conj_atom = self.atom.conjugate
            atom_cls = conj_atom.__class__
            cls = conjugate_block_pairs[self.__class__]
            atom = cls(self.primal_shape, 
                       linear_term=linear_term,
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound)

            if offset is not None and linear_term is not None:
                _constant_term = (linear_term * offset).sum()
            else:
                _constant_term = 0.
            atom.constant_term = self.constant_term - _constant_term
            self._conjugate = atom
            self._conjugate._conjugate = self
        return self._conjugate

class linf_linf(linf_l2):
    
    objective_template = r"""\|%(var)s\|_{\infty,\infty}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def __init__(self, primal_shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 linear_term=None,
                 constant_term=0.):
        block_max.__init__(self, atoms.l2norm,
                           primal_shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           linear_term=linear_term,
                           constant_term=constant_term)

    def constraint(self, x):
        norm_max = np.fabs(x).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.fabs(x).max()
        return lagrange * norm_max


    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atoms.atom.bound_prox(self, x, lipschitz=lipschitz, 
                                      bound=bound)
        # print 'bound', bound
        return np.clip(x, -bound, bound)

class l1_l2(block_sum):
    
    objective_template = r"""\|%(var)s\|_{1,2}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}


    def __init__(self, primal_shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 linear_term=None,
                 constant_term=0.):
        block_sum.__init__(self, atoms.l2norm,
                           primal_shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           linear_term=linear_term,
                           constant_term=constant_term)

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        lagrange = atoms.atom.lagrange_prox(self, x, lipschitz, lagrange)
        norm = np.sqrt((x**2).sum(1))
        mult = np.maximum(norm - lagrange, 0) / norm
        return x * mult[:, np.newaxis]

    @property
    def conjugate(self):
        if not hasattr(self, "_conjugate"):
            if self.offset is not None:
                linear_term = -self.offset
            else:
                linear_term = None
            if self.linear_term is not None:
                offset = -self.linear_term
            else:
                offset = None

            conj_atom = self.atom.conjugate
            atom_cls = conj_atom.__class__
            cls = conjugate_block_pairs[self.__class__]
            atom = cls(self.primal_shape, 
                       linear_term=linear_term,
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound)

            if offset is not None and linear_term is not None:
                _constant_term = (linear_term * offset).sum()
            else:
                _constant_term = 0.
            atom.constant_term = self.constant_term - _constant_term
            self._conjugate = atom
            self._conjugate._conjugate = self
        return self._conjugate

    def constraint(self, x):
        norm_sum = np.sqrt((x**2).sum(1)).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_sum = np.sqrt((x**2).sum(1)).sum()
        return lagrange * norm_sum

class l1_l1(l1_l2):
    
    objective_template = r"""\|%(var)s\|_{1,1}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def __init__(self, primal_shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 linear_term=None,
                 constant_term=0.):
        block_sum.__init__(self, atoms.l2norm,
                           primal_shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           linear_term=linear_term,
                           constant_term=constant_term)

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        lagrange = atoms.atom.lagrange_prox(self, x, lipschitz, lagrange)
        norm = np.fabs(x)
        return np.maximum(norm - lagrange, 0) * np.sign(x)

    def constraint(self, x):
        norm_sum = np.fabs(x).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_sum = np.fabs(x).sum()
        return lagrange * norm_sum




conjugate_block_pairs = {}
for n1, n2 in [(block_max, block_sum),
               (l1_l2, linf_l2),
               (l1_l1, linf_linf)
               ]:
    conjugate_block_pairs[n1] = n2
    conjugate_block_pairs[n2] = n1

