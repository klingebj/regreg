"""
This module contains the implementation of block norms, i.e.
l1/l*, linf/l* norms. These are used in multiresponse LASSOs.

"""

from md5 import md5
import numpy as np
from projl1 import projl1
import atoms

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
        self.primal_shape = primal_shape
        self.atom = atom_cls(primal_shape[1:], lagrange=1.,
                             bound=bound,
                             offset=offset,
                             constant_term=constant_term)

    
    def seminorms(self, x, lagrange=None, check_feasibility=False):
        value = np.empty(self.primal_shape[0])
        for i in range(self.primal_shape[0]):
            value[i] = self.atom.seminorm(x[i], lagrange=lagrange,
                                          check_feasibility=False)
        return value

    def seminorm(self, x, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, X, lagrange=lagrange,
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
        #HMM... requires a little thought
        pass 

class block_max(block_sum):

    objective_template = r"""\|%(var)s\|_{\infty,h}"""
    _doc_dict = copy(block_sum._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atoms.atom.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.max(self.seminorms(x,  
                                                lagrange=1.,
                                                check_feasibility=check_feasibility))

    def constraint(self, x, bound=None):
        bound = atoms.atom.constraint(self, X, bound=bound)
        # XXX should we check feasibility here?
        v = np.max(self.seminorms(x, lagrange=1., check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf
                    
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        #HMM... requires a little thought
        pass

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atoms.atom.bound_prox(self, x, lipschitz=lipschitz, 
                                      bound=bound)
        v = np.empty(x.shape)
        for i in xrange(self.primal_shape[0]):
            v[i] = self.atom.bound_prox(x[i], lipschitz=lipschitz,
                                        bound=bound)
        return v




