"""
This module contains the implementation of block norms, i.e.
l1/l*, linf/l* norms. These are used in multiresponse LASSOs.

"""

import warnings

import numpy as np
import seminorms
from ..identity_quadratic import identity_quadratic
from ..problems.composite import smooth_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

@objective_doc_templater()
class block_sum(seminorms.seminorm):

    _doc_dict = {'linear':r' + \text{Tr}(\eta^T X)',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':r'p \times q',
                 'var':r'X'}

    objective_template = r"""\|%(var)s\|_{1,h}"""
    objective_vars = {'var': r'X + A'}

    def __init__(self, atom_cls, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        seminorms.seminorm.__init__(self,
                            shape,
                            quadratic=quadratic,
                            offset=offset,
                            initial=initial,
                            lagrange=lagrange,
                            bound=bound)

        self.atom = atom_cls(shape[1:], lagrange=lagrange,
                             bound=bound,
                             offset=None,
                             quadratic=quadratic)

    def seminorms(self, x, lagrange=None, check_feasibility=False):
        value = np.empty(self.shape[0])
        for i in range(self.shape[0]):
            value[i] = self.atom.seminorm(x[i], lagrange=lagrange,
                                          check_feasibility=False)
        return value

    def seminorm(self, x, check_feasibility=False,
                 lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.sum( \
            self.seminorms(x, check_feasibility=check_feasibility,
                           lagrange=1.))

    def constraint(self, x):
        # XXX should we check feasibility here?
        x = x.reshape(self.shape)
        v = np.sum(self.seminorms(x, check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        v = np.empty(x.shape)
        for i in xrange(self.shape[0]):
            v[i] = self.atom.lagrange_prox(x[i], lipschitz=lipschitz,
                                           lagrange=lagrange)
        return v

    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        warnings.warn('bound_prox of block_sum requires a little thought -- should be like l1prox')
        return 0 * x

    def get_lagrange(self):
        return self.atom.lagrange
    def set_lagrange(self, lagrange):
        self.atom.lagrange = lagrange
    lagrange = property(get_lagrange, set_lagrange)

    def get_bound(self):
        return self.atom.bound
    def set_bound(self, bound):
        self.atom.bound = bound
    bound = property(get_bound, set_bound)

    @property
    def conjugate(self):

        if self.quadratic.coef == 0:
            if self.offset is not None:
                if self.quadratic.linear_term is not None:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term + (self.offset * self.quadratic.linear_term).sum())
                else:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term)
            else:
                outq = identity_quadratic(0, 0, 0, -self.quadratic.constant_term)
            if self.quadratic.linear_term is not None:
                offset = -self.quadratic.linear_term
            else:
                offset = None

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate
            atom_cls = conj_atom.__class__

            atom = cls(atom_cls, 
                       self.shape, 
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate


@objective_doc_templater()
class block_max(block_sum):

    objective_template = r"""\|%(var)s\|_{\infty,h}"""
    objective_vars = {'var': r'X + A'}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.max(self.seminorms(x,  
                                                lagrange=1.,
                                                check_feasibility=check_feasibility))

    def constraint(self, x, bound=None):
        x = x.reshape(self.shape)
        bound = seminorms.seminorm.constraint(self, x, bound=bound)
        # XXX should we check feasibility here?
        v = np.max(self.seminorms(x, lagrange=1., check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        warnings.warn('lagrange_prox of block_max requires a little thought -- should be like l1prox')
        return 0 * x

    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        bound = seminorms.seminorm.bound_prox(self, x,
                                      bound=bound)
        v = np.empty(x.shape)
        for i in xrange(self.shape[0]):
            v[i] = self.atom.bound_prox(x[i], 
                                        bound=bound)
        return v


@objective_doc_templater()
class linf_l2(block_max):

    objective_template = r"""\|%(var)s\|_{\infty,2}"""
    objective_vars = {'var': r'X + A'}

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_max.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    def constraint(self, x):
        x = x.reshape(self.shape)
        norm_max = np.sqrt((x**2).sum(1)).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.sqrt((x**2).sum(1)).max()
        return lagrange * norm_max

    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        norm = np.sqrt((x**2).sum(1))
        bound = seminorms.seminorm.bound_prox(self, x,
                                      bound=bound)
        v = x.copy()
        v[norm >= bound] *= bound / norm[norm >= bound][:,np.newaxis]
        return v

    @property
    def conjugate(self):

        if self.quadratic.coef == 0:
            if self.offset is not None:
                if self.quadratic.linear_term is not None:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term + (self.offset * self.quadratic.linear_term).sum())
                else:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term)
            else:
                outq = identity_quadratic(0, 0, 0, -self.quadratic.constant_term)
            if self.quadratic.linear_term is not None:
                offset = -self.quadratic.linear_term
            else:
                offset = None

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound,
                       quadratic=outq)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate


@objective_doc_templater()
class linf_linf(linf_l2):

    objective_template = r"""\|%(var)s\|_{\infty,\infty}"""
    objective_vars = {'var': r'X + A'}

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_max.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    def constraint(self, x):
        x = x.reshape(self.shape)
        norm_max = np.fabs(x).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.fabs(x).max()
        return lagrange * norm_max


    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        bound = seminorms.seminorm.bound_prox(self, x,
                                      bound=bound)
        # print 'bound', bound
        return np.clip(x, -bound, bound)


@objective_doc_templater()
class l1_l2(block_sum):

    objective_template = r"""\|%(var)s\|_{1,2}"""
    objective_vars = {'var': r'X + A'}

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_sum.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)


    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        norm = np.sqrt((x**2).sum(1))
        mult = np.maximum(norm - lagrange / lipschitz, 0) / norm
        return x * mult[:, np.newaxis]

    @property
    def conjugate(self):

        if self.quadratic.coef == 0:
            if self.offset is not None:
                if self.quadratic.linear_term is not None:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term + (self.offset * self.quadratic.linear_term).sum())
                else:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term)
            else:
                outq = identity_quadratic(0, 0, 0, -self.quadratic.constant_term)
            if self.quadratic.linear_term is not None:
                offset = -self.quadratic.linear_term
            else:
                offset = None

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound,
                       quadratic=outq)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    def constraint(self, x):
        x = x.reshape(self.shape)
        norm_sum = np.sqrt((x**2).sum(1)).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_sum = np.sum(np.sqrt((x**2).sum(1)))
        return lagrange * norm_sum


@objective_doc_templater()
class l1_l1(l1_l2):

    objective_template = r"""\|%(var)s\|_{1,1}"""
    objective_vars = {'var': r'X + A'}

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_sum.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        norm = np.fabs(x)
        return np.maximum(norm - lagrange, 0) * np.sign(x)

    def constraint(self, x):
        x = x.reshape(self.shape)
        norm_sum = np.fabs(x).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
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
