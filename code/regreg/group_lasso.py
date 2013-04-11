from copy import copy
import warnings

import numpy as np

from .composite import composite, nonsmooth, smooth_conjugate
from .affine import linear_transform, identity as identity_transform, selector
from .identity_quadratic import identity_quadratic
from .atoms import _work_out_conjugate, atom #, conjugate_seminorm_pairs
from .smooth import affine_smooth
from .cones import cone# , conjugate_cone_pairs

from .objdoctemplates import objective_doc_templater
from .doctemplates import (doc_template_user, doc_template_provider)

from .mixed_lasso_cython import (mixed_lasso_bound_prox,
                                 mixed_lasso_epigraph)

#@objective_doc_templater()
class group_lasso(atom):

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    """
    A class that defines the API for cone constraints.
    """
    tol = 1.0e-05

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        primal_shape = np.asarray(groups).shape
        atom.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial,
                      lagrange=lagrange,
                      bound=bound)

        self.weights = weights
        self.groups = groups
        self._groups = np.zeros(primal_shape, np.int)

        sg = sorted(np.unique(groups))
        self._weight_array = np.ones(len(sg))
        
        for i, g in enumerate(sorted(np.unique(groups))):
            self._groups[(groups == g)] = i
            self._weight_array[i] = self.weights.get(g, np.sqrt((groups == g).sum()))

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.primal_shape == other.primal_shape and 
                    np.all(self.groups == other.groups)
                    and np.all(self.weights == other.weights)
                    and self.lagrange == other.lagrange)
        return False

    def __copy__(self):
        return self.__class__(copy(self.groups),
                              lagrange=self.lagrange,
                              bound=self.bound,
                              weights=self.weights,
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic.iszero:
                return "%s(%s, lagrange=%s, weights=%s, offset=%s)" % \
                    (self.__class__.__name__,
                     `self.groups`,
                     self.lagrange,
                     `self.weights`,
                     str(self.offset))
            else:
                return "%s(%s, lagrange=%s, weights=%s, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     `self.groups`,
                     self.lagrange, 
                     `self.weights`,
                     str(self.quadratic))
        if self.bound is not None:
            if self.quadratic.iszero:
                return "%s(%s, bound=%s, weights=%s, offset=%s)" % \
                    (self.__class__.__name__,
                     `self.groups`,
                     self.bound,
                     `self.weights`,
                     str(self.offset))
            else:
                return "%s(%s, bound=%s, weights=%s, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     `self.groups`,
                     self.bound,
                     `self.weights`,
                     str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = conjugate_seminorm_pairs[self.__class__]
            if self.bound is None:
                atom = cls(self.groups,
                           weights=self.weights,
                           bound=self.lagrange, 
                           lagrange=None,
                           quadratic=outq,
                           offset=offset)
            else:
                atom = cls(self.groups,
                           weights=self.weights,
                           lagrange=self.bound,
                           bound=None,
                           quadratic=outq,
                           offset=offset)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def dual(self):
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.primal_shape)
        return self._linear_transform
    
    def latexify(self, var='x', idx=''):
        d = {}
        if self.offset is None:
            d['var'] = var
        else:
            d['var'] = var + r'+\alpha_{%s}' % str(idx)

        obj = self.objective_template % d

        if not self.quadratic.iszero:
            return ' + '.join([self.quadratic.latexify(var=var,idx=idx),obj])
        return obj

    @doc_template_provider
    def constraint(self, x, bound=None):
        r"""
        Verify :math:`\cdot %(objective)s \leq \lambda`, where :math:`\lambda`
        is bound, :math:`\alpha` is self.offset (if any).

        If True, returns 0, else returns np.inf.

        The class atom's constraint just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            raise ValueError('bound must be suppled')
        x_offset = self.apply_offset(x)
        return self.seminorm(x_offset) <= bound

    def nonsmooth_objective(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        v = self.seminorm(x_offset, check_feasibility=check_feasibility)
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, lagrange=1., check_feasibility=False):
        x_offset = self.apply_offset(x)
        value = 0
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            group = x[self._groups == i]
            value += self._weight_array[i] * np.linalg.norm(group)
        return value * lagrange

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        result = np.zeros_like(x)
        lf = self.lagrange / lipschitz
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            s = self._groups == i
            xi = x[s]
            normx = np.linalg.norm(xi)
            factor = max(1. - self._weight_array[i] * lf / normx, 0)
            result[s] = xi * factor
        return result

    @doc_template_user
    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        bound = atom.bound_prox(self, x, lipschitz, bound)
        x = np.asarray(x, np.float)

        return mixed_lasso_bound_prox(x, float(bound),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      self._groups,
                                      self._weight_array)

#@objective_doc_templater()
class group_lasso_conjugate(group_lasso):

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    """
    Conjugate of the group lasso seminorm (in bound form only for now).
    """
    tol = 1.0e-05

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        primal_shape = np.asarray(groups).shape
        atom.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial,
                      lagrange=lagrange,
                      bound=bound)

        self.weights = weights
        self.groups = groups
        self._groups = np.zeros(primal_shape, np.int)

        sg = sorted(np.unique(groups))
        self._weight_array = np.ones(len(sg))
        
        for i, g in enumerate(sorted(np.unique(groups))):
            self._groups[(groups == g)] = i
            self._weight_array[i] = self.weights.get(g, np.sqrt((groups == g).sum()))

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.primal_shape == other.primal_shape and 
                    np.all(self.groups == other.groups)
                    and np.all(self.weights == other.weights)
                    and self.bound == other.bound)
        return False

    def __copy__(self):
        return self.__class__(copy(self.groups),
                              self.bound,
                              weights=self.weights,
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, weights=%s, offset=%s)" % \
                (self.__class__.__name__,
                 self.bound,
                 `self.groups`,
                 `self.weights`,
                 str(self.offset))
        else:
            return "%s(%s, %s, weights=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 self.bound, 
                 `self.groups`,
                 `self.weights`,
                 str(self.quadratic))

    def latexify(self, var='x', idx=''):
        d = {}
        if self.offset is None:
            d['var'] = var
        else:
            d['var'] = var + r'+\alpha_{%s}' % str(idx)

        obj = self.objective_template % d

        if not self.quadratic.iszero:
            return ' + '.join([self.quadratic.latexify(var=var,idx=idx),obj])
        return obj

    @doc_template_user
    def constraint(self, x, bound=None):
        if bound is None:
            raise ValueError('bound must be suppled')
        x_offset = self.apply_offset(x)
        return self.seminorm(x_offset) <= bound

    def nonsmooth_objective(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        if check_feasibility:
            v = self.constraint(x_offset, self.bound)
        else:
            v = 0
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, lagrange=1, check_feasibility=False):

        x_offset = self.apply_offset(x)
        value = 0
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            group = x[self._groups == i]
            value = max(value, np.linalg.norm(group) / self._weight_array[i])
        return value * lagrange

    @doc_template_user
    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        The proximal operator. If the atom is in
        Bound mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        bound = atom.bound_prox(self, x, lipschitz, bound)
        result = np.zeros_like(x)
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            s = self._groups == i
            group = x[s]
            ngroup = np.linalg.norm(group)
            group = group / ngroup
            result[s] = group * min(self.bound * self._weight_array[i], ngroup)
        return result

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        r = mixed_lasso_bound_prox(x, lagrange / lipschitz,
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   self._groups,
                                   self._weight_array)
        return x - r

# the conjugate method needs to be slightly modified
class group_lasso_cone(cone):
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, weights=%s, offset=%s)" % \
                (self.__class__.__name__,
                 `self.groups`,
                 `self.weights`,
                 str(self.offset))
        else:
            return "%s(%s, weights=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 `self.groups`,
                 `self.weights`,
                 str(self.offset),
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.groups,
                       self.weights,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def weights(self):
        return self.snorm.weights

    @property
    def groups(self):
        return self.snorm.groups

class group_lasso_epigraph(group_lasso_cone):

    """
    The group LASSO epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_{G,2})})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        primal_shape = groups.shape[0]+1
        cone.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_{G,2})

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """

        return mixed_lasso_epigraph(x,
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    self.snorm._groups,
                                    self.snorm._weight_array)

#@objective_doc_templater()
class group_lasso_epigraph_polar(group_lasso_cone):

    """
    The group LASSO epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_{G,2})})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        primal_shape = groups.shape[0]+1
        cone.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

           FIX

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """

        return mixed_lasso_epigraph(x,
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    self.snorm._groups,
                                    self.snorm._weight_array) - x


#@objective_doc_templater()
class group_lasso_conjugate_epigraph(group_lasso_cone):

    """
    The group LASSO conjugate epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_{G,2,*})})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        primal_shape = groups.shape[0]+1
        cone.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso_conjugate(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_{G,2,*})

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """

        return x + mixed_lasso_epigraph(-x,
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         self.snorm._groups,
                                         self.snorm._weight_array)

#@objective_doc_templater()
class group_lasso_conjugate_epigraph_polar(group_lasso_cone):

    """
    The group LASSO conjugate epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_{G,2,*})})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        primal_shape = groups.shape[0]+1
        cone.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso_conjugate(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_{G,2,*})

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        return - mixed_lasso_epigraph(-x,
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       self.snorm._groups,
                                       self.snorm._weight_array)

conjugate_seminorm_pairs = {}
conjugate_seminorm_pairs[group_lasso_conjugate] = group_lasso
conjugate_seminorm_pairs[group_lasso] = group_lasso_conjugate

conjugate_cone_pairs = {}
conjugate_cone_pairs[group_lasso_epigraph] = group_lasso_epigraph_polar
conjugate_cone_pairs[group_lasso_conjugate_epigraph_polar] = group_lasso_conjugate_epigraph

conjugate_cone_pairs[group_lasso_conjugate_epigraph] = group_lasso_conjugate_epigraph_polar
