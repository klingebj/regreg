from copy import copy
import warnings

from scipy import sparse
import numpy as np

from ..problems.composite import nonsmooth, smooth_conjugate
from ..affine import linear_transform, identity as identity_transform
from ..identity_quadratic import identity_quadratic
from ..smooth import affine_smooth
from ..atoms import _work_out_conjugate, atom, affine_atom
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

from .projl1_cython import projl1_epigraph

@objective_doc_templater()
class cone(atom):

    """
    A class that defines the API for cone constraints.
    """

    tol = 1.0e-05

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.shape == other.shape
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 str(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 str(self.offset),
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.shape, 
                       offset=offset,
                       quadratic=outq)
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
            self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    
    @doc_template_provider
    def constraint(self, x):
        """
        The constraint

        .. math::

           %(objective)s
        """
        raise NotImplementedError

    @doc_template_provider
    def nonsmooth_objective(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        if check_feasibility:
            v = self.constraint(x_offset)
        else:
            v = 0
        v += self.quadratic.objective(x, 'func')
        return v

    @doc_template_provider
    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator. 

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^{%(shape)s}} \frac{L}{2}
           \|x-v\|^2_2 + %(objective)s + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        debug = False
        if debug:
            print '='*80
            print 'x :', x
            print 'grad: ', grad
            print 'cone: ', self
            print 'proxq: ', proxq
            print 'proxarg: ', prox_arg
            print 'totalq: ', totalq

        eta = self.cone_prox(prox_arg)
        if offset is None:
            return eta
        else:
            return eta - offset

    @doc_template_provider
    def cone_prox(self, x):
        r"""
        Return (unique) minimizer

        .. math::

           %(var)s^{\lambda}(u) = \text{argmin}_{%(var)s \in \mathbb{R}^%(shape)s} 
           \frac{1}{2} \|%(var)s-u\|^2_2 + %(objective)s

        """
        raise NotImplementedError

    @classmethod
    def linear(cls, linear_operator, diag=False,
               offset=None,
               quadratic=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        cone = cls(l.output_shape, 
                   offset=offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

    @classmethod
    def affine(cls, linear_operator, offset, diag=False,
               quadratic=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        cone = cls(l.output_shape, 
                   offset=offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)


class affine_cone(affine_atom):

    def __repr__(self):
        return "affine_cone(%s, %s)" % (repr(self.atom),
                                        repr(self.linear_transform.linear_operator))

class nonnegative(cone):
    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """

    objective_template = r"""I^{\infty}(%(var)s \succeq 0)"""

    @doc_template_user
    def constraint(self, x):
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(x, -tol_lim))
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return np.maximum(x, 0)


class nonpositive(nonnegative):

    """
    The non-positive cone constraint (which is the support
    function of the non-negative cone constraint).
    """

    objective_template = r"""I^{\infty}(%(var)s \preceq 0)"""

    @doc_template_user
    def constraint(self, x):
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.less_equal(x, tol_lim))
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return np.minimum(x, 0)


class zero(cone):
    """
    The zero seminorm, support function of :math:\{0\}
    """

    objective_template = r"""{\cal Z}(%(var)s)"""

    @doc_template_user
    def constraint(self, x):
        return 0.

    @doc_template_user
    def cone_prox(self, x):
        return x

class zero_constraint(cone):
    """
    The zero constraint, support function of :math:`\mathbb{R}`^p
    """

    objective_template = r"""I^{\infty}(%(var)s = 0)"""

    @doc_template_user
    def constraint(self, x):
        if not np.linalg.norm(x) <= self.tol:
            return np.inf
        return 0.

    @doc_template_user
    def cone_prox(self, x):
        return np.zeros(np.asarray(x).shape)

class l2_epigraph(cone):

    """
    The l2_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_2 \leq %(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):
        
        incone = np.linalg.norm(x[1:]) / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    @doc_template_user
    def cone_prox(self, x):
        norm = x[-1]
        coef = x[:-1]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(x)
        result[:-1] = coef / norm_coef * max(norm_coef - thold, 0)
        result[-1] = max(norm + thold, 0)
        return result

class l2_epigraph_polar(cone):

    """
    The polar of the l2_epigraph constraint, which is the negative of the 
    l2 epigraph..
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_2 \in -%(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):
        incone = np.linalg.norm(x[1:]) / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    @doc_template_user
    def cone_prox(self, x):
        norm = -x[-1]
        coef = -x[:-1]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(x)
        result[:-1] = coef / norm_coef * max(norm_coef - thold, 0)
        result[-1] = max(norm + thold, 0)
        return x + result


class l1_epigraph(cone):

    """
    The l1_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_1 \leq %(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):
        incone = np.fabs(x[1:]).sum() / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    @doc_template_user
    def cone_prox(self, x):
        return projl1_epigraph(x)

class l1_epigraph_polar(cone):

    """
    The polar l1_epigraph constraint which is just the
    negative of the linf_epigraph.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{\infty} \leq - %(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):
        
        incone = np.fabs(-x[1:]).max() / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    @doc_template_user
    def cone_prox(self, x):
        return projl1_epigraph(x) - x

class linf_epigraph(cone):

    """
    The $\ell_{\nfty}$ epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{\infty} \leq %(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):

        incone = np.fabs(x[1:]).max() / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return x + projl1_epigraph(-x)

class linf_epigraph_polar(cone):

    """
    The polar linf_epigraph constraint which is just the
    negative of the l1_epigraph.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_1 \leq -%(var)s[-1])"""

    @doc_template_user
    def constraint(self, x):
        
        incone = np.fabs(-x[1:]).sum() / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    @doc_template_user
    def cone_prox(self, x):
        return -projl1_epigraph(-x)


conjugate_cone_pairs = {}
for n1, n2 in [(nonnegative,nonpositive),
               (zero, zero_constraint),
               (l1_epigraph, l1_epigraph_polar),
               (l2_epigraph, l2_epigraph_polar),
               (linf_epigraph, linf_epigraph_polar)
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
