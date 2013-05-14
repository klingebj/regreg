from copy import copy
import warnings

import numpy as np

from .identity_quadratic import identity_quadratic
from .affine import (linear_transform, identity as identity_transform)
from .objdoctemplates import objective_doc_templater
from .doctemplates import (doc_template_user, doc_template_provider)
from .atoms import atom, _work_out_conjugate
from .projl1_cython import projl1
from .piecewise_linear import find_solution_piecewise_linear_c

@objective_doc_templater()
class seminorm(atom):
    """
    An atom that can be in lagrange or bound form.
    """

    def __init__(self, primal_shape, lagrange=None, bound=None,
                 offset=None, quadratic=None, initial=None):

        atom.__init__(self, primal_shape, offset,
                      quadratic, initial)

        if not (bound is None or lagrange is None):
            raise ValueError('An atom must be either in Lagrange form or ' 
                             + 'bound form. Only one of the parameters '
                             + 'in the constructor can not be None.')
        if bound is None and lagrange is None:
            raise ValueError('Atom must be in lagrange or bound form, '
                             + 'as specified by the choice of one of'
                             + 'the keyword arguments.')
        if bound is not None and bound < 0:
            raise ValueError('Bound on the seminorm should be non-negative')
        if lagrange is not None and lagrange < 0:
            raise ValueError('Lagrange multiplier should be non-negative')

        if lagrange is not None:
            self._lagrange = lagrange
            self._bound = None
        if bound is not None:
            self._bound = bound
            self._lagrange = None

    @doc_template_provider
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        r"""
        Return :math:`\lambda \cdot %(objective)s`, where
        :math:`\lambda` is lagrange. If `check_feasibility`
        is True, and seminorm is unbounded, will return ``np.inf``
        if appropriate.

        The class atom's seminorm just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a ' + 
                             'keyword "lagrange" argument must be supplied')
        return lagrange

    @doc_template_provider
    def constraint(self, arg, bound=None):
        r"""
        Verify :math:`\cdot %(objective)s \leq \lambda`, where :math:`\lambda`
        is bound, :math:`\alpha` is self.offset (if any).

        If True, returns 0, else returns np.inf.

        The class atom's constraint just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode' 
                             + 'or a keyword "bound" argument must be supplied')
        return bound

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if self.bound is not None:
                return self.bound == other.bound
            return self.lagrange == other.lagrange
        return False

    def __copy__(self):
        return self.__class__(copy(self.primal_shape),
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset),
                              quadratic=copy(self.quadratic))

    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic.iszero:
                return "%s(%s, lagrange=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.primal_shape), 
                     self.lagrange,
                     str(self.offset))
            else:
                return "%s(%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.primal_shape), 
                     self.lagrange,
                     str(self.offset),
                     str(self.quadratic))

        else:
            if self.quadratic.iszero:
                return "%s(%s, bound=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.primal_shape), 
                     self.bound,
                     str(self.offset))

            else:
                return "%s(%s, bound=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.primal_shape), 
                     self.bound,
                     str(self.offset),
                     str(self.quadratic))

    def get_conjugate(self):
        """
        Return the conjugate of an given atom.

        >>> penalty = l1norm(30, lagrange=3.4)
        >>> penalty.get_conjugate()
        supnorm((30,), bound=3.400000, offset=0)

        """
        self.quadratic.zeroify()
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            if self.bound is None:
                cls = conjugate_seminorm_pairs[self.__class__]
                conjugate_atom = cls(self.primal_shape,  \
                           bound=self.lagrange, 
                           lagrange=None,
                           quadratic=outq,
                           offset=offset)
            else:
                cls = conjugate_seminorm_pairs[self.__class__]
                conjugate_atom = cls(self.primal_shape, \
                           lagrange=self.bound, 
                           bound=None,
                           quadratic=outq,
                           offset=offset)

        else:
            conjugate_atom = smooth_conjugate(self)
        self._conjugate = conjugate_atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    def get_lagrange(self):
        """
        Get method of the lagrange property.

        >>> penalty = l1norm(30, lagrange=3.4)
        >>> penalty.lagrange
        3.4

        """
        return self._lagrange
    def set_lagrange(self, lagrange):
        """
        Set method of the lagrange property.

        >>> penalty = l1norm(30, lagrange=3.4)
        >>> penalty.lagrange
        3.4
        >>> penalty.lagrange = 2.3
        >>> penalty.lagrange
        2.3

        """
        if self.bound is None:
            self._lagrange = lagrange
            self.conjugate._bound = lagrange
        else:
            raise AttributeError("atom is in bound mode")
    lagrange = property(get_lagrange, set_lagrange)

    def get_bound(self):
        """
        Get method of the bound property.

        >>> penalty = l1norm(30, bound=2.3)
        >>> penalty.bound
        2.3

        """
        return self._bound

    def set_bound(self, bound):
        """
        Set method of the lagrange property.

        >>> penalty = l1norm(30, bound=3.4)
        >>> penalty.bound
        3.4
        >>> penalty.bound = 2.3
        >>> penalty.bound
        2.3

        """
        if self.lagrange is None:
            self._bound = bound
            self.conjugate._lagrange = bound
        else:
            raise AttributeError("atom is in bound mode")
    bound = property(get_bound, set_bound)

    @doc_template_provider
    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.linear_transform and
        :math:`\eta` is self.linear_term.

        If the atom is in bound mode, then this has the form

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
            print 'atom: ', self
            print 'proxq: ', proxq
            print 'proxarg: ', prox_arg
            print 'totalq: ', totalq

        if self.bound is not None:
            eta = self.bound_prox(prox_arg, 
                                  lipschitz=totalq.coef, 
                                  bound=self.bound)
        else:
            eta = self.lagrange_prox(prox_arg, 
                                     lipschitz=totalq.coef, 
                                     lagrange=self.lagrange)

        if offset is None:
            return eta
        else:
            return eta - offset

    @doc_template_provider
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        r"""
        Return unique minimizer

        .. math::

           %(var)s^{\lambda}(u) =
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|u-%(var)s\|^2_2 %(linear)s %(constant)s \ 
            + \lambda   %(objective)s 

        Above, :math:`\lambda` is the Lagrange parameter,
        :math:`\alpha` is self.offset (if any), 
        :math:`\eta` is self.linear_term (if any)
        and :math:`\tau` is self.constant_term.

        If the argument `lagrange` is None and the atom is in lagrange mode,
        ``self.lagrange`` is used as the lagrange parameter, else an exception
        is raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange '
                             + 'mode or a keyword "lagrange" '
                             + 'argument must be supplied')
        return lagrange

    @doc_template_provider
    def bound_prox(self, arg, lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

           %(var)s^{\lambda}(u) \in
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|u-%(var)s\|^2_2 %(linear)s %(constant)s \ 
           \text{s.t.} \   %(objective)s \leq \lambda

        Above, :math:`\lambda` is the bound parameter, :math:`\alpha` is
        self.offset (if any), :math:`\eta` is self.linear_term (if any) and
        :math:`\tau` is self.constant_term (if any).

        If the argument `bound` is None and the atom is in bound mode,
        ``self.bound`` is used as the bound parameter, else an exception is
        raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or '
                             + 'a keyword "bound" argument must be supplied')
        return bound


    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        The nonsmooth objective function of the atom.
        Includes the quadratic term of the atom.

        Abstract method: subclasses must implement.
        """
        x_offset = self.apply_offset(arg)

        if self.bound is not None:
            if check_feasibility:
                v = self.constraint(x_offset)
            else:
                v = 0
        else:
            v = self.seminorm(x_offset, check_feasibility=check_feasibility)
        v += self.quadratic.objective(arg, 'func')
        return v


    @classmethod
    def affine(cls, linear_operator, offset, lagrange=None, diag=False,
               bound=None, quadratic=None):
        """
        This is the same as the linear class method but with offset as a positional argument
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        new_atom = cls(l.dual_shape, lagrange=lagrange, bound=bound,
                   offset=offset,
                   quadratic=quadratic)
        return affine_atom(new_atom, l)

    @classmethod
    def linear(cls, linear_operator, lagrange=None, diag=False,
               bound=None, quadratic=None, offset=None):
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        new_atom = cls(l.dual_shape, lagrange=lagrange, bound=bound,
                   quadratic=quadratic, offset=offset)
        return affine_atom(new_atom, l)


    @classmethod
    def shift(cls, offset, lagrange=None, diag=False,
              bound=None, quadratic=None):
        new_atom = cls(offset.shape, lagrange=lagrange, bound=bound,
                   quadratic=quadratic, offset=offset)
        return new_atom


@objective_doc_templater()
class l1norm(seminorm):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_1"""
    objective_vars = {'var': r'x + \alpha'}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.fabs(arg).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = np.fabs(arg).sum() <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        return np.sign(arg) * np.maximum(np.fabs(arg)-lagrange/lipschitz, 0)

    @doc_template_user
    def bound_prox(self, arg, lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        arg = np.asarray(arg, np.float)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(bound, 0, absarg)
        if cut < np.inf:
            return np.sign(arg) * (absarg - cut) * (absarg > cut)
        return arg

@objective_doc_templater()
class supnorm(seminorm):

    r"""
    The :math:`\ell_{\infty}` norm
    """

    objective_template = r"""\|%(var)s\|_{\infty}"""
    objective_vars = {'var': r'\beta + \alpha'}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.fabs(arg).max()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = np.product(np.less_equal(np.fabs(arg), bound * (1+self.tol)))
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(lagrange / lipschitz, 0, absarg)
        if cut < np.inf:
            d = np.sign(arg) * (absarg - cut) * (absarg > cut)
        else:
            d = arg
        return arg - d

    @doc_template_user
    def bound_prox(self, arg, lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        return np.clip(arg, -bound, bound)


@objective_doc_templater()
class l2norm(seminorm):

    """
    The l2 norm
    """

    objective_template = r"""\|%(var)s\|_1"""
    objective_vars = {'var': r'x + \alpha'}


    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.linalg.norm(arg)

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inball = (np.linalg.norm(arg) <= bound * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        n = np.linalg.norm(arg)
        if n <= lagrange / lipschitz:
            proj = arg
        else:
            proj = (lagrange / (lipschitz * n)) * arg
        return arg - proj

    @doc_template_user
    def bound_prox(self, arg,  lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        n = np.linalg.norm(arg)
        if n <= bound:
            return arg
        else:
            return (bound / n) * arg


def positive_part_lagrange(primal_shape, lagrange,
                           offset=None, quadratic=None, initial=None):
    r'''
    The positive_part atom in lagrange form can be represented
    by an l1norm atom with the addition of a linear term
    and half the lagrange parameter. This reflects the fact that
    :math:`[0,1]^p = [-1/2,1/2]^p + 1/2 \pmb{1}`.

    '''
    lin = np.ones(primal_shape) * .5 * lagrange
    linq = identity_quadratic(0,0,lin,0)
    if quadratic is not None:
        linq = linq + quadratic
    return l1norm(primal_shape, lagrange=0.5*lagrange,
                  offset=offset, quadratic=linq,
                  initial=initial)


class positive_part(seminorm):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """

    objective_template = r"""\sum_{i=1}^{%(shape)s} %(var)s_i^+"""
    objective_vars = {'var': r'x + \alpha',
                      'shape': seminorm._doc_dict['shape']}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.maximum(arg, 0).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inside = np.maximum(x, 0).sum() <= bound * (1 + self.tol)
        if inside:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = arg.copy()
        pos = v > 0
        v = np.atleast_1d(v)
        v[pos] = np.maximum(v[pos] - lagrange/lipschitz, 0)
        return v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg,  lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        arg = np.asarray(arg)
        v = x.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(arg.shape)


@objective_doc_templater()
class constrained_max(seminorm):
    r"""
    The seminorm x.max() s.t. x geq 0.
    """

    objective_template = (r"""\|%(var)s\|_{\infty} + \sum_{i=1}^{%(shape)s} """
                          + r"""\delta_{[0,+\infty)}(%(var)s_i)) """)
    objective_vars = {'var': r'x + \alpha',
                      'shape': seminorm._doc_dict['shape']}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(arg < 0 + self.tol)
        v = lagrange * np.max(arg)
        if not anyneg or not check_feasibility:
            return v
        return np.inf

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        anyneg = np.any(arg < 0 + self.tol)
        inside = np.max(arg) <= bound * (1 + self.tol)
        if inside and not anyneg:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        x = np.asarray(x)
        v = x.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange/lipschitz)
        return arg - v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg,  lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        return np.clip(arg, 0, bound)


@objective_doc_templater()
class constrained_positive_part(seminorm):

    r"""
    Support function of $[-\infty,1]^p$
    """

    objective_template = (r"""\|%(var)s\|_{1} + \sum_{i=1}^{%(shape)s} """
                          + r"""\delta_{[0,+\infty]}(%(var)s_i)""")
    objective_vars = {'var': r'x + \alpha',
                      'shape': seminorm._doc_dict['shape']}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(arg < 0 + self.tol)
        v = np.maximum(arg, 0).sum()
        if not anyneg or not check_feasibility:
            return v * lagrange
        return np.inf

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        value = self.seminorm(arg, lagrange=1, check_feasibility=True)
        if value >= bound * (1 + self.tol):
            return np.inf
        return 0

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = np.maximum(arg[pos] - lagrange/lipschitz, 0)
        return v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg,  lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape, np.float)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(arg.shape)


@objective_doc_templater()
class max_positive_part(seminorm):

    """
    support function of the standard simplex
    """

    objective_template = r"""\|%(var)s^+\|_{\infty}"""
    objective_vars = {'var': r'x + \alpha',
                      'shape': seminorm._doc_dict['shape']}

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return np.max(np.maximum(x,0)) * lagrange

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        v = np.max(np.maximum(arg,0))
        if v >= bound * (1 + self.tol):
            return np.inf
        return 0

    @doc_template_user
    def bound_prox(self, arg, lipschitz=1, bound=None):
        bound = seminorm.bound_prox(self, arg, lipschitz, bound)
        return np.clip(arg, -np.inf, bound)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape, np.float)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange / lipschitz)
        return arg - v.reshape(arg.shape)


conjugate_seminorm_pairs = {}
for n1, n2 in [(l1norm,supnorm),
               (l2norm,l2norm),
               (positive_part, constrained_max),
               (constrained_positive_part, max_positive_part)]:
    conjugate_seminorm_pairs[n1] = n2
    conjugate_seminorm_pairs[n2] = n1

nonpaired_atoms = [positive_part_lagrange]
