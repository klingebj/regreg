import numpy as np
from scipy import sparse
from identity_quadratic import identity_quadratic
from composite import nonsmooth, smooth
from affine import (linear_transform, identity as identity_transform, 
                    affine_transform, selector)
from smooth import affine_smooth
from copy import copy
import warnings

try:
    from projl1_cython import projl1
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1

class atom(nonsmooth):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def __init__(self, primal_shape, lagrange=None, bound=None, 
                 linear_term=None,
                 constant_term=0., offset=None,
                 quadratic=None, initial=None):

        self.offset = offset
        self.constant_term = constant_term
        if offset is not None:
            self.offset = np.array(offset)

        self.linear_term = None
        if linear_term is not None:
            self.linear_term = np.array(linear_term)

        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.dual_shape = self.primal_shape

        if not (bound is None or lagrange is None):
            raise ValueError('An atom must be either in Lagrange form or bound form. Only one of the parameters in the constructor can not be None.')
        if bound is None and lagrange is None:
            raise ValueError('Atom must be in lagrange or bound form, as specified by the choice of one of the keyword arguments.')
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
        
        if quadratic is not None:
            self.set_quadratic(quadratic.coef, quadratic.offset,
                               quadratic.linear_term, 
                               quadratic.constant_term)

        if initial is None:
            self.coefs = np.zeros(self.primal_shape)
        else:
            self.coefs = initial.copy()

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if self.bound is not None:
                return self.bound == other.bound
            return self.lagrange == other.lagrange
        return False

    def __copy__(self):
        new_atom = self.__class__(copy(self.primal_shape),
                                  linear_term=copy(self.linear_term),
                                  constant_term=copy(self.constant_term),
                                  bound=copy(self.bound),
                                  lagrange=copy(self.lagrange),
                                  offset=copy(self.offset))
        q = self.quadratic
        if q is not None:
            new_atom.set_quadratic(q.coef, q.offset, q.linear_term, q.constant_term)
        return new_atom

    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic is None or not self.quadratic.anything_to_return:
                return "%s(%s, lagrange=%f, linear_term=%s, offset=%s, constant_term=%f)" % \
                    (self.__class__.__name__,
                     `self.primal_shape`, 
                     self.lagrange,
                     str(self.linear_term),
                     str(self.offset),
                     self.constant_term)
            else:
                return "%s(%s, lagrange=%f, linear_term=%s, offset=%s, constant_term=%f, quadratic=%s)" % \
                    (self.__class__.__name__,
                     `self.primal_shape`, 
                     self.lagrange,
                     str(self.linear_term),
                     str(self.offset),
                     self.constant_term,
                     str(self.quadratic))

        else:
            if self.quadratic is None or not self.quadratic.anything_to_return:
                return "%s(%s, bound=%f, linear_term=%s, offset=%s, constant_term=%f)" % \
                    (self.__class__.__name__,
                     `self.primal_shape`, 
                     self.bound,
                     str(self.linear_term),
                     str(self.offset),
                     self.constant_term)

            else:
                return "%s(%s, bound=%f, linear_term=%s, offset=%s, constant_term=%f, quadratic=%s)" % \
                    (self.__class__.__name__,
                     `self.primal_shape`, 
                     self.bound,
                     str(self.linear_term),
                     str(self.offset),
                     self.constant_term,
                     str(self.quadratic))
    
    def get_conjugate(self):

        if self.quadratic is None:
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
                           bound=self.lagrange, 
                           lagrange=None,
                           linear_term=linear_term,
                           offset=offset)
            else:
                cls = conjugate_seminorm_pairs[self.__class__]
                atom = cls(self.primal_shape, 
                           lagrange=self.bound, 
                           bound=None,
                           linear_term=linear_term,
                           offset=offset)

            if offset is not None and linear_term is not None:
                _constant_term = (linear_term * offset).sum()
            else:
                _constant_term = 0.
            atom.constant_term = self.constant_term - _constant_term
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)
    
    def get_lagrange(self):
        return self._lagrange
    def set_lagrange(self, lagrange):
        if self.bound is None:
            self._lagrange = lagrange
            self.conjugate._bound = lagrange
        else:
            raise AttributeError("atom is in bound mode")
    lagrange = property(get_lagrange, set_lagrange)

    def get_bound(self):
        return self._bound

    def set_bound(self, bound):
        if self.lagrange is None:
            self._bound = bound
            self.conjugate._lagrange = bound
        else:
            raise AttributeError("atom is in lagrange mode")
    bound = property(get_bound, set_bound)

    @property
    def dual(self):
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.primal_shape)
        return self._linear_transform
    
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        """
        Return :math:`\lambda \cdot %(objective)s`, where
        :math:`\lambda` is lagrange. If check_feasibility
        is True, and seminorm is unbounded, will return np.inf
        if appropriate.

        The class atom's seminorm just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return lagrange

    def constraint(self, x, bound=None):
        """
        Verify :math:`\cdot %(objective)s \leq \lambda`, where
        :math:`\lambda` is bound,
        :math:`\alpha` is self.offset (if any). 

        If True, returns 0,
        else returns np.inf.

        The class atom's constraint just returns the appropriate bound
        parameter for use by the subclasses.

        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return bound


    def nonsmooth_objective(self, x, check_feasibility=False):
        if self.offset is not None:
            x_offset = x + self.offset
        else:
            x_offset = x
        if self.bound is not None:
            if check_feasibility:
                v = self.constraint(x_offset)
            else:
                v = 0
        else:
            v = self.seminorm(x_offset, check_feasibility=check_feasibility)
        if self.linear_term is None:
            v += self.constant_term
        else:
            v += (self.linear_term * x).sum() + self.constant_term
        if self.quadratic is not None:
            v += self.quadratic.objective(x, 'func')
        return v

    def proximal(self, lipschitz, x, grad):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        If the atom is in bound mode, then this has the form
        
        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """

        proxq = identity_quadratic(lipschitz, -x, grad)
        if self.linear_term is not None:
            linearq = identity_quadratic(0, 0, self.linear_term, 0)
            proxq = proxq + linearq

        if self.quadratic is not None:
            totalq = self.quadratic + proxq
        else:
            totalq = proxq.collapsed()

        if self.offset is None or np.all(np.equal(self.offset, 0)):
            offset = None
        else:
            offset = self.offset

        if offset is not None:
            totalq.offset = -offset
            totalq = totalq.collapsed()

        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        debug = False
        if debug:
            print '='*80
            print 'x :', x
            print 'grad: ', grad
            print 'atom: ', self
            print 'proxq: ', proxq
            print 'proxarg: ', prox_arg
            print 'totalq: ', totalq

        if self.bound is not None:
            eta = self.bound_prox(prox_arg, lipschitz=totalq.coef, bound=self.bound)
        else:
            eta = self.lagrange_prox(prox_arg, lipschitz=totalq.coef, lagrange=self.lagrange)

        if offset is None:
            return eta
        else:
            return eta - offset

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
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

        If the argument lagragne is None and the atom is in lagrange mode, 
        self.lagrange is used as the lagrange parameter, 
        else an exception is raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return lagrange

    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

           %(var)s^{\lambda}(u) \in 
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|u-%(var)s\|^2_2 %(linear)s %(constant)s \ 
           \text{s.t.} \   %(objective)s \leq \lambda

        Above, :math:`\lambda` is the bound parameter,
        :math:`\alpha` is self.offset (if any), 
        :math:`\eta` is self.linear_term (if any)
        and :math:`\tau` is self.constant_term (if any).

        If the argument is bound None and the atom is in bound mode, 
        self.bound is used as the bound parameter, 
        else an exception is raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.

        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return bound

    @classmethod
    def affine(cls, linear_operator, offset, lagrange=None, diag=False,
               bound=None, linear_term=None):
        """
        This is the same as the linear class method but with offset as a positional argument
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        atom = cls(l.primal_shape, lagrange=lagrange, bound=bound,
                   linear_term=linear_term, offset=offset)
        return affine_atom(atom, l)


    @classmethod
    def linear(cls, linear_operator, lagrange=None, diag=False,
               bound=None, linear_term=None, offset=None):
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        atom = cls(l.primal_shape, lagrange=lagrange, bound=bound,
                   linear_term=linear_term, offset=offset)
        return affine_atom(atom, l)


    @classmethod
    def shift(cls, offset, lagrange=None, diag=False,
               bound=None, linear_term=None):
        atom = cls(offset.shape, lagrange=lagrange, bound=bound,
                   linear_term=linear_term, offset=offset)
        return atom

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
        '''
        conjugate_atom = copy(self.conjugate)
        sq = smoothing_quadratic
        if sq.coef in [None, 0]:
            raise ValueError('quadratic term of smoothing_quadratic must be non 0')
        total_q = sq

        if conjugate_atom.quadratic is not None:
            total_q = sq + conjugate_atom.quadratic
        conjugate_atom.set_quadratic(total_q.coef, total_q.offset,
                                     total_q.linear_term, 
                                     total_q.constant_term)
        smoothed_atom = conjugate_atom.conjugate
        return smoothed_atom

    
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
        return lagrange * np.fabs(x).sum()
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inbox = np.fabs(x).sum() <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        return np.sign(x) * np.maximum(np.fabs(x)-lagrange/lipschitz, 0)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        x = np.asarray(x, np.float)
        return projl1(x, self.bound)
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
        return lagrange * np.fabs(x).max()
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inbox = np.product(np.less_equal(np.fabs(x), bound * (1+self.tol)))
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        x = np.asarray(x, np.float)
        d = projl1(x, lagrange/lipschitz)
        return x - d
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        return np.clip(x, -bound, bound)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class l2norm(atom):

    """
    The l2 norm
    """
    
    objective_template = r"""\|%(var)s\|_1"""
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}


    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.linalg.norm(x)
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inball = (np.linalg.norm(x) <= bound * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        n = np.linalg.norm(x)
        if n <= lagrange / lipschitz:
            proj = x
        else:
            proj = (self.lagrange / (lipschitz * n)) * x
        return x - proj * (1 - l2norm.tol)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x,  lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        n = np.linalg.norm(x)
        if n <= bound:
            return x
        else:
            return (bound / n) * x
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class positive_part(atom):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """
    
    objective_template = r"""\sum_{i=1}^{%(shape)s} %(var)s_i^+"""
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha',
                                                   'shape':_doc_dict['shape']}


    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.maximum(x, 0).sum()
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        inside = np.maximum(x, 0).sum() <= bound * (1 + self.tol)
        if inside:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        x = np.asarray(x)
        v = x.copy()
        pos = v > 0
        v = np.atleast_1d(v)
        v[pos] = np.maximum(v[pos] - lagrange/lipschitz, 0)
        return v.reshape(x.shape)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x,  lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        x = np.asarray(x)
        v = x.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(x.shape)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class constrained_max(atom):
    """
    The seminorm x.max() s.t. x geq 0.
    """

    objective_template = r"""\|%(var)s\|_{\infty} + \sum_{i=1}^{%(shape)s} \delta_{[0,+\infty)}(%(var)s_i)) """
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha',
                                                   'shape':_doc_dict['shape']}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(x < 0 + self.tol)
        v = lagrange * np.max(x)
        if not anyneg or not check_feasibility:
            return v
        return np.inf
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        anyneg = np.any(x < 0 + self.tol)
        inside = np.max(x) <= bound * (1 + self.tol)
        if inside and not anyneg:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        x = np.asarray(x)
        v = x.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange/lipschitz)
        return x-v.reshape(x.shape)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x,  lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        return np.clip(x, 0, bound)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class constrained_positive_part(atom):

    """
    Support function of (-\infty,0]^p
    """

    objective_template = r"""\|%(var)s\|_{1} + \sum_{i=1}^{%(shape)s} \delta_{[0,+\infty)}(%(var)s_i)) """
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha',
                                                   'shape':_doc_dict['shape']}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(x < 0 + self.tol)
        v = np.maximum(x, 0).sum()
        if not anyneg or not check_feasibility:
            return v * lagrange
        return np.inf
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        anyneg = np.any(x < 0 + self.tol)
        v = np.maximum(x, 0).sum()
        if anyneg or v >= bound * (1 + self.tol):
            return np.inf
        return 0
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        x = np.asarray(x)
        v = np.zeros(x.shape)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = np.maximum(x[pos] - lagrange/lipschitz, 0)
        return v
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, x,  lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        x = np.asarray(x)
        v = np.zeros(x.shape, np.float)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(x.shape)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

class max_positive_part(atom):

    """
    support function of the standard simplex
    """

    objective_template = r"""\|%(var)s^+\|_{\infty}"""
    _doc_dict = copy(atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = atom.seminorm(self, x, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return np.max(np.maximum(x,0)) * lagrange
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, x, bound=None):
        bound = atom.constraint(self, x, bound=bound)
        v = np.max(np.maximum(x,0))
        if v >= bound * (1 + self.tol):
            return np.inf
        return 0
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

    def bound_prox(self, x, lipschitz=1, bound=None):
        bound = atom.bound_prox(self, x, lipschitz, bound)
        return np.clip(x, -np.inf, bound)
    bound_prox.__doc__ = atom.bound_prox.__doc__ % _doc_dict

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        lagrange = atom.lagrange_prox(self, x, lipschitz, lagrange)
        x = np.asarray(x)
        v = np.zeros(x.shape, np.float)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange / lipschitz)
        return x-v.reshape(x.shape)
    lagrange_prox.__doc__ = atom.lagrange_prox.__doc__ % _doc_dict

class affine_atom(object):

    """
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.
    :math:`\beta \mapsto h_K(\beta)`
    this class creates a new seminorm 
    that evaluates :math:`h_K(D\beta+\alpha)`

    This class does not have a prox, but its
    dual does. The prox of the dual is

    .. math::

       \text{minimize} \frac{1}{2} \|y-x\|^2_2 + x^T\alpha
       \ \text{s.t.} \ x \in \lambda K
    
    """

    def __init__(self, atom_obj, atransform):
        self.atom = copy(atom_obj)
        # if the affine transform has an offset, put it into
        # the atom. in this way, the affine_transform is actually
        # always linear
        if atransform.affine_offset is not None:
            if self.atom.offset is not None:
                self.atom.offset += atransform.affine_offset
            else:
                self.atom.offset = atransform.affine_offset
            ltransform = affine_transform(atransform.linear_operator, None,
                                          diag=atransform.diagD)
        else:
            ltransform = atransform
        self.linear_transform = ltransform
        self.primal_shape = self.linear_transform.primal_shape
        self.dual_shape = self.linear_transform.dual_shape

    def __repr__(self):
        return "affine_atom(%s, %s)" % (`self.atom`,
                                        `self.linear_transform.linear_operator`)

    @property
    def dual(self):
        tmpatom = copy(self.atom)
        tmpatom.primal_shape = tmpatom.dual_shape = self.dual_shape
        return self.linear_transform, tmpatom.conjugate

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Return self.atom.seminorm(self.linear_transform.linear_map(x))
        """
        return self.atom.nonsmooth_objective(self.linear_transform.linear_map(x),
                                             check_feasibility=check_feasibility)

    def get_lagrange(self):
        return self.atom._lagrange
    def set_lagrange(self, lagrange):
        if self.bound is None:
            self.atom._lagrange = lagrange
            self.atom.conjugate._bound = lagrange
        else:
            raise AttributeError("atom is in bound mode")
    lagrange = property(get_lagrange, set_lagrange)

    def get_bound(self):
        return self.atom._bound

    def set_bound(self, bound):
        if self.lagrange is None:
            self.atom._bound = bound
            self.atom.conjugate._lagrange = bound
        else:
            raise AttributeError("atom is in lagrange mode")
    bound = property(get_bound, set_bound)

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
        '''
        ltransform, conjugate_atom = self.dual
        if conjugate_atom.quadratic is not None:
            total_q = smoothing_quadratic + conjugate_atom.quadratic
        else:
            total_q = smoothing_quadratic
        if total_q.coef in [None, 0]:
            raise ValueError('quadratic term of smoothing_quadratic must be non 0')
        conjugate_atom.set_quadratic(total_q.coef, total_q.offset,
                                     total_q.linear_term, 
                                     total_q.constant_term)
        smoothed_atom = conjugate_atom.conjugate
        return affine_smooth(smoothed_atom, ltransform)

class smooth_conjugate(smooth):

    def __init__(self, atom, quadratic=None):
        """
        Given an atom,
        compute the conjugate of this atom plus 
        an identity_quadratic which will be 
        a smooth version of the conjugate of the atom.

        """
        # this holds a pointer to the original atom,
        # but will be replaced later
        self._conjugate = atom

        if quadratic is not None:
            self.set_quadratic(quadratic.coef,
                               quadratic.offset,
                               quadratic.linear_term,
                               quadratic.constant_term)

        # this ensures that the atom has
        # quadratic_spec (coef, None, None, 0)
        # and makes a copy of the atom
        self.atom = collapse_linear_terms(atom)

        totalq = self.atom.quadratic + self.quadratic  

        if totalq.coef in [0,None]:
            raise ValueError('the atom must have non-zero quadratic term to compute ensure smooth conjugate')
        if self.atom.linear_term is not None:
            if totalq.linear_term is not None:
                self.atom.linear_term += totalq.linear_term
        elif not np.all(totalq.linear_term == 0):
            self.atom.linear_term = totalq.linear_term
        self.set_quadratic(totalq.coef, None, None, 0)
        self.atom.set_quadratic(None, None, None, 0)
        self.primal_shape = atom.primal_shape

    def __repr__(self):
        return 'smooth_conjugate(%s,%s)' % (str(self.atom), str(self.quadratic))

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        q = self.quadratic
        prox_arg = x / q.coef

        if mode == 'both':
            argmin, optimal_value = self.atom.proximal_optimum(q.coef, prox_arg, 0)
            objective = q.coef / 2. * np.linalg.norm(prox_arg)**2 - optimal_value
            # retain a reference
            self.argmin = argmin
            return objective, argmin
        elif mode == 'grad':
            argmin = self.atom.proximal(q.coef, prox_arg, 0)
            # retain a reference
            self.argmin = argmin
            return argmin
        elif mode == 'func':
            argmin, optimal_value = self.atom.proximal_optimum(q.coef, prox_arg, 0)
            objective = q.coef / 2. * np.linalg.norm(prox_arg)**2 - optimal_value
            # retain a reference
            self.argmin = argmin
            return objective
        else:
            raise ValueError("mode incorrectly specified")

    def nonsmooth_objective(self, x, check_feasibilty=False):
        return self.atom.quadratic.objective(x, 'func')

    def proximal(self, lipschitz, x, grad):
        raise ValueError('no proximal defined')

conjugate_seminorm_pairs = {}
for n1, n2 in [(l1norm,supnorm),
               (l2norm,l2norm),
               (positive_part, constrained_max),
               (constrained_positive_part, max_positive_part)]:
    conjugate_seminorm_pairs[n1] = n2
    conjugate_seminorm_pairs[n2] = n1

def collapse_linear_terms(atom):
    # check the constant term is handled correctly
    atom_copy = copy(atom)
    q = atom.quadratic
    aq = identity_quadratic(None, None, atom.linear_term, atom.constant_term)
    tq = aq + q
    if not np.all(np.equal(atom_copy.linear_term,0)):
        atom_copy.linear_term = tq.linear_term
    tq.linear_term = None
    atom_copy.constant_term = tq.constant_term
    if q is not None:
        atom_copy.set_quadratic(q.coef, None, None, 0)
    else:
        atom_copy.set_quadratic(0, None, None, 0)
    return atom_copy
