import numpy as np
from scipy import sparse
from composite import composite, nonsmooth
from affine import linear_transform, identity as identity_transform
from projl1 import projl1
from copy import copy

class atom(nonsmooth):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def __init__(self, primal_shape, lagrange=None, bound=None, 
                 linear_term=None,
                 constant_term=0., offset=None):

        self.offset = None
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


    

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if self.bound is not None:
                return self.bound == other.bound
            return self.lagrange == other.lagrange
        return False

    def __copy__(self):
        return self.__class__(copy(self.primal_shape),
                              linear_term=copy(self.linear_term),
                              constant_term=copy(self.constant_term),
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange))
    
    def __repr__(self):
        if self.lagrange is not None:
            return "%s(%s, lagrange=%f, linear_term=%s, offset=%s)" % (self.__class__.__name__,
                                                                       `self.primal_shape`, 
                                                                       self.lagrange,
                                                                       str(self.linear_term),
                                                                       str(self.offset))

        else:
            return "%s(%s, bound=%f, linear_term=%s, offset=%s)" % (self.__class__.__name__,
                                                         `self.primal_shape`, 
                                                         self.bound,
                                                         str(self.linear_term),
                                                         str(self.offset))
    
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
            if self.bound is None:
                atom = primal_dual_seminorm_pairs[self.__class__](self.primal_shape, 
                                                                  bound=self.lagrange, 
                                                                  lagrange=None,
                                                                  linear_term=linear_term,
                                                                  offset=offset)
            else:
                atom = primal_dual_seminorm_pairs[self.__class__](self.primal_shape, 
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
    
    def seminorm(self, x, check_feasibility=False):
        """
        Abstract method. Evaluate the norm of x.
        """
        raise NotImplementedError

    def constraint(self, x):
        """
        Abstract method. Evaluate the constraint on the dual norm of x.
        """
        raise NotImplementedError

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
            return v + self.constant_term
        else:
            return v + (self.linear_term * x).sum() + self.constant_term
        
    def proximal(self, x, lipschitz=1):
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
        if self.offset is not None:
            offset = self.offset
        else:
            offset = 0
        if self.linear_term is not None:
            shift = offset - self.linear_term / lipschitz
        else:
            shift = offset

        if not np.all(np.equal(shift, 0)):
            x = x + shift
        if np.all(np.equal(offset, 0)):
            offset = None

        if self.bound is not None:
            eta = self.bound_prox(x, lipschitz=lipschitz, bound=self.bound)
        else:
            eta = self.lagrange_prox(x, lipschitz=lipschitz, lagrange=self.lagrange)

        if offset is None:
            return eta
        else:
            return eta - offset

    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` is the support function of self (with a
        Lagrange multiplier of 1 in front) and :math:`\lambda` is the Lagrange parameter.
        If the argument is None and the atom is in Lagrange mode, this parameter
        is used for the proximal operator, else an exception is raised.
        
        """
        raise NotImplementedError
    
    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

           v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
           \|u-'v\|^2_2  \ \text{s.t.} \   h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of the support function of self (with a Lagrange multiplier of 1 in front).
        and :math:`\lambda` is the bound parameter.
        If the argument is None and the atom is in bound mode, this parameter
        is used for the proximal operator, else an exception is raised.
        """
        raise NotImplementedError

    def adjoint_map(self, x, copy=True):
        r"""
        Return :math:`u`

        This routine is currently a matrix multiplication in the subclass
        affine_transform, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        return self.linear_transform.adjoint_map(x, copy=copy)

    def linear_map(self, x, copy=True):
        r"""
        Return :math:`x`

        This routine is subclassed in affine_transform
        as a matrix multiplications, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        return self.linear_transform.linear_map(x, copy)
                                                              
    @classmethod
    def linear(cls, linear_operator, lagrange=None, diag=False,
               bound=None, args=(), keywords=None,
               linear_term=None, offset=None):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        if keywords is None:
            keywords = {}
        l = linear_transform(linear_operator, diag=diag)
        atom = cls(l.primal_shape, lagrange=lagrange, bound=bound,
                   linear_term=linear_term, offset=offset)
                   
        return affine_atom(atom, l)
    
class l1norm(atom):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    def seminorm(self, x, check_feasibility=False):
        """
        The L1 norm of x.
        """
        return self.lagrange * np.fabs(x).sum()

    def constraint(self, x):
        inbox = np.fabs(x).sum() <= self.bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|v\|_1

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange.
        This is just soft thresholding with an affine shift

        .. math::

            v^{\lambda}(x) = \text{sign}(x) \max(|x|-\lambda/L, 0)
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return np.sign(x) * np.maximum(np.fabs(x)-lagrange/lipschitz, 0)


    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{1} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is solved with a binary search.
        """
        
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        return projl1(x, self.bound)

class supnorm(atom):

    """
    The :math:`\ell_{\infty}` norm
    """

    def seminorm(self, x, check_feasibility=False):
        """
        The l-infinity norm of x.
        """
        return self.lagrange * np.fabs(x).max()

    def constraint(self, x):
        inbox = np.product(np.less_equal(np.fabs(x), self.bound * (1+self.tol)))
        if inbox:
            return 0
        else:
            return np.inf


    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|v\|_{\infty}

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange.
        This is the residual
        after projecting :math:`x` onto
        :math:`\lambda/L` times the :math:`\ell_1` ball

        .. math::

            v^{\lambda}(x) = x - P_{\lambda/L B_{\ell_1}}(x)
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        d = projl1(x, lagrange/lipschitz)
        return x - d

    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange.
        This is just truncation: np.clip(x, -self.lagrange/L, self.lagrange/L)
        """
        
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        return np.clip(x, -bound, bound)

class l2norm(atom):

    """
    The l2 norm
    """
    
    def seminorm(self, x, check_feasibility=False):
        """
        The L2 norm of x.
        """
        return self.lagrange * np.linalg.norm(x)

    def constraint(self, x):
        inball = (np.linalg.norm(x) <= self.bound * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|v\|_2

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(x) = \max\left(1 - \frac{\lambda/L}{\|x\|_2}, 0\right) x
        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        n = np.linalg.norm(x)
        if n <= lagrange / lipschitz:
            proj = x
        else:
            proj = (self.lagrange / (lipschitz * n)) * x
        return x - proj * (1 - l2norm.tol)

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 s.t. \|v\|_2 \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(x) = \min\left(1, \frac{\lambda/L}{\|u\|_2}\right) u
        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        n = np.linalg.norm(x)
        if n <= bound:
            return x
        else:
            return (bound / n) * x

class nonnegative(atom):

    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """
    
    def seminorm(self, x, check_feasibility=False):
        """
        The non-negative constraint of x.
        """
        if not check_feasibility:
            return 0
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(x, -tol_lim))
        if incone:
            return 0
        return np.inf

    def constraint(self, x):
        """
        The non-negative constraint of u.
        """
        return self.seminorm(x)


    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  (v)_i \geq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \max(x_i, 0)

        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        return np.maximum(x, 0)


    def bound_prox(self, x, lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \geq 0

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(x)_i = \max(u_i, 0)
        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        return self.lagrange_prox(x, lipschitz, lagrange=bound)

class nonpositive(nonnegative):

    """
    The non-positive cone constraint (which is the support
    function of the non-negative cone constraint).
    """
    
    def seminorm(self, x, check_feasibility=False):
        """
        The non-positive constraint of x.
        """
        if not check_feasibility:
            return 0
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.less_equal(x, tol_lim))
        if incone:
            return 0
        return np.inf

    def constraint(self, x):
        """
        The non-positive constraint of u.
        """
        return self.seminorm(x)

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v_i \leq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \min(x_i, 0)

        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        return np.minimum(x, 0)

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(x)_i = \min(u_i, 0)
        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        # XXX  being a cone, the first two arguments are not needed
        return self.lagrange_prox(x, lagrange=bound)

class positive_part(atom):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """
    
    def seminorm(self, x, check_feasibility=False):
        """
        The sum of the positive parts of x.
        """
        return self.lagrange * np.maximum(x, 0).sum()

    def constraint(self, x):
        inside = np.maximum(x, 0).sum() <= self.bound * (1 + self.tol)
        if inside:
            return 0
        else:
            return np.inf

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  + \sum_i \lambda \max(v_i, 0)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just soft-thresholding
        positive values and leaving negative values untouched.

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \max(x_i - \frac{\lambda}{L}, 0) & x_i \geq 0 \\
            x_i & x_i \leq 0.  
            \end{cases} 

        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        x = np.asarray(x)
        v = x.copy()
        pos = v > 0
        v = np.atleast_1d(v)
        v[pos] = np.maximum(v[pos] - lagrange/lipschitz, 0)
        return v.reshape(x.shape)

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        x = np.asarray(x)
        v = x.copy()
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(x.shape)

class constrained_max(atom):
    """
    The seminorm x.max() s.t. x geq 0.
    """

    def seminorm(self, x, check_feasibility=False):
        """
        The sum of the positive parts of x.
        """
        anyneg = np.any(x < 0 + self.tol)
        v = self.lagrange * np.max(x)
        if not anyneg or not check_feasibility:
            return v
        return np.inf

    def constraint(self, x):
        anyneg = np.any(x < 0 + self.tol)
        inside = np.max(x) <= self.bound * (1 + self.tol)
        if inside and not anyneg:
            return 0
        else:
            return np.inf

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  + \sum_i \lambda \max(v_i, 0)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just soft-thresholding
        positive values and leaving negative values untouched.

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \max(x_i - \frac{\lambda}{L}, 0) & x_i \geq 0 \\
            x_i & x_i \leq 0.  
            \end{cases} 

        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        x = np.asarray(x)
        v = x.copy()
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange/lipschitz)
        return x-v.reshape(x.shape)

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        return np.clip(x, 0, bound)

class constrained_positive_part(atom):

    """
    Support function of (-\infty,0]^p
    """

    def seminorm(self, x, check_feasibility=False):
        anyneg = np.any(x < 0 + self.tol)
        v = np.maximum(x, 0).sum()
        if not anyneg or not check_feasibility:
            return v * self.lagrange
        return np.inf

    def constraint(self, x):
        anyneg = np.any(x < 0 + self.tol)
        v = np.maximum(x, 0).sum()
        if anyneg or v >= self.bound * (1 + self.tol):
            return np.inf
        return 0

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        x = np.asarray(x)
        v = np.zeros(x.shape)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = np.maximum(x[pos] - lagrange/lipschitz, 0)
        return v

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        x = np.asarray(x)
        v = np.zeros(x.shape)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(x.shape)

class max_positive_part(atom):

    """
    support function of the standard simplex
    """
    def seminorm(self, x, check_feasibility=False):
        return np.max(np.maximum(x,0)) * self.lagrange

    def constraint(self, x):
        v = np.max(np.maximum(x,0))
        if v >= self.bound * (1 + self.tol):
            return np.inf
        return 0

    def bound_prox(self, x, lipschitz=1, bound=None):
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        return np.clip(x, -np.inf, bound)

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(x) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in lagrange mode or a keyword "lagrange" argument must be supplied')

        x = np.asarray(x)
        v = np.zeros(x.shape)
        v = np.atleast_1d(v)
        pos = x > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange / lipschitz)
        return x-v.reshape(x.shape)


class projection_atom(atom):

    """
    An atom representing a linear constraint.
    It is specified via a matrix that is assumed
    to be an set of row vectors spanning the space.
    """

    #XXX this is broken currently
    def __init__(self, primal_shape, basis, lagrange=None):
        self.basis = basis

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; x \in \text{row}(L)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange 
        and :math:`L` = self.basis.

        This is just projection onto :math:`\text{row}(L)`.

        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')

        coefs = np.dot(self.basis, x)
        return np.dot(coefs, self.basis)

    def bound_prox(self, x,  lipschitz=1, bound=None):
        r"""

        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; x \in \text{row}(L)^{\perp}

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange 
        and :math:`L` = self.basis.

        This is just projection onto :math:`\text{row}(L)^{\perp}`.

        """

        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')

        # XXX being a cone, the two arguments are not needed
        return self.lagrange_prox(x, lipschitz=lipschitz,lagrange=bound)

class zero(atom):
    """
    The zero seminorm, support function of :math:\{0\}
    """

    def seminorm(self, x, check_feasibility=False):
        return 0.

    def constraint(self, x):
        return 0.

    def bound_prox(self, x, lipschitz=1, bound=None):
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return x

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in lagrange mode or a keyword "lagrange" argument must be supplied')
        return x
    
class zero_constraint(atom):
    """
    The zero constraint, support function of :math:`\mathbb{R}`^p
    """

    tol = 1.0e-05
    def seminorm(self, x, check_feasibility=False):
        if not check_feasibility:
            return 0.
        elif not np.linalg.norm(x) <= self.tol:
                return np.inf
        return 0.

    def constraint(self, x):
        if not np.linalg.norm(x) <= self.tol:
            return np.inf
        return 0.

    def bound_prox(self, x, lipschitz=1, bound=None):
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return np.zeros(x.shape)

    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
        r"""
        """

        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in lagrange mode or a keyword "lagrange" argument must be supplied')
        return np.zeros(x.shape)

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

    # if atom_obj is a class, an object is created
    # atom_obj(*args, **keywords)
    # else, it is assumed to be an instance of atom
 
    def __init__(self, atom_obj, atransform):
        self.atom = copy(atom_obj)
        # self.linear_term = self.offset = None
        # if the affine transform has an offset, put it into
        # the atom. in this way, the affine_transform is actually
        # always linear
        if atransform.affine_offset is not None:
            self.atom.offset += atransform.affine_offset
            ltransform = affine_transform(atransform.linear_operator, None,
                                          diag=atransform.diag)
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
        return self.linear_transform, self.atom.conjugate

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



primal_dual_seminorm_pairs = {}
for n1, n2 in [(l1norm,supnorm),
               (l2norm,l2norm),
               (nonnegative,nonpositive),
               (positive_part, constrained_max),
               (constrained_positive_part, max_positive_part),
               (zero, zero_constraint)]:
    primal_dual_seminorm_pairs[n1] = n2
    primal_dual_seminorm_pairs[n2] = n1
