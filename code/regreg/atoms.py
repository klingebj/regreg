import numpy as np
from scipy import sparse
from problem import dummy_problem
from affine import affine_transform, identity

class Atom(object):

    """
    A class that defines the API for support functions.
    """

    def __init__(self, primal_shape, lagrange=None, bound=None):

        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.dual_shape = self.primal_shape
        self.lagrange = lagrange
        self.bound = bound
        if not (self.bound is None or self.lagrange is None):
            raise ValueError('An atom must be either in Lagrange form or bound form. Only one of the parameters in the constructor can not be None.')
        self.affine_transform = identity(self.primal_shape)
        self.atoms = [self]

    @property
    def constraint(self):
        if self.bound is not None:
            return True
        return False
    
    @property
    def conjugate(self):
        if not self.constraint:
            atom = primal_dual_seminorm_pairs[self.__class__](self.primal_shape, bound=self.lagrange, lagrange=None)
        else:
            atom = primal_dual_seminorm_pairs[self.__class__](self.primal_shape, lagrange=self.bound, bound=None)
        return atom
    
    #XXX this will fail for things that are constraints...
    @property
    def dual_seminorm(self):
        return primal_dual_seminorm_pairs[self.__class__](self.primal_shape, lagrange=1./self.lagrange)

    def evaluate_seminorm(self, x):
        """
        Abstract method. Evaluate the norm of x.
        """
        raise NotImplementedError

    def evaluate_dual_constraint(self, u):
        """
        Abstract method. Evaluate the constraint on the dual norm of x.
        """
        raise NotImplementedError

    def primal_prox(self, x, lipschitz):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.evaluate_seminorm(v).
        """
        raise NotImplementedError


    def primal_prox_optimum(self, x, lipschitz):
        """
        Returns
        
        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.evaluate_seminorm(v).

        """
        argmin = self.primal_prox(x, lipschitz)
        return argmin, lipschitz * np.linalg.norm(x-argmin)**2 / 2. + self.evaluate_seminorm(argmin)
    
    def dual_prox(self, u, lipschitz):
        r"""
        Return unique minimizer

        .. math::

           v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
           \|u-'v\|^2_2  \ \text{s.t.} \   h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of self.evaluate_seminorm.
        """
        raise NotImplementedError

    def dual_prox_optimum(self, x, lipschitz):
        """
        Returns
        
        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 \ \text{s.t.} \   h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of self.evaluate_seminorm and :math:`\lambda` = self.lagrange.

        """
        argmin = self.dual_prox(x, lipschitz)
        return argmin, lipschitz * np.linalg.norm(x-argmin)**2 / 2.
    

    def affine_objective(self, u):
        """
        Return :math:`\alpha'u`. 
        """
        return self.affine_transform.affine_objective(u)

    def adjoint_map(self, u, copy=True):
        r"""
        Return :math:`u`

        This routine is currently a matrix multiplication in the subclass
        affine_transform, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        return self.affine_transform.adjoint_map(u, copy=copy)

    def linear_map(self, x, copy=True):
        r"""
        Return :math:`x`

        This routine is subclassed in affine_transform
        as a matrix multiplications, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        return self.affine_transform.linear_map(x, copy)
                                                              
    def affine_map(self, x, copy=True):
        """
        Return x + self.affine_offset. If copy: then x is copied if
        affine_offset is None.
        """
        return self.affine_transform.affine_map(x, copy)

    def primal_problem(self, smooth_func, smooth_multiplier=1., initial=None):
        """
        Return a problem instance 
        """
        prox = self.primal_prox
        nonsmooth = self.evaluate_seminorm
        if initial is None:
            initial = np.random.standard_normal(self.primal_shape)
        return dummy_problem(smooth_func.smooth_eval, nonsmooth, prox, initial, smooth_multiplier)

    def dual_problem(self, smooth_func, smooth_multiplier=1., initial=None):
        """
        Return a problem instance 
        """
        prox = self.dual_prox
        nonsmooth = self.evaluate_dual_constraint
        if initial is None:
            initial = np.random.standard_normal(self.dual_shape)
        return dummy_problem(smooth_func, nonsmooth, prox, initial, smooth_multiplier)

    @classmethod
    def affine(cls, linear_operator, affine_offset, lagrange=None,
               bound=None, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, linear_operator, affine_offset, diag=diag,
                           lagrange=lagrange, bound=bound, args=args, keywords=keywords)
    
    @classmethod
    def linear(cls, linear_operator, lagrange=None, diag=False,
               bound=None, args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, linear_operator, None, diag=diag,
                           lagrange=lagrange, args=args, keywords=keywords,
                           bound=bound)
    
    @classmethod
    def shift(cls, affine_offset, lagrange=None, diag=False,
              bound=None, args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, None, affine_offset, diag=diag,
                           lagrange=lagrange, args=args, keywords=keywords,
                           bound=bound)
    

class l1norm(Atom):

    """
    The l1 norm
    """
    tol = 1e-5
    def evaluate_seminorm(self, x):
        """
        The L1 norm of x.
        """
        return self.lagrange * np.fabs(x).sum()

    def evaluate_dual_constraint(self, u):
        inbox = np.product(np.less_equal(np.fabs(u), self.lagrange * (1+self.tol)))
        if inbox:
            return 0
        else:
            return np.inf


    def primal_prox(self, x,  lipschitz=1):
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

        return np.sign(x) * np.maximum(np.fabs(x)-self.lagrange/lipschitz, 0)

    def dual_prox(self, u, lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange.
        This is just truncation: np.clip(u, -self.lagrange/L, self.lagrange/L)
        """
        return np.clip(u, -self.lagrange, self.lagrange) 

class maxnorm(Atom):

    """
    The :math:`\ell_{\infty}` norm
    """

    tol = 1e-5
    prox_tol = 1.0e-10
    def evaluate_seminorm(self, x):
        """
        The l-infinity norm of x.
        """
        return self.lagrange * np.fabs(x).max()

    def evaluate_dual_constraint(self, u):
        inbox = np.fabs(u).sum() <= self.lagrange * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
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

        d = self.dual_prox(x,lipschitz)
        u = x - d
        return u

    def dual_prox(self, u, lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{1} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is solved with a binary search.
        """
        
        #XXX TO DO, make this efficient
        fabsu = np.fabs(u)
        l = self.lagrange / lipschitz
        upper = fabsu.sum()
        lower = 0.

        if upper <= l:
            return u

        # else, do a bisection search
        def _st_l1(ll):
            """
            the ell1 norm of a soft-thresholded vector
            """
            return np.maximum(fabsu-ll,0).sum()

        # XXX this code will be changed by Brad -- names for l, ll?
        ll = upper / 2.
        val = _st_l1(ll)
        max_iters = 30000; itercount = 0
        while np.fabs(val-l) >= upper * self.prox_tol:
            if itercount > max_iters:
                break
            itercount += 1
            val = _st_l1(ll)
            if val > l:
                lower = ll
            else:
                upper = ll
            ll = (upper + lower) / 2.
        return np.maximum(fabsu - ll, 0) * np.sign(u)


class l2norm(Atom):

    """
    The l2 norm
    """
    tol = 1e-5
    
    def evaluate_seminorm(self, x):
        """
        The L2 norm of x.
        """
        return self.lagrange * np.linalg.norm(x)

    def evaluate_dual_constraint(self, u):
        inball = (np.linalg.norm(u) <= self.lagrange * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|v\|_2

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(x) = \max\left(1 - \frac{\lambda/L}{\|x\|_2}, 0\right) x
        """

        n = np.linalg.norm(x)
        if n <= self.lagrange / lipschitz:
            proj = x
        else:
            proj = (self.lagrange / (lipschitz * n)) * x
        return x - proj * (1 - l2norm.tol)

    def dual_prox(self, u,  lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 s.t. \|v\|_2 \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(u) = \min\left(1, \frac{\lambda/L}{\|u\|_2}\right) u
        """
        n = np.linalg.norm(u)
        if n <= self.lagrange:
            return u
        else:
            return (self.lagrange / n) * u

class nonnegative(Atom):

    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """
    tol = 1e-05
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of x.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(x, -tol_lim))
        if incone:
            return 0
        return np.inf

    def evaluate_dual_constraint(self, u):
        """
        The non-positive constraint of u.
        """
        tol_lim = np.fabs(u).max() * self.tol
        indual = np.all(np.less_equal(u, tol_lim))
        if indual:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
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

        return np.maximum(x, 0)


    def dual_prox(self, u,  lipschitz=1):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(u)_i = \min(u_i, 0)
        """
        return np.minimum(u, 0)

class nonpositive(nonnegative):

    """
    The non-positive cone constraint (which is the support
    function of the non-negative cone constraint).
    """
    tol = 1e-05
    
    def evaluate_seminorm(self, x):
        """
        The non-positive constraint of x.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.less_equal(self.affine_map(x), tol_lim))
        if incone:
            return 0
        return np.inf

    def evaluate_dual_constraint(self, u):
        """
        The non-negative constraint of u.
        """
        tol_lim = np.fabs(u).max() * self.tol
        indual = np.all(np.greater_equal(u, -tol_lim))
        if indual:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
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

        return np.minimum(x, 0)

    def dual_prox(self, u,  lipschitz=1):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \geq 0

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 

        .. math::

            v^{\lambda}(u)_i = \max(u_i, 0)
        """
        return np.maximum(u, 0)

class positive_part(Atom):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of x.
        """
        return self.lagrange * np.maximum(x, 0).sum()

    def evaluate_dual_constraint(self, u):
        inbox = np.product(np.less_equal(u, self.lagrange) * np.greater_equal(u, 0))
        if inbox:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
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

        x = np.asarray(x)
        v = x.copy()
        pos = v > 0
        v = np.at_least1d(v)
        v[pos] = np.maximum(v[pos] - self.lagrange, 0)
        return v.reshape(x.shape)

    def dual_prox(self, u,  lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(u)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """
        u = np.asarray(u)
        v = u.copy()
        v = np.atleast_1d(v)
        neg = v < 0
        v[neg] = 0
        v[~neg] = np.minimum(self.lagrange, u[~neg])
        return v.reshape(u.shape)

class constrained_positive_part(Atom):

    """
    The constrained positive part seminorm (which is the support
    function of [-np.inf,l]^p). The value
    is np.inf if any coordinates are negative.
    """
    tol = 1e-10
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of x.
        """
        anyneg = np.any(x < -self.tol)
        if not anyneg:
            return self.lagrange * np.maximum(x, 0).sum()
        return np.inf
    
    def evaluate_dual_constraint(self, u):
        inside = np.product(np.less_equal(u, self.lagrange))
        if inside:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  lipschitz=1):
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
        x = np.asarray(x)
        v = x.copy()
        v = np.at_least1d(v)
        pos = v > 0
        v[pos] = np.maximum(v[pos] - self.lagrange, 0)
        v[~pos] = 0.
        return v.reshape(x.shape)

    def dual_prox(self, u,  lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation

        .. math::

            v^{\lambda}(u)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """
        u = np.asarray(u)
        v = u.copy()
        v = np.atleast_1d(v)
        pos = v > 0
        v[pos] = np.minimum(self.lagrange, u[pos])
        return v.reshape(u.shape)

class linear_atom(Atom):

    """
    An atom representing a linear constraint.
    It is specified via a matrix that is assumed
    to be an set of row vectors spanning the space.
    """

    #XXX this is broken currently
    def __init__(self, primal_shape, basis, lagrange=None):
        self.basis = basis

    def primal_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; x \in \text{row}(L)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange 
        and :math:`L` = self.basis.

        This is just projection onto :math:`\text{row}(L)`.

        """
        coefs = np.dot(self.basis, x)
        return np.dot(coefs, self.basis)

    def dual_prox(self, u,  lipschitz=1):
        r"""

        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; x \in \text{row}(L)^{\perp}

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange 
        and :math:`L` = self.basis.

        This is just projection onto :math:`\text{row}(L)^{\perp}`.

        """
        return u - self.primal_prox(u)

class affine_atom(Atom):

    """
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.
    :math:`\beta \mapsto h_K(\beta)`
    this class creates a new seminorm 
    that evaluates :math:`h_K(D\beta+\alpha)`

    The dual prox is unchanged, though the instance
    gets a affine_offset which shows up in the
    gradient of the dual problem for this atom.

    The dual problem is

    .. math::

       \text{minimize} \frac{1}{2} \|y-D^Tu\|^2_2 + u^T\alpha
       \ \text{s.t.} \ u \in \lambda K
    
    """

    # if smooth_obj is a class, an object is created
    # smooth_obj(*args, **keywords)
    # else, it is assumed to be an instance of smooth_function
 
    def __init__(self, atom_obj, linear_operator, affine_offset, diag=False, lagrange=None, args=(), keywords={}, bound=None):
        self.affine_transform = affine_transform(linear_operator, affine_offset, diag)
        self.primal_shape = self.affine_transform.primal_shape
        self.dual_shape = self.affine_transform.dual_shape

        # overwrite keyword arguments for bound, lagrange
        # not quite kosher...
        keywords = keywords.copy()
        keywords['lagrange'] = lagrange
        keywords['bound'] = bound

        if type(atom_obj) == type(type): # it is a class
            atom_class = atom_obj
            self.atom = atom_class(self.dual_shape, *args, **keywords)
        else:
            self.atom = atom_obj
        self.atoms = [self]
        

    @property
    def conjugate(self):
        if not self.constraint:
            atom = primal_dual_seminorm_pairs[self.atom.__class__](self.primal_shape, bound=self.lagrange, lagrange=None)
        else:
            atom = primal_dual_seminorm_pairs[self.atom.__class__](self.primal_shape, lagrange=self.bound, bound=None)
        return atom

    @property
    def dual_seminorm(self):
        return primal_dual_seminorm_pairs[self.atom.__class__](self.dual_shape, 1./self.lagrange)

    def _getlagrange(self):
        return self.atom.lagrange

    def _setlagrange(self, lagrange):
        self.atom.lagrange = lagrange
    lagrange = property(_getlagrange, _setlagrange)

    def _getbound(self):
        return self.atom.bound

    def _setbound(self, bound):
        self.atom.bound = bound
    bound = property(_getbound, _setbound)

    def evaluate_seminorm(self, x):
        """
        Return self.atom.evaluate_seminorm(self.affine_map(x))

        """
        return self.atom.evaluate_seminorm(self.affine_map(x))

    def evaluate_dual_constraint(self, u):
        return self.atom.evaluate_dual_constraint(u)

    def primal_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda h_K(Dv+\alpha)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        This is just self.atom.primal_prox(x + self.affine_offset, L) + self.affine_offset
        """
        if self.affine_transform.linear_operator is None:
            if self.affine_transform.affine_offset is not None:
                return self.atom.primal_prox(x + self.affine_transform.affine_offset, lipschitz) - self.affine_transform.affine_offset
            else:
                return self.atom.primal_prox(x, lipschitz)
        else:
            raise NotImplementedError('when linear_operator is not None, primal_prox is not implemented, can be done with FISTA')

    def dual_prox(self, u, lipschitz=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.lagrange. 
        This is just truncation: np.clip(u, -self.lagrange/L, self.lagrange/L).
        """
        return self.atom.dual_prox(u, lipschitz)

primal_dual_seminorm_pairs = {}
for n1, n2 in [(l1norm,maxnorm),
               (l2norm,l2norm),
               (nonnegative,nonpositive),
               (positive_part, constrained_positive_part)]:
    primal_dual_seminorm_pairs[n1] = n2
    primal_dual_seminorm_pairs[n2] = n1
