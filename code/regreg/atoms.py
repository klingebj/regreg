import numpy as np
from scipy import sparse
from problem import dummy_problem

class seminorm_atom(object):

    """
    A seminorm atom class
    """

    #XXX spec as 1d array could mean weights?
    #XXX matrix multiply should be sparse if possible

    def __init__(self, spec, l=1.):
        if type(spec) == type(1):
            self.p = self.m = spec
            self.D = None
        else:
            D = spec
            if D.ndim == 1:
                D = D.reshape((1,-1))
            self.D = D
            self.m, self.p = D.shape
        self.l = l
        if self.D is not None:
            self.noneD = False
            self.sparseD = sparse.isspmatrix(self.D)
        else:
            self.noneD = True
        self.atoms = [self]
        
    @property
    def dual_constraint(self):
        return primal_dual_pairs[self.__class__](self.m, self.l)

    @property
    def dual(self):
        return self.dual_constraint

    def evaluate(self, x):
        """
        Abstract method. Evaluate the norm of x.
        """
        raise NotImplementedError

    def evaluate_dual(self, u):
        """
        Abstract method. Evaluate the constraint on the dual norm of x.
        """
        raise NotImplementedError

    def primal_prox(self, x, L):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(Dv)

        where *p*=x.shape[0] and :math:`h(v)` = self.evaluate(v).
        """
        raise NotImplementedError

    def dual_prox(self, u, L):
        r"""
        Return a minimizer

        .. math::

           v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
           \|u-D'v\|^2_2  \ \text{s.t.} \   h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of self.evaluate.
        """
        raise NotImplementedError

    def multiply_by_DT(self, u):
        r"""
        Return :math:`D^Tu`

        This routine is currently a matrix multiplication, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        if not self.noneD:
            if self.sparseD:
                return u * self.D
            else:
                return np.dot(u, self.D)
        else:
                return u
                                 
    def multiply_by_D(self, x):
        r"""
        Return :math:`Dx`

        This routine is currently a matrix multiplications, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        if not self.noneD:
            if self.sparseD:
                return self.D * x
            else:
                return np.dot(self.D, x)
        else:
                return x
                                                              
    def problem(self, smooth_func, smooth_multiplier=1., initial=None):
        """
        Return a problem instance 
        """
        prox = self.primal_prox
        nonsmooth = self.evaluate
        if initial is None:
            initial = np.random.standard_normal(self.p)
        return dummy_problem(smooth_func, nonsmooth, prox, initial, smooth_multiplier)

class l1norm(seminorm_atom):

    """
    The l1 norm
    """

    def evaluate(self, x):
        """
        The L1 norm of Dx.
        """
        return self.l * np.fabs(self.multiply_by_D(x)).sum()

    def evaluate_dual(self, u):
        inbox = np.product(np.less_equal(np.fabs(u), self.l))
        if inbox:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|Dv\|_1

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is just soft thresholding

        .. math::

            v^{\lambda}(x) = \text{sign}(x) \max(|x|-\lambda/L, 0)
        """

        if self.D is None:
            return np.sign(x) * np.maximum(np.fabs(x)-self.l/L, 0)
        else:
            raise NotImplementedError

    def dual_prox(self, u, L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation: np.clip(u, -self.l/L, self.l/L).
        """
        return np.clip(u, -self.l, self.l)

class l2norm(seminorm_atom):

    """
    The l2 norm
    """
    tol = 1e-10
    
    def evaluate(self, x):
        """
        The L2 norm of Dx.
        """
        return self.l * np.linalg.norm(self.multiply_by_D(x))

    def evaluate_dual(self, u):
        inball = (np.linalg.norm(u) <= self.l * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|Dv\|_2

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is just a "James-Stein" estimator

        .. math::

            v^{\lambda}(x) = \max\left(1 - \frac{\lambda/L}{\|x\|_2}, 0\right) x
        """

        if self.D is None:
            n = np.linalg.norm(x)
            if n >= self.l / L:
                return np.zeros(x.shape)
            else:
                return (1 - self.l / (L*n) * (1 - l2norm.tol)) * x
        else:
            raise NotImplementedError


    def dual_prox(self, u,  L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 + \lambda \|v\|_2

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation

        .. math::

            v^{\lambda}(u) = \min\left(1, \frac{\lambda/L}{\|u\|_2}\right) u
        """
        n = np.linalg.norm(u)
        if n < self.l:
            return u
        else:
            return (self.l / n) * u

class nonnegative(seminorm_atom):

    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """
    tol = 1e-05
    
    def evaluate(self, x):
        """
        The non-negative constraint of Dx.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(self.multiply_by_D(x), -tol_lim))
        if incone:
            return 0
        return np.inf

    def evaluate_dual(self, u):
        """
        The non-positive constraint of u.
        """
        tol_lim = np.fabs(u).max() * self.tol
        indual = np.all(np.less_equal(u, tol_lim))
        if indual:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  (Dv)_i \geq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \max(x_i, 0)

        """

        if self.D is None:
            return np.maximum(x, 0)
        else:
            raise NotImplementedError


    def dual_prox(self, u,  L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda` = self.l. 

        .. math::

            v^{\lambda}(u)_i = \min(u_i, 0)
        """
        return np.minimum(u, 0)

class positive_part(seminorm_atom):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """
    tol = 1e-10
    
    def evaluate(self, x):
        """
        The non-negative constraint of Dx.
        """
        Dx = self.multiply_by_D(x)
        return self.l * np.maximum(Dx, 0).sum()

    def evaluate_dual(self, u):
        inbox = np.product(np.less_equal(u, self.l) * np.greater_equal(u, 0))
        if inbox:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  + \sum_i \lambda \max(Dv_i, 0)

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is just soft-thresholding
        positive values and leaving negative values untouched.

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \max(x_i - \frac{\lambda}{L}, 0) & x_i \geq 0 \\
            x_i & x_i \leq 0.  
            \end{cases} 

        """

        if self.D is None:
            pos = x > 0
            v = x.copy()
            v[pos] -= np.maximum(v[pos] - self.l, 0)
            return v
        else:
            raise NotImplementedError

    def dual_prox(self, u,  L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  0 \leq v_i \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation

        .. math::

            v^{\lambda}(u)_i = \begin{cases}
            \min(u_i, \lambda) & u_i \geq 0 \\
            0 & u_i \leq 0.  
            \end{cases} 

        """
        neg = u < 0
        v = u.copy()
        v[neg] = 0
        v[~neg] = np.minimum(self.l, u[~neg])
        return v

class zero(seminorm_atom):

    """
    The zero seminorm
    """

    def evaluate(self, x):
        """
        The zero normL1 norm of Dx.
        """
        return np.zeros(x.shape)

    def evaluate_dual(self, u):
        iszero = np.equal(u, 0)
        iszero[iszero] = np.inf
        return iszero

    def primal_prox(self, x,  L=1):
        r"""
        Return x
        """
        return x

    def dual_prox(self, u, L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation: np.clip(u, -self.l/L, self.l/L).
        """
        return np.clip(u, -self.l, self.l)
    

class constraint_atom(object):

    def __init__(self, p, l, dual_seminorm):
        if type(p) == type(1):
            self.p = self.m = p
        else:
            raise ValueError("constraints cannot be specified with D at this time")
        self.l = l
        self.noneD = True
        self.dual_seminorm = dual_seminorm(self.p, self.l)
        self.dual = self.dual_seminorm
        
    def evaluate(self, x):
        """
        Abstract method. Evaluate the constraint on the norm of x.
        """
        return self.dual_seminorm.evaluate_dual(x)

    def evaluate_dual(self, u):
        """
        Abstract method. Evaluate the dual norm of x.
        """
        return self.dual_seminorm.evaluate(u)

    def primal_prox(self, x, L):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 \ \text{s.t.} \|v\| \leq \lambda

        where *p*=x.shape[0] and :math:`\lambda` = self.l
        and the norm is the dual of self.evaluate.
        """
        return self.dual_seminorm.dual_prox(x, L)

    def dual_prox(self, u, L):
        r"""
        Return a minimizer

        .. math::

           v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
           \|u-D'v\|^2_2  + \lambda \|v\|^*

        where *p*=u.shape[0], :math:`\lambda` = self.l and :math:`h^*` is the 
        conjugate of self.evaluate, i.e. a seminorm.
        """
        return self.dual_seminorm.primal_prox(u, L)

    def random_initial(self):
        """
        Return a random feasible point for use as an initial condition.
        """
        Z = np.random.standard_normal(self.p)
        return self.dual_seminorm.dual_prox(Z, 1)

def box_constraint(p, l=1):
    return constraint_atom(p, l, l1norm)

def l2_constraint(p, l=1):
    return constraint_atom(p, l, l2norm)

def negative_constraint(p, l=1):
    return constraint_atom(p, l, nonnegative)

def negative_part_constraint(constraint_atom):
    return constraint_atom(p, l, positive_part)

primal_dual_pairs = {l1norm:box_constraint,
                     l2norm:l2_constraint,
                     negative_constraint:nonnegative,
                     negative_part_constraint:positive_part}



