import numpy as np
from scipy import sparse

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
        
    def evaluate(self, x):
        """
        Abstract method. Evaluate the norm of x.
        """
        raise NotImplementedError

    def primal_prox(self, x, L):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(Dv)

        where *p*=x.shape[0] and :math:`h(v)`=self.evaluate(v).
        """
        raise NotImplementedError

    def dual_prox(self, u, L):
        r"""
        Return a minimizer

        .. math::

           v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
           \|u-D'v\|^2_2  s.t.  h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of self.evaluate.
        """
        raise NotImplementedError

    #XXX These routines are currently matrix multiplications, but could also call
    #FFTs if D is a DFT matrix, etc.
    def multiply_by_DT(self, u):
        if not self.noneD:
            if self.sparseD:
                return u * self.D
            else:
                return np.dot(u, self.D)
        else:
                return u
                                 
    def multiply_by_D(self, x):
        if not self.noneD:
            if self.sparseD:
                return self.D * x
            else:
                return np.dot(self.D, x)
        else:
                return x
                                                              

    def problem(self, smooth, grad_smooth, smooth_multiplier=1., initial=None):
        """
        Return a problem instance 
        """
        prox = self.primal_prox
        nonsmooth = self.evaluate
        if initial is None:
            initial = np.random.standard_normal(self.p)
        return dummy_problem(smooth, grad_smooth, nonsmooth, prox, initial, smooth_multiplier)


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

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
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
            \|u-v\|^2_2 s.t. \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
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

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
        If :math:`D=I` this is just a "James-Stein" estimator

        .. math::

            v^{\lambda}(x) = \max(1 - \frac{\lambda}{\|x\|_2}, 0) x
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

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
        This is just truncation

        .. math::

            v^{\lambda}(u) = \min(1, \frac{\lambda/L}{\|u\|_2}) u
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
            \|x-v\|^2_2 s.t. (Dv)_i \geq 0.

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
        If :math:`D=I` this is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \min(x_i, 0)

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
            \|u-v\|^2_2 s.t. v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
        This is just truncation

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

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
        If :math:`D=I` this is just soft-thresholding
        positive values and leaving negative values untouched.

        .. math::

            v^{\lambda}(x)_i = \begin{cases}
            \max(x_i - \lambda, 0) & x_i \geq 0 \\
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
            \|u-v\|^2_2 s.t. v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
        This is just truncation

        .. math::

            v^{\lambda}(u)_i = \min(u_i, 0)
        """
        neg = u < 0
        v = u.copy()
        v[neg] = 0
        v[~neg] = np.minimum(self.l, u[~neg])
        return v
