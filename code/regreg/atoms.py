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
        # this affine term appears in the gradient of the dual
        # problem of the atom
        # for affine seminorms, this can be an array of
        # shape (self.p,)
        self.affine_term = None
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
        return primal_dual_constraint_pairs[self.__class__](self.m, self.l)

    @property
    def dual_seminorm(self):
        return primal_dual_seminorm_pairs[self.__class__](self.m, 1./self.l)

    @property
    def dual(self):
        return self.dual_constraint

    def primal_from_dual(self, u):
        return self.multiply_by_DT(u)

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

    def primal_prox(self, x, L):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(Dv)

        where *p*=x.shape[0] and :math:`h(v)` = self.evaluate_seminorm(v).
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

    # JT: these two methods are used in 
    # computing the gradient
    # and proximal problems
    # currently, the affine_term
    # are handled in those modules but
    # probably should be handled in these two methods

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
            # this might have to be a copy
            # but we only multiply by D.T when
            # computing gradient -- 
            # this currently doesn't happen in seminorm or
            # smoothed_seminorm
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
            # this has to be a copy
            # because the array can later be modified
            # in place -- see the smoothed_seminorm
            return x.copy()
                                                              
    def affine_map(self, x):
        """
        Return :math:`Dx+\alpha`.
        """
        Dx = self.multiply_by_D(x)
        if self.affine_term is not None:
            return Dx + self.affine_term
        return Dx
    
    def problem(self, smooth_func, smooth_multiplier=1., initial=None):
        """
        Return a problem instance 
        """
        prox = self.primal_prox
        nonsmooth = self.evaluate_seminorm
        if initial is None:
            initial = np.random.standard_normal(self.p)
        return dummy_problem(smooth_func, nonsmooth, prox, initial, smooth_multiplier)

    @classmethod
    def affine(cls, spec, alpha, l=1):
        atom = cls(spec, l=l)
        return affine_atom(atom, spec, alpha)
    
class l1norm(seminorm_atom):

    """
    The l1 norm
    """

    def evaluate_seminorm(self, x):
        """
        The L1 norm of Dx.
        """
        return self.l * np.fabs(self.affine_map(x)).sum()

    def evaluate_dual_constraint(self, u):
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

class maxnorm(seminorm_atom):

    """
    The :math:`\ell_{\infty}` norm
    """

    tol = 1.0e-06
    def evaluate_seminorm(self, x):
        """
        The l-infinity norm of Dx.
        """
        return self.l * np.fabs(self.affine_map(x)).max()

    def evaluate_dual_constraint(self, u):
        inbox = np.fabs(u).sum() <= self.l
        if inbox:
            return 0
        else:
            return np.inf

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|Dv\|_{\infty}

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is the residual
        after projecting onto :math:`\lambda/L` times the :math:`\ell_1` ball

        .. math::

            v^{\lambda}(x) = x - P_{\lambda/L B_{\ell_1}}(x)
        """

        if self.D is None:
            d = self.dual_prox(x,L)
            u = x - d
            # print np.fabs(d).sum(), self.l / L, (u*d).sum(), self.l/L * np.fabs(u).max()
            return u
        else:
            raise NotImplementedError

    def dual_prox(self, u, L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{1} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is solved with a binary search.
        """
        
        #XXX TO DO, make this efficient
        fabsu = np.fabs(u)
        l = self.l / L
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

        ll = upper / 2.
        val = _st_l1(ll)
        max_iters = 30; itercount = 0
        while np.fabs(val-l) >= upper * self.tol and itercount <= max_iters:
            itercount += 1
            val = _st_l1(ll)
            if val > l:
                lower = ll
            else:
                upper = ll
            ll = (upper + lower) / 2.
        return np.maximum(fabsu - ll, 0) * np.sign(u)


class l2norm(seminorm_atom):

    """
    The l2 norm
    """
    tol = 1e-10
    
    def evaluate_seminorm(self, x):
        """
        The L2 norm of Dx.
        """
        return self.l * np.linalg.norm(self.affine_map(x))

    def evaluate_dual_constraint(self, u):
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
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of Dx.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(self.affine_map(x), -tol_lim))
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
        Return unique minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \leq 0

        where *m*=u.shape[0], :math:`\lambda` = self.l. 

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
        The non-positive constraint of Dx.
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

    def primal_prox(self, x,  L=1):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  (Dv)_i \leq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.l. 
        If :math:`D=I` this is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \min(x_i, 0)

        """

        if self.D is None:
            return np.minimum(x, 0)
        else:
            raise NotImplementedError

    def dual_prox(self, u,  L=1):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  v_i \geq 0

        where *m*=u.shape[0], :math:`\lambda` = self.l. 

        .. math::

            v^{\lambda}(u)_i = \max(u_i, 0)
        """
        return np.maximum(u, 0)

class positive_part(seminorm_atom):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of Dx.
        """
        Dx = self.affine_map(x)
        return self.l * np.maximum(Dx, 0).sum()

    def evaluate_dual_constraint(self, u):
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

        x = np.asarray(x)
        if self.D is None:
            v = x.copy()
            pos = v > 0
            v = np.at_least1d(v)
            v[pos] = np.maximum(v[pos] - self.l, 0)
            return v.reshape(x.shape)
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
        u = np.asarray(u)
        v = u.copy()
        v = np.atleast_1d(v)
        neg = v < 0
        v[neg] = 0
        v[~neg] = np.minimum(self.l, u[~neg])
        return v.reshape(u.shape)

class constrained_positive_part(seminorm_atom):

    """
    The constrained positive part seminorm (which is the support
    function of [-np.inf,l]^p). The value
    is np.inf if any coordinates are negative.
    """
    tol = 1e-10
    
    def evaluate_seminorm(self, x):
        """
        The non-negative constraint of Dx.
        """
        Dx = self.affine_map(x)
        anyneg = np.any(Dx < -self.tol)
        if not anyneg:
            return self.l * np.maximum(Dx, 0).sum()
        return np.inf
    
    def evaluate_dual_constraint(self, u):
        inside = np.product(np.less_equal(u, self.l))
        if inside:
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
        x = np.asarray(x)
        if self.D is None:
            v = x.copy()
            v = np.at_least1d(v)
            pos = v > 0
            v[pos] = np.maximum(v[pos] - self.l, 0)
            v[~pos] = 0.
            return v.reshape(x.shape)
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
        u = np.asarray(u)
        v = u.copy()
        v = np.atleast_1d(v)
        pos = v > 0
        v[pos] = np.minimum(self.l, u[pos])
        return v.reshape(u.shape)

class affine_atom(seminorm_atom):

    """
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.
    :math:`\beta \mapsto h_K(D\beta)`
    this class creates a new seminorm 
    that evaluates :math:`h_K(D\beta+\alpha)`

    The dual prox is unchanged, though the instance
    gets a affine_term which shows up in the
    gradient of the dual problem for this atom.

    The dual problem is

    .. math::

       \text{minimize} \frac{1}{2} \|y-D^Tu\|^2_2 + u^T\alpha
       \ \text{s.t.} \ u \in \lambda K
    
    """

    def __init__(self, atom, spec, alpha):
        self.atom = atom
        seminorm_atom.__init__(self, spec)
        self.l = atom.l
        # an instance with D=I
        # all atoms should (?) be such that this
        # is the conjugate of atom.evaluate
        self.affine_term = alpha
        self.pure_atom = atom.__class__(atom.m, l=atom.l)

    def _getl(self):
        return self.atom.l

    def _setl(self, l):
        self.atom.l = l
    l = property(_getl, _setl)

    def evaluate_seminorm(self, x):
        """
        Return self.atom_I(np.dot(self.atom.D, x)+self.affine_term)

        """
        return self.pure_atom.evaluate_seminorm(self.affine_map(x))

    def evaluate_dual_constraint(self, u):
        return self.atom.evaluate_dual_constraint(u)

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda h_K(Dv+\alpha)

        where *p*=x.shape[0], :math:`\lambda` = self.l. 

        This is just self.atom.primal_prox(x - self.affine_term, L) + self.affine_term
        """

        return self.atom.primal_prox(x - self.affine_term, L) + self.affine_term

    @property
    def dual_constraint(self):
        return primal_dual_constraint_pairs[self.atom.__class__](self.m, self.l)

    @property
    def dual_seminorm(self):
        return primal_dual_seminorm_pairs[self.atom.__class__](self.m, self.l)

    def dual_prox(self, u, L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation: np.clip(u, -self.l/L, self.l/L).
        """
        return self.atom.dual_prox(u, L)

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
        
    def evaluate_constraint(self, x):
        """
        Abstract method. Evaluate the constraint on the norm of x.
        """
        return self.dual_seminorm.evaluate_dual_constraint(x)

    def evaluate_dual_seminorm(self, u):
        """
        Abstract method. Evaluate the dual norm of x.
        """
        return self.dual_seminorm.evaluate_seminorm(u)

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

primal_dual_constraint_pairs = {l1norm:box_constraint,
                               l2norm:l2_constraint,
                               negative_constraint:nonnegative,
                               negative_part_constraint:positive_part}

primal_dual_seminorm_pairs = {}
for n1, n2 in [(l1norm,maxnorm),
               (l2norm,l2norm),
               (nonnegative,nonpositive),
               (positive_part, constrained_positive_part)]:
    primal_dual_seminorm_pairs[n1] = n2
    primal_dual_seminorm_pairs[n2] = n1
