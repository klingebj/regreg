import numpy as np

from regression import FISTA, ISTA
from problems import linmodel

class problem(object):

    """
    A problem class with a smooth component, and a seminorm component stored in self.semi
    """

    def __init__(self, data, semi):
        self.semi = semi
        self.initialize(data)

    @property
    def output(self):
        r = self.Y - np.dot(self.X, self.coefs)
        return self.coefs.copy(), r
    
    def set_coefs(self, coefs):
        self._coefs = coefs

    def get_coefs(self):
        return self._coefs
    coefs = property(get_coefs, set_coefs)

    def set_response(self,Y):
        self._Y = Y

    def get_response(self):
        return self._Y
    Y = property(get_response, set_response)

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    def obj(self, beta):
        return self.obj_smooth(beta) + self.obj_rough(beta)

    def obj_rough(self, beta):
        return self.semi.evaluate(beta)

    def proximal(self, coefs, grad, L):
        return self.semi.proximal(coefs, grad, L)

class squaredloss(problem):

    """
    A class for combining squared error loss with a general seminorm
    """


    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
            self.n, self.p = self.X.shape
            self.semi.p = self.p
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)
            

    def obj_smooth(self, beta):
        #Smooth part of objective
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. 

    def grad(self, beta):
        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta)) 
        return XtXbeta - np.dot(self.Y,self.X) 

  
class dummy_problem(object):
    """
    A generic way to specify a problem
    """
    def __init__(self, smooth, grad_smooth, nonsmooth, prox, initial, L):
        self.coefs = initial * 1.
        self.obj_smooth = smooth
        self.nonsmooth = nonsmooth
        self.grad = grad_smooth
        self._prox = prox
        self.L = L

    def obj(self, x):
        return self.obj_smooth(x) + self.nonsmooth(x)

    def proximal(self, x, g, L):
        z = x - g / L
        return self._prox(z, L)

class seminorm_atom(object):

    """
    A seminorm atom class
    """

    def __init__(self, spec, l=1.):
        if type(spec) == type(1):
            self.p = self.m = spec
            self.D = None
        else:
            D = spec
            D = D.reshape((1,-1))
            self.D = D
            self.m, self.p = D.shape
        self.l = l
        
    def evaluate(self, x):
        """
        Abstract method. Evaluate the norm of x.
        """
        raise NotImplementedError

    def primal_prox(self, x, L):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \real^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(Dv)

        where *p*=x.shape[0] and :math:`h(v)`=self.evaluate(v).
        """
        raise NotImplementedError

    def dual_prox(self, u, L):
        """
        Return a minimizer

        .. math::

           v^{\lambda}(u) \in \text{argmin}_{v \in \real^m} \frac{L}{2}
           \|u-D'v\|^2_2  s.t.  h^*(v) \leq \lambda

        where *m*=u.shape[0] and :math:`h^*` is the 
        conjugate of self.evaluate.
        """
        raise NotImplementedError

    def multiply_by_DT(self, u):
        if self.D is not None:
            return np.dot(u, self.D)
        else:
            return u

    def multiply_by_D(self, x):
        if self.D is not None:
            return np.dot(self.D, x)
        else:
            return x

class l1norm(seminorm_atom):

    """
    The l1 norm
    """

    def evaluate(self, x):
        """
        The L1 norm of Dx.
        """
        if self.D is None:
            return self.l * np.fabs(x).sum()
        else:
            return self.l * np.fabs(np.dot(self.D, x)).sum()

    def primal_prox(self, x,  L):
        """
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \real^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|Dv\|_1

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
        If :math:`D=I` this is just soft thresholding

        .. math::

            v^{\lambda}(x) = \text{sign}(x) \max(|x|-\lambda/L, 0)
        """

        if self.D is None:
            return np.sign(v) * np.maximum(np.fabs(v)-self.l/L, 0)
        else:
            return FISTAsoln # will barf

    def dual_prox(self, u,  L):
        """
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \real^m} \frac{L}{2}
            \|u-D'v\|^2_2 s.t. \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
        This is just truncation: np.clip(u, -self.l/L, self.l/L).
        """
        return np.clip(u, -self.l/L, self.l/L)

class l2norm(seminorm_atom):

    """
    The l1 norm
    """

    def __init__(self, spec, l=1.):
        if type(spec) == type(1):
            self.p = self.m = spec
            self.D = None
        else:
            D = spec
            D = D.reshape((1,-1))
            self.D = D
            self.m, self.p = D.shape
        self.l = l
        
    def evaluate(self, x):
        """
        The L2 norm of Dx.
        """
        if self.D is not None:
            return self.l * np.sqrt((np.dot(self.D,x)**2).sum())
        else:
            return self.l * np.sqrt((x**2).sum())

    def primal_prox(self, x,  L):
        """
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \real^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda \|Dv\|_2

        where *p*=x.shape[0], :math:`\lambda`=self.l. 
        If :math:`D=I` this is just a "James-Stein" estimator

        .. math::

            v^{\lambda}(x) = \max(1 - \frac{\lambda/L}{\|x\|_2}, 0) x
        """

        if self.D is None:
            return x - self.dual_prox(x, L)
        else:
            return FISTAsoln

    def dual_prox(self, u,  L):
        """
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \real^m} \frac{L}{2}
            \|u-D'v\|^2_2 + \lambda \|v\|_2

        where *m*=u.shape[0], :math:`\lambda`=self.l. 
        This is just truncation

        .. math::

            v^{\lambda}(u) = \min(1, \frac{\lambda/L}{\|u\|_2}) u
        """
        n = self.evaluate(u)
        return min(1, (self.l/L) / n) * u

class seminorm(object):
    """
    A seminorm container class for storing/combining seminorm_atom classes
    """
    def __init__(self, *atoms):
        self.atoms = []
        self.primal_dim = -1
        self.segments = []
        idx = 0
        for atom in atoms:
            if self.primal_dim < 0:
                self.primal_dim = atom.p
            else:
                if atom.p != self.primal_dim:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
            self.segments.append(slice(idx, idx+atom.m))
            idx += atom.m
        self.total_dual = idx

    def __add__(self,y):
        #Combine two seminorms
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return seminorm(*atoms())

    def evaluate(self, x):
        out = 0.
        for atom in self.atoms:
            out += atom.evaluate(x)
        return out
    
    def dual_prox(self, u, L):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{L}{2}
           \|v-u\|^2_2  s.t.  h^*_i(v) \leq \infty, 0 \leq i \leq M-1

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h^*_i` is the conjugate of 
        self.atoms[i].l * self.atoms[i].evaluate and 
        :math:`\lambda_i`=self.atoms[i].l.

        This is used in the inner loop with :math:`u=z-g/L` when finding
        self.primal_prox, i.e., the signal approximator problem.
        """
        v = np.empty(u.shape)
        for atom, segment in zip(self.atoms, self.segments):
            v[segment] = atom.dual_prox(v[segment], atom.l/L)
        return v

    def primal_prox(self, x, L, solver=ISTA):
        dualp = self.dual_problem(x, L=L, solver=solver)
        dualp.debug = True
        dualp.fit(max_its=20000)
        return self.primal_from_dual(x, dualp.problem.coefs)

    def primal_from_dual(self, u, v):
        """

        """
        x = u * 1.
        for atom, segment in zip(self.atoms, self.segments):
            x -= atom.multiply_by_DT(v[segment])
        return x

    def dual_problem(self, y, L=1, solver=FISTA, initial=None):
        """
        Return a problem instance of the dual
        prox problem with a given y value.
        """
        def smooth(v,y=y):
            primal = self.primal_from_dual(y, v)
            return L * (primal**2).sum() / 2.
        def grad_smooth(v,y=y):
            primal = self.primal_from_dual(y, v)
            g = np.zeros(self.total_dual)
            for atom, segment in zip(self.atoms, self.segments):
                g[segment] = atom.multiply_by_D(primal)
            g *= -L
            return g
        prox = self.dual_prox
        nonsmooth = self.evaluate #XXX this should be the constraint one
        if initial is None:
            initial = np.random.standard_normal(self.total_dual)
        return solver(dummy_problem(smooth, grad_smooth, nonsmooth, prox, initial, L))

