import numpy as np

from regression import FISTA
from problems import linmodel
from signal_approximator import signal_approximator

class group_approximator(signal_approximator):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - D'u||^{2}_{2} s.t. \|u\|_{\infty} \leq  \lambda_{1}
       \end{eqnarray}

    as a function of u and returns (y - u, u) as output.
    """

    @property
    def default_penalties(self):
        return {}

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            penalties = {}
            self.Ds = []
            self.segments = []
            idx = 0
            for i, v in enumerate(data[0]):
                D, penalty = v
                D = np.atleast_2d(D)
                self.Ds.append(D) 
                self.segments.append(slice(idx, idx+D.shape[0]))
                idx += D.shape[0]
                penalties['V%d' % i] =penalty
            self.assign_penalty(**penalties)
            self.Y = data[1]
            self.D = np.vstack(self.Ds)
            self.n = self.Y.shape[0]
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    @property
    def default_coefs(self):
        return np.zeros(self.D.shape[0])

    def compute_penalty(self, beta):
        pen = 0
        for i, D in enumerate(self.Ds):
            pen += self.penalties['V%d' % i] * norm2(np.dot(D, beta))
        return pen

    def obj(self, dual):
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2. + self.compute_penalty(beta)

    def grad(self, dual):
        dual = np.asarray(dual)
        return np.dot(self.D, np.dot(dual, self.D) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        for i, segment in enumerate(self.segments):
            l = self.penalties['V%d' % i]
            v[segment] = truncate(v[segment], l/L)
        return v

    def f(self, dual):
        #Smooth part of objective
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2.
                            

    @property
    def output(self):
        r = np.dot(self.coefs, self.D) 
        return self.Y - r, r

class group_lasso(linmodel):

    dualcontrol = {'max_its':50,
                   'tol':1.0e-06}

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 3:
            self.X = data[0]
            self.Dv = data[1]
            penalties = {}
            for i, v in enumerate(data[1]):
                _, penalty = v
                penalties['V%d' % i] =penalty
            self.assign_penalty(**penalties)
            self.Y = data[2]
            self.n, self.p = self.X.shape
        else:
            raise ValueError("Data tuple not as expected")

        self.dual = group_approximator((self.Dv, self.Y))
        self.dualopt = FISTA(self.dual)

        self.m = self.dual.D.shape[0]

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    @property
    def default_penalties(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #XXX maybe use a recarray for the penalties
        return {}

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    # this is the core generalized LASSO functionality

    def obj(self, beta):
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + self.dual.compute_penalty(beta)

    def grad(self, beta):
        return np.dot(self.X.T, np.dot(self.X, beta) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        self.dual.set_response(v)
        #XXX this is painful -- maybe do it with a recarray multiplication?
        penalties = {}
        for i in range(len(self.dual.Ds)):
            penalties['V%d' % i] = self.penalties['V%d' % i] / L
        self.dual.assign_penalty(**penalties)
        self.dualopt.fit(**self.dualcontrol)
        return self.dualopt.output[0]

    @property
    def output(self):
        r = self.Y - np.dot(self.X, self.coefs) 
        return self.coefs, r


def norm2(V):
    """
    The Euclidean norm of a vector.
    """
    return np.sqrt((V**2).sum())

def truncate(V, l):
    """
    Vector truncated to have norm <= l (projection onto
    Euclidean ball of radius l.
    """
    normV = norm2(V)
    if normV <= l:
        return V
    else:
        return V * (l / normV)

def james_stein(V, l):
    """
    James-Stein estimator:

    V - truncate(V, l)
    """
    normV = norm2(V)
    return max(1 - l / normV, 0) * V

# The API is to have a gengrad class in each module.
# In this module, this is the signal_approximator

gengrad = group_lasso
