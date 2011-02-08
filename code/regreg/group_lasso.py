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

    def set_coefficients(self, coefs):
        self.dual = coefs

    def get_coefficients(self):
        return self.dual
    coefficients = property(get_coefficients, set_coefficients)

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            self.penalties = {}
            self.Ds = []
            self.segments = []
            idx = 0
            for i, v in enumerate(data[0]):
                D, penalty = v
                self.Ds.append(D) 
                self.segments.append(slice(idx, idx+D.shape[0]))
                self.assign_penalties(**{'V%d' % i:penalty})
            self.Y = data[1]
            self.D = np.vstack(self.Ds)
            self.n = self.Y.shape[0]
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        else:
            self.set_coefficients(self.default_coefs)

    @property
    def default_coefs(self):
        return np.zeros(self.D.shape[0])

    def obj(self, dual):
        beta = self.Y - np.dot(dual, self.D)
        pen = 0
        for i, D in enumerate(self.Ds):
            pen += self.penalties['V%d' % i] * np.sqrt(np.sum((np.dot(D, beta))**2))
        return ((self.Y - beta)**2).sum() / 2. + pen

    def grad(self, dual):
        dual = np.asarray(dual)
        return np.dot(self.D, np.dot(dual, self.D) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L

        l1 = self.penalties['l1']
        return np.clip(v, -l1/L, l1/L)

    @property
    def output(self):
        r = np.dot(self.dual, self.D) 
        return self.Y - r, r

def james_stein(V, l):
    normV = np.sqrt((V**2).sum())
    return max(1 - l / normV, 0) * V
