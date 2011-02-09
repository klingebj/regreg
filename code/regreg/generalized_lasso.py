import numpy as np

from regression import FISTA
from problems import linmodel
from signal_approximator import signal_approximator

class generalized_lasso(linmodel):

    dualcontrol = {'max_its':50,
                   'tol':1.0e-06}

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 3:
            self.X = data[0]
            self.D = data[1]
            self.Y = data[2]
            self.n, self.p = self.X.shape
            self.m = self.D.shape[0]
        else:
            raise ValueError("Data tuple not as expected")

        self.dual = signal_approximator((self.D, self.Y))
        self.dualopt = FISTA(self.dual)
        self.dualM = np.linalg.eigvalsh(np.dot(self.dual.D.T, self.dual.D)).max() 

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
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    # this is the core generalized LASSO functionality

    def obj(self, beta):
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + np.sum(np.fabs(np.dot(self.D, beta))) * self.penalties['l1']

    def grad(self, beta):
        return np.dot(self.X.T, np.dot(self.X, beta) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        self.dual.set_response(v)
        self.dual.assign_penalty(l1=self.penalties['l1'] / L)
        self.dualopt.fit(self.dualM, **self.dualcontrol)
        return self.dualopt.output[0]

    @property
    def output(self):
        r = self.Y - np.dot(self.X, self.coefs) 
        return self.coefs, r

# The API is to have a gengrad class in each module.
# In this module, this is the signal_approximator

gengrad = generalized_lasso

