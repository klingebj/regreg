import numpy as np

from regression import FISTA
from problems import glasso_signal_approximator, linmodel

class glasso(linmodel):

    dualcontrol = {'max_its':50,
                   'tol':1.0e-06}

    def output(self):
        r = self.Y - np.dot(self.X, self.coefficients) 
        return self.coefficients, r

    def set_coefficients(self, coefs):
        if coefs is not None:
            self.beta = coefs.copy()

    def get_coefficients(self):
        return self.beta.copy()

    coefficients = property(get_coefficients, set_coefficients)

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 3:
            self.X = data[0]
            self.pinvX = np.linalg.pinv(self.X)
            self.D = data[1]
            self.Y = data[2]
            self.n, self.p = self.X.shape
            self.m = self.D.shape[0]
        else:
            raise ValueError("Data tuple not as expected")

        self.dual = glasso_signal_approximator((self.D, self.Y))
        self.dualopt = FISTA(self.dual)
        self.dualM = np.linalg.eigvalsh(np.dot(self.dual.D.T, self.dual.D)).max() 
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(self.p))

    def obj(self, beta):
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + np.sum(np.fabs(np.dot(self.D, beta))) * self.penalties['l1']

    def grad(self, beta):
        return np.dot(self.X, np.dot(beta, self.X) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        self.dual.set_response(v)
        self.dual.assign_penalty(l1=self.penalties['l1'] / L)
        self.dualopt.fit(self.dualM, **self.dualcontrol)
        return self.dualopt.output()[0]

    def assign_penalty(self, **params):
        """
        Abstract method for assigning penalty parameters.
        """
        penalties = self.penalties.copy()
        for key in params:
            penalties[key] = params[key]
        self.penalties = penalties
