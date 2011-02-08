import numpy as np

# Local imports

import subfunctions as sf
import updates
import l1smooth
from problems import linmodel

class lasso(linmodel):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}
       \end{eqnarray}

    as a function of beta.
    """

    @property
    def default_penalties(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([('l1', np.float)]))

    def obj(self, beta):
        beta = np.asarray(beta)
        return ((self.Y - np.dot(self.X, beta))**2).sum() / (2.*len(self.Y)) + np.sum(np.fabs(beta)) * self.penalties['l1']

class gengrad(lasso):
    
    def grad(self,beta):
        beta = np.asarray(beta)
        return (sf.multlist(self.X,np.dot(self.X, beta),transpose=True) - np.dot(self.Y,self.X)) / (1.*len(self.Y))

    def proximal(self, z, g, L):
        v = z - g / L
        return np.sign(v) * np.maximum(np.fabs(v)-self.penalties['l1']/L, 0)

class gengrad_smooth(gengrad):

    def smooth(self, L, epsilon):
        return l1smooth.l1smooth(self.grad, L, epsilon, l1=self.penalties['l1'], f=self.obj)


class cwpath(lasso):

    def __init__(self, data, penalties={}, initial_coefs=None):
        lasso.__init__(self, data, penalties=penalties,
                       initial_coefs=initial_coefs)

    # Differences to the standard initialization
        
    def initialize(self, data):
        lasso.initialize(self, data)
        self._ssq = sf.col_ssq(self.X)

    # Residuals must be updated if coefs change

    def set_coefficients(self, coefs):
        if coefs is not None:
            self.beta = coefs
            self.update_residuals()

    def get_coefficients(self):
        return self.beta
    coefficients = property(get_coefficients, set_coefficients)

    def update_residuals(self):
        self.r = self.Y - np.dot(self.X,self.beta)

    # Inner products also must be updated if inner response changes
    def set_response(self,Y):
        if Y is not None:
            self.Y = Y
            self.update_residuals()
            self.inner = sf.col_inner(self.X,self.Y)

    def get_response(self):
        return self.Y
    response = property(get_response, set_response)

    def update_cwpath(self,
                      active,
                      nonzero,
                      inner_its = 1,
                      permute = False,
                      update_nonzero = False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            updates._update_lasso_cwpath(active,
                                         self.penalties,
                                         nonzero,
                                         self.beta,
                                         self.r,
                                         self.X,
                                         self._ssq,
                                         inner_its,
                                         update_nonzero)

    def set_rowweights(self, weights):
        if weights is not None:
            self.rowwts = weights
            if self.update_resids:
                self.update_residuals()
            if hasattr(self,'_ssq'):
                self._ssq = sf.col_ssq(self.X,self.rowwts)

    def get_rowweights(self):
        if hasattr(self,'rowwts'):
            return self.rowwts.copy()
        else:
            return None
    rowweights = property(get_rowweights, set_rowweights)
