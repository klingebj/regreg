import numpy as np

# Local imports

import subfunctions as sf
import updates
import l1smooth

class linmodel(object):

    def output(self):
        return self.coefficients, self.r

    def __init__(self, data, **kwargs):
        
        self.penalties = self.default_penalty()
        if 'penalties' in kwargs:
            self.assign_penalty(**kwargs['penalties'])
        if 'initial_coefs' in kwargs:
            self.initial_coefs = kwargs['initial_coefs']
        if 'update_resids' in kwargs:
            self.update_resids = kwargs['update_resids']
        else:
            self.update_resids = True

        self.initialize(data, **kwargs)
    
    def assign_penalty(self, **params):
        """
        Abstract method for assigning penalty parameters.
        """
        penalties = self.penalties.copy()
        for key in params:
            penalties[key] = params[key]
        self.penalties = penalties
        
    def set_coefficients(self, coefs):
        if coefs is not None:
            self.beta = coefs.copy()
            if self.update_resids:    
                self.update_residuals()

    def get_coefficients(self):
        return self.beta.copy()
    coefficients = property(get_coefficients, set_coefficients)

    def set_response(self,Y):
        if Y is not None:
            self.Y = Y
            if self.update_resids:    
                self.update_residuals()
            if hasattr(self,'inner'):
                self.inner = sf.col_inner(self.X,self.Y)

    def get_response(self):
        return self.Y.copy()

    response = property(get_response, set_response)

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

    def update_residuals(self):
        if hasattr(self,'rowwts'):
            self.r = self.Y - self.rowwts * np.dot(self.X,self.beta)    
        else:
            self.r = self.Y - np.dot(self.X,self.beta)

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

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        
        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
            self.n, self.p = self.X.shape
        else:
            raise ValueError("Data tuple not as expected")

        self._ssq = sf.col_ssq(self.X)
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
        beta = np.asarray(beta)
        return ((self.Y - np.dot(self.X, beta))**2).sum() / (2.*len(self.Y)) + np.sum(np.fabs(beta)) * self.penalties['l1']

    def grad(self,beta):
        beta = np.asarray(beta)
        return (sf.multlist(self.X,np.dot(self.X, beta),transpose=True) - np.dot(self.Y,self.X)) / (1.*len(self.Y))

    def proximal(self, z, g, L):
        v = z - g / L
        return np.sign(v) * np.maximum(np.fabs(v)-self.penalties['l1']/L, 0)

    def smooth(self, L, epsilon):
        return l1smooth.l1smooth(self.grad, L, epsilon, l1=self.penalties['l1'], f=self.obj)

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

class glasso_dual(linmodel):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - D'u||^{2}_{2} s.t. \|u\|_{\infty} \leq  \lambda_{1}
       \end{eqnarray}

    as a function of u.
    """

    def output(self):
        r = np.dot(self.coefficients, self.D) 
        return self.Y - r, r

    def set_coefficients(self, coefs):
        if coefs is not None:
            self.u = coefs.copy()

    def get_coefficients(self):
        return self.u.copy()

    coefficients = property(get_coefficients, set_coefficients)

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            self.D = data[0]
            self.Y = data[1]
            self.n, self.p = self.D.shape
        else:
            raise ValueError("Data tuple not as expected")

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
        self.set_coefficients(np.zeros(self.n))

    def obj(self, u):
        u = np.asarray(u)
        beta = self.Y - np.dot(u, self.D)
        return ((self.Y - beta)**2).sum() / 2. + np.sum(np.fabs(np.dot(self.D, beta))) * self.penalties['l1']

    def grad(self, u):
        u = np.asarray(u)
        return np.dot(self.D, np.dot(u, self.D) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        l1 = self.penalties['l1']
        return np.clip(v, -l1, l1)

