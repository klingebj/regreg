import numpy as np

# Local imports

from problems import linmodel

problem_statement = r"""
    The signal approximator problem minimizes the following
    as a function of :math:`\beta`

    .. math::

       \frac{1}{2}\|y - \beta\|^{2}_{2}  + \lambda_1 \|D\beta\|_1

    It does this by solving the dual problem, which minimizes
    the following as a function of *u*

    .. math::

       \frac{1}{2}\|y - D'u\|^{2}_{2}  \ \ \text{s.t.} \ \ \|u\|_{\infty}
       \leq  \lambda_1

"""

class signal_approximator(linmodel):

    __doc__ = problem_statement

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            self.D = data[0]
            self.Y = data[1]
            self.m, self.n = self.D.shape
        else:
            raise ValueError("Data tuple not as expected")

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
        return np.zeros(self.m)

    def obj(self, dual):
        dual = np.asarray(dual)
        if np.max(np.fabs(dual)) <= self.penalties['l1']:
            return self.f(dual)
        else:
            return np.inf
    
    def f(self, dual):
        dual = np.asarray(dual)
        return ((self.Y - np.dot(self.D.T, dual))**2).sum() / 2.
    
    def grad(self, dual):
        dual = np.asarray(dual)
        return np.dot(self.D, np.dot(self.D.T, dual)) - np.dot(self.Y,self.D.T)
    
    def proximal(self, z, g, L):
        v = z - g / L
        l1 = self.penalties['l1']
        return np.clip(v, -l1, l1)
    
    @property
    def output(self):
        r = np.dot(self.coefs, self.D) 
        return self.Y - r, r

class signal_approximator_sparse(signal_approximator):

    __doc__ = problem_statement + """

    This class allows *D* to be a *scipy.sparse* matrix.

    """

    @property
    def output(self):
        r = self.DT.matvec(self.coefs)
        return self.Y - r, r

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            self.D = data[0]
            self.DT = self.D.transpose()
            self.Y = data[1]
            self.m, self.n = self.D.shape
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    def obj(self, u):
        u = np.asarray(u)
        beta = self.Y - self.DT.matvec(u)
        Dbeta = self.D.matvec(beta)
        return ((self.Y - beta)**2).sum() / 2. + np.sum(np.fabs(Dbeta)) * self.penalties['l1']

    def grad(self, u):
        u = np.asarray(u)
        return self.D.matvec(self.DT.matvec(u) - self.Y)

# The API is to have a gengrad class in each module.
# In this module, this is the signal_approximator

gengrad = signal_approximator

# If the matrix D is sparse, use gengrad_sparse

gengrad_sparse = signal_approximator_sparse
