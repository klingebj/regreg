import numpy as np

# Local imports

from problems import linmodel

problem_statement = """
    The signal approximator problem minimizes the following
    as a function of :math:`beta`

    .. math::

       \begin{eqnarray}
       \frac{1}{2}\|y - \beta\|^{2}_{2}  + \lambda_1 \|D\beta\|_1\|
       \end{eqnarray}

    It does this by solving the dual problem, which minimizes
    the following as a function of *u*

    .. math::

       \begin{eqnarray}
       \frac{1}{2}\|y - D'u\|^{2}_{2}  \ \ \text{s.t.} \ \ \|u\|_{\infty}
       \leq  \lambda_1
       \end{eqnarray}
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
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2. + np.sum(np.fabs(np.dot(self.D, beta))) * self.penalties['l1']

    def f(self, dual):
        dual = np.asarray(dual)
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2. 


    def grad(self, dual):
        dual = np.asarray(dual)
        return np.dot(self.D, np.dot(dual, self.D) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        l1 = self.penalties['l1']
        return np.clip(v, -l1/L, l1/L)

    @property
    def output(self):
        r = np.dot(self.coefs, self.D) 
        return self.Y - r, r

class signal_approximator_sparse(signal_approximator):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - D'u||^{2}_{2} s.t. \|u\|_{\infty} \leq  \lambda_{1}
       \end{eqnarray}

    as a function of u.
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
