import numpy as np
from problems import glasso_signal_approximator as glasso_dense

class glasso_signal_approximator(glasso_dense):

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
        r = self.DT.matvec(self.coefficients)
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
            self.DT = self.D.transpose()
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
        beta = self.Y - self.DT.matvec(u)
        Dbeta = self.D.matvec(beta)
        return ((self.Y - beta)**2).sum() / 2. + np.sum(np.fabs(Dbeta)) * self.penalties['l1']

    def grad(self, u):
        u = np.asarray(u)
        return self.D.matvec(self.DT.matvec(u) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        l1 = self.penalties['l1']
        return np.clip(v, -l1, l1)
