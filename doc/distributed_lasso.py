"""

From http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf.
Specifically, on p. 70 \S 8.1.3, making the substitutions
:math:`x_i=\beta_i, z_i=\mu_i`.

.. math::

   \newcommand{\argmin}{\mathop{argmin}}
   \begin{aligned}
   \beta_i^{k+1} &= \argmin_{\beta_i} \left(\frac{\rho}{2} \|X_i\beta_i-X_i\beta_i^k - \bar{\mu}^k + \bar{X\beta}^k + u^k\|^2_2 + \lambda\|\beta_i\|_1 \right) \\
   \bar{\mu}^{k+1} &= \frac{1}{N+\rho} \left(y + \rho \bar{X\beta}^k + \rho u^k\right) \\
   u^{k+1} &= u^k + \bar{X\beta}^k - \bar{\mu}^{k+1}
   \end{aligned}
"""

import regreg.api as R
import numpy as np

# np.random.seed(1)  # for debugging

class LassoNode(object):

    """
    A class that can handle a LASSO problem, with
    changeable responses and Lagrange parameter
    """
    def __init__(self, X, initial=None, lagrange=1, rho=1):
        self.X = R.affine_transform(X, None)
        self.atom = R.l1norm(X.shape[1], lagrange)
        self.rho = rho
        self.loss = R.l2normsq.affine(X, -np.zeros(X.shape[0]), lagrange=rho/2.)
        self.lasso = R.container(self.loss, self.atom)
        self.solver = R.FISTA(self.lasso.problem())

        if initial is None:
            self.beta[:] = np.random.standard_normal(self.atom.primal_shape)
        else:
            self.beta[:] = initial

    def set_response(self, response):
        self.loss.affine_transform.affine_offset[:] = -response

    def get_response(self):
        return -self.loss.affine_transform.affine_offset
    response = property(get_response, set_response)

    # The Lagrange parameter
    def get_lagrange(self):
        return self.atom.lagrange
    
    def set_lagrange(self, lagrange):
        self.atom.lagrange = lagrange
    lagrange = property(get_lagrange, set_lagrange)

    # The coefficients for this node
    @property
    def beta(self):
        return self.solver.problem.coefs

    # The fitted values for this node
    @property
    def fitted(self):
        return self.X.linear_map(self.beta)


if __name__ == '__main__':
    import os.path

    # Load IPython parallel resources, a cluster should have been instantiated
    from IPython.parallel import Client
    rc = Client()
    # We actually work with the view, in this case using all engines
    view = rc[:]
    # Use the view in synchronous mode, we don't have any major space below for
    # overlapping local and remote computation, and this is easier to work with
    view.block = True

    @view.remote()
    def update_lasso_nodes(pseudo_response, tol):
        node.response = node.fitted + pseudo_response
        node.solver.fit(max_its=1000, min_its=10, tol=tol)
        beta[:] = node.beta
        return node.fitted

    def update_global_variables(lasso_fits, y, u, rho):
        # this is a reduction operation
        Xbeta_bar = np.mean(lasso_fits, 0)

        N = len(lasso_fits)
        mu_bar = (y + rho * (Xbeta_bar + u)) / (N + rho)
        u = u + Xbeta_bar - mu_bar
        return Xbeta_bar, mu_bar, u

    def objective(beta, X, Y, l):
        return np.linalg.norm(Y - np.dot(X, beta))**2 / 2. + \
               l * np.fabs(beta).sum()

    # generate a data matrix
    n, p = (500, 400)
    X = np.random.standard_normal((n, p))
    beta = 10 * np.ones(p)
    beta[200:] = 0
    Y = np.dot(X, beta) + np.random.standard_normal(n)

    # Scatter works on the first dimension, and we want to break things up by
    # columns, so we transpose before scattering and transpo
    view.scatter('Xt', X.T)

    @view.remote()
    def node_init(lagrange, path):
        global node, beta
        import os
        import numpy as np
        os.chdir(path)
        import distributed_lasso as dl

        #np.random.seed(1) # for debugging
        node = dl.LassoNode(Xt.T, lagrange=lagrange)
        beta = np.empty(Xt.shape[0])

    # the Lagrange penalty parameter, lambda
    lagrange = 40.
    mu_bar = 0 * Y
    Xbeta_bar = 0 * Y
    u = 0 * Y
    rho = 1.
    tol = 1.0e-10

    # This initializes all the nodes and creates the Lasso objects
    node_init(lagrange, os.path.dirname(os.path.abspath(__file__)))
    
    old_obj = np.inf
    for i in range(2000):
        lasso_fits = update_lasso_nodes(mu_bar - Xbeta_bar - u, tol)
        Xbeta_bar, mu_bar, u = update_global_variables(lasso_fits, Y, u, rho)
        beta = view.gather('beta')
        new_obj = objective(beta, X, Y, lagrange)
        if np.fabs(old_obj-new_obj) / np.fabs(new_obj) < tol:
            break
        old_obj = new_obj
        print 'Iteration %d, objective %0.2f' % (i, new_obj)

    #np.random.seed(1) # for debugging

    penalty = R.l1norm(p, lagrange=lagrange)
    loss = R.l2normsq.affine(-X, Y, lagrange=0.5)
    lasso = R.container(loss, penalty)
    solver = R.FISTA(lasso.problem())
    solver.fit(tol=tol)

    lasso_soln = solver.problem.coefs
    distributed_soln = beta
