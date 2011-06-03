"""

From http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf.
Specifically, on p. 70 \S 8.1.3, making the substitutions
:math:`x_i=\beta_i, z_i=\mu_i`.

.. math::

   \newcommand{\argmin}{\mathop{argmin}}
   \begin{aligned}
   \beta_i^{k+1} &= \argmin_{\beta_i} \left(\frac{\rho}{2} \|X_i\beta_i-X_i\beta_i^k - \bar{\mu}^k + \bar{X\beta}^k + u^k\|^2_2 + \lambda\|\beta\|_1 \right) \\
   \bar{\mu}^{k+1} &= \frac{1}{N+\rho} \left(y + \rho \bar{X\beta}^k + \rho u^k\right) \\
   u^{k+1} &= u^k + \bar{X\beta}^k - \bar{\mu}^{k+1}
   \end{aligned}
"""

import regreg.api as R
import numpy as np

# generate a data matrix

n, p = (500, 400)
X = np.random.standard_normal((n, p))
beta = 10 * np.ones(p)
beta[200:] = 0
Y = np.dot(X, beta) + np.random.standard_normal(n)

groups = [slice(0,200), slice(200,300), range(300,400)]
Xs = [X[:,group] for group in groups]

class LassoNode(object):

    """
    A class that can handle a LASSO problem, with
    changeable responses and Lagrange parameter
    """
    def __init__(self, X, initial=None, l=1, rho=1):
        self.X = R.affine_transform(X, None)
        self.atom = R.l1norm(X.shape[1], l)
        self.rho = rho
        self.loss = R.l2normsq.affine(X, -np.zeros(X.shape[0]), l=rho/2.)
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
    def get_l(self):
        return self.atom.l
    
    def set_l(self, l):
        self.atom.l = l
    l = property(get_l, set_l)

    # The coefficients for this node
    @property
    def beta(self):
        return self.solver.problem.coefs

    # The fitted values for this node
    @property
    def fitted(self):
        return self.X.linear_map(self.beta)

# the Lagrange penalty parameter, lambda
l = 40.

# the different nodes, living on different nodes 
# in an ipcluster. 
# the matrix X would have to be split up
# onto the different nodes...
lasso_nodes = [LassoNode(x, l=l) for x in Xs]

def update_lasso_nodes(lasso_nodes, mu_bar, Xbeta_bar, u):
    # this is a scatter -- no serial dependencies
    for node in lasso_nodes:
        node.response = node.fitted + mu_bar - Xbeta_bar - u
        node.solver.fit(max_its=1000, min_its=10, tol=1.0e-06)

def update_global_variables(lasso_nodes, y, u, rho):
    # this is a gather
    Xbeta_bar = np.mean([node.fitted for node in lasso_nodes], 0)

    N = len(lasso_nodes)
    mu_bar = (y + rho * (Xbeta_bar + u)) / (N + rho)
    u = u + Xbeta_bar - mu_bar
    return Xbeta_bar, mu_bar, u

mu_bar = 0 * Y
Xbeta_bar = 0 * Y
u = 0 * Y
rho = 1.

def objective(beta, X, Y, l):
    return np.linalg.norm(Y - np.dot(X, beta))**2 / 2. + l * np.fabs(beta).sum()

tol = 1.0e-10
old_obj = np.inf
for i in range(2000):
    update_lasso_nodes(lasso_nodes, mu_bar, Xbeta_bar, u)
    Xbeta_bar, mu_bar, u = update_global_variables(lasso_nodes, Y, u, rho)
    for g, n in zip(groups, lasso_nodes):
        beta[g] = n.beta
    new_obj = objective(beta, X, Y, l)
    if np.fabs(old_obj-new_obj) / np.fabs(new_obj) < tol:
        break
    old_obj = new_obj
    print 'Iteration %d, objective %0.2f' % (i, new_obj)

penalty = R.l1norm(p, l=l)
loss = R.l2normsq.affine(-X, Y, l=0.5)
lasso = R.container(loss, penalty)
solver = R.FISTA(lasso.problem())
solver.fit(tol=1.0e-10)

lasso_soln = solver.problem.coefs
distributed_soln = beta
