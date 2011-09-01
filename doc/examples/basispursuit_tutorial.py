
import numpy as np
import scipy.linalg
import pylab

X = np.random.standard_normal((500,1000))

beta = np.zeros(1000)
beta[:100] = 3 * np.sqrt(2 * np.log(1000))

Y = np.random.standard_normal((500,)) + np.dot(X, beta)
Xnorm = scipy.linalg.eigvalsh(np.dot(X.T,X), eigvals=(998,999)).max()

import regreg.api as R
from regreg.smooth import linear
smooth_linf_constraint = R.smoothed_atom(R.supnorm(1000, bound=1),
                                        epsilon=0.01,
                                        store_argmin=True)
transform = R.linear_transform(-X.T)
loss = R.affine_smooth(smooth_linf_constraint, transform)


norm_Y = np.linalg.norm(Y)
l2_constraint_value = np.sqrt(0.1) * norm_Y
l2_lagrange = R.l2norm(500, lagrange=l2_constraint_value)

basis_pursuit = R.container(loss, linear(Y), l2_lagrange)
solver = R.FISTA(basis_pursuit)
tol = 1.0e-08

for epsilon in [0.6**i for i in range(20)]:
   smooth_linf_constraint.epsilon = epsilon
   solver.composite.lipschitz = 1.1/epsilon * Xnorm
   solver.fit(max_its=2000, tol=tol, min_its=10, backtrack=False)

basis_pursuit_soln = smooth_linf_constraint.argmin

sparsity = R.l1norm(1000, bound=np.fabs(basis_pursuit_soln).sum())
loss = R.quadratic.affine(X, -Y)
lasso = R.container(loss, sparsity)
lasso_solver = R.FISTA(lasso)
lasso_solver.fit(max_its=2000, tol=1.0e-10)
lasso_soln = lasso.coefs

pylab.plot(basis_pursuit_soln, label='Basis pursuit')
pylab.plot(lasso_soln, label='LASSO')
pylab.legend()
