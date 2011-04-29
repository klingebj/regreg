import pylab
import numpy as np
import scipy.sparse

from regreg.algorithms import FISTA
from regreg.smooth import l2normsq
from regreg.atoms import l1norm, maxnorm
from regreg.seminorm import seminorm


sparsity = l1norm(500, l=1.3)
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = scipy.sparse.csr_matrix(D)
fused = l1norm(D, l=20)

penalty = seminorm(sparsity,fused)


Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = l2normsq.shift(-Y, l=0.5)
problem = loss.add_seminorm(penalty)
solver = FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = solver.problem.coefs

import pylab
pylab.scatter(np.arange(Y.shape[0]), Y, c='r')
pylab.plot(solution, color='yellow', linewidth=5)

l1_fused = np.fabs(D * solution).sum()
l1_sparsity = np.fabs(solution).sum()

new_fused = l1norm(D, l=l1_fused)
new_sparsity = l1norm(500, l=l1_sparsity)
conjugate = l2normsq.shift(Y, l=0.5)
from regreg.constraint import constraint
loss_constraint = constraint(conjugate, new_fused, new_sparsity)
constrained_solver = FISTA(loss_constraint.dual_problem())
constrained_solver.debug = True
constrained_solver.fit(max_its=2000, tol=1e-10)
constrained_solution = loss_constraint.primal_from_dual(constrained_solver.problem.coefs)
pylab.plot(constrained_solution, color='black', linewidth=3)

