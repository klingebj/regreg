import pylab
import numpy as np
import scipy.sparse

from regreg.algorithms import FISTA
from regreg.smooth import l2normsq
from regreg.atoms import l1norm, maxnorm
from regreg.seminorm import seminorm

D = (np.diag(np.ones(500)) - np.diag(np.ones(499),1))[:-1]
DT = scipy.sparse.csr_matrix(D.T)
D = scipy.sparse.csr_matrix(D)

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = l2normsq.shift(-Y, l=0.5)
penalty = l1norm(D, l=20)

problem = loss.add_seminorm(seminorm(penalty))
solver = FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = solver.problem.coefs

l1_soln = np.fabs(D * solution).sum()

tfocs_penalty = maxnorm(499, l=l1_soln)
tfocs_loss = l2normsq.affine(DT, -Y, l=0.5)
tfocs_loss.coefs = np.zeros(499)
tfocs_problem = tfocs_loss.add_seminorm(tfocs_penalty)
tfocs_solver = FISTA(tfocs_problem)
tfocs_solver.debug = True
tfocs_solver.fit(max_its=1000, tol=1e-10)
tfocs_dual_solution = tfocs_problem.coefs
tfocs_primal_solution = Y - DT * tfocs_dual_solution

import pylab
pylab.scatter(np.arange(Y.shape[0]), Y, c='r')
pylab.plot(solution, color='yellow', linewidth=5)
pylab.plot(tfocs_primal_solution, color='black', linewidth=3)


newl1 = l1norm(D, l=l1_soln)
conjugate = l2normsq.shift(Y, l=0.5)
from regreg.constraint import constraint
loss_constraint = constraint(conjugate, newl1)
new_solver = FISTA(loss_constraint.dual_problem())
new_solver.debug = True
new_solver.fit(max_its=1000, tol=1e-10)
soln3 = loss_constraint.primal_from_dual(new_solver.problem.coefs)
pylab.plot(soln3, color='gray', linewidth=1)

