import pylab
import numpy as np
import scipy.sparse

from regreg.algorithms import FISTA
from regreg.smooth import l2normsq
from regreg.atoms import l1norm, maxnorm
from regreg.seminorm import seminorm

D = (np.diag(np.ones(500)) - np.diag(np.ones(499),1))[:-1]
DT = scipy.sparse.csr_matrix(D.T)

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = l2normsq.shift(-Y)
penalty = l1norm(D, l=25)

problem = loss.add_seminorm(seminorm(penalty))
solver = FISTA(problem)
solver.fit(max_its=1000, tol=1e-10)
solution = solver.problem.coefs

l1_soln = np.fabs(np.dot(D, solution)).sum()

tfocs_penalty = maxnorm(499, l=l1_soln)
tfocs_loss = l2normsq.affine(DT, -Y)
tfocs_loss.coefs = np.zeros(499)
tfocs_problem = tfocs_loss.add_seminorm(tfocs_penalty)
tfocs_solver.debug = True
tfocs_solver = FISTA(tfocs_problem)
tfocs_solver.fit(maxx_its=1000, tol=1e-10, monotonicity_restart=False)
tfocs_solution = tfocs_problem.coefs

