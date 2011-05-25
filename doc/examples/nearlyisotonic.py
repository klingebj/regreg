import numpy as np
import pylab
from scipy import sparse

from regreg.algorithms import FISTA
from regreg.atoms import positive_part
from regreg.container import container
from regreg.smooth import l2normsq


n = 100
Y = np.random.standard_normal(n)
Y[:-30] += np.arange(n-30) * 0.2

D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
nisotonic = positive_part.linear(-sparse.csr_matrix(D), l=3)


loss = l2normsq.shift(-Y,l=0.5)
p = container(loss, nisotonic)
solver=FISTA(p.problem())

vals = solver.fit(max_its=25000, tol=1e-05)
soln = solver.problem.coefs.copy()

nisotonic.atoms[0].l = 100.
solver.fit(max_its=25000, tol=1e-05)
soln2 = solver.problem.coefs.copy()

nisotonic.atoms[0].l = 1000.
solver.fit(max_its=25000, tol=1e-05)
soln3 = solver.problem.coefs.copy()

X = np.arange(n)
pylab.clf()
pylab.scatter(X, Y)
pylab.step(X, soln, 'r--', linewidth=3)
pylab.step(X, soln2, 'g--', linewidth=3)
pylab.step(X, soln3, 'y--', linewidth=3)

