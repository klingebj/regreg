import numpy as np
import pylab
from scipy import sparse

import regreg.api as R

n = 100
Y = np.random.standard_normal(n)
Y[:-30] += np.arange(n-30) * 0.2

D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
nisotonic = R.positive_part.linear(-sparse.csr_matrix(D), lagrange=1)

loss = R.l2normsq.shift(-Y,coef=0.5)
p = R.container(loss, nisotonic)
solver = R.FISTA(p.composite())

vals = solver.fit(max_its=25000, tol=1e-05)
soln = solver.composite.coefs.copy()

nisotonic.atom.lagrange = 2.
p = R.container(loss, nisotonic)
solver = R.FISTA(p.composite())
solver.fit(max_its=25000, tol=1e-05)
soln2 = solver.composite.coefs.copy()

nisotonic.atom.lagrange = 100.
p = R.container(loss, nisotonic)
solver = R.FISTA(p.composite())
solver.fit(max_its=25000, tol=1e-05)
soln3 = solver.composite.coefs.copy()

X = np.arange(n)
pylab.clf()
pylab.scatter(X, Y)
pylab.step(X, soln, 'r--', linewidth=3, label='l=1')
pylab.step(X, soln2, 'g--', linewidth=3, label='l=2')
pylab.step(X, soln3, 'y--', linewidth=3, label='l=100')
pylab.legend()
