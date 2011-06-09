import numpy as np
import pylab
from scipy import sparse

import regreg.api as R

n = 100
Y = np.random.standard_normal(n)
Y[:-30] += np.arange(n-30) * 0.2

D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]


isotonic = R.nonnegative.linear(sparse.csr_matrix(D), lagrange=1.)
loss = R.l2normsq.shift(-Y, coef=0.5)
p = R.container(loss, isotonic)
solver=R.FISTA(p.composite(initial=np.zeros(n)))
solver.debug=True

vals = solver.fit(max_its=25000, tol=1e-08, backtrack=True)
soln = solver.composite.coefs

X = np.arange(n)
pylab.clf()
pylab.scatter(X, Y)
pylab.step(X, soln, 'r--')
        
