import numpy as np
import pylab
from scipy import sparse

from regreg.algorithms import FISTA
from regreg.atoms import nonnegative
from regreg.seminorm import seminorm
from regreg.smooth import signal_approximator, smooth_function

n = 100
Y = np.random.standard_normal(n)
Y[:-30] += np.arange(n-30) * 0.2

D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]


isotonic = seminorm(nonnegative(sparse.csr_matrix(D)))
loss = smooth_function(signal_approximator(Y))
p = loss.add_seminorm(isotonic, initial=np.ones(Y.shape)*Y.mean())
p.L = isotonic.power_LD()
solver=FISTA(p)

vals = solver.fit(max_its=25000, tol=1e-05, backtrack=False)
soln = solver.problem.coefs

X = np.arange(n)
pylab.clf()
pylab.scatter(X, Y)
pylab.step(X, soln, 'r--')
        
