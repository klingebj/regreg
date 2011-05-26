import numpy as np
import pylab
from scipy import sparse

from regreg.algorithms import FISTA
from regreg.atoms import nonnegative
from regreg.container import container
from regreg.smooth import signal_approximator, smooth_function

n = 100
Y = np.random.standard_normal(n)
Y[:-30] += np.arange(n-30) * 0.2

D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]


isotonic = nonnegative.linear(sparse.csr_matrix(D))
loss = signal_approximator(Y)
p = container(loss, isotonic)
solver=FISTA(p.problem(initial=np.zeros(n)))
solver.debug=True

vals = solver.fit(max_its=25000, tol=1e-08, backtrack=True)
soln = solver.problem.coefs

X = np.arange(n)
pylab.clf()
pylab.scatter(X, Y)
pylab.step(X, soln, 'r--')
        
