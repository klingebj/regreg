
import numpy as np
import pylab	
from scipy import sparse

import regreg.api as R
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = R.signal_approximator(Y)
sparsity = R.l1norm(len(Y), lagrange=0.8)
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)
problem = R.container(loss, sparsity, fused)
solver = R.admm_problem(problem)
solver.fit(max_its=1000, tol=1e-8)
solution = solver.beta
pylab.plot(solution, c='g', linewidth=3)	
pylab.scatter(np.arange(Y.shape[0]), Y)
