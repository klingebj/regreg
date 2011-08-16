
import numpy as np
import pylab	
from scipy import sparse

import regreg.api as rr

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.quadratic.shift(-Y, coef=0.5)
sparsity = rr.l1norm(len(Y), lagrange=0.8)
sparsity.lagrange += 1
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = rr.l1norm.linear(D, lagrange=25.5)
problem = rr.container(sparsity, fused, loss)
solver = rr.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = problem.coefs
pylab.plot(solution, c='g', linewidth=3)	
pylab.scatter(np.arange(Y.shape[0]), Y)


