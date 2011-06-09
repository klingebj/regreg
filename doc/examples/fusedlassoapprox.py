import numpy as np
import pylab
from scipy import sparse

import regreg.api as R

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

sparsity = R.l1norm(500, lagrange=1.3)
#Create D
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)

loss = signal_approximator(Y)

problem = R.container(loss, sparsity, fused)
solver = R.FISTA(problem.composite())
solver.fit(max_its=800,tol=1e-10)
soln = solver.composite.coefs

#plot solution
pylab.figure(num=1)
pylab.clf()
pylab.plot(soln, c='g')
pylab.scatter(np.arange(Y.shape[0]), Y)
    

