import numpy as np
import pylab
from scipy import sparse

from regreg.algorithms import FISTA
from regreg.atoms import l1norm
from regreg.container import container
from regreg.smooth import signal_approximator

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

sparsity = l1norm(500, l=1.3)
#Create D
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = l1norm.linear(D, l=25.5)

loss = signal_approximator(Y)

problem = container(loss, sparsity, fused)
solver = FISTA(problem.problem())
solver.fit(max_its=800,tol=1e-10)
soln = solver.problem.coefs

#plot solution
pylab.figure(num=1)
pylab.clf()
pylab.plot(soln, c='g')
pylab.scatter(np.arange(Y.shape[0]), Y)
    

