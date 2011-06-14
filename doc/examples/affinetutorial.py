import numpy as np
import pylab	
from scipy import sparse

np.random.seed(40)
import regreg.api as R

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

alpha = np.linspace(0,10,500)
Y += alpha
loss = R.l2normsq.shift(-Y.copy(), coef=0.5)

shrink_to_alpha = R.l1norm(Y.shape, offset=-alpha, lagrange=3.)

D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)

cont = R.container(loss, shrink_to_alpha, fused)
solver = R.FISTA(cont)
solver.debug = True
solver.fit(max_its=200, tol=1e-10)
solution = solver.composite.coefs

block_soln = R.blockwise([shrink_to_alpha, fused], Y, max_its=500, tol=1.0e-10)
np.linalg.norm(block_soln - solution) / np.linalg.norm(solution)
cont.objective(block_soln), cont.objective(solution)

pylab.clf()
pylab.plot(solution, c='g', linewidth=6, label=r'$\hat{Y}$')	
pylab.plot(alpha, c='black', linewidth=3, label=r'$\alpha$')	
pylab.scatter(np.arange(Y.shape[0]), Y, facecolor='red', label=r'$Y$')
pylab.plot(block_soln, c='yellow', linewidth=2, label='blockwise')	
pylab.legend()


pylab.gca().set_xlim([0,650])
pylab.legend()
