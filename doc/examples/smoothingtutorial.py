import numpy as np
import pylab	
from scipy import sparse

import regreg.api as R

# generate the data

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

loss = R.l2normsq.shift(-Y, coef=1)
sparsity = R.l1norm(len(Y), lagrange=1.8)

# fused
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)

smoothed_sparsity = R.smoothed_atom(sparsity, epsilon=0.01)
smoothed_fused = R.smoothed_atom(fused, epsilon=0.01)

problem = R.smooth_function(loss, smoothed_sparsity, smoothed_fused)
solver = R.FISTA(problem)

solns = []
for eps in [.5**i for i in range(15)]:
   smoothed_fused.epsilon = smoothed_sparsity = eps
   solver.fit()
   solns.append(solver.composite.coefs.copy())
   pylab.plot(solns[-1])
