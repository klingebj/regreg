import numpy as np
import regreg.api as rr

np.random.seed(400)

N = 1000
P = 200

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]


X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)

s = rr.selector(slice(0,P), (P+1,))
sparsity = rr.l1norm.linear(s, lagrange=0.2)
quadratic = rr.l2normsq.linear(s, coef=0.5)
problem = rr.container(quadratic, hinge_loss, sparsity)
solver = rr.FISTA(problem)
solver.debug = True
solver.fit()
solver.composite.coefs


fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
print accuracy



## X_1 = np.hstack([X, np.ones((N,1))])
## transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
## C = 0.2
## hinge = rr.positive_part(N, lagrange=C)
## hinge_loss = rr.linear_atom(hinge, transform)
## smoothed_hinge_loss = rr.smoothed_atom(hinge_loss)

## sparsity = rr.l1norm(P, lagrange=2)

## problem = rr.container(quadratic, 
##                       smoothed_hinge_loss, sparsity)
## solver = rr.FISTA(problem)
## solver.fit()
## solver.problem.coefs

