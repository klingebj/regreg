import numpy as np
import regreg.api as rr

np.random.seed(400)

N = 500
P = 2

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([3,-2])[np.newaxis,:]

X_1 = np.hstack([X, np.ones((N,1))])
X_1_signs = -Y[:,np.newaxis] * X_1
transform = rr.affine_transform(X_1_signs, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)

quadratic = rr.l2normsq.linear(rr.selector(slice(0,P), (P+1,)), coef=0.5)
problem = rr.container(quadratic, hinge_loss)
solver = rr.FISTA(problem)
solver.fit()

import pylab
pylab.clf()
pylab.scatter(X[Y==1,0],X[Y==1,1], facecolor='red')
pylab.scatter(X[Y==-1,0],X[Y==-1,1], facecolor='blue')

fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1

pointX = [X[:,0].min(), X[:,0].max()]
pointY = [-(pointX[0]*problem.coefs[0]+problem.coefs[2])/problem.coefs[1],
          -(pointX[1]*problem.coefs[0]+problem.coefs[2])/problem.coefs[1]]
pylab.plot(pointX, pointY, linestyle='--', label='Separating hyperplane')
pylab.title("Accuracy = %0.1f %%" % (100-100 * np.fabs(labels - Y).sum() / (2 * N)))
#pylab.show()
