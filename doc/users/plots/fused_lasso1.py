import numpy as np
import pylab

import regreg.regression as regreg
import regreg.signal_approximator as sapprox
        
import nose.tools

def test_fused_lasso(n=400, l1=0.2):

    l1 *= n
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 
    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.

    problem = sapprox.gengrad((D, Y))
    problem.assign_penalty(l1=l1)
    opt = regreg.FISTA(problem)
    opt.debug = True
    opt.fit(M,tol=1.0e-06, max_its=1000)
    beta1 = problem.output[0].copy()

    problem.assign_penalty(l1=2*l1)
    opt.fit(M,tol=1.0e-06, max_its=1000)
    beta2 = problem.output[0].copy()
    
    X = np.linspace(0,1,n)
    pylab.scatter(X, Y)
    pylab.step(X, beta1, linewidth=3, c='red')
    pylab.step(X, beta2, linewidth=3, c='yellow')


test_fused_lasso()
pylab.show()

