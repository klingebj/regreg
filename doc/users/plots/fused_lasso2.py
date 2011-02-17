import numpy as np
import pylab

import regreg.regression as regreg
import regreg.signal_approximator as sapprox
        
import nose.tools

def test_fused_lasso(n=100, l1=0.02, ratio=.1):

    l1 *= n
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D = np.vstack([D, ratio*np.identity(n)])
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 
    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.

    p1 = sapprox.gengrad((D, Y), L=M)
    p1.assign_penalty(l1=l1)
    opt1 = regreg.FISTA(p1)
    opt1.fit(tol=1e-10, max_its=1000)
    beta1 = p1.output[0].copy()

    p1.assign_penalty(l1=2*l1)
    opt1.fit(tol=1e-06, max_its=1000)
    beta2 = p1.output[0].copy()
    
    X = np.linspace(0,1,n)
    pylab.scatter(X, Y)
    pylab.step(X, beta1, linewidth=3, c='red')
    pylab.step(X, beta2, linewidth=3, c='yellow')
    pylab.show()

test_fused_lasso()


