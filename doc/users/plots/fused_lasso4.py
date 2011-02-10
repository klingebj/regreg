import numpy as np
import scipy.sparse
import pylab

import regreg.regression as regreg
import regreg.signal_approximator as sapprox
        
import nose.tools

def test_fused_lasso(n=500, l1=10., ratio=0.1):

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D2 = np.vstack([D2, ratio*np.identity(n)])
    M = np.linalg.eigvalsh(np.dot(D2.T, D2)).max() 
    print M
    Dsp = scipy.sparse.lil_matrix(D2)

    Y = np.random.standard_normal(n) * 0.2
    X = np.linspace(0,1,n)
    mu = 0 * Y
    mu[int(0.1*n):int(0.3*n)] += (X[int(0.1*n):int(0.3*n)] - X[int(0.1*n)]) * 6
    mu[int(0.3*n):int(0.5*n)] += (X[int(0.3*n):int(0.5*n)] - X[int(0.3*n)]) * (-6) + 2
    Y += mu
    p1 = sapprox.gengrad_sparse((Dsp, Y))
    p1.assign_penalty(l1=l1)
    
    opt1 = regreg.FISTA(p1)
    opt1.debug = True
    opt1.fit(M,tol=1.0e-06, max_its=1000)
    beta1, _ = opt1.output

    pylab.clf()
    pylab.plot(X, beta1, linewidth=3, c='red')
    pylab.scatter(X, Y)

    l1 /= 3; 
    p1.assign_penalty(l1=l1)
    opt2 = regreg.FISTA(p1)
    opt2.debug = True
    opt2.fit(M,tol=1.0e-06, max_its=1000)
    beta2, _ = opt2.output
    
    pylab.scatter(X, Y)
    pylab.plot(X, beta2, linewidth=3, c='orange')


test_fused_lasso()
