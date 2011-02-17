"""
Fit generalized lasso using a scipy.sparse D matrix
"""

import numpy as np
import pylab, time
import scipy.sparse

import regreg.regression as regreg
import regreg.signal_approximator as sapprox
        
control = {'max_its':1500,
           'tol':1.0e-10,
           'plot':True}

def fused_lasso(n=100,l1=2.,**control):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    Dsp = scipy.sparse.lil_matrix(D)
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.

    p1 = sapprox.gengrad_sparse((Dsp, Y),L=M)
    p1.assign_penalty(l1=l1)

    p2 = sapprox.gengrad((D, Y),L=M)
    p2.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.debug=True
    opt1.fit(tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(tol=control['tol'], max_its=control['max_its'])
    beta2 = opt2.problem.coefs
    t2 = time.time()
    ts2 = t2-t1

    beta1, _ = opt1.output
    beta2, _ = opt2.output
    np.testing.assert_almost_equal(beta1, beta2)

    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.step(X, beta1, linewidth=3, c='red')
        pylab.step(X, beta2, linewidth=3, c='green')
        pylab.scatter(X, Y)
        pylab.show()

def test_fused_lasso():
    p = control['plot']
    control['plot'] = False
    fused_lasso(**control)
    control['plot'] = p

def sparse_fused_lasso(n=100,l1=5.,ratio=0.2,**control):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D = np.vstack([D, ratio*np.identity(n)])
    Dsp = scipy.sparse.lil_matrix(D)
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.
    p1 = sapprox.gengrad_sparse((Dsp, Y),L=M)
    p1.assign_penalty(l1=l1)

    p2 = sapprox.gengrad((D, Y),L=M)
    p2.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(tol=control['tol'], max_its=control['max_its'])
    beta2 = opt2.problem.coefs
    t2 = time.time()
    ts2 = t2-t1

    beta1, _ = opt1.output
    beta2, _ = opt2.output
    np.testing.assert_almost_equal(beta1, beta2)

    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.step(X, beta1, linewidth=3, c='red')
        pylab.step(X, beta2, linewidth=3, c='green')
        pylab.scatter(X, Y)
        pylab.show()

        

def test_sparse_fused_lasso():
    p = control['plot']
    control['plot'] = False
    sparse_fused_lasso(**control)
    control['plot'] = p
