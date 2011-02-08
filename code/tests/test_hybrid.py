import numpy as np
import pylab, time

import regreg.regression as regreg
import regreg.hybrid as hybrid
reload(regreg); reload(hybrid)
import regreg.problems as problems
reload(problems)
        
control = {'max_its':1500,
           'tol':1.0e-10,
           'plot':True}

def test_fused_lasso(n,l1=2.,**control):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.

    p1 = problems.glasso_signal_approximator((D, Y))
    p1.assign_penalty(l1=l1)
    
    p2 = hybrid.glasso((np.identity(n), D, Y))
    p2.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol']*1.0e-04, max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts2 = t2-t1

    beta1, _ = opt1.output()
    beta2, _ = opt2.output()
    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.step(X, beta1, linewidth=3, c='red')
        pylab.step(X, beta2, linewidth=3, c='blue')
        pylab.scatter(X, Y)
        pylab.show()

