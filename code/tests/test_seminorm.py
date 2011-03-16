import numpy as np
import scipy.sparse
import pylab, time
pylab.ion()
import regreg.regression as regreg
import regreg.signal_approximator as sapprox
from regreg.seminorm import l1norm, genl1norm, squaredloss, seminorm

import nose.tools

def test_fused_lasso(n=100, l1=15., ratio=0.1):


    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.
    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D2 = np.vstack([D1, ratio*np.identity(n)])

    M = np.linalg.eigvalsh(np.dot(D2.T, D2)).max() 
    D1sp = scipy.sparse.csr_matrix(D1)
    Dsp = scipy.sparse.csr_matrix(D2)

    D3 = np.identity(n)
    D3sp =  scipy.sparse.csr_matrix(D3)



    semi = seminorm(l1norm(l1)) #+ seminorm(genl1norm(D1sp,l1))
    p2 = squaredloss((np.eye(n),Y),semi)
    opt2 = regreg.FISTA(p2)
    opt2.debug = True
    obj2 = opt2.fit(tol=1.0e-10, max_its=5000)
    beta2, _ = opt2.output
    print beta2


    p1 = sapprox.gengrad_sparse((D3sp, Y),L=M)
    p1.assign_penalty(l1=l1)


    opt1 = regreg.FISTA(p1)
    #opt1.debug = True
    obj1 = opt1.fit(tol=1.0e-10, max_its=5000, alpha=1.05)
    beta1, _ = opt1.output
    print beta1
    
    print np.allclose(beta1,beta2,rtol=1e-4)

    """
    p1 = sapprox.gengrad_sparse((Dsp, Y),L=M)
    p1.assign_penalty(l1=l1)

    p2 = sapprox.gengrad_sparse((Dsp, Y),L=M)
    p2.assign_penalty(l1=l1)

    objs = []


    opt1 = regreg.ISTA(p1)
    opt1.debug = True
    obj1 = opt1.fit(tol=1.0e-8, max_its=10000, alpha=1.05)
    beta1, _ = opt1.output
    objs.append(obj1)


    opt2 = regreg.FISTA(p2)
    opt2.debug = True
    obj2 = opt2.fit(tol=1.0e-10, max_its=50000,restart=1000)
    beta1, _ = opt2.output
    objs.append(obj2)

    pylab.clf()
    pylab.plot(X, beta1, linewidth=3, c='red')
    pylab.scatter(X, Y)
    """
    #l1 /= 3; 
    #p1.assign_penalty(l1=l1)
    #opt2 = regreg.FISTA(p1)
    #opt2.debug = True
    #opt2.fit(tol=1.0e-06, max_its=1000)
    #beta2, _ = opt2.output
    
    #pylab.scatter(X, Y)
    #pylab.plot(X, beta2, linewidth=3, c='orange')

    #return objs
objs = test_fused_lasso()
#pylab.show()
