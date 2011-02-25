import numpy as np
import pylab, time, scipy
import scipy.sparse
import regreg.regression as regreg
import regreg.lasso as lasso
import regreg.graphnet as graphnet
import regreg.signal_approximator as glasso
from tests import gen_adj        
import nose.tools

control = {'max_its':5500,
           'tol':1.0e-6,
           'plot':False,
           'backtrack':True}

def test_graphnet(X=None,Y=None,l1=5.,l2=0.,l3=0., control=control):

    if X is None or Y is None:
        X = np.load('X.npy')
        Y = np.load('Y.npy')

    p = X.shape[1]
    _ , L = gen_adj(p)
    Lsparse = scipy.sparse.lil_matrix(L)

    l1 *= X.shape[0]
    p1 = graphnet.gengrad((X, Y, L))
    p1.assign_penalty(l1=l1,l2=l2,l3=l3)
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts3 = t2-t1

    p2 = graphnet.gengrad_sparse((X, Y, Lsparse))
    p2.assign_penalty(l1=l1,l2=l2,l3=l3)
    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(tol=control['tol'], max_its=control['max_its'])
    beta2 = opt2.problem.coefs
    t2 = time.time()
    ts3 = t2-t1


    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2 + l3 * np.dot(beta, np.dot(L, beta))
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)
    
    print beta1
    print beta2
    print v
    print f(beta1), f(v), ts3

