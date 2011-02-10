import numpy as np
import pylab, time, scipy

import regreg.regression as regreg
import regreg.lasso as lasso
import regreg.graphnet as graphnet
import regreg.signal_approximator as glasso
from tests import gen_adj        
import nose.tools

control = {'max_its':5500,
           'tol':1.0e-12,
           'plot':False,
           'backtrack':True}

def test_FISTA(X=None,Y=None,l1=5.,l2=0.,l3=0., control=control):

    if X or Y is None:
        X = np.load('X.npy')
        Y = np.load('Y.npy')

    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() #/ (1*len(Y))

    p = X.shape[1]
    _ , L = gen_adj(p)

    l1 *= X.shape[0]
    p3 = graphnet.gengrad((X, Y, L))
    p3.assign_penalty(l1=l1,l2=l2,l3=l3)

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    beta3 = opt3.problem.coefs
    t2 = time.time()
    ts3 = t2-t1




    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2 + l3 * np.dot(beta, np.dot(L, beta))
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)
    
    print beta3
    print v
    print f(beta3), f(v), ts3

def test_lasso(X=None,Y=None,l1=5., control=control):

    if X or Y is None:
        X = np.load('X.npy')
        Y = np.load('Y.npy')

    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() #/ (1*len(Y))
    print M
    Y += np.dot(X[:,:5], 10 * np.ones(5))

    p1 = lasso.cwpath((X, Y))
    p1.assign_penalty(l1=l1*X.shape[0])

    p2 = lasso.gengrad((X, Y))
    p2.assign_penalty(l1=l1*X.shape[0])

    p3 = lasso.gengrad((X, Y))
    p3.assign_penalty(l1=l1*X.shape[0])

    p4 = lasso.gengrad_smooth((X, Y))
    p4.assign_penalty(l1=l1*X.shape[0])

    t1 = time.time()
    opt1 = regreg.CWPath(p1)
    opt1.fit(tol=1e-8, max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.ISTA(p2)
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'],backtrack=control['backtrack'],alpha=1.25)
    beta2 = opt2.problem.coefs
    t2 = time.time()
    ts2 = t2-t1
    time.sleep(0.5)
    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'],backtrack=control['backtrack'],alpha=1.25)
    beta3 = opt3.problem.coefs
    t2 = time.time()
    ts3 = t2-t1

    epsvec = np.exp(np.arange(4,-16,-1))
    t1 = time.time()
    opt4 = regreg.NesterovSmooth(p4)
    for eps in epsvec:
        f_s = opt4.fit(M, tol=control['tol'], max_its=50,epsilon=eps)
        #    f_s = opt4.fit(M, tol=control['tol'], max_its=control['max_its'],epsilon=0.1)
    beta4 = opt4.problem.coefs
    t2 = time.time()
    ts4 = t2-t1

    print beta1
    print beta2
    print beta3
    print p2.obj(beta1), p2.obj(beta2), p3.obj(beta3)
    nose.tools.assert_true((np.fabs(beta1-beta3).sum() / np.fabs(beta1).sum() <= 1.0e-04))



    """
    print p3.obj(beta1), p3.obj(beta2), p3.obj(beta3)
    """
    print "Times", ts1, ts2, ts3, ts4

