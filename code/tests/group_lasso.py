import numpy as np
import pylab, time

import regreg.regression as regreg
import regreg.lasso as lasso
import regreg.group_lasso as group
import regreg.generalized_lasso as glasso
import regreg.signal_approximator as signal_approximator
        
import nose.tools

control = {'max_its':1500,
           'tol':1.0e-10,
           'plot':True}

def test_group_lasso_approximator1(l1=0.1,**control):

    Y = np.load('Y.npy')
    n = Y.shape[0]
    def e(i, n):
        z = np.zeros(n)
        z[i] = 1.
        return z

    Dv = [(e(i, n), l1*n) for i in range(n)]
    
    p1 = group.group_approximator((Dv, Y))

    p2 = lasso.gengrad((np.identity(n), Y))
    p2.assign_penalty(l1=l1*n)

    p3 = signal_approximator.signal_approximator((np.identity(n), Y))
    p3.assign_penalty(l1=l1*n)

    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.debug = True
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts2 = t2-t1

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    beta1, _ = opt1.output
    beta2, _ = opt2.output
    beta3, _ = opt3.output
    X = np.arange(n)

    nose.tools.assert_true((np.fabs(beta1-beta2).sum() / np.fabs(beta1).sum()) < 1.0e-04)
    nose.tools.assert_true((np.fabs(beta1-beta3).sum() / np.fabs(beta1).sum()) < 1.0e-04)

def test_group_lasso_approximator2(l1=0.1,**control):

    """
    fits a fused lasso as a group lasso approximator, i.e.
    all 2norms are one dimensional
    """
    Y = np.load('Y.npy')
    n = Y.shape[0]

    def e(i, n):
        z = np.zeros(n)
        z[i] = 1.
        z[i+1] = -1
        return z

    Dv = [(e(i, n), l1*n) for i in range(n-1)]
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 
    
    p1 = group.group_approximator((Dv, Y))

    p3 = signal_approximator.signal_approximator((D, Y))
    p3.assign_penalty(l1=l1*n)

    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.debug = True
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    beta1, _ = opt1.output
    beta3, _ = opt3.output
    X = np.arange(n)

    nose.tools.assert_true((np.fabs(beta1-beta3).sum() / np.fabs(beta1).sum()) < 1.0e-04)

def test_group_lasso(l1=0.1,**control):

    """
    fits a fused lasso as a group lasso approximator, i.e.
    all 2norms are one dimensional
    """
    Y = np.load('Y.npy')
    n = Y.shape[0]

    def e(i, n):
        z = np.zeros(n)
        z[i] = 1.
        z[i+1] = -1
        return z

    Dv = [(e(i, n), l1*n) for i in range(n-1)]
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 
    
    p1 = group.group_approximator((Dv, Y))

    p2 = group.group_lasso((np.identity(n), Dv, Y))

    p3 = signal_approximator.signal_approximator((D, Y))
    p3.assign_penalty(l1=l1*n)

    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.debug = True
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p3)
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    beta1, _ = opt1.output
    beta2, _ = opt2.output
    beta3, _ = opt3.output
    X = np.arange(n)

    nose.tools.assert_true((np.fabs(beta1-beta3).sum() / np.fabs(beta1).sum()) < 1.0e-04)
    nose.tools.assert_true((np.fabs(beta1-beta2).sum() / np.fabs(beta1).sum()) < 1.0e-04)

def test_group_lasso2(l1=1,**control):

    """
    fits a LASSO as a group lasso with 
    all 2norms are one dimensional
    """
    X = np.load("X.npy")
    Y = np.load('Y.npy')
    n, p = X.shape
    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() #/ (1*len(Y))

    def e(i, n):
        z = np.zeros(n)
        z[i] = 1.
        return z

    p1 = lasso.gengrad((X, Y))
    p1.assign_penalty(l1=l1*n)

    Dv = [(e(i, p), l1*n) for i in range(p)]
    p2 = group.group_lasso((X, Dv, Y))
    p2.dualcontrol['tol'] = 1.0e-10

    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.debug = True
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    p3 = glasso.generalized_lasso((X, np.identity(p), Y))
    p3.assign_penalty(l1=l1*n)
    p3.dualcontrol['tol'] = 1.0e-10
    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    beta1, _ = opt1.output
    beta2, _ = opt2.output
    beta3, _ = opt3.output
    X = np.arange(n)

    nose.tools.assert_true((np.fabs(beta1-beta2).sum() / np.fabs(beta1).sum()) < 5.0e-04)
    nose.tools.assert_true((np.fabs(beta1-beta3).sum() / np.fabs(beta1).sum()) < 5.0e-04)

def test_group_lasso3(l1=3,**control):

    """
    fits a LASSO as a group lasso with 
    all 2norms are one dimensional
    """
    X = np.load("X.npy")
    Y = np.load('Y.npy')
    n, p = X.shape
    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() #/ (1*len(Y))

    def e(i, n):
        z = np.zeros(n)
        z[i] = 1.
        z[i+1] = -1
        return z

    Dv = [(e(i, p), l1*n) for i in range(p-1)]
    p2 = group.group_lasso((X, Dv, Y))
    p2.dualcontrol['tol'] = 1.0e-10

    D = (np.identity(p) - np.diag(np.ones(p-1),-1))[1:]
    p3 = glasso.generalized_lasso((X, D, Y))
    p3.dualcontrol['tol'] = 1.0e-10
    p3.assign_penalty(l1=l1*n)

    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.debug = True
    opt2.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts3 = t2-t1

    beta2, _ = opt2.output
    beta3, _ = opt3.output
    X = np.arange(n)

    nose.tools.assert_true((np.fabs(beta2-beta3).sum() / np.fabs(beta2).sum()) < 5.0e-04)

