import numpy as np
import pylab, time

import regreg.regression as regreg
import regreg.lasso as lasso
import regreg.signal_approximator as sapprox
        
import nose.tools

control = {'max_its':2500,
           'tol':1.0e-10,
           'plot':False,
           'backtrack':True}

def test_FISTA(X=None,Y=None,l1=5., control=control):

    if X or Y is None:
        X = np.load('X.npy')
        Y = np.load('Y.npy')

    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() #/ (1*len(Y))

    p3 = lasso.gengrad((X, Y))
    p3.assign_penalty(l1=l1*X.shape[0])

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=control['tol'], max_its=control['max_its'])
    beta3 = opt3.problem.coefs
    t2 = time.time()
    ts3 = t2-t1


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


def test_fused_lasso(n=400,l1=2.,control=control):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 6.
    p1 = sapprox.signal_approximator((D, Y))
    p1.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    beta, _ = opt1.output
    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.step(X, beta, linewidth=3, c='red')
        pylab.scatter(X, Y)
        pylab.show()

def test_sparse_fused_lasso(n=500,l1=2.,ratio=1.,control=control):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D = np.vstack([D, ratio*np.identity(n)])
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n)
    Y[int(0.1*n):int(0.3*n)] += 3.
    p1 = sapprox.signal_approximator((D, Y))
    p1.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    beta, _ = opt1.output
    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.step(X, beta, linewidth=3, c='red')
        pylab.scatter(X, Y)
        pylab.show()

def test_linear_trend(n=500,l1=2.,control=control):

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    M = np.linalg.eigvalsh(np.dot(D2.T, D2)).max() 

    Y = np.random.standard_normal(n) * 0.2
    X = np.linspace(0,1,n)
    mu = 0 * Y
    mu[int(0.1*n):int(0.3*n)] += (X[int(0.1*n):int(0.3*n)] - X[int(0.1*n)]) * 6
    mu[int(0.3*n):int(0.5*n)] += (X[int(0.3*n):int(0.5*n)] - X[int(0.3*n)]) * (-6) + 2
    Y += mu
    p1 = sapprox.signal_approximator((D2, Y))
    p1.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    t2 = time.time()
    ts1 = t2-t1

    beta, _ = opt1.output
    X = np.arange(n)
    if control['plot']:
        pylab.clf()
        pylab.plot(X, beta, linewidth=3, c='red')
        pylab.scatter(X, Y)
        pylab.show()

def test_sparse_linear_trend(n=500,l1=2., ratio=0.1, control=control):

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D = np.vstack([D2, ratio*np.identity(n)])
    M = np.linalg.eigvalsh(np.dot(D.T, D)).max() 

    Y = np.random.standard_normal(n) * 0.1
    X = np.linspace(0,1,n)
    mu = 0 * Y
    mu[int(0.1*n):int(0.3*n)] += (X[int(0.1*n):int(0.3*n)] - X[int(0.1*n)]) * 6
    mu[int(0.3*n):int(0.5*n)] += (X[int(0.3*n):int(0.5*n)] - X[int(0.3*n)]) * (-6) + 1.2
    Y += mu
    p1 = sapprox.signal_approximator((D, Y))
    p1.assign_penalty(l1=l1)
    
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(M,tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    beta, _ = opt1.output
    if control['plot']:
        pylab.clf()
        pylab.plot(X, beta, linewidth=3, c='red')
        pylab.scatter(X, Y)
        pylab.plot(X, mu)
        pylab.show()

def all(control=control):
    test_lasso(control=control)
    for i, f in enumerate([test_fused_lasso,
                        test_sparse_fused_lasso,
                        test_linear_trend,
                        test_sparse_linear_trend]):
        if control['plot']:
            pylab.figure(num=i+1)
        f(100,control=control)
