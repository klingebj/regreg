import numpy as np
import regreg, time
import scipy.optimize

X = np.load('X.npy')
Y = np.load('Y.npy')
N = 50
p = 1000
#X = np.random.normal(0,1,p*N).reshape((N,p))
#Y = np.random.normal(0,1,N)
Xlist = [x for x in X]

XtX = np.dot(X.T, X)
M = np.linalg.eigvalsh(XtX).max() / (1*len(Y))

import regreg.regression as regreg
reload(regreg)
import regreg.problems as problems
reload(problems)
        
def test_lasso(X=X,Y=Y,l1=5.,tol=1e-10,M=M,epsilon=1e-1,testit=False):

    Y += np.dot(X[:,:5], 10 * np.ones(5))
    p1 = problems.lasso((X, Y))
    p1.assign_penalty(l1=l1)
    
    p2 = problems.lasso((X, Y))
    p2.assign_penalty(l1=l1)

    p3 = problems.lasso((X, Y))
    p3.assign_penalty(l1=l1)

    p4 = problems.lasso((X, Y))
    p4.assign_penalty(l1=l1)

    t1 = time.time()
    opt1 = regreg.CWPath(p1)
    opt1.fit(tol=1e-6, max_its=1500)
    beta1 = opt1.problem.coefficients
    t2 = time.time()
    ts1 = t2-t1

    t1 = time.time()
    opt2 = regreg.ISTA(p2)
    opt2.fit(M,tol=tol, max_its=1500)
    beta2 = opt2.problem.coefficients
    t2 = time.time()
    ts2 = t2-t1

    t1 = time.time()
    opt3 = regreg.FISTA(p3)
    opt3.fit(M,tol=tol, max_its=1500)
    beta3 = opt3.problem.coefficients
    t2 = time.time()
    ts3 = t2-t1

    epsvec = np.exp(np.arange(4,-16,-1))
    t1 = time.time()
    opt4 = regreg.NesterovSmooth(p4)
    for eps in epsvec:
        f_s = opt4.fit(M, tol=tol, max_its=50,epsilon=eps)
    f_s = opt4.fit(M, tol=tol, max_its=15000,epsilon=0.1)
    beta4 = opt4.problem.coefficients
    t2 = time.time()
    ts4 = t2-t1

    print "Times", ts1, ts2, ts3, ts4
    stop
