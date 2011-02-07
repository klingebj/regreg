import numpy as np
import regreg, time
import scipy.optimize
from mask import convert_to_array



X = np.load('X.npy')
Y = np.load('Y.npy')
N = 50
p = 100
X = np.random.normal(0,1,p*N).reshape((N,p))
Y = np.random.normal(0,1,N)
Xlist = [x for x in X]

XtX = np.dot(X.T, X)
M = np.linalg.eigvalsh(XtX).max() / (1*len(Y))

        
def test_lasso(X=X,Xlist=Xlist,Y=Y,l1=.05,tol=1e-10,M=M,epsilon=1e-1,testit=False):


    # Here we set L=M, but if this argument is not given the lasso problem uses
    # the power method to find L
    p1 = regreg.lasso((Xlist, Y))
    p1.assign_penalty(l1=l1)
    
    p2 = regreg.lasso((Xlist, Y),L=M)
    p2.assign_penalty(l1=l1)

    p3 = regreg.lasso((Xlist, Y),L=M)
    p3.assign_penalty(l1=l1)

    p4 = regreg.lasso((Xlist, Y),L=M)
    p4.assign_penalty(l1=l1)


    # solve using cwpath
    t1 = time.time()
    
    opt1 = regreg.cwpath(p1)
    opt1.fit(tol=1e-6, max_its=2500)
    beta1 = opt1.problem.coefficients

    t2 = time.time()
    ts1 = t2-t1

    # solve using ista
    t1 = time.time()

    opt2 = regreg.ista(p2)
    opt2.fit(tol=tol, max_its=2500)
    beta2 = opt2.problem.coefficients

    t2 = time.time()
    ts2 = t2-t1

    # solve using fista
    t1 = time.time()

    opt3 = regreg.fista(p3)
    opt3.fit(tol=tol, max_its=2500)
    beta3 = opt3.problem.coefficients

    t2 = time.time()
    ts3 = t2-t1

    # "solve" using Nesterov epsilon smoothing
    
    t1 = time.time()
    opt4 = regreg.nesterov_eps(p4)
    #epsvec = np.exp([-1,-3,-5,-7,-9,-11,-13,-15])
    #for eps in epsvec:
    #    f_s = opt4.fit(tol=tol, max_its=5000,epsilon=eps)
    f_s = opt4.fit(tol=tol, max_its=15000,epsilon=0.1)
    beta4 = opt4.problem.coefficients
    t2 = time.time()
    ts4 = t2-t1



    if testit:
        def f(beta):
            return np.linalg.norm(Y - np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1

        v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
        v = np.asarray(v)

        v_s = scipy.optimize.fmin_powell(f_s, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
        v_s = np.asarray(v_s)


        print beta1
        print beta2
        print beta3
        print v

        print "\n", beta4
        print  v_s

        """
        assert(np.linalg.norm(beta1 - v) / np.linalg.norm(v) < 1.0e-04)
        assert(np.linalg.norm(beta2 - v) / np.linalg.norm(v) < 1.0e-04)
        assert(np.linalg.norm(beta2 - v) / np.linalg.norm(v) < 1.0e-04)
        """

    print "\nTimes", ts1, ts2, ts3, ts4

