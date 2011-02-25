import numpy as np
import pylab, time, scipy
import scipy.sparse
import regreg.regression as regreg
import regreg.lasso as lasso
import regreg.graphroot as graphroot
import regreg.signal_approximator as glasso
from tests import gen_adj
from regreg import mask
import nose.tools

control = {'max_its':5500,
           'tol':1.0e-8,
           'plot':False,
           'backtrack':True}

def test_graphroot(X=None,Y=None,l1=5.,l2=10., control=control, mu=1.):

    if X is None or Y is None:
        X = np.load('X.npy')
        Y = np.load('Y.npy')

    p = X.shape[1]
    adj, L = gen_adj(p)
    Dsparse = mask.create_D(adj)
    D = Dsparse.toarray()
    Lsparse = scipy.sparse.lil_matrix(L)

    l1 *= X.shape[0]
    p1 = graphroot.gengrad((X, Y, D))
    p1.assign_penalty(l1=l1,l2=l2,mu=mu)
    t1 = time.time()
    opt1 = regreg.FISTA(p1)
    opt1.fit(tol=control['tol'], max_its=control['max_its'])
    beta1 = opt1.problem.coefs
    t2 = time.time()
    ts1 = t2-t1

    p2 = graphroot.gengrad_sparse((X, Y, Dsparse))
    p2.assign_penalty(l1=l1,l2=l2,mu=mu)
    t1 = time.time()
    opt2 = regreg.FISTA(p2)
    opt2.fit(tol=control['tol'], max_its=control['max_its'])
    beta2 = opt2.problem.coefs
    t2 = time.time()
    ts2 = t2-t1


    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2) + np.fabs(beta).sum()*l1  + l2 * np.sqrt(np.dot(beta, np.dot(L, beta)))


    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)
    vs = scipy.optimize.fmin_powell(p1.obj, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    vs = np.asarray(vs)
    
    print np.round(1000*beta1)/1000
    print np.round(1000*beta2)/1000
    print np.round(1000*vs)/1000
    print np.round(1000*v)/1000
    print p1.obj(beta1), p1.obj(vs), f(beta1), f(v)

    print ts1, ts2


