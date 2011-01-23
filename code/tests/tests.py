import numpy as np
import regreg
import scipy.optimize
from mask import convert_to_array


def test_all(n=1000):

    test_opt()
    print "\n\n Congratulations - nothing exploded!"

    
def test_opt():
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    Xlist = [x for x in X]

    l1vec = [1,10,100,1000,10000][::-1]
    l2vec = [1,10,100][::-1]
    l3vec = [1,10,100][::-1]
    
    cwpathtol = 1e-7

    for l1 in l1vec:
        test_lasso(X,Xlist,Y,l1,tol=cwpathtol)
        #test_lasso_wts(X,Xlist,Y,l1,tol=cwpathtol)
        for l2 in l2vec:
            for l3 in l3vec:
                test_graphnet(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)
                test_lin_graphnet(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)
                #test_graphnet_wts(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)

def test_lasso(X,Xlist,Y,l1=500.,tol=1e-4):

    print "LASSO", l1
    l = regreg.regreg((Xlist, Y),regreg.lasso,regreg.cwpath)#, initial_coefs= np.array([7.]*10))
    l.problem.assign_penalty(l1=l1)
    l.fit(tol=tol, max_its=50)
    beta = l.problem.coefficients


    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
    v = np.asarray(v)

    """
    print np.round(1000*beta)/1000
    print np.round(1000*v)/1000
    """

    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)


def test_lasso_wts(X,Xlist,Y,l1=500.,tol=1e-4):

    print "lasso_wts", l1
    n = len(Xlist)
    wts = np.random.normal(0,1,n)
    l = scca.lasso.lasso_wts(Xlist, Y,update_resids=False)#, initial_coefs= np.array([7.]*10))
    l.weights = wts
    l.assign_penalty(l1=l1)
    l.update_residuals()
    l.fit(tol=tol, max_its=50)

    beta = l.coefficients


    def f(beta):
        return np.linalg.norm(Y - wts*np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
    v = np.asarray(v)


    print np.round(1000*beta)/1000
    print np.round(1000*v)/1000


    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)


def test_graphnet(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "GraphNet", l1,l2,l3
    A = convert_to_array(regreg._create_adj(X.shape[1]))
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    l = regreg.regreg((Xlist, Y,A),regreg.graphnet,regreg.cwpath)#, initial_coefs= np.array([7.]*10))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)
    l.fit(tol=tol, inner_its=40)
    beta = l.problem.coefficients

    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)



def test_lin_graphnet(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "lin_graphnet", l1,l2,l3
    A = convert_to_array(regreg._create_adj(X.shape[1]))
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    l = regreg.regreg((Xlist, Y,A),regreg.lin_graphnet,regreg.cwpath)#, initial_coefs= np.array([7.]*10))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)
    l.fit(tol=tol,max_its=500)
    beta = l.problem.coefficients

    
    def f(beta):
        q = np.dot(X,beta)
        return -np.dot(Y,q) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)




def test_graphnet_wts(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "graphnet_wts", l1,l2,l3
    n = len(Xlist)
    wts = np.random.normal(0,1,n)
    A = convert_to_array(scca.lasso._create_adj(X.shape[1]))
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    l = scca.lasso.graphnet_wts(Xlist,Y,A,weights=wts)# initial_coefs = np.array([7.]*10))
    l.assign_penalty(l1=l1,l2=l2,l3=l3)
    l.fit(tol=tol, max_its=40)
    beta = l.coefficients

    
    def f(beta):
        return np.linalg.norm(Y - wts*np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    #print np.round(100*v)/100,'\n', np.round(100*beta)/100
    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)



def test_softThreshold(n=1):
    
    p = 1500
    def m(x,y):
        if x >= y:
            return x
        else:
            return y
    m = np.vectorize(m)
    for i in range(n):
        l = np.random.normal(5)**2
        v = np.random.normal(0,10,p)
        signs = np.sign(v)
        s = signs*m(np.fabs(v)-l,0.)
        vec = scca.softThreshold(v,l)
        assert(np.sum((vec-s)**2)<1e-6)

def test_deltaSoft(n=1):

    p = 1000
    ctype = np.zeros(4)
    for i in range(n):
        v = np.random.normal(0,10,p)
        normalize = np.bool(np.sign(np.random.normal())+1)
        if normalize:
            v = v/np.sqrt(np.sum(v**2))
        l = np.max([np.fabs(np.random.normal(np.sum(np.fabs(v)),50)),1.])
        vec = scca.deltaSoft(v,l,normalize)
        if normalize:
            n2vec = np.sum(vec**2)
            assert(np.fabs(n2vec-1.)<1e-8)
        n1v = np.sum(np.fabs(v))
        if n1v <= l:
            if normalize:
                ctype[0]+=1
            else:
                ctype[1]+=1
            assert(np.sum((v-vec)**2)<1e-8)
        else:
            if normalize:
                ctype[2]+=1
            else:
                ctype[3]+=1
            n1vec = np.sum(np.fabs(vec))
            assert(np.sum((n1vec-l)**2)<1e-8)
    #print ctype/n

def test_multImage(n=1):

    for i in range(n):
        N=10
        p=500
        X = np.random.normal(0,1,N*p).reshape((N,p))
        Xlist = [r for r in X]
        a = np.random.normal(0,1,p)
        b = np.random.normal(0,1,N)
        Xa = np.dot(X,a)
        Xb = np.dot(X.T,b)
        Xa2 = scca.multimage(Xlist,a)
        Xb2 = scca.multimage(Xlist,b,True)
        assert(np.sum((Xa-Xa2)**2) < 1e-6)
        assert(np.sum((Xb-Xb2)**2) < 1e-6)


def test_centerscaleImages(n=1):

    for i in range(n):
        N=13
        p=40
        X = np.random.normal(0,1,N*p).reshape((N,p))
        Xlist = [r for r in X]
        X = X - np.mean(X,0)
        X = X / np.std(X,0)
        scca.centerscaleImages(Xlist)
        X2 = np.vstack(Xlist)
        assert(np.allclose(X,X2))
        #assert(np.sum((X-X2)**2)<1e-6)
