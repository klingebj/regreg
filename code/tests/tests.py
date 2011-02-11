import numpy as np
import regreg, time
import scipy.optimize
from mask import convert_to_array


def test_all(n=100):

    test_soft_threshold(n)
    test_mult_Lbeta(n)
    test_opt()
    test_gen_adj(n)
    print "\n\n Congratulations - nothing exploded!"

    
def test_opt():
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    Xlist = [x for x in X]

    XtX = np.dot(X.T, X)
    M = np.linalg.eigvalsh(XtX).max() / len(Y)

    l1vec = [1,10,100,1000,10000][::-1]
    l2vec = [1,10,100][::-1]
    l3vec = [1,10,100][::-1]
    
    cwpathtol = 1e-7

    test_col_inner(X,Xlist,Y)
    for l1 in l1vec:
        #test_lasso(X,Xlist,Y,l1,tol=cwpathtol,M=M)
        #test_lasso_wts(X,Xlist,Y,l1,tol=cwpathtol)

        for l2 in l2vec:
            for l3 in l3vec:
                assert(True)
                #test_graphnet(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)
                #test_lin_graphnet(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)
                test_v_graphnet(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)
                #test_graphnet_wts(X,Xlist,Y,l1,l2,l3,tol=cwpathtol)





def test_soft_threshold(n=1):
    
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
        vec = regreg.soft_threshold(v,l)
        assert(np.sum((vec-s)**2)<1e-6)

def test_col_inner(X,Xlist,Y):

    v1 = regreg.col_inner(Xlist,Y)
    v2 = np.dot(X.T,Y)
    assert(np.allclose(v1,v2))

def test_mult_Lbeta(n):

    p = 20

    for i in range(n):
        A, Afull = gen_adj(p)
        regreg._check_adj(A)
        nadj = regreg._create_nadj(A)
        beta = np.random.normal(0,1,p)
        s1 = regreg._mult_Lbeta(A,nadj,beta)
        s2 = 2*np.dot(beta,np.dot(Afull,beta))
        assert(np.allclose(s1,s2))
                

def test_gen_adj(n):


    for i in range(n):
        p = np.random.randint(100)+2
        A, Afull = gen_adj(p)

        regreg._check_adj(A)
        v1 =  np.diag(Afull)
        v2 =  np.array([np.sum(r>=0) for r in A],dtype=int)

        assert(np.sum((v1-v2)**2)==0)
        v = np.unique(np.triu(Afull,1)+np.tril(Afull,-1))
        if len(v) == 1:
            assert(np.product(np.unique(np.triu(Afull,1)) in  [-1,0]))
        else:
            assert(np.product(np.unique(np.triu(Afull,1)) ==  [-1,0]))
        assert(np.unique([np.sum(r) for r in Afull])==[0])
        assert(np.sum(np.fabs(Afull-Afull.T))==0.)
        for j in range(A.shape[0]):
            for k in range(A.shape[1]):
                if A[j,k] > -1:
                    assert(Afull[j,A[j,k]]==-1)
    
def gen_adj(p):

    Afull = np.zeros((p,p),dtype=int)
    A = - np.ones((p,p),dtype=int)
    counts = np.zeros(p)
    for i in range(p):
        for j in range(p):
            if np.random.uniform(0,1) < 0.3:
                if i != j:
                    if Afull[i,j] == 0:
                        Afull[i,j] = -1
                        Afull[j,i] = -1
                        Afull[i,i] += 1
                        Afull[j,j] += 1
                        A[i,counts[i]] = j
                        A[j,counts[j]] = i
                        counts[i] += 1
                        counts[j] += 1
    return A, Afull

        
def test_lasso(X,Xlist,Y,l1=500.,tol=1e-4,M=0.):

    print "LASSO", l1
    #l = regreg.regreg((Xlist, Y),regreg.lasso,regreg.cwpath)#, initial_coefs= np.array([7.]*10))

    p1 = regreg.lasso((Xlist, Y))
    p1.assign_penalty(l1=l1)
    
    p2 = regreg.lasso((Xlist, Y))
    p2.assign_penalty(l1=l1)

    p3 = regreg.lasso((Xlist, Y))
    p3.assign_penalty(l1=l1)

    t1 = time.time()
    o1 = regreg.cwpath(p1)
    o1.fit(tol=tol, max_its=50)
    beta1 = o1.problem.coefficients
    t2 = time.time()
    print "CWPATH", t2-t1

    t1 = time.time()
    o2 = regreg.gengrad(p2)
    o2.fit(M,tol=1e-10, max_its=1500)
    beta2 = o2.problem.coefficients
    t2 = time.time()
    print "GENGRAD", t2-t1

    epsvec = [1e-0,1e-3,1e-6]
    t1 = time.time()
    o3 = regreg.nesterov(p3)
    for eps in epsvec:
        f_s = o3.fit(M, tol=1e-10, max_its=150,epsilon=eps)
    f_s = o3.fit(M, tol=1e-10, max_its=5000,epsilon=eps)
    beta3 = o3.problem.coefficients
    t2 = time.time()
    print "NEST", t2-t1

    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1

    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
    v = np.asarray(v)

    vs = scipy.optimize.fmin_powell(f_s, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
    vs = np.asarray(vs)

    print np.round(10000*beta1)/10000
    print np.round(10000*beta2)/10000
    print np.round(10000*beta3)/10000
    print "\n",np.round(10000*v)/10000
    print np.round(10000*vs)/10000

    print f_s(beta3), f_s(vs)

    assert(np.fabs(f(beta1) - f(beta2)) / np.fabs(f(beta1) + f(beta1)) < 1.0e-04)
    if np.linalg.norm(beta1) > 1e-8:
        assert(np.linalg.norm(beta2 - beta1) / np.linalg.norm(beta1) < 1.0e-04)
    else:
        assert(np.linalg.norm(beta2) < 1e-8)


    if f(v) < f(beta1):
        assert(np.fabs(f(v) - f(beta1)) / np.fabs(f(v) + f(beta1)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta1) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta1) < 1e-8)




def test_lasso_wts(X,Xlist,Y,l1=500.,tol=1e-4):

    print "lasso_wts", l1
    n = len(Xlist)
    wts = np.random.normal(0,1,n)
    l = regreg.lasso.lasso_wts(Xlist, Y,update_resids=False)#, initial_coefs= np.array([7.]*10))
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
    print f(beta), f(v)

    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)


def test_graphnet(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "GraphNet", l1,l2,l3
    A, Afull = gen_adj(X.shape[1])

    #l = regreg.regreg((Xlist, Y, A),regreg.graphnet,regreg.cwpath)#, initial_coefs= np.array([7.]*10))
    l = regreg.cwpath(regreg.graphnet((Xlist, Y, A)))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)

    l.fit(tol=tol, inner_its=50,max_its=5000)
    beta = l.problem.coefficients

    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    #print np.round(100*v)/100,'\n', np.round(100*beta)/100
    #print f(v), f(beta)
    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)



def test_lin_graphnet(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "lin_graphnet", l1,l2,l3
    A, Afull = gen_adj(X.shape[1])

    orth = np.random.normal(0,1,X.shape[1])
    eta = np.random.normal(0,1,1)[0]

    #l = regreg.regreg((Xlist, Y, A),regreg.lin_graphnet,regreg.cwpath)#, initial_coefs= np.array([7.]*10))
    l = regreg.cwpath(regreg.lin_graphnet((Xlist, Y, A)))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,eta=eta)
    l.problem.orth = orth
    l.fit(tol=tol,max_its=5000)
    beta = l.problem.coefficients

    
    def f(beta):
        q = np.dot(X,beta)
        return -np.dot(Y,q) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2 + eta*np.dot(beta,orth)
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print eta, np.sum(beta), np.sum(v)
    if f(v) < f(beta):
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)




def test_v_graphnet(X,Xlist,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "v_graphnet", l1,l2,l3
    A, Afull = gen_adj(X.shape[1])

    vec = np.dot(Y,X)
    l = regreg.cwpath(regreg.v_graphnet((vec,A)))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)
    l.fit(tol=tol,max_its=5000)
    beta = l.problem.coefficients

    lv2 = regreg.cwpath(regreg.lin_graphnet((Xlist,Y,A)))
    lv2.problem.assign_penalty(l1=l1,l2=l2,l3=l3)
    lv2.fit(tol=tol,max_its=5000)
    beta2 = lv2.problem.coefficients

    if np.linalg.norm(beta) > 1e-8:
        assert(np.linalg.norm(beta - beta2) / np.linalg.norm(beta) < 1.0e-04)
    else:
        assert(np.linalg.norm(beta2) < 1e-8)

    
    def f(beta):
        return -np.dot(vec,beta) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    #print np.round(100*v)/100,'\n', np.round(100*beta)/100
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
    Xlist2 = [Xlist[i]*wts[i] for i in range(n)]
    A, Afull = gen_adj(X.shape[1])

    l = regreg.cwpath(regreg.graphnet_wts((Xlist,Y,A),rowweights=wts))# initial_coefs = np.array([7.]*10))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)
    l.fit(tol=tol, max_its=400)
    beta = l.problem.coefficients

    
    def f(beta):
        return np.linalg.norm(Y - np.dot(np.vstack(Xlist2), beta))**2/(2*len(Y)) + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)


    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print f(beta), f(v)

    if f(v) < f(beta)+1e-10:
        assert(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert(np.linalg.norm(beta) < 1e-8)



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
