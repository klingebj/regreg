import numpy as np
from scipy import sparse
import time

from algorithms import FISTA
from old_framework.lasso import lasso
        
import pylab

def fused_example():

    x=np.random.standard_normal(500); x[100:150] += 7

    sparsity = l1norm(500, l=1.3)
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    fused = l1norm(D, l=10.5)

    pen = seminorm(sparsity,fused)
    soln, vals = pen.primal_prox(x, 1, with_history=True)
    
    # solution

    pylab.figure(num=1)
    pylab.clf()
    pylab.plot(soln, c='g')
    pylab.scatter(np.arange(x.shape[0]), x)

    # objective values

    pylab.figure(num=2)
    pylab.clf()
    pylab.plot(vals)

def lasso_example(compare=False):

    l1 = 20.
    sparsity = l1norm(500, l=l1/2.)
    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    regloss = squaredloss(X,Y)
    sparsity2 = l1norm(500, l=l1/2.)
    #p=regloss.add_seminorm(sparsity)
    p=regloss.add_seminorm(seminorm(sparsity,sparsity2))
    solver=FISTA(p)
    solver.debug = True
    vals = solver.fit(max_its=2000, min_its = 100)
    soln = solver.problem.coefs

    if not compare:
        # solution
        pylab.figure(num=1)
        pylab.clf()
        pylab.plot(soln, c='g')

        # objective values
        pylab.figure(num=2)
        pylab.clf()
        pylab.plot(vals)
    else:
        p2 = lasso.gengrad((X, Y))
        p2.assign_penalty(l1=l1)
        opt = FISTA(p2)
        opt.debug = True
        opt.fit(tol=1e-10,max_its=5000)
        beta = opt.problem.coefs
        print "Terminal error with seminorm:", np.min(vals), "\tTerminal error with lasso", p.obj(beta) ,"\nTerminal relative error:", (np.min(vals) - p.obj(beta))/p.obj(beta)

        pylab.figure(num=1)
        pylab.clf()
        #pylab.plot(soln, c='g')
        pylab.scatter(soln,beta)
        
        pylab.figure(num=2)
        pylab.clf()
        pylab.plot(vals)


def group_lasso_signal_approx():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [l2norm(selector(500, slice(i*100,(i+1)*100)), l=10.) for i in range(5)]
    group_lasso = seminorm(*penalties)
    x = np.random.standard_normal(500)
    a = group_lasso.primal_prox(x, 1., debug=True)
    
def lasso_via_dual_split():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [l1norm(selector(500, slice(i*100,(i+1)*100)), l=0.2) for i in range(5)]
    lasso = seminorm(*penalties)
    x = np.random.standard_normal(500)
    a = lasso.primal_prox(x, debug=True)
    np.testing.assert_almost_equal(np.maximum(np.fabs(x)-0.2, 0) * np.sign(x), a)
    
def group_lasso_example():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [l2norm(selector(500, slice(i*100,(i+1)*100)), l=.1) for i in range(5)]
    penalties[0].l = 250.
    penalties[1].l = 225.
    penalties[2].l = 150.
    penalties[3].l = 100.
    group_lasso = seminorm(*penalties)

    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(group_lasso)
    solver=FISTA(p)
    solver.debug = True
    vals = solver.fit(max_its=2000, min_its=20,tol=1e-10)
    soln = solver.problem.coefs

    # solution

    pylab.figure(num=1)
    pylab.clf()
    pylab.plot(soln, c='g')

    # objective values

    pylab.figure(num=2)
    pylab.clf()
    pylab.plot(vals)


    
def test_group_lasso_sparse(n=100):

    def selector(p, slice):
        return np.identity(p)[slice]

    def selector_sparse(p, slice):
        return sparse.csr_matrix(np.identity(p)[slice])

    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))


    penalties = [l2norm(selector(500, slice(i*100,(i+1)*100)), l=.1) for i in range(5)]
    penalties[0].l = 250.
    penalties[1].l = 225.
    penalties[2].l = 150.
    penalties[3].l = 100.
    group_lasso = seminorm(*penalties)
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(group_lasso)
    solver=FISTA(p)
    solver.debug = True
    t1 = time.time()
    vals = solver.fit(max_its=2000, min_its=20,tol=1e-8)
    soln1 = solver.problem.coefs
    t2 = time.time()
    dt1 = t2 - t1


    penalties = [l2norm(selector_sparse(500, slice(i*100,(i+1)*100)), l=.1) for i in range(5)]
    penalties[0].l = 250.
    penalties[1].l = 225.
    penalties[2].l = 150.
    penalties[3].l = 100.
    group_lasso = seminorm(*penalties)
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(group_lasso)
    solver=FISTA(p)
    solver.debug = True
    t1 = time.time()
    vals = solver.fit(max_its=2000, min_its=20,tol=1e-8)
    soln2 = solver.problem.coefs
    t2 = time.time()
    dt2 = t2- t1

    print "Times", dt1, dt2
    print soln1[range(10)]
    print soln2[range(10)]
    np.testing.assert_almost_equal(soln1,soln2)

def test_1d_fused_lasso(n=100):

    l1 = 1.


    sparsity1 = l1norm(n, l=l1)
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    extra = np.zeros(n)
    extra[0] = 1.
    D = np.vstack([D,extra])
    D = sparse.csr_matrix(D)

    fused = seminorm(l1norm(D, l=l1))

    X = np.random.standard_normal((2*n,n))
    Y = np.random.standard_normal((2*n,))
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(fused)
    solver=FISTA(p)
    solver.debug = True
    vals1 = solver.fit(max_its=25000, tol=1e-12)
    soln1 = solver.problem.coefs

    B = np.array(sparse.tril(np.ones((n,n))).todense())
    X2 = np.dot(X,B)

    time.sleep(3)
    
    D2 = np.diag(np.ones(n))
    p2 = lasso.gengrad((X2, Y))
    p2.assign_penalty(l1=l1)
    opt = FISTA(p2)
    opt.debug = True
    opt.fit(tol=1e-12,max_its=25000)
    beta = opt.problem.coefs
    soln2 = np.dot(B,beta)

    print soln1[range(10)]
    print soln2[range(10)]
    print p.obj(soln1), p.obj(soln2)
    #np.testing.assert_almost_equal(soln1,soln2)

    return vals1
def test_lasso_dual():

    """
    Check that the solution of the lasso signal approximator dual problem is soft-thresholding
    """

    l1 = .1
    sparsity = l1norm(500, l=l1)
    x = np.random.normal(0,1,500)
    pen = seminorm(sparsity)
    soln, vals = pen.primal_prox(x, 1, with_history=True, debug=True)
    st = np.maximum(np.fabs(x)-l1,0) * np.sign(x) 

    print soln[range(10)]
    print st[range(10)]
    assert(np.allclose(soln,st,rtol=1e-3,atol=1e-3))


def test_multiple_lasso_dual(n=500):

    """
    Check that the solution of the lasso signal approximator dual problem is soft-thresholding even when specified with multiple seminorms
    """

    l1 = 1
    sparsity1 = l1norm(n, l=l1*0.75)
    sparsity2 = l1norm(n, l=l1*0.25)
    x = np.random.normal(0,1,n)
    pen = seminorm(sparsity1,sparsity2)
    t1 = time.time()
    soln, vals = pen.primal_prox(x, 1, with_history=True, debug=True,tol=1e-16)
    t2 = time.time()
    print t2-t1
    st = np.maximum(np.fabs(x)-l1,0) * np.sign(x)

    print soln[range(10)]
    print st[range(10)]
    assert(np.allclose(soln,st,rtol=1e-3,atol=1e-3))


def test_lasso_dual_from_primal(l1 = .1, L = 2.):

    """
    Check that the solution of the lasso signal approximator dual problem is soft-thresholding, when call from primal problem object
    """

    sparsity = l1norm(500, l=l1)
    x = np.random.normal(0,1,500)
    y = np.random.normal(0,1,500)

    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(seminorm(sparsity))

    z = x - y/L
    soln = p.proximal(x,y,L)
    st = np.maximum(np.fabs(z)-l1/L,0) * np.sign(z)

    print x[range(10)]
    print soln[range(10)]
    print st[range(10)]
    assert(np.allclose(soln,st,rtol=1e-3,atol=1e-3))


def test_lasso(n=100):

    l1 = 1.
    sparsity1 = l1norm(n, l=l1*0.75)
    sparsity2 = l1norm(n, l=l1*0.25)
    sparsity = l1norm(n, l=l1)
    
    X = np.random.standard_normal((5000,n))
    Y = np.random.standard_normal((5000,))
    regloss = squaredloss(X,Y)


    #p=regloss.add_seminorm(sparsity)
    #p=regloss.add_seminorm(seminorm(sparsity1,sparsity2),initial=np.zeros(n))
    p=regloss.add_seminorm(seminorm(sparsity),initial=np.zeros(n))
    solver=FISTA(p)
    solver.debug = True
    t1 = time.time()
    vals1 = solver.fit(max_its=800,tol=1e-18,set_prox_tol=True)
    t2 = time.time()
    dt1 = t2 - t1
    soln = solver.problem.coefs

    time.sleep(5)


    p2 = lasso.gengrad((X, Y))#,initial_coefs = np.random.normal(0,1,n))
    p2.assign_penalty(l1=l1)
    opt = FISTA(p2)
    opt.debug = True
    t1 = time.time()
    vals2 = opt.fit(tol=1e-18,max_its=800)
    t2 = time.time()
    dt2 = t2 - t1
    beta = opt.problem.coefs


    print soln[range(10)]
    print beta[range(10)]

    print p.obj(soln), p.obj(beta)
    print p2.obj(soln), p2.obj(beta)
    print "Times", dt1, dt2
    
    return [vals1, vals2]



