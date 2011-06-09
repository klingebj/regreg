import numpy as np
from scipy import sparse
import time

import regreg.api as R
        
import pylab

def test_conjugate():
    z = np.random.standard_normal(10)
    w = np.random.standard_normal(10)
    y = np.random.standard_normal(10)

    for atom_c in [R.l1norm, R.l2norm, 
                   R.positive_part, R.maxnorm,
                   R.nonnegative, 
                   R.constrained_positive_part,
                   R.nonpositive]:
        atom = atom_c(10, linear_term=w, offset=y, lagrange=2.345)
        np.testing.assert_almost_equal(atom.conjugate.conjugate.nonsmooth_objective(z), atom.nonsmooth_objective(z), decimal=3)

def fused_example():

    x=np.random.standard_normal(500); x[100:150] += 7

    sparsity = R.l1norm(500, lagrange=1.3)
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    fused = R.l1norm.linear(D, lagrange=10.5)

    loss = R.l2normsq.shift(-x, coef=0.5)
    pen = R.container(loss, sparsity,fused)
    solver = R.FISTA(pen.composite())
    vals = solver.fit()
    soln = solver.composite.coefs
    
    # solution

    pylab.figure(num=1)
    pylab.clf()
    pylab.plot(soln, c='g')
    pylab.scatter(np.arange(x.shape[0]), x)

    # objective values

    pylab.figure(num=2)
    pylab.clf()
    pylab.plot(vals)

def lasso_example():

    l1 = 20.
    sparsity = R.l1norm(500, lagrange=l1/2.)
    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    regloss = R.l2normsq.affine(X,-Y, coef=0.5)
    sparsity2 = R.l1norm(500, lagrange=l1/2.)
    p=R.container(regloss, sparsity, sparsity2)
    solver=R.FISTA(p.composite())
    solver.debug = True
    vals = solver.fit(max_its=2000, min_its = 100)
    soln = solver.composite.coefs

    # solution
    pylab.figure(num=1)
    pylab.clf()
    pylab.plot(soln, c='g')

    # objective values
    pylab.figure(num=2)
    pylab.clf()
    pylab.plot(vals)

def group_lasso_signal_approx():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [R.l2norm(selector(500, slice(i*100,(i+1)*100)), lagrange=10.) for i in range(5)]
    loss = R.l2normsq.shift(-x, coef=0.5)
    group_lasso = R.container(loss, **penalties)
    x = np.random.standard_normal(500)
    solver = R.FISTA(group_lasso.composite())
    solver.fit()
    a = solver.composite.coefs
    
def lasso_via_dual_split():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [R.l1norm(selector(500, slice(i*100,(i+1)*100)), lagrange=0.2) for i in range(5)]
    x = np.random.standard_normal(500)
    loss = R.l2normsq.shift(-x, coef=0.5)
    lasso = R.container(loss,*penalties)
    solver = R.FISTA(lasso.composite())
    np.testing.assert_almost_equal(np.maximum(np.fabs(x)-0.2, 0) * np.sign(x), solver.composite.coefs, decimal=3)
    
def group_lasso_example():

    def selector(p, slice):
        return np.identity(p)[slice]
    penalties = [R.l2norm(selector(500, slice(i*100,(i+1)*100)), lagrange=.1) for i in range(5)]
    penalties[0].lagrange = 250.
    penalties[1].lagrange = 225.
    penalties[2].lagrange = 150.
    penalties[3].lagrange = 100.

    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    loss = R.l2normsq.affine(X, -Y, coef=0.5)
    group_lasso = R.container(loss, *penalties)

    solver=R.FISTA(group_lasso.composite())
    solver.debug = True
    vals = solver.fit(max_its=2000, min_its=20,tol=1e-10)
    soln = solver.composite.coefs

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
    loss = R.l2normsq.affine(X, -Y, coef=0.5)

    penalties = [R.l2norm.linear(selector(500, slice(i*100,(i+1)*100)), lagrange=.1) for i in range(5)]
    penalties[0].lagrange = 250.
    penalties[1].lagrange = 225.
    penalties[2].lagrange = 150.
    penalties[3].lagrange = 100.
    group_lasso = R.container(loss, *penalties)

    solver=R.FISTA(group_lasso.composite())
    solver.debug = True
    t1 = time.time()
    vals = solver.fit(max_its=2000, min_its=20,tol=1e-8)
    soln1 = solver.composite.coefs
    t2 = time.time()
    dt1 = t2 - t1



    print soln1[range(10)]

def test_1d_fused_lasso(n=100):

    l1 = 1.

    sparsity1 = R.l1norm(n, lagrange=l1)
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    extra = np.zeros(n)
    extra[0] = 1.
    D = np.vstack([D,extra])
    D = sparse.csr_matrix(D)

    fused = R.l1norm.linear(D, lagrange=l1)

    X = np.random.standard_normal((2*n,n))
    Y = np.random.standard_normal((2*n,))
    loss = R.l2normsq.affine(X, -Y, coef=0.5)
    fused_lasso = R.container(loss, fused)
    solver=R.FISTA(fused_lasso.composite())
    solver.debug = True
    vals1 = solver.fit(max_its=25000, tol=1e-12)
    soln1 = solver.composite.coefs

    B = np.array(sparse.tril(np.ones((n,n))).todense())
    X2 = np.dot(X,B)

def test_lasso_dual():

    """
    Check that the solution of the lasso signal approximator dual composite is soft-thresholding
    """

    l1 = .1
    sparsity = R.l1norm(500, lagrange=l1)
    x = np.random.normal(0,1,500)
    loss = R.l2normsq.shift(-x, coef=0.5)
    pen = R.container(loss, sparsity)
    solver = R.FISTA(pen.composite())
    solver.fit()
    soln = solver.composite.coefs
    st = np.maximum(np.fabs(x)-l1,0) * np.sign(x) 

    print soln[range(10)]
    print st[range(10)]
    np.testing.assert_almost_equal(soln,st, decimal=3)


def test_multiple_lasso_dual(n=500):

    """
    Check that the solution of the lasso signal approximator dual composite is soft-thresholding even when specified with multiple seminorms
    """

    l1 = 1
    sparsity1 = R.l1norm(n, lagrange=l1*0.75)
    sparsity2 = R.l1norm(n, lagrange=l1*0.25)
    x = np.random.normal(0,1,n)
    loss = R.l2normsq.shift(-x, coef=0.5)
    p = R.container(loss, sparsity1, sparsity2)
    t1 = time.time()
    solver = R.FISTA(p.composite())
    solver.debug = True
    vals = solver.fit(tol=1.0e-16)
    soln = solver.composite.coefs
    t2 = time.time()
    print t2-t1
    st = np.maximum(np.fabs(x)-l1,0) * np.sign(x)

    print soln[range(10)]
    print st[range(10)]
    np.testing.assert_almost_equal(soln,st, decimal=3)


def test_lasso_dual_from_primal(l1 = .1, L = 2.):

    """
    Check that the solution of the lasso signal approximator dual composite is soft-thresholding, when call from primal composite object
    """

    sparsity = R.l1norm(500, lagrange=l1)
    x = np.random.normal(0,1,500)
    y = np.random.normal(0,1,500)

    X = np.random.standard_normal((1000,500))
    Y = np.random.standard_normal((1000,))
    regloss = R.l2normsq.affine(-X,Y)
    p= R.container(regloss, sparsity)

    z = x - y/L
    soln = p.primal_prox(z,L,with_history=False)
    st = np.maximum(np.fabs(z)-l1/L,0) * np.sign(z)

    print x[range(10)]
    print soln[range(10)]
    print st[range(10)]
    np.testing.assert_almost_equal(soln,st, decimal=3)


def test_lasso(n=100):

    l1 = 1.
    sparsity = R.l1norm(n, lagrange=l1)
    
    X = np.random.standard_normal((5000,n))
    Y = np.random.standard_normal((5000,))
    regloss = R.l2normsq.affine(-X,Y)

    p=R.container(regloss, sparsity)
    solver=R.FISTA(p.composite())
    solver.debug = True
    t1 = time.time()
    vals1 = solver.fit(max_its=800,prox_tol=1e-10)
    t2 = time.time()
    dt1 = t2 - t1
    soln = solver.composite.coefs

    time.sleep(5)


    print soln[range(10)]

    print solver.composite.objective(soln)
    print "Times", dt1
    



