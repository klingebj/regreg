import numpy as np
import pylab; pylab.ion()
from scipy import sparse
import time

from algorithms import FISTA
from atoms import l1norm, l2norm, nonnegative, positive_part
from seminorm import seminorm
from smooth import squaredloss, signal_approximator, logistic_loglikelihood


import old_framework.lasso as lasso

def lasso_example(n=100):

    l1 = 1.
    X = np.random.standard_normal((5000,n))
    Y = np.random.standard_normal((5000,))
    regloss = squaredloss(X,Y)
    sparsity = l1norm(n, l=l1)
    p=regloss.add_seminorm(seminorm(sparsity),initial=np.zeros(n))
    
    solver=FISTA(p)
    solver.debug = True
    vals = solver.fit(max_its=800,tol=1e-12)
    soln = solver.problem.coefs
    return vals



def fused_approximator_example():

    x=np.random.standard_normal(500); x[100:150] += 7; x[250:300] += 14

    sparsity = l1norm(500, l=1.3)
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    D = sparse.csr_matrix(D)
    fused = l1norm(D, l=25.5)

    pen = seminorm(sparsity,fused)
    soln, vals = pen.primal_prox(x, 1, with_history=True, debug=True,tol=1e-10)
    
    # solution

    pylab.figure(num=1)
    pylab.clf()
    pylab.plot(soln, c='g')
    pylab.scatter(np.arange(x.shape[0]), x)

    # objective values

    pylab.figure(num=2)
    pylab.clf()
    pylab.plot(vals)


def fused_lasso_example(n=100):

    l1 = 1.


    sparsity1 = l1norm(n, l=l1)
    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D = sparse.csr_matrix(D)

    fused = seminorm(l1norm(D, l=l1))

    X = np.random.standard_normal((2*n,n))
    Y = np.random.standard_normal((2*n,))
    regloss = squaredloss(X,Y)
    p=regloss.add_seminorm(fused)
    solver=FISTA(p)
    solver.debug = True
    vals = solver.fit(max_its=25000, tol=1e-10)
    soln = solver.problem.coefs

    return vals


def isotonic_example(n=100, plot=True):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    isotonic = seminorm(nonnegative(sparse.csr_matrix(D)))
    Y = np.random.standard_normal(n)
    Y[:-30] += np.arange(n-30) * 0.2
    loss = signal_approximator(Y)
    p = loss.add_seminorm(isotonic, initial=np.ones(Y.shape)*Y.mean())
    p.L = isotonic.power_LD()
    solver=FISTA(p)

    solver.debug = True
    vals = solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln = solver.problem.coefs
    if plot:
        X = np.arange(n)
        pylab.clf()
        pylab.scatter(X, Y)
        pylab.step(X, soln, 'r--')

    return vals

def nearly_isotonic_example(n=100, plot=True):

    D = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    nisotonic = seminorm(positive_part(-sparse.csr_matrix(D), l=3))
    Y = np.random.standard_normal(n)
    Y[:-30] += np.arange(n-30) * 0.2
    loss = signal_approximator(Y)
    p = loss.add_seminorm(nisotonic, initial=np.ones(Y.shape)*Y.mean())
    p.L = nisotonic.power_LD()
    solver=FISTA(p)

    solver.debug = True
    vals = solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln = solver.problem.coefs.copy()

    nisotonic.atoms[0].l = 100.
    solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln2 = solver.problem.coefs.copy()

    nisotonic.atoms[0].l = 1000.
    solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln3 = solver.problem.coefs.copy()

    if plot:
        X = np.arange(n)
        pylab.clf()
        pylab.scatter(X, Y)
        pylab.step(X, soln, 'r--', linewidth=3)
        pylab.step(X, soln2, 'g--', linewidth=3)
        pylab.step(X, soln3, 'y--', linewidth=3)

    return vals

def nearly_concave_example(n=100, plot=True):

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D2 = sparse.csr_matrix(D2)
    nisotonic = seminorm(positive_part(-D2, l=3))

    Y = np.random.standard_normal(n)
    X = np.linspace(0,1,n)
    Y -= (X-0.5)**2 * 10.
    loss = signal_approximator(Y)
    p = loss.add_seminorm(nisotonic, initial=np.ones(Y.shape)*Y.mean())
    p.L = nisotonic.power_LD()
    solver=FISTA(p)

    solver.debug = True
    vals = solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln = solver.problem.coefs.copy()

    nisotonic.atoms[0].l = 100.
    solver.fit(max_its=25000, tol=1e-05, backtrack=False)
    soln2 = solver.problem.coefs.copy()


    if plot:
        X = np.arange(n)
        pylab.clf()
        pylab.scatter(X, Y)
        pylab.plot(X, soln, 'r--', linewidth=3)
        pylab.plot(X, soln2, 'y--', linewidth=3)

    return vals


def concave_example(n=100, plot=True):
    """
    second differences of fit must be non-positive
    """

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D2 = sparse.csr_matrix(D2)

    concave = seminorm(nonnegative(-D2))
    Y = np.random.standard_normal(n)
    X = np.linspace(0,1,n)
    Y -= (X-0.5)**2 * 10.
    loss = signal_approximator(Y)
    p = loss.add_seminorm(concave, initial=np.ones(Y.shape)*Y.mean())
    p.L = concave.power_LD()
    solver=FISTA(p)

    solver.debug = True
    vals = solver.fit(max_its=25000, tol=1e-05, monotonicity_restart=False)
    soln = solver.problem.coefs
    if plot:
        pylab.clf()
        pylab.scatter(X, Y)
        pylab.plot(X, soln, 'r--')

    return vals

    
def group_lasso_example():

    def selector(p, slice):
        return sparse.csr_matrix(np.identity(p)[slice])
    
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




def linear_trend_example(n=500, l1=10.):

    D1 = (np.identity(n) - np.diag(np.ones(n-1),-1))[1:]
    D2 = np.dot(D1[1:,1:], D1)
    D2 = sparse.csr_matrix(D2)


    Y = np.random.standard_normal(n) * 0.2
    X = np.linspace(0,1,n)
    mu = 0 * Y
    mu[int(0.1*n):int(0.3*n)] += (X[int(0.1*n):int(0.3*n)] - X[int(0.1*n)]) * 6
    mu[int(0.3*n):int(0.5*n)] += (X[int(0.3*n):int(0.5*n)] - X[int(0.3*n)]) * (-6) + 2
    Y += mu

    sparsity = l1norm(500, l=0.1)
    fused = l1norm(D2, l=l1)
    pen = seminorm(sparsity,fused)
    soln, vals = pen.primal_prox(Y, 1, with_history=True, debug=True,tol=1e-10)

    pylab.clf()
    pylab.plot(X, soln, linewidth=3, c='red')
    pylab.scatter(X, Y)



def logistic_regression_example(n=100):

    X = np.random.normal(0,1,n*n*5).reshape((5*n,n))
    Y = np.random.randint(0,2,5*n)

    loss = logistic_loglikelihood(X,Y,initial=np.zeros(n))

    solver = FISTA(loss)
    solver.debug = True
    vals = solver.fit(max_its=500, tol=1e-10)
    soln = solver.problem.coefs

