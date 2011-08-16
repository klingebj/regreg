import numpy as np
import regreg.api as rr
import itertools
from numpy import testing as npt
from numpy.testing import *

@dec.setastest(True)
def test_affine_linear_offset_l1norm():

    """
    Test linear, affine and offset with the l1norm atom
    """
    
    n = 1000
    p = 10
    
    X = np.random.standard_normal((n,p))
    Y = 10*np.random.standard_normal(n)
    
    coefs = []
    
    loss = rr.quadratic.affine(X,-Y, coef=0.5)
    sparsity = rr.l1norm(p, lagrange=5.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs.append(1.*solver.composite.coefs)
    
    loss = rr.quadratic.affine(X,-Y, coef=0.5)
    sparsity = rr.l1norm.linear(np.eye(p), lagrange=5.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs.append(1.*solver.composite.coefs)
    
    loss = rr.quadratic.affine(X,-Y, coef=0.5)
    sparsity = rr.l1norm.affine(np.eye(p),np.zeros(p), lagrange=5.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs.append(1.*solver.composite.coefs)
    
    loss = rr.quadratic.affine(X,-Y, coef=0.5)
    sparsity = rr.l1norm.linear(np.eye(p), lagrange=5., offset=np.zeros(p))
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs.append(1.*solver.composite.coefs)
    
    loss = rr.quadratic.affine(X,-Y, coef=0.5)
    sparsity = rr.l1norm.offset(np.zeros(p), lagrange=5.)
    problem = rr.container(loss, sparsity)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs.append(1.*solver.composite.coefs)

    for i,j in itertools.combinations(range(len(coefs)), 2):
        npt.assert_almost_equal(coefs[i], coefs[j])

