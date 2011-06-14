import numpy as np
from numpy import testing as npt
from numpy.testing import *

from scipy import sparse

import regreg.api as R

@dec.setastest(True)
def test_l1_seminorm():
    """
    Test using the l1norm in lagrange form
    """


    p = 1000
    Y = 10 * np.random.normal(0,1,p)

    loss = R.l2normsq.shift(-Y, coef=0.5)
    sparsity = R.l1norm(p, lagrange=5.)
    sparsity.lagrange *= 1.

    prob = R.container(loss, sparsity)
    problem = prob

    solver = R.FISTA(problem)
    vals = solver.fit(tol=1e-10, max_its=500)
    solution = solver.composite.coefs

    npt.assert_array_almost_equal(solution, np.maximum(np.fabs(Y) - sparsity.lagrange,0.)*np.sign(Y), 3)

@dec.setastest(True)
def test_l1_constraint():
    """
    Test using the l1norm in bound form
    """
  
    p = 1000
    Y = 10 * np.random.normal(0,1,p)

    loss = R.linear(Y, coef=0.5)
    sparsity = R.l1norm(p, bound=5.)

    prob = R.container(loss, sparsity)
    problem = prob

    solver = R.FISTA(problem)
    vals = solver.fit(tol=1e-8, max_its=500)
    solution = solver.composite.coefs

    npt.assert_almost_equal(np.fabs(solution).sum(), sparsity.bound, 3)

@dec.setastest(True)
def test_lasso_via_dual_split():
    """
    Test the lasso by breaking it up into multiple l1 atoms over the range of beta
    """

    def selector(p, slice):
        return np.identity(p)[slice]
    
    penalties = [R.l1norm.linear(selector(500, slice(i*100,(i+1)*100)), lagrange=0.2) for i in range(5)]
    x = np.random.standard_normal(500)
    loss = R.l2normsq.shift(-x, coef=0.5)
    lasso = R.container(loss,*penalties)
    solver = R.FISTA(lasso)
    solver.fit(tol=1e-8)

    npt.assert_array_almost_equal(np.maximum(np.fabs(x)-0.2, 0) * np.sign(x), solver.composite.coefs, 3)


@dec.setastest(True)
def test_multiple_lasso():

    """
    Check that the solution of the lasso signal approximator dual problem is soft-thresholding even when specified with multiple seminorms
    """
    
    
    p = 1000
    
    l1 = 2
    sparsity1 = R.l1norm(p, lagrange=l1*0.75)
    sparsity2 = R.l1norm(p, lagrange=l1*0.25)
    x = np.random.normal(0,1,p)
    loss = R.l2normsq.shift(-x, coef=0.5)
    p = R.container(loss, sparsity1, sparsity2)
    solver = R.FISTA(p)
    vals = solver.fit(tol=1.0e-10)
    soln = solver.composite.coefs
    st = np.maximum(np.fabs(x)-l1,0) * np.sign(x)
    
    npt.assert_array_almost_equal(soln, st, 3)
    

@dec.setastest(True)
def test_1d_fused_lasso():

    """
    Check the 1d fused lasso solution using an equivalent lasso formulation
    """

    n = 100
    l1 = 1.
    
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
    solver=R.FISTA(fused_lasso)
    vals1 = solver.fit(max_its=25000, tol=1e-10)
    soln1 = solver.composite.coefs
    
    B = np.array(sparse.tril(np.ones((n,n))).todense())
    X2 = np.dot(X,B)
    
    loss = R.l2normsq.affine(X2, -Y, coef=0.5)
    sparsity = R.l1norm(n, lagrange=l1)
    lasso = R.container(loss, sparsity)
    solver = R.FISTA(lasso)
    solver.fit(tol=1e-10)

    soln2 = np.dot(B, solver.composite.coefs)

    npt.assert_array_almost_equal(soln1, soln2, 3)


@dec.slow
def test_conjugate_solver():

    # Solve Lagrange problem 
    Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
    loss = R.l2normsq.shift(-Y, coef=0.5)

    sparsity = R.l1norm(len(Y), lagrange = 1.4)
    D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
    fused = R.l1norm.linear(D, lagrange = 25.5)
    problem = R.container(loss, sparsity, fused)
    
    solver = R.FISTA(problem)
    solver.fit(max_its=500, tol=1e-10)
    solution = solver.composite.coefs

    # Solve constrained version
    delta1 = np.fabs(D * solution).sum()
    delta2 = np.fabs(solution).sum()
    fused_constraint = R.l1norm.linear(D, bound = delta1)
    sparsity_constraint = R.l1norm(500, bound = delta2)
    constrained_problem = R.container(loss, fused_constraint, sparsity_constraint)
    constrained_solver = R.FISTA(constrained_problem)
    vals = constrained_solver.fit(max_its=500, tol=1e-10)
    constrained_solution = constrained_solver.composite.coefs

    npt.assert_almost_equal(np.fabs(constrained_solution).sum(), delta2, 3)
    npt.assert_almost_equal(np.fabs(D * constrained_solution).sum(), delta1, 3)


    # Solve with (shifted) conjugate function

    loss = R.l2normsq.shift(-Y, coef=0.5)
    true_conjugate = R.l2normsq.shift(Y, coef=0.5)
    problem = R.container(loss, fused_constraint, sparsity_constraint)
    solver = R.FISTA(problem.conjugate_composite(true_conjugate))
    solver.fit(max_its=500, tol=1e-10)
    conjugate_coefs = problem.conjugate_primal_from_dual(solver.composite.coefs)
                      

    # Solve with generic conjugate function

    loss = R.l2normsq.shift(-Y, coef=0.5)
    problem = R.container(loss, fused_constraint, sparsity_constraint)
    solver2 = R.FISTA(problem.conjugate_composite(conjugate_tol=1e-12))
    solver2.fit(max_its=500, tol=1e-10)
    conjugate_coefs_gen = problem.conjugate_primal_from_dual(solver2.composite.coefs)



    d1 = np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)
    d2 = np.linalg.norm(solution - conjugate_coefs) / np.linalg.norm(solution)
    d3 = np.linalg.norm(solution - conjugate_coefs_gen) / np.linalg.norm(solution)

    npt.assert_array_less(d1, 0.01)
    npt.assert_array_less(d2, 0.01)
    npt.assert_array_less(d3, 0.01)


@dec.setastest(True)
def test_admm_l1_seminorm():
    """
    Test ADMM using the l1norm in lagrange form
    """
    p = 1000
    Y = 10 * np.random.normal(0,1,p)

    loss = R.l2normsq.shift(-Y, coef=0.5)
    sparsity = R.l1norm(p, lagrange=5.)

    prob = R.container(loss, sparsity)

    solver = R.admm_problem(prob)
    solver.fit(debug=False, tol=1e-12)
    solution = solver.beta

    npt.assert_array_almost_equal(solution, np.maximum(np.fabs(Y) - sparsity.lagrange,0.)*np.sign(Y), 3)

@dec.setastest(True)
def test_admm_l1_constraint():
    """
    Test ADMM using the l1norm in bound form
    """
  
    p = 1000
    Y = 10 * np.random.normal(0,1,p)

    loss = R.linear(Y, coef=0.5)
    sparsity = R.l1norm(p, bound=5.)
    sparsity.bound *= 1.

    prob = R.container(loss, sparsity)

    solver = R.admm_problem(prob)
    solver.fit(debug=False, tol=1e-12)
    solution = solver.beta

    npt.assert_almost_equal(np.fabs(solution).sum(), sparsity.bound, 3)
