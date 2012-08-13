"""
Solving a LASSO with linear constraints using NESTA
"""

import regreg.api as rr
import numpy as np
import nose.tools as nt

def test_nesta_nonnegative():

    n, p, q = 1000, 20, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 3
    Y = np.random.standard_normal(n) + np.dot(X, beta)
    A = np.random.standard_normal((q,p))

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=0.2)
    constraint = rr.nonnegative.linear(A)

    primal, dual = rr.nesta(loss, penalty, constraint)

    assert_almost_nonnegative(np.dot(A,primal))

def test_nesta_lasso():

    n, p = 1000, 20
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 30
    Y = np.random.standard_normal(n) + np.dot(X, beta)

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=4.)

    # using nesta
    z = rr.zero(p)
    primal, dual = rr.nesta(loss, z, penalty, tol=1.e-10,
                            epsilon=2.**(-np.arange(30)))

    # using simple problem

    problem = rr.simple_problem(loss, penalty)
    problem.solve()
    nt.assert_true(np.linalg.norm(primal - problem.coefs) / np.linalg.norm(problem.coefs) < 1.e-3)

    

def assert_almost_nonnegative(b):
    return np.testing.assert_almost_equal(b[b < 0], 0)

