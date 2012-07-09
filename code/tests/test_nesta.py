"""
Solving a LASSO with linear constraints using NESTA
"""

import regreg.api as rr
import numpy as np

n, p, q = 100, 20, 5
X = np.random.standard_normal((n, p))
beta = np.zeros(p)
beta[:4] = 3
Y = np.random.standard_normal(n) + np.dot(X, beta)
A = np.random.standard_normal((q,p))

loss = rr.squared_error(X,Y)
penalty = rr.l1norm(p, lagrange=0.2)
constraint = rr.nonnegative.linear(A)

primal, dual = rr.nesta(loss, penalty, constraint)

def assert_almost_nonnegative(b):
    return np.testing.assert_almost_equal(b[b < 0], 0)

assert_almost_nonnegative(np.dot(A,primal))
