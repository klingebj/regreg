import numpy as np

import regreg.api as rr
from regreg.simple import gengrad
import nose.tools as nt

def test_simple():
    Z = np.random.standard_normal(9)
    p = rr.l1norm(9, lagrange=0.13)
    L = 0.89

    loss = rr.quadratic.shift(-Z, coef=0.5*L)
    problem = rr.simple_problem(loss, p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10, debug=True)

    simple_coef = solver.composite.coefs
    prox_coef = p.proximal(L, Z, 0)

    p = rr.l1norm(9, lagrange=0.13)
    p.set_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    simple_nonsmooth_coef = solver.composite.coefs

    p = rr.l1norm(9, lagrange=0.13)
    p.set_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    simple_nonsmooth_gengrad = gengrad(problem, L, tol=1.0e-10)

    p = rr.l1norm(9, lagrange=0.13)
    problem = rr.separable_problem.singleton(p, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    separable_coef = solver.composite.coefs

    yield np.testing.assert_allclose, prox_coef, simple_nonsmooth_gengrad
    yield np.testing.assert_allclose, prox_coef, separable_coef
    yield np.testing.assert_allclose, prox_coef, simple_nonsmooth_coef
    yield np.testing.assert_allclose, prox_coef, simple_coef


