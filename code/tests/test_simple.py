import numpy as np

import regreg.api as rr
from regreg.simple import gengrad
import nose.tools as nt

from test_seminorms import ac

from copy import copy

def test_simple():
    Z = np.random.standard_normal(100) * 4
    p = rr.l1norm(100, lagrange=0.13)
    L = 0.14

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.simple_problem(loss, p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10, debug=True)

    simple_coef = solver.composite.coefs
    prox_coef = p.proximal(rr.identity_quadratic(L, Z, 0, 0))

    p2 = rr.l1norm(100, lagrange=0.13)
    p2 = copy(p)
    p2.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, debug=True)
    simple_nonsmooth_coef = solver.composite.coefs

    p = rr.l1norm(100, lagrange=0.13)
    p.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    simple_nonsmooth_gengrad = gengrad(problem, L, tol=1.0e-10)

    p = rr.l1norm(100, lagrange=0.13)
    problem = rr.separable_problem.singleton(p, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    separable_coef = solver.composite.coefs

    loss2 = rr.quadratic.shift(-Z, coef=0.6*L)
    loss2.quadratic = rr.identity_quadratic(0.4*L, Z, 0, 0)
    p.coefs *= 0
    problem2 = rr.simple_problem(loss2, p)
    loss2_coefs = problem2.solve(coef_stop=True)
    solver2 = rr.FISTA(problem2)
    solver2.fit(tol=1.0e-10, debug=True, coef_stop=True)

    yield ac, prox_coef, simple_nonsmooth_gengrad, 'prox to nonsmooth gengrad'
    yield ac, prox_coef, separable_coef, 'prox to separable'
    yield ac, prox_coef, simple_nonsmooth_coef, 'prox to simple_nonsmooth'
    yield ac, prox_coef, simple_coef, 'prox to simple'
    yield ac, prox_coef, loss2_coefs, 'simple where loss has quadratic 1'
    yield ac, prox_coef, solver2.composite.coefs, 'simple where loss has quadratic 2'


