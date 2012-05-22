import numpy as np

import regreg.api as rr
from regreg.simple import gengrad
import nose.tools as nt

from test_seminorms import ac


from copy import copy

def test_simple():
    Z = np.random.standard_normal((10,10)) * 4
    p = rr.l1_l2((10,10), lagrange=0.13)
    dual = p.conjugate
    L = 0.23

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.simple_problem(loss, p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10, debug=True)

    simple_coef = solver.composite.coefs
    q = rr.identity_quadratic(L, Z, 0, 0)
    prox_coef = p.proximal(q)

    p2 = copy(p)
    p2.set_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, debug=True)
    simple_nonsmooth_coef = solver.composite.coefs

    p = rr.l1_l2((10,10), lagrange=0.13)
    p.set_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    simple_nonsmooth_gengrad = gengrad(problem, L, tol=1.0e-10)

    p = rr.l1_l2((10,10), lagrange=0.13)
    problem = rr.separable_problem.singleton(p, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    separable_coef = solver.composite.coefs

    ac(prox_coef, Z-simple_coef, 'prox to simple')
    ac(prox_coef, simple_nonsmooth_gengrad, 'prox to nonsmooth gengrad')
    ac(prox_coef, separable_coef, 'prox to separable')
    ac(prox_coef, simple_nonsmooth_coef, 'prox to simple_nonsmooth')

    # yield ac, prox_coef, Z - simple_dual_coef, 'prox to simple dual'
#     yield ac, prox_coef, simple_nonsmooth_gengrad, 'prox to nonsmooth gengrad'
#     yield ac, prox_coef, separable_coef, 'prox to separable'
#     yield ac, prox_coef, simple_nonsmooth_coef, 'prox to simple_nonsmooth'



