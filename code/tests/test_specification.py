
import numpy as np

from copy import copy
import scipy.optimize

import regreg.api as rr

def test_l1prox():
    '''
    this test verifies that the l1 prox in lagrange form can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=0.3)
    ww = np.random.standard_normal(4)*3
    ab = l1.proximal(rr.identity_quadratic(0.5, ww, 0,0))

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.simple_problem.nonsmooth(l1c)
    solver = rr.FISTA(a)
    solver.fit(tol=1.e-10)

    ad = a.coefs

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.dual_problem.fromseq(l1c.conjugate)
    solver = rr.FISTA(a)
    solver.fit(tol=1.0e-14)

    ac = a.primal

    np.testing.assert_allclose(ac, ab, rtol=1.0e-4)
    np.testing.assert_allclose(ac, ad, rtol=1.0e-4)


def test_l1prox_bound():
    '''
    this test verifies that the l1 prox in bound form can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, bound=2.)
    ww = np.random.standard_normal(4)*2
    ab = l1.proximal(rr.identity_quadratic(0.5, ww, 0, 0))

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.simple_problem.nonsmooth(l1c)
    solver = rr.FISTA(a)
    solver.fit()

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.dual_problem.fromseq(l1c.conjugate)
    solver = rr.FISTA(a)
    solver.fit()

    ac = a.primal

    np.testing.assert_allclose(ac, ab)


# def test_basis_pursuit():
#     '''
#     this test verifies that the smoothed
#     problem for basis pursuit in the TFOCS
#     algorithm can be solved
#     by a primal/dual specification 

#     '''

#     l1 = rr.l1norm(4, lagrange=1.)
#     l1.quadratic = rr.identity_quadratic(0.5, 0, None, 0.)

#     X = np.random.standard_normal((10,4))
#     Y = np.random.standard_normal(10) + 3
#     pY = np.dot(X, np.dot(np.linalg.pinv(X), Y))
    
#     minb = np.linalg.norm(Y-pY)
#     print minb
#     l2constraint = rr.l2norm.affine(X, -Y, bound=1.5 * minb / np.linalg.norm(Y))

#     a = rr.dual_problem.fromseq(l1.conjugate, l2constraint)
#     solver = rr.FISTA(a)
#     solver.fit(min_its=100, debug=True)

#     ac = a.primal

#     l1c = rr.l1norm(4, bound=np.fabs(ac).sum())
#     l1c.quadratic = rr.identity_quadratic(0.5, 0, None, 0.)
#     loss = rr.quadratic.affine(X, -Y, coef=0.5)

#     p2 = rr.separable_problem.singleton(l1c, loss)
#     solver2 = rr.FISTA(p2)
#     solver2.fit()

#     print solver2.composite.coefs, ac
#     stop


def test_lasso():
    '''
    this test verifies that the l1 prox can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=2.)
    l1.quadratic = rr.identity_quadratic(0.5, 0, None, 0.)

    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10) + 3
    
    loss = rr.quadratic.affine(X, -Y, coef=0.5)

    p2 = rr.separable_problem.singleton(l1, loss)
    solver2 = rr.FISTA(p2)
    solver2.fit(tol=1.0e-12)


    f = p2.objective
    ans = scipy.optimize.fmin_powell(f, np.zeros(4), ftol=1.0e-12)

    print f(solver2.composite.coefs), f(ans)
    np.testing.assert_allclose(ans, solver2.composite.coefs, rtol=1.0e-04)

