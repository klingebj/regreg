
import numpy as np

from copy import copy
import scipy.optimize

import regreg.api as rr

def test_lasso():
    '''
    this test verifies that the l1 prox can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=2.)

    l11 = rr.l1norm(4, lagrange=1.)
    l12 = rr.l1norm(4, lagrange=1.)


    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10) + 3
    
    loss = rr.quadratic.affine(X, -Y, coef=0.5)

    p1 = rr.container(l11, loss, l12)

    solver1 = rr.FISTA(p1)
    solver1.fit(tol=1.0e-12, min_its=500)

    p2 = rr.separable_problem.singleton(l1, loss)
    solver2 = rr.FISTA(p2)
    solver2.fit(tol=1.0e-12)

    f = p2.objective
    ans = scipy.optimize.fmin_powell(f, np.zeros(4), ftol=1.0e-12)
    print f(solver2.composite.coefs), f(ans)
    print f(solver1.composite.coefs), f(ans)

    np.testing.assert_allclose(ans, solver2.composite.coefs, rtol=1.0e-04)
    np.testing.assert_allclose(solver1.composite.coefs, solver2.composite.coefs, rtol=1.0e-05)
