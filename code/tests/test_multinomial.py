import numpy as np
from numpy import testing as npt
from numpy.testing import *
import regreg.api as rr

@dec.setastest(True)
def test_multinomial_vs_logistic():

    """
    Test that multinomial regression with two categories is the same as logistic regression
    """

    n = 500
    p = 10
    J = 2

    X = np.random.standard_normal(n*p).reshape((n,p))
    counts = np.random.randint(0,10,n*J).reshape((n,J)) 

    mult_x = rr.linear_transform(X, primal_shape=(p,J-1))
    loss = rr.multinomial_deviance.linear(mult_x, counts=counts)
    problem = rr.container(loss)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs1 = solver.composite.coefs

    loss = rr.logistic_deviance.linear(X, successes=counts[:,0], trials = np.sum(counts, axis=1))
    problem = rr.container(loss)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs2 = solver.composite.coefs

    loss = rr.logistic_deviance.linear(X, successes=counts[:,1], trials = np.sum(counts, axis=1))
    problem = rr.container(loss)
    solver = rr.FISTA(problem)
    solver.fit(debug=False, tol=1e-10)
    coefs3 = solver.composite.coefs

    npt.assert_equal(coefs1.shape,(p,J-1))
    npt.assert_array_almost_equal(coefs1.flatten(), coefs2.flatten(), 5)
    npt.assert_array_almost_equal(coefs1.flatten(), -coefs3.flatten(), 5)

