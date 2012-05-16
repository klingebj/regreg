import numpy as np
from numpy import testing as npt
from numpy.testing import *

from scipy import sparse

import regreg.api as rr

@dec.setastest(True)
def test_logistic_counts():
    """
    Test the equivalence of binary/count specification in logistic_deviance
    """

    #Form the count version of the problem
    trials = np.random.binomial(5,0.5,100)+1
    successes = np.random.binomial(trials,0.5,len(trials)) 
    n = len(successes)
    p = 2*n
    X = np.random.normal(0,1,n*p).reshape((n,p))

    loss = rr.logistic_deviance.linear(X, successes=successes, trials=trials)
    penalty = rr.quadratic(p, coef=1.)

    prob1 = rr.container(loss, penalty)
    solver1 = rr.FISTA(prob1)
    solver1.fit()
    solution1 = solver1.composite.coefs
    
    #Form the binary version of the problem
    Ynew = []
    Xnew = []

    for i, (s,n) in enumerate(zip(successes,trials)):
        Ynew.append([1]*s + [0]*(n-s))
        for j in range(n):
            Xnew.append(X[i,:])
    Ynew = np.hstack(Ynew)
    Xnew =  np.vstack(Xnew)


    loss = rr.logistic_deviance.linear(Xnew, successes=Ynew)
    penalty = rr.quadratic(p, coef=1.)

    prob2 = rr.container(loss, penalty)
    solver2 = rr.FISTA(prob2)
    solver2.fit()
    solution2 = solver2.composite.coefs

   
    npt.assert_array_almost_equal(solution1, solution2, 3)




@dec.setastest(True)
def test_logistic_offset():
    """
    Test the equivalence of binary/count specification in logistic_likelihood
    """

    #Form the count version of the problem
    trials = np.random.binomial(5,0.5,10)+1
    successes = np.random.binomial(trials,0.5,len(trials)) 
    n = len(successes)
    p = 2*n

    X = np.hstack([np.ones((n,1)),np.random.normal(0,1,n*p).reshape((n,p))])

    loss = rr.logistic_deviance.linear(X, successes=successes, trials=trials)
    weights = np.ones(p+1)
    weights[0] = 0.
    penalty = rr.quadratic.linear(weights, coef=.1, diag=True)

    prob1 = rr.container(loss, penalty)
    solver1 = rr.FISTA(prob1)
    vals = solver1.fit(tol=1e-12)
    solution1 = solver1.composite.coefs

    diff = 0.1

    loss = rr.logistic_deviance.affine(X, successes=successes, trials=trials, offset = diff*np.ones(n))
    weights = np.ones(p+1)
    weights[0] = 0.
    penalty = rr.quadratic.linear(weights, coef=.1, diag=True)

    prob2 = rr.container(loss, penalty)
    solver2 = rr.FISTA(prob2)
    vals = solver2.fit(tol=1e-12)
    solution2 = solver2.composite.coefs

    ind = range(1,p+1)

    print solution1[range(5)]
    print solution2[range(5)]

    npt.assert_array_almost_equal(solution1[ind], solution2[ind], 3)
    npt.assert_almost_equal(solution1[0]-diff,solution2[0], 2)



