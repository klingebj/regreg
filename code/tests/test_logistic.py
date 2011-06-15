import numpy as np
from numpy import testing as npt
from numpy.testing import *

from scipy import sparse

import regreg.api as rr

@dec.setastest(True)
def test_logistic_counts():
    """
    Test the equivalence of binary/count specification in logistic_likelihood
    """

    #Form the count version of the problem
    trials = np.random.binomial(5,0.5,100)+1
    successes = np.random.binomial(trials,0.5,len(trials)) 
    n = len(successes)
    p = 2*n
    X = np.random.normal(0,1,n*p).reshape((n,p))

    loss = rr.logistic_loglikelihood(X, successes, trials)
    penalty = rr.l2normsq(p, coef=1.)

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


    loss = rr.logistic_loglikelihood(Xnew, Ynew)
    penalty = rr.l2normsq(p, coef=1.)

    prob2 = rr.container(loss, penalty)
    solver2 = rr.FISTA(prob2)
    solver2.fit()
    solution2 = solver2.composite.coefs

   
    npt.assert_array_almost_equal(solution1, solution2, 3)
