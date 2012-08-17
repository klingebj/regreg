import numpy as np, regreg.api as rr

def test_path():
    '''
    this test looks at the paths of three different parameterizations
    of the same problem

    '''
    X = np.random.standard_normal((100,5))
    Z = np.zeros((100,10))
    Y = np.random.standard_normal(100)
    U = np.random.standard_normal((100,2))
    betaX = np.array([3,4,5,0,0])
    betaU = np.array([10,-5])
    Y += np.dot(X, betaX) + np.dot(U, betaU)
    Z[:,5:] = -X
    Z[:,:5] = X
    Z2 = np.zeros((100,8))
    Z2[:,:3] = X[:,:3]
    Z2[:,3:6] = -X[:,:3]
    Z2[:,6:] = -X[:,3:]

    lasso1 = rr.lasso.squared_error(X,Y, nstep=23)
    lasso2 = rr.lasso.squared_error(Z,Y, penalty_structure=[rr.POSITIVE_PART]*10, nstep=23)

    sol1 = lasso1.main(inner_tol=1.e-12)
    beta1 = sol1['beta'].todense()

    sol2 = lasso2.main(inner_tol=1.e-12)
    beta2 = sol2['beta'].todense()
    beta2[1:6] = beta2[1:6] - beta2[6:11]
    beta2 = beta2[:6]

    lasso3 = rr.lasso.squared_error(Z2,Y, penalty_structure=[rr.POSITIVE_PART]*6 + [rr.L1_PENALTY]*2, nstep=23)
    sol3 = lasso3.main(inner_tol=1.e-12)
    beta3 = sol3['beta'].todense()
    beta3[1:4] = beta3[1:4] - beta3[4:7]
    beta3[4:6] = - beta3[7:9]
    beta3 = beta3[:6]

    np.testing.assert_allclose(beta1, beta2)
    np.testing.assert_allclose(beta2, beta3)

def test_path_unpenalized():
    '''
    this test looks at the paths of three different parameterizations
    of the same problem

    '''
    X = np.random.standard_normal((100,5))
    U = np.random.standard_normal((100,2))
    Z = np.zeros((100,10))
    Y = np.random.standard_normal(100)
    Z[:,5:] = -X
    Z[:,:5] = X
    Z2 = np.zeros((100,8))
    Z2[:,:3] = X[:,:3]
    Z2[:,3:6] = -X[:,:3]
    Z2[:,6:] = -X[:,3:]

    lasso1 = rr.lasso.squared_error(np.hstack([X,U]),Y, penalty_structure=[rr.L1_PENALTY]*5 + [rr.UNPENALIZED]*2, nstep=23)
    lasso2 = rr.lasso.squared_error(np.hstack([Z,U]),Y, penalty_structure=[rr.POSITIVE_PART]*10 + [rr.UNPENALIZED]*2, nstep=23)

    sol1 = lasso1.main(inner_tol=1.e-12)
    beta1 = sol1['beta'].todense()

    sol2 = lasso2.main(inner_tol=1.e-12)
    beta2 = sol2['beta'].todense()
    beta2[1:6] = beta2[1:6] - beta2[6:11]
    beta2[6:8] = beta2[-2:]
    beta2 = beta2[:8]

    lasso3 = rr.lasso.squared_error(np.hstack([Z2,U]),Y, 
                                    penalty_structure=[rr.POSITIVE_PART]*6 + [rr.L1_PENALTY]*2 + [rr.UNPENALIZED]*2, nstep=23)
    sol3 = lasso3.main(inner_tol=1.e-12)
    beta3 = sol3['beta'].todense()
    beta3[1:4] = beta3[1:4] - beta3[4:7]
    beta3[4:6] = - beta3[7:9]
    beta3[6:8] = beta3[-2:]
    beta3 = beta3[:8]

    np.testing.assert_allclose(beta1, beta2)
    np.testing.assert_allclose(beta2, beta3)
