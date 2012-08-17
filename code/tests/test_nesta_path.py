import numpy as np, regreg.api as rr
X = np.random.standard_normal((100,5))
Z = np.zeros((100,10))
Y = np.random.standard_normal(100)
Z[:,5:] = -X
Z[:,:5] = X

lasso1 = rr.lasso.squared_error(X,Y, nstep=10)
lasso2 = rr.lasso.squared_error(Z,Y, positive_part=np.arange(10), nstep=10)

sol1 = lasso1.main()
beta0 = sol1['beta'].todense()[:,1]
sol2 = lasso2.main()

Z2 = np.zeros((100,8))
Z2[:,:3] = X[:,:3]
Z2[:,3:6] = -X[:,:3]
Z2[:,6:] = -X[:,3:]
lasso3 = rr.lasso.squared_error(Z2,Y, positive_part=np.arange(6), nstep=10)
sol3 = lasso3.main()

constraint_matrix = np.zeros((3,9))
constraint_matrix[2,1:6] = 1
constraint_matrix[0,6] = 1
constraint_matrix[1,7] = 1
constraint = rr.nonnegative.linear(constraint_matrix)
lasso_constraint = rr.nesta_path.squared_error(Z2, Y, constraint,
                                               epsilon=2.**(-np.arange(10)), nstep=10)
sol4 = lasso_constraint.main()
